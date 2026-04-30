# ----------------------------------------------------------------------
# Mass-balance helpers for the `tpo` step.
#
# These mirror the Fortran subroutines in
# `yelmo/src/physics/mass_conservation.f90`:
#   - `apply_tendency!`  ↔ `apply_tendency`  (line 112)
#   - `mbal_tendency!`   ↔ `calc_G_mbal`     (line 486)
#   - `resid_tendency!`  ↔ `calc_G_boundaries` (line 609)
# All three are in-place mutators that accept and return Oceananigans
# `Field`s (or any array-like with an `interior(...)` view).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, CenterField
using Oceananigans.BoundaryConditions: fill_halo_regions!

export apply_tendency!, mbal_tendency!, resid_tendency!

# Tolerance for "ice is effectively zero". Mirrors the `TOL` parameter in
# `mass_conservation.f90` (controls the `is_equal` test inside Fortran's
# `apply_tendency`, and the explicit small-thickness zeroing).
const _APPLY_TOL = 1e-9

"""
    apply_tendency!(H_ice, mb_dot, dt; adjust_mb=false, mb_lim=9999.0) -> H_ice

Apply the per-cell mass-balance tendency `mb_dot` [m/yr] to the ice
thickness `H_ice` [m] over time interval `dt` [yr]. Per cell:

  1. Clip `mb_dot[i,j]` to `[-mb_lim, +mb_lim]`.
  2. `H_ice[i,j] += dt * mb_dot[i,j]`.
  3. Clamp `H_ice` to `≥ 0` and zero values thinner than `1e-9 m`.
  4. If `adjust_mb` is `true`, rewrite `mb_dot[i,j]` to the realized
     rate `(H_new - H_prev) / dt` so the tendency reflects what was
     actually applied (after the clamp / tolerance zeroing).

No-op when `dt ≤ 0`. Port of `apply_tendency` in
`yelmo/src/physics/mass_conservation.f90:112`.
"""
function apply_tendency!(H_ice, mb_dot, dt::Real;
                         adjust_mb::Bool = false,
                         mb_lim::Float64 = 9999.0)
    dt > 0 || return H_ice

    H = interior(H_ice)
    G = interior(mb_dot)

    @inbounds for j in axes(H, 2), i in axes(H, 1)
        H_prev = H[i, j, 1]

        # Clip the requested tendency.
        g = G[i, j, 1]
        if g >  mb_lim;  g =  mb_lim;  end
        if g < -mb_lim;  g = -mb_lim;  end

        H_new = H_prev + dt * g

        # Clamp to non-negative and zero out tiny residuals.
        if H_new < 0.0
            H_new = 0.0
        elseif abs(H_new) < _APPLY_TOL
            H_new = 0.0
        end

        H[i, j, 1] = H_new

        if adjust_mb
            G[i, j, 1] = (H_new - H_prev) / dt
        end
    end

    return H_ice
end

"""
    mbal_tendency!(G_mb, H_ice, f_grnd, mbal, dt) -> G_mb

Pre-clip a raw mass-balance forcing `mbal` [m/yr] into a realisable
tendency `G_mb` [m/yr] given the current ice state. In place:

  1. `G_mb .= mbal`.
  2. Where `G_mb < 0` and `H_ice == 0`: zero `G_mb` (no melt without ice).
  3. Where `f_grnd == 0` and `H_ice == 0`: zero `G_mb` (no accumulation
     in open ocean cells).
  4. Where `H_ice + dt * G_mb < 0`: set `G_mb = -H_ice / dt`
     (don't melt more ice than is present).

Port of `calc_G_mbal` in `yelmo/src/physics/mass_conservation.f90:486`.
"""
function mbal_tendency!(G_mb, H_ice, f_grnd, mbal, dt::Real)
    G = interior(G_mb)
    H = interior(H_ice)
    Fg = interior(f_grnd)
    M = interior(mbal)

    @inbounds for j in axes(G, 2), i in axes(G, 1)
        g = M[i, j, 1]
        h = H[i, j, 1]
        fg = Fg[i, j, 1]

        # No melt where there is no ice.
        if g < 0.0 && h == 0.0
            g = 0.0
        end

        # No accumulation in open ocean (un-grounded, ice-free cells).
        if fg == 0.0 && h == 0.0
            g = 0.0
        end

        # Don't over-melt the available ice in this step.
        if dt != 0.0 && (h + dt * g) < 0.0
            g = -h / dt
        end

        G[i, j, 1] = g
    end

    return G_mb
end

"""
    resid_tendency!(G_resid, H_ice, f_ice, f_grnd, ice_allowed,
                    H_min_flt, H_min_grnd, dt) -> G_resid

Compute the residual mass-balance tendency `G_resid` [m/yr] that
removes ice in cells where the dynamic + SMB update has produced an
unphysical configuration (margins below the min-thickness threshold,
isolated ice islands, margins thicker than their neighbors, ice in
disallowed cells). The tendency is signed so that
`apply_tendency!(H_ice, G_resid, dt)` would realise the cleanup.

The factor of `1.1` on the rate is intentional: a downstream call to
`apply_tendency!(adjust_mb=true)` clips the actual delta back to what
was realised after the non-negativity clamp, so a slight overshoot in
the input tendency produces the desired final state.

Halo handling: a temporary `CenterField` (with the same boundary
conditions as `H_ice`) is used as a per-sweep snapshot. After each
sweep modifies `H_new`, the snapshot is refreshed and its halos
re-filled — so margin / island detection at boundaries respects the
field's BCs (Dirichlet zero on `H_ice` → halo = `-first_interior`,
distinct from 0 unless interior is itself 0; Periodic axes wrap).

Port of `calc_G_boundaries` in
`yelmo/src/physics/mass_conservation.f90:609`. The EISMINT-summit
averaging block (lines 651-658) and the trailing per-BC border-
zeroing switch (lines 761-805) are dropped — Oceananigans' BC
machinery now handles boundary semantics.
"""
function resid_tendency!(G_resid, H_ice, f_ice, f_grnd, ice_allowed,
                         H_min_flt::Real, H_min_grnd::Real,
                         dt::Real)
    H_in = interior(H_ice)
    Fi   = interior(f_ice)
    Fg   = interior(f_grnd)
    Ia   = interior(ice_allowed)
    G    = interior(G_resid)

    nx = size(H_in, 1)
    ny = size(H_in, 2)

    H_min_tol = 1e-6

    # Working state: `H_new` holds the in-progress thickness; `H_tmp`
    # is a per-sweep snapshot Field with halos. Allocating `H_tmp`
    # once and refilling it per sweep avoids allocating four temp
    # Matrices the way the old `copy(H_new)` pattern did, while
    # keeping halo-aware reads consistent with `H_ice`'s BCs.
    H_new   = copy(H_in)
    H_tmp_f = CenterField(H_ice.grid; boundary_conditions=H_ice.boundary_conditions)
    H_tmp   = interior(H_tmp_f)

    # ---- 1. Disallow ice where the boundary mask says so. -----------
    @inbounds for j in 1:ny, i in 1:nx
        if Ia[i, j, 1] == 0.0
            H_new[i, j, 1] = 0.0
        end
    end

    # ---- 2. Margin too-thin removal + sub-tolerance cleanup. --------
    H_tmp .= H_new
    fill_halo_regions!(H_tmp_f)
    @inbounds for j in 1:ny, i in 1:nx
        h_here = H_tmp[i, j, 1]

        # 2a. Sub-tolerance cells unconditionally zeroed.
        if h_here < H_min_tol
            H_new[i, j, 1] = 0.0
            continue
        end

        # 2b. Margin test against the snapshot.
        nW = H_tmp_f[i-1, j,   1]
        nE = H_tmp_f[i+1, j,   1]
        nS = H_tmp_f[i,   j-1, 1]
        nN = H_tmp_f[i,   j+1, 1]

        is_margin = (nW == 0.0) | (nE == 0.0) |
                    (nS == 0.0) | (nN == 0.0)

        if is_margin
            f_here  = Fi[i, j, 1]
            H_eff   = f_here > 0.0 ? h_here / f_here : h_here
            fg_here = Fg[i, j, 1]
            if fg_here == 0.0 && H_eff < H_min_flt
                H_new[i, j, 1] = 0.0
            elseif fg_here > 0.0 && H_eff < H_min_grnd
                H_new[i, j, 1] = 0.0
            end
        end
    end

    # ---- 3. Island removal (cell with H>0 but every neighbor 0). ----
    H_tmp .= H_new
    fill_halo_regions!(H_tmp_f)
    @inbounds for j in 1:ny, i in 1:nx
        h_here = H_tmp[i, j, 1]
        h_here > 0.0 || continue

        nW = H_tmp_f[i-1, j,   1]
        nE = H_tmp_f[i+1, j,   1]
        nS = H_tmp_f[i,   j-1, 1]
        nN = H_tmp_f[i,   j+1, 1]

        is_island = (nW == 0.0) & (nE == 0.0) &
                    (nS == 0.0) & (nN == 0.0)

        if is_island
            H_new[i, j, 1] = 0.0
        end
    end

    # ---- 4. Cap margin cells to the max thickness of neighbors. -----
    H_tmp .= H_new
    fill_halo_regions!(H_tmp_f)
    @inbounds for j in 1:ny, i in 1:nx
        h_here = H_tmp[i, j, 1]
        h_here > 0.0 || continue

        nW = H_tmp_f[i-1, j,   1]
        nE = H_tmp_f[i+1, j,   1]
        nS = H_tmp_f[i,   j-1, 1]
        nN = H_tmp_f[i,   j+1, 1]

        is_margin = (nW == 0.0) | (nE == 0.0) |
                    (nS == 0.0) | (nN == 0.0)

        if is_margin
            f_here = Fi[i, j, 1]
            H_eff  = f_here > 0.0 ? h_here / f_here : h_here
            H_max  = max(nW, nE, nS, nN)
            if H_eff > H_max
                H_new[i, j, 1] = H_max
            end
        end
    end

    # ---- 5. Convert the realized H delta to a tendency. -------------
    if dt != 0.0
        inv_dt = 1.0 / dt
        @inbounds for j in 1:ny, i in 1:nx
            G[i, j, 1] = 1.1 * (H_new[i, j, 1] - H_in[i, j, 1]) * inv_dt
        end
    else
        fill!(G, 0.0)
    end

    return G_resid
end
