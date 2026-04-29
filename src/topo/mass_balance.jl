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

using Oceananigans.Fields: interior

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

# Saturated lookup: returns `H[i,j,1]` if (i,j) is in bounds, else 0.0.
# Treats out-of-domain neighbors as ice-free, matching the spec's
# "grid handles BC" simplification of the Fortran neighbor lookup.
@inline function _h_or_zero(H::AbstractArray, i::Int, j::Int, nx::Int, ny::Int)
    return (1 <= i <= nx && 1 <= j <= ny) ? H[i, j, 1] : 0.0
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

Port of `calc_G_boundaries` in
`yelmo/src/physics/mass_conservation.f90:609`. Simplifications vs the
Fortran:

  - The EISMINT-summit averaging block (lines 651-658) is dropped.
  - The trailing per-BC border-zeroing switch (lines 761-805) is
    dropped — Oceananigans' Dirichlet halo on `H_ice` already enforces
    `H = 0` outside the domain. Out-of-domain neighbors are read as 0
    via `_h_or_zero` so margin / island detection is consistent with
    the BC.

`f_ice` is read for `calc_H_eff = H_ice / f_ice` at margins.
"""
function resid_tendency!(G_resid, H_ice, f_ice, f_grnd, ice_allowed,
                         # `H_ice_ref` was previously here. The Fortran
                         # `calc_G_boundaries` consumes it only in the
                         # `boundaries == "fixed"` branch of the per-BC
                         # border-zeroing switch (lines 787-793 of
                         # mass_conservation.f90), which we drop here in
                         # favor of Oceananigans' grid-level halo
                         # handling. Re-add the argument when (if) that
                         # BC mode is reintroduced.
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

    # Working copy of H. We mutate `H_new` through three sweeps; each
    # sweep reads from a snapshot `H_tmp` of the previous state.
    H_new = copy(H_in)

    # ---- 1. Disallow ice where the boundary mask says so. -----------
    @inbounds for j in 1:ny, i in 1:nx
        if Ia[i, j, 1] == 0.0
            H_new[i, j, 1] = 0.0
        end
    end

    # ---- 2. Margin too-thin removal + sub-tolerance cleanup. --------
    H_tmp = copy(H_new)
    @inbounds for j in 1:ny, i in 1:nx
        h_here = H_tmp[i, j, 1]

        # 2a. Sub-tolerance cells unconditionally zeroed.
        if h_here < H_min_tol
            H_new[i, j, 1] = 0.0
            continue
        end

        # 2b. Margin test against the snapshot.
        nW = _h_or_zero(H_tmp, i-1, j,   nx, ny)
        nE = _h_or_zero(H_tmp, i+1, j,   nx, ny)
        nS = _h_or_zero(H_tmp, i,   j-1, nx, ny)
        nN = _h_or_zero(H_tmp, i,   j+1, nx, ny)

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
    H_tmp = copy(H_new)
    @inbounds for j in 1:ny, i in 1:nx
        h_here = H_tmp[i, j, 1]
        h_here > 0.0 || continue

        nW = _h_or_zero(H_tmp, i-1, j,   nx, ny)
        nE = _h_or_zero(H_tmp, i+1, j,   nx, ny)
        nS = _h_or_zero(H_tmp, i,   j-1, nx, ny)
        nN = _h_or_zero(H_tmp, i,   j+1, nx, ny)

        is_island = (nW == 0.0) & (nE == 0.0) &
                    (nS == 0.0) & (nN == 0.0)

        if is_island
            H_new[i, j, 1] = 0.0
        end
    end

    # ---- 4. Cap margin cells to the max thickness of neighbors. -----
    H_tmp = copy(H_new)
    @inbounds for j in 1:ny, i in 1:nx
        h_here = H_tmp[i, j, 1]
        h_here > 0.0 || continue

        nW = _h_or_zero(H_tmp, i-1, j,   nx, ny)
        nE = _h_or_zero(H_tmp, i+1, j,   nx, ny)
        nS = _h_or_zero(H_tmp, i,   j-1, nx, ny)
        nN = _h_or_zero(H_tmp, i,   j+1, nx, ny)

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
