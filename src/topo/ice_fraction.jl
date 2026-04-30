# ----------------------------------------------------------------------
# Ice-area-fraction calculation `f_ice`.
#
# The fractional CISM-style version handles partially-covered cells at
# the floating margin: `f_ice = H_ice / H_eff` where `H_eff` is the
# minimum thickness over fully-covered upstream neighbours. Grounded
# margins keep the binary `f_ice = 1.0` convention (no sub-grid for
# grounded ice). Toggle via `flt_subgrid`.
#
# Port of `physics/topography.f90:369 calc_ice_fraction`. The
# `boundaries` argument is implicit — halo handling is driven by
# `H_ice`'s grid topology + BCs.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_f_ice!

# Tolerance below which `f_ice` is snapped to a small positive value
# to avoid degenerate sub-grid arithmetic. Mirrors `TOL` in
# `yelmo_defs.f90`.
const _F_ICE_TOL = 1e-5

# When the floating-margin sub-grid path runs but a cell has no
# upstream fully-covered neighbour, the fortran kernel falls back on
# `H_eff = max(H_ice, H_lim)` with `H_lim = 100 m`. This protects against
# unbounded `f_ice = H_ice / H_eff` at isolated thin shelf cells.
const _H_LIM = 100.0

"""
    calc_f_ice!(f_ice, H_ice, z_bed, z_sl, rho_ice, rho_sw;
                flt_subgrid = false) -> f_ice

Compute the per-cell ice-area-fraction `f_ice ∈ [0, 1]`:

  - `H_ice == 0`           → `f_ice = 0`
  - `H_ice > 0`, no margin → `f_ice = 1`
  - `H_ice > 0`, grounded margin → `f_ice = 1`         (CISM convention)
  - `H_ice > 0`, **floating** margin and `flt_subgrid = true` →
    `f_ice = H_ice / H_eff`, where `H_eff` is the minimum thickness
    over fully-ice-covered direct neighbours (Fortran's
    `calc_ice_fraction`). With `flt_subgrid = false` (default), the
    floating-margin branch also returns `f_ice = 1` — i.e. the binary
    fallback that matches the current default `ytopo.margin_flt_subgrid`.

`flt_subgrid = false` reproduces the previous binary behaviour. The
fractional path is enabled by setting `ytopo.margin_flt_subgrid = true`.

Halo handling: `H_ice` halos are filled via `fill_halo_regions!` so
neighbour reads honour the grid's topology + boundary conditions.

Port of `physics/topography.f90:369 calc_ice_fraction` (the active
variant, not `calc_ice_fraction_new`).
"""
function calc_f_ice!(f_ice, H_ice, z_bed, z_sl,
                     rho_ice::Real, rho_sw::Real;
                     flt_subgrid::Bool = false)
    F  = interior(f_ice)
    H  = interior(H_ice)
    Zb = interior(z_bed)
    Zs = interior(z_sl)

    nx = size(F, 1)
    ny = size(F, 2)

    # Phase 1: binary initialisation.
    @inbounds for j in 1:ny, i in 1:nx
        F[i, j, 1] = H[i, j, 1] > 0.0 ? 1.0 : 0.0
    end

    flt_subgrid || return f_ice

    # Phase 2: compute the (use_f_ice = false) flotation diagnostic
    # locally — Fortran calls calc_H_grnd with use_f_ice=FALSE to get
    # `H_grnd = H_eff_raw - rho_sw/rho_ice·max(z_sl-z_bed, 0)` with
    # `H_eff_raw = H_ice` (no margin scaling). Stored on a temporary
    # 2D Matrix; only used inside this function.
    rho_sw_ice = rho_sw / rho_ice
    H_grnd = Matrix{Float64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        depth = Zs[i, j, 1] - Zb[i, j, 1]
        H_grnd[i, j] = if depth > 0.0
            H[i, j, 1] - rho_sw_ice * depth
        else
            H[i, j, 1] + (Zb[i, j, 1] - Zs[i, j, 1])
        end
    end

    # Phase 3: count direct (4-conn) ice-covered neighbours per cell.
    # `n_ice` is later read at margin cells to decide which neighbours
    # are "fully interior" (n_ice == 4) and so reliable upstream
    # thickness sources.
    fill_halo_regions!(H_ice)
    n_ice = Matrix{Int}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        n = 0
        H_ice[i-1, j,   1] > 0.0 && (n += 1)
        H_ice[i+1, j,   1] > 0.0 && (n += 1)
        H_ice[i,   j-1, 1] > 0.0 && (n += 1)
        H_ice[i,   j+1, 1] > 0.0 && (n += 1)
        n_ice[i, j] = n
    end

    # Phase 4: margin-cell sub-grid f_ice. Read `n_ice` neighbours from
    # the local matrix; out-of-bounds defaults to 0 (matches "no ice
    # there" for the upstream filter).
    n_ice_or_zero(i, j) =
        (1 <= i <= nx && 1 <= j <= ny) ? n_ice[i, j] : 0

    @inbounds for j in 1:ny, i in 1:nx
        h = H[i, j, 1]
        h > 0.0 || continue
        n = n_ice[i, j]
        n == 4 && continue   # fully-covered interior cell — f_ice already 1

        if n == 0
            # Isolated island — assume fully covered (Fortran: ensures
            # the cell stays dynamically active).
            F[i, j, 1] = 1.0
            continue
        end

        # Direct neighbours. Halo reads of H_ice are valid here.
        H_neighb = (H_ice[i-1, j,   1],
                    H_ice[i+1, j,   1],
                    H_ice[i,   j-1, 1],
                    H_ice[i,   j+1, 1])
        n_neighb = (n_ice_or_zero(i-1, j),
                    n_ice_or_zero(i+1, j),
                    n_ice_or_zero(i,   j-1),
                    n_ice_or_zero(i,   j+1))

        if H_grnd[i, j] <= 0.0
            # Floating margin — sub-grid f_ice from upstream H_eff.
            H_eff = Inf
            n_now = 0
            for k in 1:4
                if H_neighb[k] > 0.0 && n_neighb[k] == 4
                    n_now += 1
                    H_eff = min(H_eff, H_neighb[k])
                end
            end
            if n_now == 0
                # No fully-covered upstream neighbour — fall back to
                # H_lim so f_ice doesn't blow up on isolated thin
                # shelf cells.
                H_eff = max(h, _H_LIM)
            end
        else
            # Grounded margin — Fortran sets H_eff = H_ice (CISM convention,
            # no sub-grid on grounded margins).
            H_eff = h
        end

        # Cell ice fraction from volume conservation: f = H_ice / H_eff,
        # clamped to (TOL, 1].
        if H_eff > 0.0
            f = h / H_eff
            f = min(f, 1.0)
            f < _F_ICE_TOL && (f = _F_ICE_TOL)
            F[i, j, 1] = f
        else
            F[i, j, 1] = 1.0
        end
    end

    return f_ice
end

# Convenience dispatch: refresh `tpo.f_ice` from a `YelmoModel`'s
# current state, threading through the constants and the
# `margin_flt_subgrid` parameter. Used inside `topo_step!` /
# `calving_step!` to keep the per-phase f_ice refresh calls compact.
calc_f_ice!(y::YelmoModel) =
    calc_f_ice!(y.tpo.f_ice, y.tpo.H_ice,
                y.bnd.z_bed, y.bnd.z_sl,
                y.c.rho_ice, y.c.rho_sw;
                flt_subgrid = y.p.ytopo.margin_flt_subgrid)

