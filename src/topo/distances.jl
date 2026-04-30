# ----------------------------------------------------------------------
# Distance-to-feature kernels for `tpo` diagnostics.
#
#   - `calc_distance_to_grounding_line!` — signed distance to the
#     nearest grounding line, in metres.
#   - `calc_distance_to_ice_margin!`     — signed distance to the
#     nearest ice margin, in metres.
#
# Both use a single shared two-pass forward/backward chamfer-distance
# transform with weights `{dx, √2·dx}` on the 8-neighbour stencil
# (Rosenfeld-Pfaltz 1966, Borgefors 1986). O(N) — same numerical
# output as Fortran `physics/topography.f90:1383
# calc_distance_to_grounding_line` in the iteration-converged limit,
# without the up-to-1000-pass relaxation.
#
# Boundary conditions are inherited from the input field's grid
# topology + per-side BCs (default Neumann-zero clamp on Bounded sides;
# automatic wrap on Periodic sides). Halos are filled via
# `fill_halo_regions!`.
#
# All distances are returned in **metres**. The threshold parameter
# `dist_grz` (consumed by `calc_grounding_line_zone!`) lives in km in
# the namelist; the conversion to metres happens at the call site.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, CenterField
using Oceananigans.BoundaryConditions: fill_halo_regions!

using ..YelmoCore: fill_corner_halos!

export calc_distance_to_grounding_line!, calc_distance_to_ice_margin!

const _DIAG_WEIGHT = sqrt(2.0)

"""
    calc_distance_to_grounding_line!(dist_gl, f_grnd, dx) -> dist_gl

Signed distance from each cell to the nearest grounding-line cell, in
**metres**. Source cells are grounded (`f_grnd > 0`) cells with at
least one *direct* (4-conn) floating neighbour and have distance 0.
Grounded non-source cells get positive distances; floating cells get
negative distances (Fortran convention).

Algorithm: two-pass forward/backward chamfer-distance transform with
stencil weights `{dx, √2·dx}` on the 8-neighbour stencil. O(N).

Boundary handling: out-of-domain neighbour reads use `f_grnd`'s grid
topology + boundary conditions. With the default Neumann-zero ("clamp")
BC on Bounded sides, edge cells inherit the first interior cell's
value — so a grounded cell on the domain edge does *not* spuriously
become a GL source (matches Fortran's `infinite` boundary mode).
Periodic axes wrap automatically.

Port of `physics/topography.f90:1383 calc_distance_to_grounding_line`.
"""
function calc_distance_to_grounding_line!(dist_gl, f_grnd, dx::Real)
    fill_halo_regions!(f_grnd)
    fill_corner_halos!(f_grnd)
    _chamfer_signed_distance!(dist_gl, f_grnd, Float64(dx))
    return dist_gl
end

"""
    calc_distance_to_ice_margin!(dist_mrgn, f_ice, dx) -> dist_mrgn

Signed distance from each cell to the nearest ice-margin cell, in
**metres**. Source cells are ice-covered (`f_ice > 0`) cells with at
least one direct non-ice neighbour. The Fortran routine wraps
`calc_distance_to_grounding_line` with `f_ice` swapped for `f_grnd`,
and so does this one — including boundary handling, which inherits
from `f_ice`'s grid topology and BCs.

Port of `physics/topography.f90:1362 calc_distance_to_ice_margin`.
"""
function calc_distance_to_ice_margin!(dist_mrgn, f_ice, dx::Real)
    fill_halo_regions!(f_ice)
    fill_corner_halos!(f_ice)
    _chamfer_signed_distance!(dist_mrgn, f_ice, Float64(dx))
    return dist_mrgn
end

# Shared chamfer-distance-to-zero-set kernel. Reads halos from the
# source field `F` (already filled by the public wrapper) and from the
# distance field `D` between sweeps (filled by `fill_halo_regions!(D)`).
#
# Output convention:
#   - `F[i,j] > 0` and a direct neighbour has `F == 0`: D = 0 (source)
#   - `F[i,j] > 0`, no zero direct neighbour:           D = +chamfer
#   - `F[i,j] == 0`:                                    D = -chamfer
#
# For periodic axes, the 2-pass FB chamfer can leave the wrap-around
# path unresolved if a source is near one edge and the target near the
# other. We do a third (FB) round-trip to capture one wrap; further
# wraps would need iteration to convergence (rare in practice for
# ice-sheet domains). Add explicit iteration if needed.
function _chamfer_signed_distance!(D::Field, F::Field, dx::Float64)
    Di = interior(D)
    Fi = interior(F)
    nx, ny = size(Di, 1), size(Di, 2)
    @assert size(Fi, 1) == nx && size(Fi, 2) == ny

    diag_dx = _DIAG_WEIGHT * dx

    # Phase 1: source detection. Initialise source cells to 0 and
    # everything else to +Inf. Halo reads of `F` resolve via topology
    # + BC (clamp by default on Bounded; wrap on Periodic).
    @inbounds for j in 1:ny, i in 1:nx
        if Fi[i, j, 1] > 0.0
            f_w = F[i-1, j,   1]
            f_e = F[i+1, j,   1]
            f_s = F[i,   j-1, 1]
            f_n = F[i,   j+1, 1]
            Di[i, j, 1] = (f_w == 0.0 || f_e == 0.0 ||
                           f_s == 0.0 || f_n == 0.0) ? 0.0 : Inf
        else
            Di[i, j, 1] = Inf
        end
    end

    # Phases 2a+2b: forward then backward chamfer. Refresh both the
    # orthogonal halos and the corner halos before each pass — the
    # 8-stencil reads include diagonals which `fill_halo_regions!`
    # alone leaves unfilled on Bounded sides.
    fill_halo_regions!(D); fill_corner_halos!(D)
    _chamfer_forward!(D, dx, diag_dx, nx, ny)

    fill_halo_regions!(D); fill_corner_halos!(D)
    _chamfer_backward!(D, dx, diag_dx, nx, ny)

    # One additional FB round to resolve potential wrap-around paths
    # on Periodic axes. On a fully Bounded domain this round is a
    # provable no-op (the first FB already converged) — about a 2x
    # cost on Bounded grids in exchange for correctness on Periodic.
    fill_halo_regions!(D); fill_corner_halos!(D)
    _chamfer_forward!(D, dx, diag_dx, nx, ny)

    fill_halo_regions!(D); fill_corner_halos!(D)
    _chamfer_backward!(D, dx, diag_dx, nx, ny)

    # Phase 3: sign flip for non-source cells (`F == 0`).
    @inbounds for j in 1:ny, i in 1:nx
        if Fi[i, j, 1] == 0.0
            Di[i, j, 1] = -Di[i, j, 1]
        end
    end

    return D
end

# Forward chamfer pass. Stencil reads cells already visited this pass
# (the (j-1)-row plus (i-1, j) on the current row).
function _chamfer_forward!(D::Field, dx::Float64, diag_dx::Float64,
                           nx::Int, ny::Int)
    Di = interior(D)
    @inbounds for j in 1:ny, i in 1:nx
        d = Di[i, j, 1]
        d = min(d, D[i-1, j-1, 1] + diag_dx)
        d = min(d, D[i,   j-1, 1] + dx)
        d = min(d, D[i+1, j-1, 1] + diag_dx)
        d = min(d, D[i-1, j,   1] + dx)
        Di[i, j, 1] = d
    end
end

# Backward chamfer pass. Reverse iteration order; stencil reads the
# (j+1) row plus (i+1, j).
function _chamfer_backward!(D::Field, dx::Float64, diag_dx::Float64,
                            nx::Int, ny::Int)
    Di = interior(D)
    @inbounds for j in ny:-1:1, i in nx:-1:1
        d = Di[i, j, 1]
        d = min(d, D[i+1, j,   1] + dx)
        d = min(d, D[i-1, j+1, 1] + diag_dx)
        d = min(d, D[i,   j+1, 1] + dx)
        d = min(d, D[i+1, j+1, 1] + diag_dx)
        Di[i, j, 1] = d
    end
end
