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
# All distances are returned in **metres**. The threshold parameter
# `dist_grz` (consumed by `calc_grounding_line_zone!`) lives in km in
# the namelist; the conversion to metres happens at the call site.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_distance_to_grounding_line!, calc_distance_to_ice_margin!

const _DIAG_WEIGHT = sqrt(2.0)

# Out-of-bounds source-fraction read. `:zero` returns 0 (halo is
# phantom ocean / non-ice — boundary cells with positive fraction
# are sources). `:mirror` clamps to the nearest interior cell
# (zero-flux Neumann — boundary cells inherit the interior value).
@inline function _f_halo(F::AbstractMatrix, i::Int, j::Int,
                         nx::Int, ny::Int, bc::Symbol)
    (1 <= i <= nx && 1 <= j <= ny) && return @inbounds F[i, j]
    bc === :zero   && return 0.0
    bc === :mirror && return @inbounds F[clamp(i, 1, nx), clamp(j, 1, ny)]
    error("calc_distance_*: bc=:$(bc) not supported (use :zero or :mirror)")
end

# Out-of-bounds distance read during the chamfer sweep. `:zero`
# returns Inf (halo is opaque — no propagation through it).
# `:mirror` clamps (boundary acts like a zero-flux mirror).
@inline function _d_halo(D::AbstractMatrix, i::Int, j::Int,
                         nx::Int, ny::Int, bc::Symbol)
    (1 <= i <= nx && 1 <= j <= ny) && return @inbounds D[i, j]
    bc === :zero   && return Inf
    bc === :mirror && return @inbounds D[clamp(i, 1, nx), clamp(j, 1, ny)]
    error("calc_distance_*: bc=:$(bc) not supported (use :zero or :mirror)")
end

"""
    calc_distance_to_grounding_line!(dist_gl, f_grnd, dx; bc=:zero) -> dist_gl

Signed distance from each cell to the nearest grounding-line cell, in
**metres**. Source cells are grounded (`f_grnd > 0`) cells with at
least one *direct* (4-conn) floating neighbour and have distance 0.
Grounded non-source cells get positive distances; floating cells get
negative distances (Fortran convention).

Algorithm: two-pass forward/backward chamfer-distance transform with
stencil weights `{dx, √2·dx}` on the 8-neighbour stencil. O(N), exact
match to the iteration-converged Fortran routine.

`bc` selects the boundary mode for out-of-bounds reads:

  - `:zero`   — halo source = 0 (phantom ocean), halo distance = Inf
                (opaque boundary — no propagation through halo).
                Grounded edge cells become GL sources.
  - `:mirror` — halo = nearest interior cell (zero-flux Neumann).
                Grounded edge cells stay non-GL if the interior
                neighbour is also grounded. Matches Fortran's
                `infinite` boundary mode.

Port of `physics/topography.f90:1383 calc_distance_to_grounding_line`.
"""
function calc_distance_to_grounding_line!(dist_gl, f_grnd, dx::Real;
                                          bc::Symbol = :zero)
    D = @view interior(dist_gl)[:, :, 1]
    F = @view interior(f_grnd)[:, :, 1]
    _chamfer_signed_distance!(D, F, Float64(dx), bc)
    return dist_gl
end

"""
    calc_distance_to_ice_margin!(dist_mrgn, f_ice, dx; bc=:zero) -> dist_mrgn

Signed distance from each cell to the nearest ice-margin cell, in
**metres**. Source cells are ice-covered (`f_ice > 0`) cells with at
least one direct non-ice neighbour. The Fortran routine wraps
`calc_distance_to_grounding_line` with `f_ice` swapped for `f_grnd`,
and so does this one.

`bc` keyword as in `calc_distance_to_grounding_line!`.

Port of `physics/topography.f90:1362 calc_distance_to_ice_margin`.
"""
function calc_distance_to_ice_margin!(dist_mrgn, f_ice, dx::Real;
                                      bc::Symbol = :zero)
    D = @view interior(dist_mrgn)[:, :, 1]
    F = @view interior(f_ice)[:, :, 1]
    _chamfer_signed_distance!(D, F, Float64(dx), bc)
    return dist_mrgn
end

# Shared chamfer-distance-to-zero-set kernel. Operates on plain 2D
# arrays. Public wrappers above are responsible for unwrapping
# Oceananigans Field interiors.
#
# Output convention:
#   - `F[i,j] > 0` and a direct neighbour has `F == 0`: D = 0 (source)
#   - `F[i,j] > 0`, no zero direct neighbour:           D = +chamfer
#   - `F[i,j] == 0`:                                    D = -chamfer
function _chamfer_signed_distance!(D::AbstractMatrix, F::AbstractMatrix,
                                   dx::Float64, bc::Symbol)
    nx, ny = size(D)
    @assert size(F) == (nx, ny)
    bc in (:zero, :mirror) ||
        error("calc_distance_*: bc=:$(bc) not supported (use :zero or :mirror)")

    diag_dx = _DIAG_WEIGHT * dx

    # Phase 1: source detection. Initialize source cells to 0 and
    # everything else to +Inf; the sign flip for non-source cells
    # happens in phase 3.
    @inbounds for j in 1:ny, i in 1:nx
        if F[i, j] > 0.0
            f_w = _f_halo(F, i-1, j,   nx, ny, bc)
            f_e = _f_halo(F, i+1, j,   nx, ny, bc)
            f_s = _f_halo(F, i,   j-1, nx, ny, bc)
            f_n = _f_halo(F, i,   j+1, nx, ny, bc)
            D[i, j] = (f_w == 0.0 || f_e == 0.0 || f_s == 0.0 || f_n == 0.0) ?
                      0.0 : Inf
        else
            D[i, j] = Inf
        end
    end

    # Phase 2a: forward chamfer pass. Stencil reads cells already
    # visited this pass (j-1 row in full, plus (i-1, j) on current row).
    @inbounds for j in 1:ny, i in 1:nx
        d = D[i, j]
        d = min(d, _d_halo(D, i-1, j-1, nx, ny, bc) + diag_dx)
        d = min(d, _d_halo(D, i,   j-1, nx, ny, bc) + dx)
        d = min(d, _d_halo(D, i+1, j-1, nx, ny, bc) + diag_dx)
        d = min(d, _d_halo(D, i-1, j,   nx, ny, bc) + dx)
        D[i, j] = d
    end

    # Phase 2b: backward chamfer pass. Reverse iteration order;
    # stencil reads the (j+1) row plus (i+1, j).
    @inbounds for j in ny:-1:1, i in nx:-1:1
        d = D[i, j]
        d = min(d, _d_halo(D, i+1, j,   nx, ny, bc) + dx)
        d = min(d, _d_halo(D, i-1, j+1, nx, ny, bc) + diag_dx)
        d = min(d, _d_halo(D, i,   j+1, nx, ny, bc) + dx)
        d = min(d, _d_halo(D, i+1, j+1, nx, ny, bc) + diag_dx)
        D[i, j] = d
    end

    # Phase 3: sign flip for non-source cells (`F == 0`).
    @inbounds for j in 1:ny, i in 1:nx
        if F[i, j] == 0.0
            D[i, j] = -D[i, j]
        end
    end

    return D
end
