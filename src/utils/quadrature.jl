# ----------------------------------------------------------------------
# Gauss-Legendre tensor-product quadrature on a 2D reference square.
#
# Mirrors the Fortran `gq2D_class` / `gq2D_init` defined in
# `fesm-utils/utils/src/gaussian_quadrature.f90`. The 2-point
# Gauss-Legendre rule places 4 quadrature nodes on the reference
# square `[-1, 1]²` at
#
#     (xr, yr) ∈ {(±1/√3, ±1/√3)}                      (counter-clockwise
#                                                       from SW corner)
#
# all with weight 1.0 and total weight `wt_tot = 4.0`. This rule is
# exact for bilinear functions on the square (degree 3 in each
# variable separately).
#
# Backed by [FastGaussQuadrature.jl] — `gausslegendre(n)` returns the
# n-point 1D Gauss-Legendre nodes and weights on `[-1, 1]`, which we
# tensor-product into 2D.
#
# Originally lived in `src/dyn/quadrature.jl`. Moved to `src/utils/`
# in PR7-cleanup so `dyn`, `mat`, and `thrm` can all consume them
# without an `dyn ← thrm` reverse-import smell.
#
# Used by:
#   - dyn:  basal-drag (`calc_beta_aa_*`), viscosity
#           (`calc_visc_eff_3D_nodes!`), uz (`_calc_uz_3D_kernel!`).
#   - thrm: basal frictional heating (`calc_basal_heating_nodes!`).
# ----------------------------------------------------------------------

using FastGaussQuadrature: gausslegendre

# Const NTuple{4,Float64} tables for the 2-point Gauss-Legendre rule
# in counter-clockwise corner order — see `gq2d_nodes_2pt` below for
# the contract. Exposing these as `const` Julia globals avoids the
# per-call `Vector{Float64}` allocations that `gq2d_nodes(2)` performs.
const _R_GQ2_2PT     = 1.0 / sqrt(3.0)
const _XR_GQ2_2PT    = (-_R_GQ2_2PT, +_R_GQ2_2PT, +_R_GQ2_2PT, -_R_GQ2_2PT)
const _YR_GQ2_2PT    = (-_R_GQ2_2PT, -_R_GQ2_2PT, +_R_GQ2_2PT, +_R_GQ2_2PT)
const _WT_GQ2_2PT    = (1.0, 1.0, 1.0, 1.0)
const _WTTOT_GQ2_2PT = 4.0

"""
    gq2d_nodes(n::Int = 2)
        -> (xr::Vector{Float64}, yr::Vector{Float64},
            wt::Vector{Float64}, wt_tot::Float64)

n-point Gauss-Legendre tensor-product quadrature on the reference
square `[-1, 1]²`. Returns the 2D node coordinates, weights, and
total weight. For `n = 2` (Yelmo's default — exact for bilinear
functions on the square) returns 4 nodes ordered counter-clockwise
from the SW corner:

    xr = [-r,  +r,  +r,  -r]
    yr = [-r,  -r,  +r,  +r]
    wt = [ 1,   1,   1,   1]
    wt_tot = 4

with `r = 1/√3`. Matches the Fortran `gq2D_init` convention in
`fesm-utils/utils/src/gaussian_quadrature.f90`.

For `n > 2`, the tensor product gives `n²` nodes, ordered by
column-major scan over the 1D nodes — i.e. the j-th 1D y-node sweeps
over all i 1D x-nodes (`p = (j-1) * n + i`). This is *not* the
counter-clockwise order; only the 2-point rule has the
counter-clockwise ordering match the column-major scan because each
"row" has only two nodes. Most Yelmo physics call sites assume `n = 2`
and the counter-clockwise corner order — extend with care.
"""
function gq2d_nodes(n::Int = 2)
    n ≥ 1 || error("gq2d_nodes: n must be ≥ 1; got $n")

    nodes_1d, weights_1d = gausslegendre(n)

    if n == 2
        # Match Fortran's counter-clockwise corner ordering exactly.
        # nodes_1d is sorted ascending: [-1/√3, +1/√3].
        r = nodes_1d[2]   # +1/√3
        xr = [-r, +r, +r, -r]
        yr = [-r, -r, +r, +r]
        wt = [weights_1d[1] * weights_1d[1],   # SW
              weights_1d[2] * weights_1d[1],   # SE
              weights_1d[2] * weights_1d[2],   # NE
              weights_1d[1] * weights_1d[2]]   # NW
        wt_tot = sum(wt)
        return xr, yr, wt, wt_tot
    end

    # General n: column-major tensor product (j outer, i inner).
    npts = n * n
    xr = Vector{Float64}(undef, npts)
    yr = Vector{Float64}(undef, npts)
    wt = Vector{Float64}(undef, npts)
    @inbounds for j in 1:n, i in 1:n
        p = (j - 1) * n + i
        xr[p] = nodes_1d[i]
        yr[p] = nodes_1d[j]
        wt[p] = weights_1d[i] * weights_1d[j]
    end
    return xr, yr, wt, sum(wt)
end

"""
    gq2d_nodes_2pt()
        -> (xr::NTuple{4,Float64}, yr::NTuple{4,Float64},
            wt::NTuple{4,Float64}, wt_tot::Float64)

Allocation-free 2-point Gauss-Legendre rule on `[-1, 1]²`, returned as
`NTuple{4,Float64}` instead of `Vector{Float64}`. Counter-clockwise
corner order matches `gq2d_nodes(2)`.
"""
@inline gq2d_nodes_2pt() = (_XR_GQ2_2PT, _YR_GQ2_2PT,
                            _WT_GQ2_2PT, _WTTOT_GQ2_2PT)

"""
    gq2d_shape_functions(xr::Real, yr::Real) -> NTuple{4, Float64}

Bilinear shape functions `(N1, N2, N3, N4)` of the reference square,
evaluated at point `(xr, yr) ∈ [-1, 1]²`. Counter-clockwise corner
ordering matches `gq2d_nodes`.
"""
@inline function gq2d_shape_functions(xr::Real, yr::Real)
    N1 = (1 - xr) * (1 - yr) / 4
    N2 = (1 + xr) * (1 - yr) / 4
    N3 = (1 + xr) * (1 + yr) / 4
    N4 = (1 - xr) * (1 + yr) / 4
    return (N1, N2, N3, N4)
end

"""
    gq2d_interp_to_node(v_ab::NTuple{4,Float64}, xr::Real, yr::Real) -> Float64

Interpolate a 4-corner field `v_ab = (v_SW, v_SE, v_NE, v_NW)` to a
quadrature node at `(xr, yr) ∈ [-1, 1]²` using the bilinear shape
functions. Counter-clockwise corner ordering matches `gq2d_nodes`.
"""
@inline function gq2d_interp_to_node(v_ab::NTuple{4,Float64},
                                     xr::Real, yr::Real)
    N1, N2, N3, N4 = gq2d_shape_functions(xr, yr)
    return N1 * v_ab[1] + N2 * v_ab[2] + N3 * v_ab[3] + N4 * v_ab[4]
end
