# ----------------------------------------------------------------------
# Gauss-Legendre tensor-product quadrature on a 2D reference square.
#
# Mirrors the Fortran `gq2D_class` / `gq2D_init` defined in
# `fesm-utils/utils/src/gaussian_quadrature.f90` (Yelmo Fortran's
# `gaussian_quadrature` module). The 2-point Gauss-Legendre rule places
# 4 quadrature nodes on the reference square `[-1, 1]²` at
#
#     (xr, yr) ∈ {(±1/√3, ±1/√3)}                      (counter-clockwise
#                                                       from SW corner)
#
# all with weight 1.0 and total weight `wt_tot = 4.0`. This rule is
# exact for bilinear functions on the square (degree 3 in each variable
# separately).
#
# Backed by [FastGaussQuadrature.jl](https://juliaapproximation.github.io/FastGaussQuadrature.jl/)
# — `gausslegendre(n)` returns the n-point 1D Gauss-Legendre nodes and
# weights on `[-1, 1]`, which we tensor-product into 2D.
#
# The Fortran node ordering (counter-clockwise from SW corner) is:
#
#       N4----N3
#       |     |
#       |     |
#       N1----N2
#
#     N1 = (-r, -r),  N2 = (+r, -r),  N3 = (+r, +r),  N4 = (-r, +r)
#
# where r = 1/√3 for the 2-point rule. We match that ordering exactly so
# Yelmo.jl viscosity / basal-drag kernels can read the nodes in the same
# index order as their Fortran counterparts (no node-permutation logic
# at call sites).
#
# This file is included from `YelmoModelDyn.jl` and used by the SSA
# basal-drag helper `calc_beta_aa_power_plastic` and the SSA viscosity
# helper `calc_visc_eff_3D_nodes!`. The Fortran SIA solver does not use
# `gq2D`, so this module is SSA-only.
# ----------------------------------------------------------------------

using FastGaussQuadrature: gausslegendre

export gq2d_nodes

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

Smoke-test self-consistency:

  - Sum of weights equals `wt_tot` (= n² for the [-1, 1] base rule).
  - For `n = 2`: ∫∫ 1 = 4, ∫∫ x = 0, ∫∫ x² = 4/3, all reproduced
    exactly via `sum(f(xr, yr) .* wt)`.
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
    gq2d_shape_functions(xr::Real, yr::Real)
        -> NTuple{4, Float64}

Bilinear shape functions `(N1, N2, N3, N4)` of the reference square,
evaluated at point `(xr, yr) ∈ [-1, 1]²`. Counter-clockwise corner
ordering matches `gq2d_nodes`:

    N1(x, y) = (1 - x)(1 - y) / 4    (SW corner is 1, others 0)
    N2(x, y) = (1 + x)(1 - y) / 4    (SE corner is 1, others 0)
    N3(x, y) = (1 + x)(1 + y) / 4    (NE corner is 1, others 0)
    N4(x, y) = (1 - x)(1 + y) / 4    (NW corner is 1, others 0)

Mirrors the Fortran `gq2D%N(:, p)` matrix from
`fesm-utils/utils/src/gaussian_quadrature.f90:191-196`. Used by
`gq2d_interp_to_node` to interpolate a corner value to a quadrature
node.
"""
@inline function gq2d_shape_functions(xr::Real, yr::Real)
    N1 = (1 - xr) * (1 - yr) / 4
    N2 = (1 + xr) * (1 - yr) / 4
    N3 = (1 + xr) * (1 + yr) / 4
    N4 = (1 - xr) * (1 + yr) / 4
    return (N1, N2, N3, N4)
end

"""
    gq2d_interp_to_node(v_ab::NTuple{4,Float64}, xr::Real, yr::Real)
        -> Float64

Interpolate a 4-corner field `v_ab = (v_SW, v_SE, v_NE, v_NW)` to a
quadrature node at `(xr, yr) ∈ [-1, 1]²` using the bilinear shape
functions. Counter-clockwise corner ordering matches `gq2d_nodes`.

Mirrors Fortran's `gq2D_to_nodes` body (`gaussian_quadrature.f90:497-516`):
`gq.v(p) = sum_n gq.N(n, p) * gq.v_ab(n)`.
"""
@inline function gq2d_interp_to_node(v_ab::NTuple{4,Float64},
                                     xr::Real, yr::Real)
    N1, N2, N3, N4 = gq2d_shape_functions(xr, yr)
    return N1 * v_ab[1] + N2 * v_ab[2] + N3 * v_ab[3] + N4 * v_ab[4]
end
