## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3d / PR-A.1 unit tests for the SSA helper layer:
#
#   Commit 0 — Gauss quadrature dependency + helper:
#     - `gq2d_nodes(n)` matches Fortran `gq2D_class` ordering and weights
#     - 2-point rule integrates polynomials up to bilinear exactly
#     - bilinear shape functions reproduce corner values
#     - bilinear interpolation helper round-trips
#
#   (Commits 1, 2 — additional testsets appended in subsequent commits.)
#
# YelmoMirror lockstep cross-checks are deferred to PR-C; these are
# pure unit tests against hand-derived expected values.

using Test
using Yelmo
using Oceananigans
using Oceananigans: interior

_bounded_2d(Nx, Ny; dx=1.0) = RectilinearGrid(size=(Nx, Ny),
                                                x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                                topology=(Bounded, Bounded, Flat))

# ======================================================================
# Commit 0 — Gauss-Legendre quadrature helper
# ======================================================================

@testset "quadrature: gq2d_nodes(2) matches Fortran gq2D_class" begin
    xr, yr, wt, wt_tot = gq2d_nodes(2)
    r = 1.0 / sqrt(3.0)

    # Counter-clockwise from SW corner — matches the Fortran
    # `gq2D_init` convention (gaussian_quadrature.f90:182-186).
    @test xr ≈ [-r, +r, +r, -r]   atol=1e-15
    @test yr ≈ [-r, -r, +r, +r]   atol=1e-15
    @test wt ≈ [1.0, 1.0, 1.0, 1.0] atol=1e-15
    @test wt_tot ≈ 4.0
end

@testset "quadrature: 2-point rule integrates polynomials exactly" begin
    xr, yr, wt, wt_tot = gq2d_nodes(2)

    # ∫∫_[-1,1]² 1 dxdy = 4
    @test sum(wt) ≈ 4.0 atol=1e-15
    # ∫∫_[-1,1]² x dxdy = 0  (odd in x, even domain)
    @test sum(wt .* xr) ≈ 0.0 atol=1e-15
    # ∫∫_[-1,1]² y dxdy = 0
    @test sum(wt .* yr) ≈ 0.0 atol=1e-15
    # ∫∫_[-1,1]² xy dxdy = 0  (odd in x and y separately)
    @test sum(wt .* xr .* yr) ≈ 0.0 atol=1e-15
    # ∫∫_[-1,1]² x² dxdy = 4/3
    @test sum(wt .* xr.^2) ≈ 4/3 atol=1e-15
    # ∫∫_[-1,1]² y² dxdy = 4/3
    @test sum(wt .* yr.^2) ≈ 4/3 atol=1e-15
    # ∫∫_[-1,1]² (x² + y²) dxdy = 8/3
    @test sum(wt .* (xr.^2 .+ yr.^2)) ≈ 8/3 atol=1e-15
end

@testset "quadrature: shape functions reproduce corner values" begin
    # The bilinear shape functions Ni evaluated at corner i give 1,
    # at all other corners give 0. Since gq2d_nodes returns interior
    # points (not corners), test the corner-eval property directly.
    sf = Yelmo.YelmoModelDyn.gq2d_shape_functions
    # Corners (ordered SW, SE, NE, NW).
    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    for (i, (cx, cy)) in enumerate(corners)
        N = sf(cx, cy)
        for j in 1:4
            expected = (i == j) ? 1.0 : 0.0
            @test N[j] ≈ expected atol=1e-15
        end
    end
end

@testset "quadrature: gq2d_interp_to_node bilinear interpolation" begin
    interp = Yelmo.YelmoModelDyn.gq2d_interp_to_node
    # Constant field: every node returns the same value.
    v_ab = (3.7, 3.7, 3.7, 3.7)
    @test interp(v_ab, 0.123, -0.456) ≈ 3.7
    # Linear in x: SW=0, SE=1, NE=1, NW=0 → interp(x, y) = (1+x)/2.
    v_ab = (0.0, 1.0, 1.0, 0.0)
    @test interp(v_ab, -1.0, 0.0) ≈ 0.0
    @test interp(v_ab,  0.0, 0.0) ≈ 0.5
    @test interp(v_ab, +1.0, 0.0) ≈ 1.0
    # Linear in y: SW=0, SE=0, NE=1, NW=1 → interp(x, y) = (1+y)/2.
    v_ab = (0.0, 0.0, 1.0, 1.0)
    @test interp(v_ab, 0.0, -1.0) ≈ 0.0
    @test interp(v_ab, 0.0,  0.0) ≈ 0.5
    @test interp(v_ab, 0.0, +1.0) ≈ 1.0
end

@testset "quadrature: gq2d_nodes(n=3) sanity" begin
    # 3-point Gauss-Legendre: 9 nodes, total weight = 4.
    xr, yr, wt, wt_tot = gq2d_nodes(3)
    @test length(xr) == 9
    @test length(yr) == 9
    @test length(wt) == 9
    @test wt_tot ≈ 4.0
    @test sum(wt) ≈ 4.0
    # 3-point rule integrates degree-5 in each dim; check x⁴.
    # ∫∫_[-1,1]² x⁴ dxdy = 2 · ∫_[-1,1] x⁴ dx · 1 (the 1 = ∫dy / ∫dy)
    # ∫_{-1}^{1} x⁴ dx = 2/5; ∫_[-1,1] dy = 2 → ∫∫ x⁴ = 4/5
    @test sum(wt .* xr.^4) ≈ 4/5 atol=1e-12
end
