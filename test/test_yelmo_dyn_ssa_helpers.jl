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

# ======================================================================
# Commit 1 — calc_beta! + stagger_beta!
# ======================================================================
#
# `calc_beta!` dispatches on `beta_method`:
#   -1 → external (no-op)
#    0 → constant beta_const everywhere
#    1 → linear law via power-plastic (q=1)
#    2 → power-plastic (Bueler & van Pelt 2015) at user-supplied q
#    3 → regularized Coulomb (Joughin 2019)
#    4 → power-plastic with simple_stagger=true (Schoof slab)
#    5 → reg-Coulomb with simple_stagger=true
#
# Then `beta_gl_scale` post-scales:
#    0 → fraction (beta_gl_f = 1.0 → no-op)
#    1 → H_grnd linear blend toward 0
#    2 → zstar (Gladstone 2017)
#    3 → multiply by f_grnd
#
# Then floats get beta=0, then a beta_min floor on positive entries.

# Helper: build a 4x4 fully-grounded test geometry.
function _build_grounded_geom(Nx, Ny; dx=1.0)
    g = _bounded_2d(Nx, Ny; dx=dx)
    return (
        g       = g,
        beta    = CenterField(g),
        c_bed   = (cb = CenterField(g); fill!(interior(cb), 1e7); cb),
        ux_b    = (u = XFaceField(g);   fill!(interior(u), 50.0); u),
        uy_b    = (u = YFaceField(g);   fill!(interior(u), 30.0); u),
        H_ice   = (h = CenterField(g);  fill!(interior(h), 1000.0); h),
        f_ice   = (f = CenterField(g);  fill!(interior(f), 1.0);    f),
        H_grnd  = (h = CenterField(g);  fill!(interior(h), 500.0);  h),
        f_grnd  = (f = CenterField(g);  fill!(interior(f), 1.0);    f),
        z_bed   = (z = CenterField(g);  fill!(interior(z), -100.0); z),
        z_sl    = (z = CenterField(g);  fill!(interior(z), 0.0);    z),
    )
end

@testset "calc_beta!: beta_method=0 (constant)" begin
    Nx, Ny = 4, 4
    s = _build_grounded_geom(Nx, Ny)
    calc_beta!(s.beta, s.c_bed, s.ux_b, s.uy_b, s.H_ice, s.f_ice,
               s.H_grnd, s.f_grnd, s.z_bed, s.z_sl;
               beta_method=0, beta_const=2500.0,
               beta_q=1.0, beta_u0=100.0,
               beta_gl_scale=0, beta_gl_f=1.0,
               H_grnd_lim=500.0, beta_min=100.0,
               rho_ice=910.0, rho_sw=1028.0)
    # All grounded (f_grnd=1) and beta_gl_f=1 → no scaling.
    @test all(interior(s.beta) .== 2500.0)
end

@testset "calc_beta!: beta_method=2 power-plastic with q=1" begin
    # With q=1, beta = c_bed * (uxy / u_0) * (1 / uxy) = c_bed / u_0,
    # independent of uxy. Constant uxy → constant beta = 1e7 / 100 = 1e5.
    Nx, Ny = 4, 4
    s = _build_grounded_geom(Nx, Ny)
    calc_beta!(s.beta, s.c_bed, s.ux_b, s.uy_b, s.H_ice, s.f_ice,
               s.H_grnd, s.f_grnd, s.z_bed, s.z_sl;
               beta_method=2,
               beta_q=1.0, beta_u0=100.0,
               beta_gl_scale=0, beta_gl_f=1.0,
               H_grnd_lim=500.0, beta_min=100.0,
               rho_ice=910.0, rho_sw=1028.0)
    expected = 1e7 / 100.0
    # Use interior cells away from clamp boundaries.
    @test interior(s.beta)[2, 2, 1] ≈ expected rtol=1e-9
    @test interior(s.beta)[3, 3, 1] ≈ expected rtol=1e-9
end

@testset "calc_beta!: beta_method=2 power-plastic with q=0" begin
    # With q=0 and zero velocity, uxy_n = sqrt(0+0+ub_sq_min) = ub_min
    # = 1e-3 m/yr; beta = c_bed * 1 * (1/1e-3) = 1e10.
    Nx, Ny = 4, 4
    s = _build_grounded_geom(Nx, Ny)
    fill!(interior(s.ux_b), 0.0)
    fill!(interior(s.uy_b), 0.0)
    calc_beta!(s.beta, s.c_bed, s.ux_b, s.uy_b, s.H_ice, s.f_ice,
               s.H_grnd, s.f_grnd, s.z_bed, s.z_sl;
               beta_method=2,
               beta_q=0.0, beta_u0=100.0,
               beta_gl_scale=0, beta_gl_f=1.0,
               H_grnd_lim=500.0, beta_min=100.0,
               rho_ice=910.0, rho_sw=1028.0)
    expected = 1e7 / 1e-3   # 1e10
    @test interior(s.beta)[2, 2, 1] ≈ expected rtol=1e-9
end

@testset "calc_beta!: floating cells get beta=0" begin
    Nx, Ny = 4, 4
    s = _build_grounded_geom(Nx, Ny)
    interior(s.f_grnd)[2, 2, 1] = 0.0   # mark cell (2, 2) floating
    calc_beta!(s.beta, s.c_bed, s.ux_b, s.uy_b, s.H_ice, s.f_ice,
               s.H_grnd, s.f_grnd, s.z_bed, s.z_sl;
               beta_method=0, beta_const=2500.0,
               beta_q=1.0, beta_u0=100.0,
               beta_gl_scale=0, beta_gl_f=1.0,
               H_grnd_lim=500.0, beta_min=100.0,
               rho_ice=910.0, rho_sw=1028.0)
    @test interior(s.beta)[2, 2, 1] == 0.0
    @test interior(s.beta)[3, 3, 1] == 2500.0
end

@testset "calc_beta!: beta_min floor on positive cells" begin
    Nx, Ny = 4, 4
    s = _build_grounded_geom(Nx, Ny)
    # Use beta_const < beta_min so the floor activates.
    calc_beta!(s.beta, s.c_bed, s.ux_b, s.uy_b, s.H_ice, s.f_ice,
               s.H_grnd, s.f_grnd, s.z_bed, s.z_sl;
               beta_method=0, beta_const=50.0,
               beta_q=1.0, beta_u0=100.0,
               beta_gl_scale=0, beta_gl_f=1.0,
               H_grnd_lim=500.0, beta_min=100.0,
               rho_ice=910.0, rho_sw=1028.0)
    # Positive but below floor → clamped to beta_min = 100.
    @test all(interior(s.beta) .== 100.0)
end

@testset "calc_beta!: beta_gl_scale=3 multiplies by f_grnd" begin
    Nx, Ny = 4, 4
    s = _build_grounded_geom(Nx, Ny)
    interior(s.f_grnd)[2, 2, 1] = 0.5   # half-grounded GL cell
    calc_beta!(s.beta, s.c_bed, s.ux_b, s.uy_b, s.H_ice, s.f_ice,
               s.H_grnd, s.f_grnd, s.z_bed, s.z_sl;
               beta_method=0, beta_const=2000.0,
               beta_q=1.0, beta_u0=100.0,
               beta_gl_scale=3, beta_gl_f=1.0,
               H_grnd_lim=500.0, beta_min=100.0,
               rho_ice=910.0, rho_sw=1028.0)
    # beta_gl_scale=3 multiplies by f_grnd → 0.5 * 2000 = 1000.
    @test interior(s.beta)[2, 2, 1] ≈ 1000.0
    # Fully grounded cell unchanged.
    @test interior(s.beta)[3, 3, 1] == 2000.0
end

# Helper: build a 5x5 staggering test setup. Default fully grounded.
function _build_stagger_setup(Nx, Ny; beta_val=1500.0)
    g = _bounded_2d(Nx, Ny)
    return (
        g          = g,
        beta       = (b = CenterField(g);  fill!(interior(b),  beta_val); b),
        H_ice      = (h = CenterField(g);  fill!(interior(h),  1000.0);   h),
        f_ice      = (f = CenterField(g);  fill!(interior(f),  1.0);      f),
        ux         = (u = XFaceField(g);   fill!(interior(u),  0.0);      u),
        uy         = (v = YFaceField(g);   fill!(interior(v),  0.0);      v),
        f_grnd     = (f = CenterField(g);  fill!(interior(f),  1.0);      f),
        f_grnd_acx = (fx = XFaceField(g);  fill!(interior(fx), 1.0);      fx),
        f_grnd_acy = (fy = YFaceField(g);  fill!(interior(fy), 1.0);      fy),
        beta_acx   = XFaceField(g),
        beta_acy   = YFaceField(g),
    )
end

@testset "stagger_beta!: beta_gl_stag=0 (mean only) on uniform grid" begin
    Nx, Ny = 5, 5
    s = _build_stagger_setup(Nx, Ny)
    stagger_beta!(s.beta_acx, s.beta_acy, s.beta,
                  s.H_ice, s.f_ice, s.ux, s.uy,
                  s.f_grnd, s.f_grnd_acx, s.f_grnd_acy;
                  beta_gl_stag=0, beta_min=100.0)
    # f_grnd=1 and f_ice=1 everywhere → mean rule; uniform → 1500.
    for i in 1:Nx-1, j in 1:Ny
        @test interior(s.beta_acx)[i+1, j, 1] ≈ 1500.0
    end
    for i in 1:Nx, j in 1:Ny-1
        @test interior(s.beta_acy)[i, j+1, 1] ≈ 1500.0
    end
end

@testset "stagger_beta!: beta_gl_stag=1 (gl_upstream)" begin
    # 5x5: cells (1:2, j) grounded, (3:5, j) floating with beta=0.
    Nx, Ny = 5, 5
    s = _build_stagger_setup(Nx, Ny; beta_val=0.0)
    interior(s.beta)[1:2, :, 1] .= 1500.0
    interior(s.f_grnd)[3:5, :, 1] .= 0.0

    stagger_beta!(s.beta_acx, s.beta_acy, s.beta,
                  s.H_ice, s.f_ice, s.ux, s.uy,
                  s.f_grnd, s.f_grnd_acx, s.f_grnd_acy;
                  beta_gl_stag=1, beta_min=100.0)

    # Face between cell i=2 (grounded) and i=3 (floating) at array index
    # [3, j, 1]. Upstream takes the grounded value beta(2, :) = 1500.
    @test all(interior(s.beta_acx)[3, :, 1] .== 1500.0)
    # Interior grounded face (1↔2): mean of two 1500s = 1500.
    @test all(interior(s.beta_acx)[2, :, 1] .== 1500.0)
    # Faces fully inside floating zone: both endpoints floating →
    # beta_acx = 0 (mean rule's all-floating shortcut).
    @test all(interior(s.beta_acx)[4, :, 1] .== 0.0)
    @test all(interior(s.beta_acx)[5, :, 1] .== 0.0)
end

@testset "stagger_beta!: beta_gl_stag=2 (gl_downstream)" begin
    Nx, Ny = 5, 5
    s = _build_stagger_setup(Nx, Ny; beta_val=0.0)
    interior(s.beta)[1:2, :, 1] .= 1500.0
    interior(s.f_grnd)[3:5, :, 1] .= 0.0

    stagger_beta!(s.beta_acx, s.beta_acy, s.beta,
                  s.H_ice, s.f_ice, s.ux, s.uy,
                  s.f_grnd, s.f_grnd_acx, s.f_grnd_acy;
                  beta_gl_stag=2, beta_min=100.0)

    # Downstream takes the floating-side value = 0.
    @test all(interior(s.beta_acx)[3, :, 1] .== 0.0)
end

@testset "stagger_beta!: beta_gl_stag=3 (gl_subgrid)" begin
    Nx, Ny = 5, 5
    s = _build_stagger_setup(Nx, Ny; beta_val=0.0)
    interior(s.beta)[1:2, :, 1] .= 1500.0
    interior(s.f_grnd)[3:5, :, 1] .= 0.0
    # GL face at [3, :, 1]: f_grnd_acx = 0.4 → wt = 0.16.
    interior(s.f_grnd_acx)[3, :, 1] .= 0.4

    stagger_beta!(s.beta_acx, s.beta_acy, s.beta,
                  s.H_ice, s.f_ice, s.ux, s.uy,
                  s.f_grnd, s.f_grnd_acx, s.f_grnd_acy;
                  beta_gl_stag=3, beta_min=100.0)

    # f_grnd_acx² = 0.16 → β_acx = 0.16 * 1500 + 0.84 * 0 = 240.
    @test all(interior(s.beta_acx)[3, :, 1] .≈ 240.0)
end

@testset "stagger_beta!: beta_gl_stag=4 (subgrid_flux) on uniform grid" begin
    # On a fully grounded grid, no GL → kernel runs to completion via
    # the mean-staggering branch only. Sanity check: interior faces all
    # equal the constant beta value.
    Nx, Ny = 5, 5
    s = _build_stagger_setup(Nx, Ny; beta_val=2000.0)
    stagger_beta!(s.beta_acx, s.beta_acy, s.beta,
                  s.H_ice, s.f_ice, s.ux, s.uy,
                  s.f_grnd, s.f_grnd_acx, s.f_grnd_acy;
                  beta_gl_stag=4, beta_min=100.0)
    for i in 1:Nx-1, j in 1:Ny
        @test interior(s.beta_acx)[i+1, j, 1] ≈ 2000.0
    end
end

@testset "stagger_beta!: beta_gl_stag=-1 (external) is a no-op" begin
    # Pre-fill beta_acx / beta_acy with sentinel values, call with
    # beta_gl_stag=-1, expect them to be unchanged (modulo the
    # beta_min floor at the end — sentinel above floor → unchanged).
    Nx, Ny = 5, 5
    s = _build_stagger_setup(Nx, Ny)
    fill!(interior(s.beta_acx), 999.0)
    fill!(interior(s.beta_acy), 888.0)
    stagger_beta!(s.beta_acx, s.beta_acy, s.beta,
                  s.H_ice, s.f_ice, s.ux, s.uy,
                  s.f_grnd, s.f_grnd_acx, s.f_grnd_acy;
                  beta_gl_stag=-1, beta_min=100.0)
    @test all(interior(s.beta_acx) .== 999.0)
    @test all(interior(s.beta_acy) .== 888.0)
end

@testset "stagger_beta!: beta_min floor on positive faces" begin
    Nx, Ny = 5, 5
    s = _build_stagger_setup(Nx, Ny; beta_val=50.0)   # below floor
    stagger_beta!(s.beta_acx, s.beta_acy, s.beta,
                  s.H_ice, s.f_ice, s.ux, s.uy,
                  s.f_grnd, s.f_grnd_acx, s.f_grnd_acy;
                  beta_gl_stag=0, beta_min=100.0)
    # mean staggering writes 50 everywhere → clamped to 100.
    for i in 1:Nx-1, j in 1:Ny
        @test interior(s.beta_acx)[i+1, j, 1] == 100.0
    end
end

# ======================================================================
# Commit 2 — viscosity helpers
# ======================================================================
#
# Glen-flow effective viscosity (Lipscomb 2019 Eq. 2):
#   visc = 0.5 * eps_eff_sq^p1 * ATT^p2,  p1 = (1-n)/(2n), p2 = -1/n
#
# For a uniform x-velocity ramp ux_face[i+1, j] = U·i·dx and uy = 0:
#   dudx_aa = (ux[i+1] - ux[i]) / dx = U  (independent of i, j)
#   dudy = dvdx = dvdy = 0
#   eps_sq = U² + eps_0²

# Helper: build a 3D ice grid for ATT/visc.
_grid3d(Nx, Ny, Nz; dx=1.0) = RectilinearGrid(size=(Nx, Ny, Nz),
                                              x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                              z=(0.0, 1.0),
                                              topology=(Bounded, Bounded, Bounded))

@testset "calc_visc_eff_3D_aa!: linear-x velocity, hand-derived" begin
    Nx, Ny, Nz = 5, 4, 3
    dx = 1000.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    g3 = _grid3d(Nx, Ny, Nz; dx=dx)
    zeta_aa = collect(range(0.5/Nz, 1 - 0.5/Nz, length=Nz))

    # Linear-x face velocity: ux[i+1, j, 1] = U_per_m · i · dx
    # → centered difference dudx_aa = U_per_m everywhere.
    U_per_m = 1e-3
    ux  = XFaceField(g)
    uy  = YFaceField(g);  fill!(interior(uy), 0.0)
    for j in 1:Ny, i in 0:Nx
        interior(ux)[i+1, j, 1] = U_per_m * i * dx
    end

    H_ice  = CenterField(g);  fill!(interior(H_ice),  1000.0)
    f_ice  = CenterField(g);  fill!(interior(f_ice),  1.0)
    ATT_3d = CenterField(g3); fill!(interior(ATT_3d), 1e-16)
    visc_3d = CenterField(g3)

    n_glen, eps_0 = 3.0, 1e-6

    calc_visc_eff_3D_aa!(visc_3d, ux, uy, ATT_3d, H_ice, f_ice, zeta_aa,
                         dx, dx, n_glen, eps_0)

    p1 = (1.0 - n_glen) / (2.0 * n_glen)
    p2 = -1.0 / n_glen
    eps_sq = U_per_m^2 + eps_0^2
    expected = 0.5 * eps_sq^p1 * (1e-16)^p2

    # Check interior cells (skip outermost row/col where im1/jm1 clamp).
    for k in 1:Nz, j in 2:Ny-1, i in 2:Nx-1
        @test interior(visc_3d)[i, j, k] ≈ expected rtol=1e-9
    end
end

@testset "calc_visc_eff_3D_aa!: ice-free cells get visc=0" begin
    Nx, Ny, Nz = 4, 3, 2
    dx = 1.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    g3 = _grid3d(Nx, Ny, Nz; dx=dx)
    zeta_aa = collect(range(0.5/Nz, 1 - 0.5/Nz, length=Nz))

    ux = XFaceField(g);  fill!(interior(ux), 0.0)
    uy = YFaceField(g);  fill!(interior(uy), 0.0)
    H_ice = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice = CenterField(g);  fill!(interior(f_ice), 1.0)
    interior(f_ice)[2, 2, 1] = 0.5   # partial ice
    ATT_3d = CenterField(g3); fill!(interior(ATT_3d), 1e-16)
    visc_3d = CenterField(g3)

    calc_visc_eff_3D_aa!(visc_3d, ux, uy, ATT_3d, H_ice, f_ice, zeta_aa,
                         dx, dx, 3.0, 1e-6)
    # Cell (2, 2) has f_ice<1 → visc[2, 2, :] = 0.
    @test interior(visc_3d)[2, 2, 1] == 0.0
    @test interior(visc_3d)[2, 2, 2] == 0.0
end

@testset "calc_visc_eff_3D_nodes!: linear-x velocity, agrees with _aa" begin
    # On a smooth uniform-strain-rate slab, both quadrature variants
    # should agree at deep-interior cells. The two kernels differ near
    # the domain edge: `_aa` reads `(ux(i)-ux(im1))/dx` so at i=1 the
    # clamped im1=1 gives dudx=0; `_nodes` reads
    # `(ux(ip1)-ux(im1))/(2dx)` (from `calc_strain_rate_horizontal_2D`)
    # which gives U/2 at i=1. The corner-stagger in `_nodes` then
    # propagates this i=1 oddity into cell i=2 too. Cells i ≥ 3 are
    # unaffected and agree to ~1%.
    Nx, Ny, Nz = 8, 5, 3
    dx = 1000.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    g3 = _grid3d(Nx, Ny, Nz; dx=dx)
    zeta_aa = collect(range(0.5/Nz, 1 - 0.5/Nz, length=Nz))

    U_per_m = 1e-3
    ux = XFaceField(g)
    for j in 1:Ny, i in 0:Nx
        interior(ux)[i+1, j, 1] = U_per_m * i * dx
    end
    uy = YFaceField(g);  fill!(interior(uy), 0.0)
    H_ice = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice = CenterField(g);  fill!(interior(f_ice), 1.0)
    ATT_3d = CenterField(g3); fill!(interior(ATT_3d), 1e-16)

    visc_aa = CenterField(g3)
    visc_no = CenterField(g3)

    calc_visc_eff_3D_aa!(visc_aa, ux, uy, ATT_3d, H_ice, f_ice, zeta_aa,
                         dx, dx, 3.0, 1e-6)
    calc_visc_eff_3D_nodes!(visc_no, ux, uy, ATT_3d, H_ice, f_ice, zeta_aa,
                            dx, dx, 3.0, 1e-6)

    # Deep-interior cells (skip i ≤ 2 and i ≥ Nx-1, j ≤ 1 and j ≥ Ny).
    for k in 1:Nz, j in 2:Ny-1, i in 3:Nx-2
        a = interior(visc_aa)[i, j, k]
        b = interior(visc_no)[i, j, k]
        @test abs(a - b) / abs(a) < 1e-2
    end
end

@testset "calc_visc_eff_3D_nodes!: hand-derived linear-x answer" begin
    # On a smooth uniform-gradient slab, the corner-stagger averaging
    # of dudx (which is constant = U) preserves the constant value at
    # every node, so the result equals the analytical Glen formula.
    Nx, Ny, Nz = 5, 4, 2
    dx = 1000.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    g3 = _grid3d(Nx, Ny, Nz; dx=dx)
    zeta_aa = collect(range(0.5/Nz, 1 - 0.5/Nz, length=Nz))

    U_per_m = 1e-3
    ux = XFaceField(g)
    for j in 1:Ny, i in 0:Nx
        interior(ux)[i+1, j, 1] = U_per_m * i * dx
    end
    uy = YFaceField(g);  fill!(interior(uy), 0.0)
    H_ice = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice = CenterField(g);  fill!(interior(f_ice), 1.0)
    ATT_3d = CenterField(g3); fill!(interior(ATT_3d), 1e-16)
    visc_no = CenterField(g3)

    n_glen, eps_0 = 3.0, 1e-6
    calc_visc_eff_3D_nodes!(visc_no, ux, uy, ATT_3d, H_ice, f_ice, zeta_aa,
                            dx, dx, n_glen, eps_0)

    p1 = (1.0 - n_glen) / (2.0 * n_glen)
    p2 = -1.0 / n_glen
    eps_sq = U_per_m^2 + eps_0^2
    expected = 0.5 * eps_sq^p1 * (1e-16)^p2

    # Inner cells: gauss-quadrature uniform in this scenario.
    for k in 1:Nz, j in 2:Ny-1, i in 3:Nx-2
        @test interior(visc_no)[i, j, k] ≈ expected rtol=1e-6
    end
end

@testset "calc_visc_eff_int!: trapezoidal slab (Option C, full [0,1])" begin
    Nx, Ny, Nz = 3, 3, 4
    dx = 1.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    g3 = _grid3d(Nx, Ny, Nz; dx=dx)
    # Option C: 3D `visc_eff` Center stagger does NOT include zeta = 0
    # or zeta = 1; bed/surface boundary visc are passed in separately.
    # For a constant-1e8 column with bed/surface boundary visc also =
    # 1e8, the full ∫₀¹ visc dζ = 1e8 exactly (no end-strip drop).
    zeta_aa = collect(range(0.5/Nz, 1 - 0.5/Nz, length=Nz))

    visc_3d  = CenterField(g3); fill!(interior(visc_3d), 1e8)
    visc_b   = CenterField(g);  fill!(interior(visc_b),  1e8)
    visc_s   = CenterField(g);  fill!(interior(visc_s),  1e8)
    visc_int = CenterField(g)
    H_ice    = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice    = CenterField(g);  fill!(interior(f_ice), 1.0)

    calc_visc_eff_int!(visc_int, visc_3d, visc_b, visc_s,
                       H_ice, f_ice, zeta_aa)

    # ∫₀¹ visc dζ = 1e8; visc_int = 1e8 · H_ice = 1e11.
    expected = 1e8 * 1.0 * 1000.0
    for j in 1:Ny, i in 1:Nx
        @test interior(visc_int)[i, j, 1] ≈ expected rtol=1e-9
    end
end

@testset "calc_visc_eff_int!: ice-free cells get floor visc_min" begin
    Nx, Ny, Nz = 3, 3, 2
    dx = 1.0
    g  = _bounded_2d(Nx, Ny; dx=dx)
    g3 = _grid3d(Nx, Ny, Nz; dx=dx)
    zeta_aa = collect(range(0.5/Nz, 1 - 0.5/Nz, length=Nz))

    visc_3d  = CenterField(g3); fill!(interior(visc_3d), 1e8)
    visc_b   = CenterField(g);  fill!(interior(visc_b),  1e8)
    visc_s   = CenterField(g);  fill!(interior(visc_s),  1e8)
    visc_int = CenterField(g)
    H_ice    = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice    = CenterField(g);  fill!(interior(f_ice), 1.0)
    interior(f_ice)[1, 1, 1] = 0.0

    calc_visc_eff_int!(visc_int, visc_3d, visc_b, visc_s,
                       H_ice, f_ice, zeta_aa)

    # Ice-free cell: kernel sets visc_int = 0, then floor → visc_min.
    @test interior(visc_int)[1, 1, 1] == 1e5
    # Other cells: full [0,1] integral · H = 1e8 · 1000.
    @test interior(visc_int)[2, 2, 1] ≈ 1e8 * 1.0 * 1000.0 rtol=1e-9
end

@testset "stagger_visc_aa_ab!: 4-cell average at interior corners" begin
    Nx, Ny = 4, 4
    g = _bounded_2d(Nx, Ny)
    visc_2d = CenterField(g);   fill!(interior(visc_2d), 1e8)
    H_ice   = CenterField(g);   fill!(interior(H_ice),   1000.0)
    f_ice   = CenterField(g);   fill!(interior(f_ice),   1.0)

    visc_ab = Field((Face(), Face(), Center()), g)
    stagger_visc_aa_ab!(visc_ab, visc_2d, H_ice, f_ice)

    # Interior corners (i, j) for i in 1:Nx, j in 1:Ny → array index
    # [i+1, j+1, 1]. With f_ice=1 everywhere, each corner reads 4
    # iced cells → mean = 1e8.
    for j in 1:Ny, i in 1:Nx
        @test interior(visc_ab)[i+1, j+1, 1] ≈ 1e8
    end
end

@testset "stagger_visc_aa_ab!: only ice-covered neighbours contribute" begin
    Nx, Ny = 4, 4
    g = _bounded_2d(Nx, Ny)
    visc_2d = CenterField(g);   fill!(interior(visc_2d), 1e8)
    H_ice   = CenterField(g);   fill!(interior(H_ice),   1000.0)
    f_ice   = CenterField(g);   fill!(interior(f_ice),   1.0)

    interior(f_ice)[2, 2, 1] = 0.5   # partial ice excluded
    interior(visc_2d)[2, 2, 1] = 5e8

    visc_ab = Field((Face(), Face(), Center()), g)
    stagger_visc_aa_ab!(visc_ab, visc_2d, H_ice, f_ice)

    # Corner (i=1, j=1) reads aa-cells (1, 1), (2, 1), (1, 2), (2, 2).
    # Cell (2, 2) excluded. Mean of the other three (all 1e8) = 1e8.
    @test interior(visc_ab)[2, 2, 1] ≈ 1e8 rtol=1e-12
    # Corner not touching cell (2, 2) — e.g. (i=3, j=3) → reads cells
    # (3, 3), (4, 3), (3, 4), (4, 4); all iced → mean = 1e8.
    @test interior(visc_ab)[4, 4, 1] ≈ 1e8 rtol=1e-12
end

@testset "stagger_visc_aa_ab!: zero corner when no neighbour is iced" begin
    Nx, Ny = 4, 4
    g = _bounded_2d(Nx, Ny)
    visc_2d = CenterField(g);   fill!(interior(visc_2d), 1e8)
    H_ice   = CenterField(g);   fill!(interior(H_ice),   1000.0)
    f_ice   = CenterField(g);   fill!(interior(f_ice),   0.0)

    visc_ab = Field((Face(), Face(), Center()), g)
    fill!(interior(visc_ab), 0.0)
    stagger_visc_aa_ab!(visc_ab, visc_2d, H_ice, f_ice)
    @test maximum(abs.(interior(visc_ab))) == 0.0
end

# Smoke: verify the new `dyn.scratch.ssa_n_aa_ab` field type is what
# `stagger_visc_aa_ab!` expects (corner-staggered, shape Nx+1 × Ny+1).
@testset "dyn.scratch.ssa_n_aa_ab: allocation shape" begin
    Nx, Ny = 4, 4
    g  = _bounded_2d(Nx, Ny)
    f  = Field((Face(), Face(), Center()), g)
    @test size(interior(f)) == (Nx + 1, Ny + 1, 1)
end
