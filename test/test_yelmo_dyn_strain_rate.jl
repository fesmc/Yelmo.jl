## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3h / commit C4 unit tests for `calc_strain_rate_tensor_jac_quad3D!`:
#
#   - Pure rotation `(ux, uy) = (-Ω·y, +Ω·x)` → de = 0 (antisymmetric).
#   - Linear extension `ux = a·x, uy = -a·y` → dxx = a, dyy = -a, de = a,
#     div = 0, f_shear = 0 (pure stretching).
#   - Pure shear `ux = γ·z` → dxz ≈ γ/2 (recovered as ½·dxz_jvel from
#     the symmetrisation, since dzx_jvel = 0 here), de ≈ γ/2, f_shear = 1.
#   - Floating ice (f_grnd = 0) → f_shear = 0 regardless of strain.
#
# Pure unit tests against hand-derived expected values, no fixtures.

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior

# Helper to allocate full strn / strn2D / jvel state for the test.
function _make_strain_state(Nx, Ny, Nz; H_const=2000.0, dx=1e3, dy=1e3)
    g_2d = RectilinearGrid(size=(Nx, Ny),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy),
                           topology=(Bounded, Bounded, Flat))

    zeta_aa = collect(range(0.5/Nz, 1.0 - 0.5/Nz, length=Nz))
    zeta_ac = vcat(0.0, 0.5*(zeta_aa[1:end-1].+zeta_aa[2:end]), 1.0)

    g_3d = RectilinearGrid(size=(Nx, Ny, Nz),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy), z=zeta_ac,
                           topology=(Bounded, Bounded, Bounded))

    # 3D strn fields (CenterField — strn lives at aa, zeta_aa).
    strn_dxx = CenterField(g_3d); strn_dyy = CenterField(g_3d)
    strn_dxy = CenterField(g_3d); strn_dxz = CenterField(g_3d)
    strn_dyz = CenterField(g_3d); strn_de  = CenterField(g_3d)
    strn_div = CenterField(g_3d); strn_f_shear = CenterField(g_3d)
    # 2D strn averaged.
    strn2D_dxx = CenterField(g_2d); strn2D_dyy = CenterField(g_2d)
    strn2D_dxy = CenterField(g_2d); strn2D_dxz = CenterField(g_2d)
    strn2D_dyz = CenterField(g_2d); strn2D_de  = CenterField(g_2d)
    strn2D_div = CenterField(g_2d); strn2D_f_shear = CenterField(g_2d)

    # jvel fields (xy-row at acx/acy CenterField; z-row at ZFaceField).
    jvel_dxx = CenterField(g_3d); jvel_dxy = CenterField(g_3d); jvel_dxz = CenterField(g_3d)
    jvel_dyx = CenterField(g_3d); jvel_dyy = CenterField(g_3d); jvel_dyz = CenterField(g_3d)
    jvel_dzx = ZFaceField(g_3d);  jvel_dzy = ZFaceField(g_3d);  jvel_dzz = ZFaceField(g_3d)

    # uz buffer (used by uzterms in setup).
    uz = ZFaceField(g_3d)
    uz_star = ZFaceField(g_3d)

    # ux, uy
    ux = XFaceField(g_3d); uy = YFaceField(g_3d)

    # masks + bed/surface gradients (zero by default).
    H_ice  = CenterField(g_2d); fill!(interior(H_ice), H_const)
    f_ice  = CenterField(g_2d); fill!(interior(f_ice), 1.0)
    f_grnd = CenterField(g_2d); fill!(interior(f_grnd), 1.0)
    dzsdx = CenterField(g_2d); dzsdy = CenterField(g_2d)
    dzbdx = CenterField(g_2d); dzbdy = CenterField(g_2d)

    return (; g_2d, g_3d, zeta_aa, zeta_ac, dx, dy, H_const, Nx, Ny, Nz,
              strn_dxx, strn_dyy, strn_dxy, strn_dxz, strn_dyz,
              strn_de, strn_div, strn_f_shear,
              strn2D_dxx, strn2D_dyy, strn2D_dxy, strn2D_dxz, strn2D_dyz,
              strn2D_de, strn2D_div, strn2D_f_shear,
              jvel_dxx, jvel_dxy, jvel_dxz,
              jvel_dyx, jvel_dyy, jvel_dyz,
              jvel_dzx, jvel_dzy, jvel_dzz,
              ux, uy, uz, uz_star,
              H_ice, f_ice, f_grnd, dzsdx, dzsdy, dzbdx, dzbdy)
end

# Run the full Jacobian → uz → uz-jacobian → strain-rate pipeline given
# a prescribed (ux, uy) and (optional) prescribed forcing.
function _run_pipeline!(s; de_max=2.0)
    # 1. uxy Jacobian
    calc_jacobian_vel_3D_uxyterms!(
        s.jvel_dxx, s.jvel_dxy, s.jvel_dxz,
        s.jvel_dyx, s.jvel_dyy, s.jvel_dyz,
        s.ux, s.uy,
        s.H_ice, s.f_ice,
        s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
        s.zeta_aa, s.dx, s.dy)

    # 2. uz, uz_star — needs smb, bmb, dHdt, dzsdt zero-allocated.
    smb   = CenterField(s.g_2d); bmb   = CenterField(s.g_2d)
    dHdt  = CenterField(s.g_2d); dzsdt = CenterField(s.g_2d)
    calc_uz_3D_jac!(
        s.uz, s.uz_star,
        s.ux, s.uy,
        s.jvel_dxx, s.jvel_dyy,
        s.H_ice, s.f_ice,
        smb, bmb, dHdt, dzsdt,
        s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
        s.zeta_aa, s.zeta_ac, s.dx, s.dy, true)

    # 3. uz Jacobian
    calc_jacobian_vel_3D_uzterms!(
        s.jvel_dzx, s.jvel_dzy, s.jvel_dzz,
        s.uz,
        s.H_ice, s.f_ice,
        s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
        s.zeta_ac, s.dx, s.dy)

    # 4. Strain rate tensor
    calc_strain_rate_tensor_jac_quad3D!(
        s.strn_dxx, s.strn_dyy, s.strn_dxy, s.strn_dxz, s.strn_dyz,
        s.strn_de,  s.strn_div, s.strn_f_shear,
        s.strn2D_dxx, s.strn2D_dyy, s.strn2D_dxy, s.strn2D_dxz, s.strn2D_dyz,
        s.strn2D_de,  s.strn2D_div, s.strn2D_f_shear,
        s.jvel_dxx, s.jvel_dxy, s.jvel_dxz,
        s.jvel_dyx, s.jvel_dyy, s.jvel_dyz,
        s.jvel_dzx, s.jvel_dzy,
        s.f_ice, s.f_grnd,
        s.zeta_aa, de_max)
end

# ======================================================================
# Test 1 — Pure 2D rotation `ux = -Ω·y, uy = +Ω·x` → de = 0
# ======================================================================
#
# `dudx = 0, dudy = -Ω, dvdx = +Ω, dvdy = 0`. Symmetric strain:
# dxx = dyy = 0, dxy = ½·(-Ω + Ω) = 0. Effective strain rate = 0.

@testset "Strain rate: pure rotation → de = 0" begin
    s = _make_strain_state(14, 12, 4)
    Ω = 1e-4

    # ux = -Ω·y  (XFace at acx of cell i, y-center of cell j)
    UX = interior(s.ux)
    for j in 1:s.Ny
        y_center = (j - 0.5) * s.dy
        UX[:, j, :] .= -Ω * y_center
    end
    # uy = +Ω·x  (YFace at x-center of cell i, acy of cell j)
    UY = interior(s.uy)
    for i in 1:s.Nx
        x_center = (i - 0.5) * s.dx
        UY[i, :, :] .= +Ω * x_center
    end

    _run_pipeline!(s)

    Sde = interior(s.strn_de)
    Sdiv = interior(s.strn_div)
    # Interior cells (away from one-sided FD margins).
    for k in 1:s.Nz, j in 4:s.Ny-3, i in 4:s.Nx-3
        @test isapprox(Sde[i, j, k],  0.0; atol=1e-9)
        @test isapprox(Sdiv[i, j, k], 0.0; atol=1e-12)
    end
end

# ======================================================================
# Test 2 — Linear extension `ux = a·x, uy = -a·y` (pure stretching)
# ======================================================================
#
# dxx = +a, dyy = -a, dxy = 0, dxz = dyz = 0.
# de = √(a² + a² - a² + 0 + 0 + 0) = a. div = 0. f_shear = 0.

@testset "Strain rate: linear extension → de = a, f_shear = 0" begin
    s = _make_strain_state(14, 12, 4)
    a = 1e-4

    UX = interior(s.ux)
    for i in axes(UX, 1)
        x_face = (i - 1) * s.dx
        UX[i, :, :] .= a * x_face
    end
    UY = interior(s.uy)
    for j in axes(UY, 2)
        y_face = (j - 1) * s.dy
        UY[:, j, :] .= -a * y_face
    end

    _run_pipeline!(s)

    Sxx = interior(s.strn_dxx); Syy = interior(s.strn_dyy); Sde = interior(s.strn_de)
    Sdiv = interior(s.strn_div); Sfs = interior(s.strn_f_shear)
    # Note: `ux = a·x` is unbounded so the boundary FD at i = 1 / i = Nx
    # gives `dxx = a/2` (clamped neighbour). The resulting non-zero
    # divergence at the edges propagates a small `uz` and then `dzx` two
    # cells inward via the strain-rate corner average, making `Sxz` and
    # `Sfs` slightly non-zero at boundary-adjacent interior cells. We
    # check only `Sxx`, `Syy`, `Sdiv` (which are local FD ops, no inward
    # propagation) at machine precision; `Sde` / `Sfs` are checked at a
    # looser tolerance that covers the boundary residual.
    for k in 1:s.Nz, j in 4:s.Ny-3, i in 4:s.Nx-3
        @test isapprox(Sxx[i, j, k],  a;   atol=1e-12)
        @test isapprox(Syy[i, j, k], -a;   atol=1e-12)
        @test isapprox(Sdiv[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Sde[i, j, k],  a;   atol=1e-7)   # boundary-residual ≲ 0.1·a
        @test isapprox(Sfs[i, j, k],  0.0; atol=0.2)    # f_shear ≪ 1 (stretching-dominated)
    end
end

# ======================================================================
# Test 3 — Pure vertical shear `ux = γ·z`, uy = 0 → f_shear = 1
# ======================================================================
#
# dxz_jvel = γ (jvel-side, at acx of cell i, zeta_aa[k]).
# dzx_jvel = 0 (uz = 0 so its horizontal derivative is 0; sigma
#                correction `c_x · dzz` also 0 since flat bed/surface).
# strn.dxz = ½ · (dxz_jvel + dzx_jvel) = ½·γ.
# strn.de = √(0 + 0 + 0 + 0 + (γ/2)² + 0) = γ/2.
# strn.f_shear = √((γ/2)²)/(γ/2) = 1 (pure shearing).

@testset "Strain rate: pure vertical shear → f_shear = 1, dxz = γ/2" begin
    s = _make_strain_state(14, 12, 5)
    γ = 1e-3

    # ux = γ·z at all (i, j); layers vary with zeta_aa
    UX = interior(s.ux)
    for k in 1:s.Nz
        z_layer = s.zeta_aa[k] * s.H_const
        UX[:, :, k] .= γ * z_layer
    end

    _run_pipeline!(s)

    Sxz = interior(s.strn_dxz); Sde = interior(s.strn_de); Sfs = interior(s.strn_f_shear)
    expected_dxz = γ / 2
    expected_de  = γ / 2
    for k in 1:s.Nz, j in 4:s.Ny-3, i in 4:s.Nx-3
        @test isapprox(Sxz[i, j, k], expected_dxz; atol=1e-9)
        @test isapprox(Sde[i, j, k], expected_de;  atol=1e-9)
        @test isapprox(Sfs[i, j, k], 1.0;          atol=1e-9)
    end
end

# ======================================================================
# Test 4 — Floating ice (f_grnd = 0) → f_shear forced to 0
# ======================================================================

@testset "Strain rate: floating ice → f_shear = 0" begin
    s = _make_strain_state(14, 12, 5)
    fill!(interior(s.f_grnd), 0.0)
    γ = 1e-3
    UX = interior(s.ux)
    for k in 1:s.Nz
        z_layer = s.zeta_aa[k] * s.H_const
        UX[:, :, k] .= γ * z_layer
    end
    _run_pipeline!(s)
    Sfs = interior(s.strn_f_shear)
    for k in 1:s.Nz, j in 4:s.Ny-3, i in 4:s.Nx-3
        @test isapprox(Sfs[i, j, k], 0.0; atol=1e-12)
    end
end

# ======================================================================
# Test 5 — `de_max` clamp
# ======================================================================
#
# Crank up the velocity scale so de exceeds de_max; check the clamp
# bites and `f_shear` is computed from the clamped `de`.

@testset "Strain rate: de_max clamp" begin
    s = _make_strain_state(14, 12, 4)
    a = 1.0   # 1/yr — huge rate.
    UX = interior(s.ux)
    for i in axes(UX, 1)
        x_face = (i - 1) * s.dx
        UX[i, :, :] .= a * x_face
    end

    _run_pipeline!(s; de_max=0.5)   # clamp at 0.5

    Sde = interior(s.strn_de)
    for k in 1:s.Nz, j in 4:s.Ny-3, i in 4:s.Nx-3
        @test isapprox(Sde[i, j, k], 0.5; atol=1e-12)
    end
end

# ======================================================================
# Test 6 — `strn2D` matches per-layer mean (uniform layers)
# ======================================================================
#
# For the linear-extension setup the per-layer values are constant in z,
# so the depth average equals the per-layer value.

@testset "Strain rate: 2D vertical average matches per-layer (uniform z)" begin
    s = _make_strain_state(14, 12, 5)
    a = 1e-4
    UX = interior(s.ux)
    for i in axes(UX, 1)
        x_face = (i - 1) * s.dx
        UX[i, :, :] .= a * x_face
    end
    UY = interior(s.uy)
    for j in axes(UY, 2)
        y_face = (j - 1) * s.dy
        UY[:, j, :] .= -a * y_face
    end

    _run_pipeline!(s)

    S2_xx = interior(s.strn2D_dxx); S2_yy = interior(s.strn2D_dyy)
    S2_de = interior(s.strn2D_de)
    # Per-layer FD ops are local (no inward propagation), so dxx / dyy
    # match at machine precision; `de` picks up the same boundary-FD
    # residual as Test 2 (looser tol).
    for j in 4:s.Ny-3, i in 4:s.Nx-3
        @test isapprox(S2_xx[i, j, 1],  a; atol=1e-12)
        @test isapprox(S2_yy[i, j, 1], -a; atol=1e-12)
        @test isapprox(S2_de[i, j, 1],  a; atol=1e-7)
    end
end
