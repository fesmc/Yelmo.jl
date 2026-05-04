## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3h / commit C1 unit tests for the velocity Jacobian
# xy-row terms (`calc_jacobian_vel_3D_uxyterms!`):
#
#   - Linear extension `ux = a·x, uy = -a·y` → `dxx=a, dyy=-a`,
#     all cross / vertical terms zero.
#   - Vertical shear `ux = γ·z` → `dxz=γ` interior; layer-uniform.
#   - Cross-direction gradient `ux = β·y` → `dxy=β`.
#   - Sigma-coordinate correction: with non-zero `dzbdx` and a
#     vertical shear in `ux`, the corrected `dxx` picks up
#     `c_x · dxz` per Fortran lines 823.
#
# Pure unit tests against hand-derived expected values, no fixtures.

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior

# Helper: build a small (Nx, Ny, Nz) Bounded×Bounded×Bounded grid plus
# the associated 2D grid, plus all field allocations the Jacobian
# routine needs. Returns a NamedTuple of fields.
function _make_uxy_jacobian_state(Nx, Ny, Nz; H_const=2000.0, dx=1e3, dy=1e3)
    g_2d = RectilinearGrid(size=(Nx, Ny),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy),
                           topology=(Bounded, Bounded, Flat))

    zeta_aa = collect(range(0.5/Nz, 1.0 - 0.5/Nz, length=Nz))
    zeta_ac = vcat(0.0, 0.5*(zeta_aa[1:end-1].+zeta_aa[2:end]), 1.0)

    g_3d = RectilinearGrid(size=(Nx, Ny, Nz),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy), z=zeta_ac,
                           topology=(Bounded, Bounded, Bounded))

    ux = XFaceField(g_3d); uy = YFaceField(g_3d)

    # All 6 jvel xy-row fields are CenterField (no `_acx` suffix in
    # the schema; cell-center indexing for face-located values per the
    # `dzsdx` convention).
    jvel_dxx = CenterField(g_3d); jvel_dxy = CenterField(g_3d); jvel_dxz = CenterField(g_3d)
    jvel_dyx = CenterField(g_3d); jvel_dyy = CenterField(g_3d); jvel_dyz = CenterField(g_3d)

    H_ice = CenterField(g_2d); fill!(interior(H_ice), H_const)
    f_ice = CenterField(g_2d); fill!(interior(f_ice), 1.0)
    dzsdx = CenterField(g_2d); dzsdy = CenterField(g_2d)
    dzbdx = CenterField(g_2d); dzbdy = CenterField(g_2d)

    return (; g_2d, g_3d, zeta_aa, zeta_ac,
              ux, uy,
              jvel_dxx, jvel_dxy, jvel_dxz, jvel_dyx, jvel_dyy, jvel_dyz,
              H_ice, f_ice, dzsdx, dzsdy, dzbdx, dzbdy,
              dx, dy, H_const, Nx, Ny, Nz)
end

_call_uxyterms!(s) = calc_jacobian_vel_3D_uxyterms!(
    s.jvel_dxx, s.jvel_dxy, s.jvel_dxz,
    s.jvel_dyx, s.jvel_dyy, s.jvel_dyz,
    s.ux, s.uy,
    s.H_ice, s.f_ice,
    s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
    s.zeta_aa, s.dx, s.dy)

# ======================================================================
# Test 1 — Linear extension `ux = a·x, uy = -a·y`
# ======================================================================
#
# Incompressible 2D flow in horizontal plane, layer-uniform.
# Expected: dxx = +a, dyy = -a, all cross / vertical terms zero,
# and the depth-averaged Jacobian matches the layer values exactly.

@testset "Jacobian uxy: linear extension" begin
    s = _make_uxy_jacobian_state(8, 6, 5)
    a = 1e-4   # 1/yr

    # Set ux = a·x at every face slot. XFace under Bounded-x has Nx+1=9
    # face slots at x = 0, dx, 2dx, ..., 8dx.
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

    _call_uxyterms!(s)

    Dxx = interior(s.jvel_dxx); Dxy = interior(s.jvel_dxy); Dxz = interior(s.jvel_dxz)
    Dyx = interior(s.jvel_dyx); Dyy = interior(s.jvel_dyy); Dyz = interior(s.jvel_dyz)

    # Interior cells (away from one-sided FD margins).
    for k in 1:s.Nz, j in 2:s.Ny-1, i in 2:s.Nx-1
        @test isapprox(Dxx[i, j, k],  a; atol=1e-12)
        @test isapprox(Dyy[i, j, k], -a; atol=1e-12)
        @test isapprox(Dxy[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dyx[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dxz[i, j, k], 0.0; atol=1e-9)   # 3-point Lagrange residual
        @test isapprox(Dyz[i, j, k], 0.0; atol=1e-9)
    end
end

# ======================================================================
# Test 2 — Vertical shear `ux = γ·z`, `uy = 0`, flat bed/surface
# ======================================================================
#
# `z = zeta · H_const`, so ux at layer center k = γ · zeta_aa[k] · H_const.
# 3-point Lagrange recovers `dxz = γ` exactly for a linear-in-z field
# regardless of layer spacing (linear is in the basis). Other Jacobian
# terms vanish.

@testset "Jacobian uxy: pure vertical shear" begin
    s = _make_uxy_jacobian_state(8, 6, 5)
    γ = 1e-4   # 1/yr at z = 1m equivalent; arbitrary

    UX = interior(s.ux)
    for k in 1:s.Nz
        z_layer = s.zeta_aa[k] * s.H_const
        UX[:, :, k] .= γ * z_layer
    end
    # uy stays zero.

    _call_uxyterms!(s)

    Dxx = interior(s.jvel_dxx); Dxy = interior(s.jvel_dxy); Dxz = interior(s.jvel_dxz)
    Dyx = interior(s.jvel_dyx); Dyy = interior(s.jvel_dyy); Dyz = interior(s.jvel_dyz)

    # Interior cells.
    for k in 1:s.Nz, j in 2:s.Ny-1, i in 2:s.Nx-1
        # Note: dxx and dxy include the sigma-correction term `c_x · dxz`.
        # With dzbdx = dzsdx = 0 (flat bed/surface), c_x = 0, so dxx = 0.
        @test isapprox(Dxx[i, j, k], 0.0; atol=1e-9)
        @test isapprox(Dxy[i, j, k], 0.0; atol=1e-9)
        @test isapprox(Dyx[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dyy[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dxz[i, j, k], γ;   atol=1e-9)   # 3-point Lagrange linear-in-z is exact
        @test isapprox(Dyz[i, j, k], 0.0; atol=1e-12)
    end
end

# ======================================================================
# Test 3 — Cross gradient `ux = β·y`, `uy = 0`
# ======================================================================
#
# Layer-uniform, varies in y only. Expected: dxy = β, all others zero.

@testset "Jacobian uxy: cross gradient (∂ux/∂y)" begin
    s = _make_uxy_jacobian_state(8, 6, 5)
    β = 2e-4   # 1/yr per m

    UX = interior(s.ux)
    for j in 1:s.Ny
        y_center = (j - 0.5) * s.dy   # ux at acx of cell i, center of cell j
        UX[:, j, :] .= β * y_center
    end

    _call_uxyterms!(s)

    Dxy = interior(s.jvel_dxy)
    Dxx = interior(s.jvel_dxx); Dxz = interior(s.jvel_dxz)
    Dyx = interior(s.jvel_dyx); Dyy = interior(s.jvel_dyy); Dyz = interior(s.jvel_dyz)

    for k in 1:s.Nz, j in 2:s.Ny-1, i in 2:s.Nx-1
        @test isapprox(Dxy[i, j, k], β;   atol=1e-12)
        @test isapprox(Dxx[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dyx[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dyy[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dxz[i, j, k], 0.0; atol=1e-9)
        @test isapprox(Dyz[i, j, k], 0.0; atol=1e-12)
    end
end

# ======================================================================
# Test 4 — Sigma-coordinate correction: tilted bed × vertical shear
# ======================================================================
#
# Prescribe `ux = γ·z`, `uy = 0`, plus a uniform bed slope
# `dzbdx = α` (and matching `dzsdx = α` for a tilted slab), `dzbdy=dzsdy=0`.
# Then `c_x = -((1-ζ)·α + ζ·α) = -α`, so the sigma-corrected
# `dxx = 0 + c_x · dxz = -α·γ` per layer.
#
# This isolates the sigma-correction branch of the Step-2 logic.

@testset "Jacobian uxy: sigma correction (tilted slab × vertical shear)" begin
    s = _make_uxy_jacobian_state(8, 6, 5)
    γ = 1e-4
    α = 0.01   # bed slope, dimensionless

    UX = interior(s.ux)
    for k in 1:s.Nz
        z_layer = s.zeta_aa[k] * s.H_const
        UX[:, :, k] .= γ * z_layer
    end

    fill!(interior(s.dzbdx), α)
    fill!(interior(s.dzsdx), α)
    # dzbdy / dzsdy stay zero.

    _call_uxyterms!(s)

    Dxx = interior(s.jvel_dxx); Dxz = interior(s.jvel_dxz)
    Dxy = interior(s.jvel_dxy)
    Dyx = interior(s.jvel_dyx); Dyy = interior(s.jvel_dyy); Dyz = interior(s.jvel_dyz)

    # Interior cells: dxx = c_x · dxz = -α · γ.
    expected_dxx = -α * γ
    for k in 1:s.Nz, j in 2:s.Ny-1, i in 2:s.Nx-1
        @test isapprox(Dxx[i, j, k], expected_dxx; atol=1e-12)
        @test isapprox(Dxz[i, j, k], γ;            atol=1e-9)
        # dxy, dyx, dyy still zero (no y-gradients).
        @test isapprox(Dxy[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dyx[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dyy[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dyz[i, j, k], 0.0; atol=1e-12)
    end
end
