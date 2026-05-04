## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3h / commit C3 unit tests for the velocity Jacobian
# z-row terms (`calc_jacobian_vel_3D_uzterms!`):
#
#   - Linear-in-zeta uz `uz = γ·z` → `dzz = γ`, dzx/dzy = 0.
#   - Horizontal x-gradient `uz = β·x` → `dzx = β`, others zero.
#   - Horizontal y-gradient `uz = β·y` → `dzy = β`, others zero.
#   - Sigma correction: tilted slab × vertical shear in uz →
#     `dzx = -α·γ` (since centered FD gives 0, then `c_x · dzz`
#     contributes -α·γ).
#
# Pure unit tests against hand-derived expected values, no fixtures.

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior

function _make_uzterms_state(Nx, Ny, Nz; H_const=2000.0, dx=1e3, dy=1e3)
    g_2d = RectilinearGrid(size=(Nx, Ny),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy),
                           topology=(Bounded, Bounded, Flat))

    zeta_aa = collect(range(0.5/Nz, 1.0 - 0.5/Nz, length=Nz))
    zeta_ac = vcat(0.0, 0.5*(zeta_aa[1:end-1].+zeta_aa[2:end]), 1.0)

    g_3d = RectilinearGrid(size=(Nx, Ny, Nz),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy), z=zeta_ac,
                           topology=(Bounded, Bounded, Bounded))

    uz       = ZFaceField(g_3d)
    jvel_dzx = ZFaceField(g_3d)
    jvel_dzy = ZFaceField(g_3d)
    jvel_dzz = ZFaceField(g_3d)

    H_ice = CenterField(g_2d); fill!(interior(H_ice), H_const)
    f_ice = CenterField(g_2d); fill!(interior(f_ice), 1.0)
    dzsdx = CenterField(g_2d)
    dzsdy = CenterField(g_2d)
    dzbdx = CenterField(g_2d)
    dzbdy = CenterField(g_2d)

    return (; g_2d, g_3d, zeta_aa, zeta_ac, dx, dy, H_const, Nx, Ny, Nz,
              uz, jvel_dzx, jvel_dzy, jvel_dzz,
              H_ice, f_ice, dzsdx, dzsdy, dzbdx, dzbdy)
end

_call_uzterms!(s) = calc_jacobian_vel_3D_uzterms!(
    s.jvel_dzx, s.jvel_dzy, s.jvel_dzz,
    s.uz,
    s.H_ice, s.f_ice,
    s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
    s.zeta_ac, s.dx, s.dy)

# ======================================================================
# Test 1 — Linear-in-zeta uz: `uz(zeta) = γ·zeta·H` (so `uz(z) = γ·z`)
# ======================================================================
#
# Expected: dzz = γ (3-point Lagrange is exact for linear-in-z fields
# regardless of layer spacing). dzx = dzy = 0 (no horizontal gradient,
# no sigma correction since flat bed/surface).

@testset "Jacobian uz: linear-in-z uz → dzz = γ" begin
    s = _make_uzterms_state(8, 6, 5)
    γ = 1e-3   # 1/yr at z = 1m

    UZ = interior(s.uz)
    for k in 1:length(s.zeta_ac)
        z_face = s.zeta_ac[k] * s.H_const
        UZ[:, :, k] .= γ * z_face
    end

    _call_uzterms!(s)

    Dzz = interior(s.jvel_dzz)
    Dzx = interior(s.jvel_dzx)
    Dzy = interior(s.jvel_dzy)

    for k in 1:length(s.zeta_ac), j in 2:s.Ny-1, i in 2:s.Nx-1
        @test isapprox(Dzz[i, j, k], γ;   atol=1e-9)
        @test isapprox(Dzx[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dzy[i, j, k], 0.0; atol=1e-12)
    end
end

# ======================================================================
# Test 2 — Horizontal x-gradient `uz = β·x`, layer-uniform
# ======================================================================
#
# Expected: dzx = β, dzy = dzz = 0 (no z-dependence, no y-dependence).

@testset "Jacobian uz: horizontal x-gradient → dzx = β" begin
    s = _make_uzterms_state(10, 6, 5)
    β = 1e-5   # 1/yr per m

    UZ = interior(s.uz)
    for i in 1:s.Nx
        x_center = (i - 0.5) * s.dx
        UZ[i, :, :] .= β * x_center
    end

    _call_uzterms!(s)

    Dzx = interior(s.jvel_dzx)
    Dzy = interior(s.jvel_dzy)
    Dzz = interior(s.jvel_dzz)

    # Interior cells (away from one-sided FD margins).
    for k in 1:length(s.zeta_ac), j in 2:s.Ny-1, i in 3:s.Nx-2
        @test isapprox(Dzx[i, j, k], β;   atol=1e-12)
        @test isapprox(Dzy[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dzz[i, j, k], 0.0; atol=1e-9)
    end
end

# ======================================================================
# Test 3 — Horizontal y-gradient `uz = β·y`, layer-uniform
# ======================================================================
#
# Expected: dzy = β, dzx = dzz = 0.

@testset "Jacobian uz: horizontal y-gradient → dzy = β" begin
    s = _make_uzterms_state(8, 10, 5)
    β = 2e-5

    UZ = interior(s.uz)
    for j in 1:s.Ny
        y_center = (j - 0.5) * s.dy
        UZ[:, j, :] .= β * y_center
    end

    _call_uzterms!(s)

    Dzx = interior(s.jvel_dzx)
    Dzy = interior(s.jvel_dzy)
    Dzz = interior(s.jvel_dzz)

    for k in 1:length(s.zeta_ac), j in 3:s.Ny-2, i in 2:s.Nx-1
        @test isapprox(Dzy[i, j, k], β;   atol=1e-12)
        @test isapprox(Dzx[i, j, k], 0.0; atol=1e-12)
        @test isapprox(Dzz[i, j, k], 0.0; atol=1e-9)
    end
end

# ======================================================================
# Test 4 — Sigma correction: tilted slab × vertical shear in uz
# ======================================================================
#
# Prescribe `uz = γ·z`, uniform bed slope `dzbdx = α` and matching
# `dzsdx = α` (parallel-bed-and-surface tilted slab). Then:
#
#   dzz_layer        = γ                     (linear-in-z is exact)
#   dzx_centered     = 0                     (no horizontal gradient)
#   c_x at zeta_ac[k] = -((1-zk)·α + zk·α) = -α   (uniform slope)
#   dzx_corrected    = 0 + c_x · dzz = -α·γ
#
# This isolates the sigma-correction branch of the routine.

@testset "Jacobian uz: sigma correction (tilted slab × vertical shear)" begin
    s = _make_uzterms_state(8, 6, 5)
    γ = 1e-3
    α = 0.01

    UZ = interior(s.uz)
    for k in 1:length(s.zeta_ac)
        z_face = s.zeta_ac[k] * s.H_const
        UZ[:, :, k] .= γ * z_face
    end

    fill!(interior(s.dzbdx), α)
    fill!(interior(s.dzsdx), α)

    _call_uzterms!(s)

    Dzx = interior(s.jvel_dzx)
    Dzy = interior(s.jvel_dzy)
    Dzz = interior(s.jvel_dzz)

    expected_dzx = -α * γ
    for k in 1:length(s.zeta_ac), j in 2:s.Ny-1, i in 2:s.Nx-1
        @test isapprox(Dzz[i, j, k], γ;            atol=1e-9)
        @test isapprox(Dzx[i, j, k], expected_dzx; atol=1e-9)
        @test isapprox(Dzy[i, j, k], 0.0;          atol=1e-12)
    end
end
