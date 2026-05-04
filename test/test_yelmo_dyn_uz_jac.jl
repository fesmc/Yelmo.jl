## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3h / commit C2 unit tests for `calc_uz_3D_jac!`. Confirms
# the kinematic BC sign convention, the column-integration of
# divergence, and the sigma-coordinate `uz_star` correction terms.
#
# Audit of the Fortran source (saved 2026-05-04) flagged a possible
# sign issue in `+ f_bmb*bmb`. The user clarified: not a bug — Greve-
# Blatter Eq. 5.31 uses `-a_b` with `a_b` ablation-positive, while
# Yelmo uses `bmb` accumulation-positive, so `-a_b = +bmb`. The
# bmb-sign test here confirms the convention is consistent: with
# `bmb = -1 m/yr` (basal melt), expected `uz_b = -1 m/yr` (ice
# descends to fill the melted volume).

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior

# Build a small test state with prescribed forcing for uz tests.
function _make_uz_state(Nx, Ny, Nz; H_const=2000.0, dx=1e3, dy=1e3)
    g_2d = RectilinearGrid(size=(Nx, Ny),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy),
                           topology=(Bounded, Bounded, Flat))

    zeta_aa = collect(range(0.5/Nz, 1.0 - 0.5/Nz, length=Nz))
    zeta_ac = vcat(0.0, 0.5*(zeta_aa[1:end-1].+zeta_aa[2:end]), 1.0)

    g_3d = RectilinearGrid(size=(Nx, Ny, Nz),
                           x=(0.0, Nx*dx), y=(0.0, Ny*dy), z=zeta_ac,
                           topology=(Bounded, Bounded, Bounded))

    ux        = XFaceField(g_3d)
    uy        = YFaceField(g_3d)
    jvel_dxx  = CenterField(g_3d)   # acx-staggered, CenterField storage
    jvel_dyy  = CenterField(g_3d)   # acy-staggered
    uz        = ZFaceField(g_3d)
    uz_star   = ZFaceField(g_3d)

    H_ice  = CenterField(g_2d); fill!(interior(H_ice), H_const)
    f_ice  = CenterField(g_2d); fill!(interior(f_ice), 1.0)
    smb    = CenterField(g_2d)
    bmb    = CenterField(g_2d)
    dHdt   = CenterField(g_2d)
    dzsdt  = CenterField(g_2d)
    dzsdx  = CenterField(g_2d)
    dzsdy  = CenterField(g_2d)
    dzbdx  = CenterField(g_2d)
    dzbdy  = CenterField(g_2d)

    return (; g_2d, g_3d, zeta_aa, zeta_ac, dx, dy, H_const, Nx, Ny, Nz,
              ux, uy, jvel_dxx, jvel_dyy, uz, uz_star,
              H_ice, f_ice, smb, bmb, dHdt, dzsdt,
              dzsdx, dzsdy, dzbdx, dzbdy)
end

_call_uz!(s; use_bmb=true) = calc_uz_3D_jac!(
    s.uz, s.uz_star,
    s.ux, s.uy,
    s.jvel_dxx, s.jvel_dyy,
    s.H_ice, s.f_ice,
    s.smb, s.bmb, s.dHdt, s.dzsdt,
    s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
    s.zeta_aa, s.zeta_ac,
    s.dx, s.dy, use_bmb)

# ======================================================================
# Test 1 — bmb-sign convention confirmation (audit follow-up)
# ======================================================================
#
# Static slab, flat bed, no flow, all forcing zero EXCEPT `bmb = -1 m/yr`
# (basal melt). Yelmo convention: bmb positive = accumulation, so
# `-1` means melt. The kinematic BC at a stationary lower surface with
# basal accumulation `a_b = +bmb`:
#
#     uz_b = ∂z_b/∂t + u·∇z_b - a_b
#          = 0 + 0 - bmb
#          = +1 m/yr  (in accumulation-positive convention this is wrong)
#
# But Yelmo's Fortran formula (with `+f_bmb*bmb` per user clarification)
# gives:
#
#     uz_b = dzbdt + uz_grid + f_bmb·bmb + ux·∇z_b
#          = 0 + 0 + (-1) + 0
#          = -1 m/yr
#
# Physical interpretation in Yelmo's convention: as the bed melts away
# (mass loss at base), the ice column above descends to fill the void,
# so `uz_b < 0`. The test below confirms `uz_b = -1` (Yelmo's sign).

@testset "uz_jac: bmb-sign convention (basal melt → uz_b = -bmb)" begin
    s = _make_uz_state(6, 5, 4)
    fill!(interior(s.bmb), -1.0)   # -1 m/yr basal melt

    _call_uz!(s; use_bmb=true)

    UZ = interior(s.uz)
    # Interior cells (away from any margin effects)
    for j in 2:s.Ny-1, i in 2:s.Nx-1
        @test isapprox(UZ[i, j, 1], -1.0; atol=1e-12)
    end

    # use_bmb = false should give uz_b = 0 everywhere (bmb gated off).
    fill!(interior(s.uz), 0.0)
    _call_uz!(s; use_bmb=false)
    UZ = interior(s.uz)
    for j in 2:s.Ny-1, i in 2:s.Nx-1
        @test isapprox(UZ[i, j, 1], 0.0; atol=1e-12)
    end
end

# ======================================================================
# Test 2 — Static slab, no forcing (regression)
# ======================================================================
#
# All inputs zero; expected: uz = 0, uz_star = 0 everywhere.

@testset "uz_jac: static slab (all zero forcing)" begin
    s = _make_uz_state(6, 5, 4)
    _call_uz!(s)
    @test all(isapprox.(interior(s.uz),      0.0; atol=1e-12))
    @test all(isapprox.(interior(s.uz_star), 0.0; atol=1e-12))
end

# ======================================================================
# Test 3 — Incompressible 2D extension `ux = a·x, uy = -a·y`
# ======================================================================
#
# Divergence-free in the horizontal: dudx = +a, dvdy = -a, sum = 0.
# Flat bed/surface, dHdt = dzsdt = bmb = smb = 0 → uz = 0 throughout.
# uz_star also zero (no time-varying terms, no slope-driven c_x/c_y).

@testset "uz_jac: incompressible 2D extension → uz = 0" begin
    s = _make_uz_state(8, 6, 5)
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

    # Populate jvel_dxx, jvel_dyy via the existing kernel (since uz_jac
    # consumes precomputed jvel).
    jvel_dxy = CenterField(s.g_3d); jvel_dxz = CenterField(s.g_3d)
    jvel_dyx = CenterField(s.g_3d); jvel_dyz = CenterField(s.g_3d)
    calc_jacobian_vel_3D_uxyterms!(
        s.jvel_dxx, jvel_dxy, jvel_dxz,
        jvel_dyx, s.jvel_dyy, jvel_dyz,
        s.ux, s.uy,
        s.H_ice, s.f_ice,
        s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
        s.zeta_aa, s.dx, s.dy)

    _call_uz!(s)

    UZ = interior(s.uz); UZS = interior(s.uz_star)
    # Interior cells (away from one-sided FD margins).
    for k in 1:length(s.zeta_ac), j in 3:s.Ny-2, i in 3:s.Nx-2
        @test isapprox(UZ[i, j, k],  0.0; atol=1e-9)
        @test isapprox(UZS[i, j, k], 0.0; atol=1e-9)
    end
end

# ======================================================================
# Test 4 — Pure horizontal divergence `ux = a·x, uy = 0`
# ======================================================================
#
# dudx = +a (positive divergence), dvdy = 0. Continuity:
#
#     uz(zeta_ac[k]) = uz_b - H · zeta_ac[k] · a
#
# With uz_b = 0 (no bed motion, no bmb, no slope) and zeta_ac[1] = 0:
#   uz at face k: -H · zeta_ac[k] · a (linear in zeta).

@testset "uz_jac: divergent column → linear uz profile" begin
    s = _make_uz_state(10, 6, 6)
    a = 1e-4
    H_const = s.H_const

    UX = interior(s.ux)
    for i in axes(UX, 1)
        x_face = (i - 1) * s.dx
        UX[i, :, :] .= a * x_face
    end
    # uy stays zero.

    jvel_dxy = CenterField(s.g_3d); jvel_dxz = CenterField(s.g_3d)
    jvel_dyx = CenterField(s.g_3d); jvel_dyz = CenterField(s.g_3d)
    calc_jacobian_vel_3D_uxyterms!(
        s.jvel_dxx, jvel_dxy, jvel_dxz,
        jvel_dyx, s.jvel_dyy, jvel_dyz,
        s.ux, s.uy,
        s.H_ice, s.f_ice,
        s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
        s.zeta_aa, s.dx, s.dy)

    _call_uz!(s)

    UZ = interior(s.uz)
    # Check interior columns
    for j in 3:s.Ny-2, i in 4:s.Nx-3
        @test isapprox(UZ[i, j, 1], 0.0; atol=1e-9)
        for k in 2:length(s.zeta_ac)
            expected = -H_const * s.zeta_ac[k] * a
            @test isapprox(UZ[i, j, k], expected; atol=1e-7)
        end
    end
end

# ======================================================================
# Test 5 — Inflating column kinematic test
# ======================================================================
#
# No flow (ux = uy = 0), ice surface lifting at +1 m/yr while bed
# stationary: dzsdt = +1, dHdt = +1 → dzbdt = dzsdt - dHdt = 0.
# bmb = smb = 0. Expected:
#   - uz_b = 0 (kinematic BC: stationary bed, no melt, no flow).
#   - uz at every face above bed: 0 (no horizontal divergence).
#   - uz_star: c_t = -((1-zk)*0 + zk*1) = -zk; uz_star = 0 + 0 + uz - zk
#     → at zeta=0 (k=1): uz_star = -0 = 0
#     → at zeta=1 (k=Nz_ac): uz_star = -1 m/yr

@testset "uz_jac: inflating column kinematic (uz_star surface = -dzsdt)" begin
    s = _make_uz_state(6, 5, 4)
    fill!(interior(s.dzsdt), +1.0)
    fill!(interior(s.dHdt),  +1.0)

    _call_uz!(s)

    UZ = interior(s.uz); UZS = interior(s.uz_star)
    Nz_ac = length(s.zeta_ac)

    for j in 2:s.Ny-1, i in 2:s.Nx-1
        # uz = 0 throughout (no flow, no bed motion).
        for k in 1:Nz_ac
            @test isapprox(UZ[i, j, k], 0.0; atol=1e-12)
        end
        # uz_star at the bed face (zeta = 0): c_t = 0 → uz_star = 0.
        @test isapprox(UZS[i, j, 1], 0.0; atol=1e-12)
        # uz_star at the surface face (zeta = 1): c_t = -dzsdt = -1.
        @test isapprox(UZS[i, j, Nz_ac], -1.0; atol=1e-12)
    end
end

# ======================================================================
# Test 6 — Tilted slab + uniform sliding (advection-of-bed term)
# ======================================================================
#
# `ux = u0` constant, sloped bed `dzbdx = α`, all other forcing zero.
# Kinematic BC: uz_b = 0 + 0 + 0 + u0·α + 0 = u0·α.
# (Tests that the advection-of-bed terms `ux·dzbdx` are wired correctly.)

@testset "uz_jac: sliding over tilted bed → uz_b = u0·α" begin
    s = _make_uz_state(8, 6, 4)
    u0 = 100.0   # m/yr
    α  = 0.005   # bed slope, dimensionless

    fill!(interior(s.ux), u0)
    fill!(interior(s.dzbdx), α)
    # Surface stays flat for this test.
    # Set H_ice and ice-mask present everywhere already.

    # Compute jvel for completeness — though for uniform ux, dxx = 0.
    jvel_dxy = CenterField(s.g_3d); jvel_dxz = CenterField(s.g_3d)
    jvel_dyx = CenterField(s.g_3d); jvel_dyz = CenterField(s.g_3d)
    calc_jacobian_vel_3D_uxyterms!(
        s.jvel_dxx, jvel_dxy, jvel_dxz,
        jvel_dyx, s.jvel_dyy, jvel_dyz,
        s.ux, s.uy,
        s.H_ice, s.f_ice,
        s.dzsdx, s.dzsdy, s.dzbdx, s.dzbdy,
        s.zeta_aa, s.dx, s.dy)

    _call_uz!(s)

    UZ = interior(s.uz)
    expected_uz_b = u0 * α   # 0.5 m/yr — within ±10 limit
    for j in 3:s.Ny-2, i in 3:s.Nx-2
        @test isapprox(UZ[i, j, 1], expected_uz_b; atol=1e-9)
    end
end

# ======================================================================
# Test 7 — Ice-free branch
# ======================================================================
#
# `f_ice = 0` everywhere → uz = dzbdt - max(smb, 0) per layer.
# Set dzsdt = +0.5, dHdt = +0.2 → dzbdt = +0.3. smb = +0.1 → max = 0.1.
# Expected uz = 0.3 - 0.1 = 0.2 m/yr at every face.

@testset "uz_jac: ice-free branch → uz = dzbdt - max(smb, 0)" begin
    s = _make_uz_state(6, 5, 4)
    fill!(interior(s.f_ice), 0.0)        # no ice
    fill!(interior(s.dzsdt), 0.5)
    fill!(interior(s.dHdt),  0.2)
    fill!(interior(s.smb),   0.1)

    _call_uz!(s)

    UZ = interior(s.uz); UZS = interior(s.uz_star)
    Nz_ac = length(s.zeta_ac)
    expected = (0.5 - 0.2) - max(0.1, 0.0)   # 0.2 m/yr
    for j in 2:s.Ny-1, i in 2:s.Nx-1, k in 1:Nz_ac
        @test isapprox(UZ[i, j, k],  expected; atol=1e-12)
        @test isapprox(UZS[i, j, k], expected; atol=1e-12)   # uz_star = uz here
    end
end

# ======================================================================
# Test 8 — Error stubs for uz_method ∈ {1, 2}
# ======================================================================

@testset "uz_jac: legacy uz_method stubs throw" begin
    @test_throws ErrorException calc_uz_3D!()
    @test_throws ErrorException calc_uz_3D_aa!()
end
