## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# mat 1.3 unit tests for the rate-factor family.
#
# All three Fortran routines are `elemental`, so the Julia ports are
# pure per-cell formulas wrapped in 3D `CenterField` kernels. Tests
# mirror the structure: hand-derived expected values from a chosen
# `(T_ice, T_pmp, enh, T0)` triple, plus the documented edge cases.
#
#   - calc_rate_factor!
#       * branch threshold at T_prime = 263.15 K (low/high-T Arrhenius)
#       * enh linearity:  ATT(c·enh) = c · ATT(enh)
#       * T_prime FLOOR clamp at 220 K (Fortran line 414)
#       * T_prime CEILING clamp at T_pmp (Fortran line 415)
#
#   - calc_rate_factor_eismint!
#       * branch threshold at T_prime = 263.15 K
#       * NO T_prime clamps — Fortran lines 451-455
#
#   - scale_rate_factor_water!
#       * omega = 0 → unchanged
#       * omega > 0 → ATT *= (1 + 181.25·omega)  (in-place)

using Test
using Yelmo
using Oceananigans
using Oceananigans.Fields: interior


# Fortran-side Arrhenius constants (mirrored here so tests are
# independent of the Julia internals).
const T_LIM = 263.15
const R     = 8.314

# Greve & Blatter constants
const GB_A0_1 = 1.25671e-5
const GB_A0_2 = 6.0422976e10
const GB_Q_1  =  60.0e3
const GB_Q_2  = 139.0e3

# EISMINT2 constants
const E2_A0_1 = 1.139205e-5
const E2_A0_2 = 5.459348198e10
const E2_Q_1  =  60.0e3
const E2_Q_2  = 139.0e3


# Allocate a tiny 3D CenterField stack of (ATT, T_ice, T_pmp, enh) for
# unit tests.
function _make_rf_state(Nx, Ny, Nz)
    g_3d = RectilinearGrid(size=(Nx, Ny, Nz),
                           x=(0.0, 1.0), y=(0.0, 1.0), z=(0.0, 1.0),
                           topology=(Bounded, Bounded, Bounded))
    return (; g_3d, Nx, Ny, Nz,
              ATT   = CenterField(g_3d),
              T_ice = CenterField(g_3d),
              T_pmp = CenterField(g_3d),
              enh   = CenterField(g_3d),
              omega = CenterField(g_3d))
end


@testset "calc_rate_factor! — Greve & Blatter Arrhenius branches" begin
    s = _make_rf_state(2, 2, 2)
    T0    = 273.15
    T_pmp = 273.15           # standard pressure-melting point
    enh   = 1.0

    # Pick T_ice so T_prime sits below T_LIM (low-T branch).
    T_low  = 240.0           # T_prime = T_low - T_pmp + T0 = 240.0
    fill!(interior(s.T_pmp), T_pmp)
    fill!(interior(s.enh),   enh)
    fill!(interior(s.T_ice), T_low)

    calc_rate_factor!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)
    expected_low = enh * GB_A0_1 * exp(-GB_Q_1 / (R * T_low))
    @test all(interior(s.ATT) .≈ expected_low)

    # Now high-T branch: pick T_ice so T_prime > T_LIM. Note the
    # `min(T_prime, T_pmp)` clamp will fire if T_prime > T_pmp, so
    # stay below T_pmp. Use T_prime = 270 K.
    T_high = 270.0
    fill!(interior(s.T_ice), T_high)
    calc_rate_factor!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)
    expected_high = enh * GB_A0_2 * exp(-GB_Q_2 / (R * T_high))
    @test all(interior(s.ATT) .≈ expected_high)
end

@testset "calc_rate_factor! — enh linearity" begin
    s1 = _make_rf_state(2, 2, 2)
    s2 = _make_rf_state(2, 2, 2)
    T0    = 273.15
    T_pmp = 273.15
    T_ice = 250.0

    fill!(interior(s1.T_pmp), T_pmp); fill!(interior(s1.T_ice), T_ice)
    fill!(interior(s2.T_pmp), T_pmp); fill!(interior(s2.T_ice), T_ice)
    fill!(interior(s1.enh),   1.0)
    fill!(interior(s2.enh),   2.5)

    calc_rate_factor!(s1.ATT, s1.T_ice, s1.T_pmp, s1.enh, T0)
    calc_rate_factor!(s2.ATT, s2.T_ice, s2.T_pmp, s2.enh, T0)

    @test all(interior(s2.ATT) .≈ 2.5 .* interior(s1.ATT))
end

@testset "calc_rate_factor! — T_prime floor at 220 K" begin
    # T_prime would be 219 K without clamp (T_ice=219, T_pmp=273.15,
    # T0=273.15). With the floor, the kernel evaluates Arrhenius at 220 K.
    s = _make_rf_state(1, 1, 1)
    T0    = 273.15
    T_pmp = 273.15
    T_ice = 219.0
    fill!(interior(s.T_pmp), T_pmp)
    fill!(interior(s.T_ice), T_ice)
    fill!(interior(s.enh),   1.0)

    calc_rate_factor!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)

    # T_prime is clamped to 220 K, which sits in the low-T branch.
    expected = GB_A0_1 * exp(-GB_Q_1 / (R * 220.0))
    @test interior(s.ATT)[1, 1, 1] ≈ expected
end

@testset "calc_rate_factor! — T_prime ceiling at T_pmp" begin
    # T_prime would be 280 K without clamp (T_ice=280, T_pmp=273.15,
    # T0=273.15). With the cap, the kernel evaluates Arrhenius at T_pmp.
    s = _make_rf_state(1, 1, 1)
    T0    = 273.15
    T_pmp = 273.15
    T_ice = 280.0
    fill!(interior(s.T_pmp), T_pmp)
    fill!(interior(s.T_ice), T_ice)
    fill!(interior(s.enh),   1.0)

    calc_rate_factor!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)

    # T_prime clamped to T_pmp = 273.15 K (high-T branch since > T_LIM).
    expected = GB_A0_2 * exp(-GB_Q_2 / (R * T_pmp))
    @test interior(s.ATT)[1, 1, 1] ≈ expected
end


@testset "calc_rate_factor_eismint! — Arrhenius branches" begin
    s = _make_rf_state(2, 2, 2)
    T0    = 273.15
    T_pmp = 273.15
    enh   = 1.0

    # Low-T branch: T_prime = 240 K.
    T_low = 240.0
    fill!(interior(s.T_pmp), T_pmp); fill!(interior(s.enh), enh)
    fill!(interior(s.T_ice), T_low)
    calc_rate_factor_eismint!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)
    expected_low = enh * E2_A0_1 * exp(-E2_Q_1 / (R * T_low))
    @test all(interior(s.ATT) .≈ expected_low)

    # High-T branch: T_prime = 270 K.
    T_high = 270.0
    fill!(interior(s.T_ice), T_high)
    calc_rate_factor_eismint!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)
    expected_high = enh * E2_A0_2 * exp(-E2_Q_2 / (R * T_high))
    @test all(interior(s.ATT) .≈ expected_high)
end

@testset "calc_rate_factor_eismint! — no T_prime clamps" begin
    # Same temperatures as the GB-clamp tests above. EISMINT2 must
    # evaluate Arrhenius at the unclamped T_prime values.
    s = _make_rf_state(1, 1, 1)
    T0    = 273.15
    T_pmp = 273.15

    # T_prime = 219 K — below the GB floor of 220, but EISMINT2 doesn't
    # clamp, so the kernel should evaluate at 219 K.
    fill!(interior(s.T_pmp), T_pmp)
    fill!(interior(s.enh),   1.0)
    fill!(interior(s.T_ice), 219.0)
    calc_rate_factor_eismint!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)
    expected_under = E2_A0_1 * exp(-E2_Q_1 / (R * 219.0))
    @test interior(s.ATT)[1, 1, 1] ≈ expected_under

    # T_prime = 280 K — above T_pmp, but EISMINT2 doesn't cap.
    fill!(interior(s.T_ice), 280.0)
    calc_rate_factor_eismint!(s.ATT, s.T_ice, s.T_pmp, s.enh, T0)
    expected_over = E2_A0_2 * exp(-E2_Q_2 / (R * 280.0))
    @test interior(s.ATT)[1, 1, 1] ≈ expected_over
end


@testset "scale_rate_factor_water! — omega = 0 leaves ATT unchanged" begin
    s = _make_rf_state(2, 2, 2)
    fill!(interior(s.ATT),   3.0e-17)
    fill!(interior(s.omega), 0.0)
    A_before = copy(interior(s.ATT))

    scale_rate_factor_water!(s.ATT, s.omega)

    @test all(interior(s.ATT) .== A_before)
end

@testset "scale_rate_factor_water! — omega > 0 scales by 1+181.25·omega" begin
    s = _make_rf_state(2, 2, 2)
    fill!(interior(s.ATT),   3.0e-17)
    omega_val = 0.01
    fill!(interior(s.omega), omega_val)

    scale_rate_factor_water!(s.ATT, s.omega)

    expected_factor = 1.0 + 181.25 * omega_val
    @test all(interior(s.ATT) .≈ 3.0e-17 * expected_factor)
end

@testset "scale_rate_factor_water! — mixed mask (some cells gated)" begin
    # omega < 0 (or == 0) cells are gated — only positive-omega cells
    # are scaled. Use a 2x2x1 grid with three different omega values.
    s = _make_rf_state(2, 2, 1)
    A = interior(s.ATT)
    Ω = interior(s.omega)

    A[1, 1, 1] = 1.0e-17;  Ω[1, 1, 1] = 0.0
    A[2, 1, 1] = 2.0e-17;  Ω[2, 1, 1] = -0.05    # gated (≤ 0)
    A[1, 2, 1] = 3.0e-17;  Ω[1, 2, 1] = 0.005
    A[2, 2, 1] = 4.0e-17;  Ω[2, 2, 1] = 0.02

    scale_rate_factor_water!(s.ATT, s.omega)

    @test A[1, 1, 1] ≈ 1.0e-17                               # untouched (omega=0)
    @test A[2, 1, 1] ≈ 2.0e-17                               # untouched (omega<0)
    @test A[1, 2, 1] ≈ 3.0e-17 * (1.0 + 181.25 * 0.005)
    @test A[2, 2, 1] ≈ 4.0e-17 * (1.0 + 181.25 * 0.02)
end
