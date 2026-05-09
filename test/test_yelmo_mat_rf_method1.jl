## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Wiring test for `rf_method = 1` in `mat_step!`
# (`src/mat/YelmoModelMat.jl`).
#
# `rf_method = 1` is the temperature- (and optionally water-content-)
# coupled Arrhenius branch. It reads `y.thrm.{T_ice, T_pmp, omega}` and
# `y.mat.enh` and writes `y.mat.{ATT, ATT_b, ATT_s, ATT_bar}`. The per-
# cell formula and clamps are unit-tested in
# `test_yelmo_mat_rate_factor.jl`; this test exercises the full
# `mat_step!` dispatch end-to-end and checks:
#
#   - The right Arrhenius variant is selected by `rf_use_eismint2`
#     (Greve & Blatter 2009 vs EISMINT2 / Payne 2000).
#   - `rf_with_water = true` triggers the Lliboutry & Duval water scale.
#   - `ATT_b` / `ATT_s` are refreshed (Path B constant-extrapolation off
#     the basal / surface 3D layers).
#   - `ATT_bar` is the depth-average of `ATT` with explicit boundaries —
#     same kernel `mat_step!` calls. The test reproduces the call so a
#     wrong field / argument order in the wiring is caught.
#
# Test inputs are uniform thermal fields (`T_ice`, `T_pmp`, `omega`)
# overwritten directly on `y.thrm` after fixture load, so the
# rate-factor output can be checked against the per-cell formula
# without depending on the therm port producing a specific T-profile.
# A full Mirror-lockstep test against an EISMINT2-expA fixture is the
# planned follow-up.

using Test
using Statistics
using NCDatasets
using Yelmo
using Yelmo.YelmoModelPar: YelmoModelParameters, ymat_params, ydyn_params,
                           ytill_params, yneff_params, ytopo_params,
                           ytherm_params
using Oceananigans.Fields: interior
using Oceananigans.Grids: znodes, Center

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "benchmarks", "fixtures"))
const FIXTURE_PATH = joinpath(FIXTURES_DIR, "eismint_moving_t25000.nc")

# Greve & Blatter (2009) Arrhenius constants — duplicated here so the
# expected-value computation is independent of the Julia internals.
const _GB_A0_1, _GB_A0_2 = 1.25671e-5,    6.0422976e10
const _GB_Q_1,  _GB_Q_2  = 60.0e3,        139.0e3
# EISMINT2 / Payne (2000) Arrhenius constants.
const _E2_A0_1, _E2_A0_2 = 1.139205e-5,   5.459348198e10
const _E2_Q_1,  _E2_Q_2  = 60.0e3,        139.0e3
const _R                 = 8.314
const _T_LIM             = 263.15
const _T_FLOOR           = 220.0           # GB-only T_prime floor
const _WATER_FACTOR      = 181.25

function _expected_att_gb(T_ice::Float64, T_pmp::Float64,
                          enh::Float64, T0::Float64)
    Tp = clamp(T_ice - T_pmp + T0, _T_FLOOR, T_pmp)
    if Tp ≤ _T_LIM
        return enh * _GB_A0_1 * exp(-_GB_Q_1 / (_R * Tp))
    else
        return enh * _GB_A0_2 * exp(-_GB_Q_2 / (_R * Tp))
    end
end

function _expected_att_eismint(T_ice::Float64, T_pmp::Float64,
                               enh::Float64, T0::Float64)
    Tp = T_ice - T_pmp + T0
    if Tp ≤ _T_LIM
        return enh * _E2_A0_1 * exp(-_E2_Q_1 / (_R * Tp))
    else
        return enh * _E2_A0_2 * exp(-_E2_Q_2 / (_R * Tp))
    end
end

# Build a YelmoModelParameters with `rf_method = 1` and the requested
# Arrhenius / water-scale toggles. `enh_method = "simple"` plus a
# uniform `enh_*` makes the post-`mat_step!` `enh` field uniformly
# `enh_uniform`, so the expected ATT depends only on T and the
# Arrhenius constants.
function _rf1_params(; use_eismint2::Bool, with_water::Bool,
                     enh_uniform::Float64 = 1.0)
    return YelmoModelParameters("mat_rf_method1";
        ydyn  = ydyn_params(solver = "sia",
                            uz_method = 3,
                            visc_method = 1,
                            eps_0 = 1e-6),
        yneff = yneff_params(method = 0, const_ = 1.0),
        ytill = ytill_params(method = -1),
        ytopo = ytopo_params(solver = "expl"),
        ymat  = ymat_params(
            rf_method       = 1,
            rf_use_eismint2 = use_eismint2,
            rf_with_water   = with_water,
            n_glen          = 3.0,
            visc_min        = 1e3,
            de_max          = 0.5,
            enh_method      = "simple",
            enh_shear       = enh_uniform,
            enh_stream      = enh_uniform,
            enh_shlf        = enh_uniform,
        ),
        # Therm is a no-op; T_ice / T_pmp / omega are overwritten
        # explicitly below so the test does not depend on the therm
        # path producing a specific profile.
        ytherm = ytherm_params(method = "fixed"),
    )
end

# Build a YelmoModel from the EISMINT-moving fixture, then overwrite
# `y.thrm.{T_ice, T_pmp, omega}` with uniform values so the per-cell
# expected ATT is a single constant.
function _setup_rf1_model(; T_ice::Float64, T_pmp::Float64, omega::Float64,
                          use_eismint2::Bool, with_water::Bool,
                          enh_uniform::Float64 = 1.0,
                          alias::String = "mat_rf1")
    p = _rf1_params(; use_eismint2 = use_eismint2,
                      with_water    = with_water,
                      enh_uniform   = enh_uniform)
    y = YelmoModel(FIXTURE_PATH, 25000.0;
                   alias  = alias,
                   p      = p,
                   strict = false)
    fill!(interior(y.thrm.T_ice), T_ice)
    fill!(interior(y.thrm.T_pmp), T_pmp)
    fill!(interior(y.thrm.omega), omega)
    return y
end

@testset "mat: rf_method=1 wiring (Greve & Blatter, no water)" begin
    isfile(FIXTURE_PATH) || error(
        "rf_method=1 test: EISMINT-moving fixture missing at $FIXTURE_PATH. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "eismint_moving --overwrite` first.")

    # Pick T such that T_prime > 263.15 K to exercise the high-T
    # Arrhenius branch (the more-active branch in the EISMINT regime).
    # `enh_uniform = 1e6` keeps ATT above `TOL_UNDERFLOW = 1e-15` in
    # `vert_int_trapz_boundary!` so `ATT_bar` is non-trivial and the
    # depth-average wiring can be exercised.
    T_ice = 263.0   # K
    T_pmp = 273.15  # K
    omega = 0.0
    enh   = 1.0e6
    y = _setup_rf1_model(; T_ice = T_ice, T_pmp = T_pmp, omega = omega,
                          use_eismint2 = false, with_water = false,
                          enh_uniform  = enh)

    Yelmo.mat_step!(y, 0.0)

    T0       = y.c.T0
    expected = _expected_att_gb(T_ice, T_pmp, enh, T0)
    ATT      = Array(interior(y.mat.ATT))
    ATT_b    = Array(interior(y.mat.ATT_b))[:, :, 1]
    ATT_s    = Array(interior(y.mat.ATT_s))[:, :, 1]
    ATT_bar  = Array(interior(y.mat.ATT_bar))[:, :, 1]

    @info "rf_method=1 GB: ATT extrema" minimum(ATT) maximum(ATT) expected
    @test all(isfinite, ATT)
    @test isapprox(maximum(ATT), expected; rtol = 1e-12)
    @test isapprox(minimum(ATT), expected; rtol = 1e-12)

    # Path B refresh: ATT_b == ATT[:, :, 1], ATT_s == ATT[:, :, Nz].
    Nz = size(ATT, 3)
    @test ATT_b == ATT[:, :, 1]
    @test ATT_s == ATT[:, :, Nz]

    # ATT_bar: depth-average of a uniform 3D ATT (with matching
    # boundaries) collapses to the same constant.
    @test all(isapprox.(ATT_bar, expected; rtol = 1e-12))
end

@testset "mat: rf_method=1 wiring (EISMINT2 variant)" begin
    T_ice = 263.0
    T_pmp = 273.15
    omega = 0.0
    enh   = 1.0e6
    y = _setup_rf1_model(; T_ice = T_ice, T_pmp = T_pmp, omega = omega,
                          use_eismint2 = true, with_water = false,
                          enh_uniform  = enh,
                          alias = "mat_rf1_eismint")

    Yelmo.mat_step!(y, 0.0)

    T0       = y.c.T0
    expected = _expected_att_eismint(T_ice, T_pmp, enh, T0)
    ATT      = Array(interior(y.mat.ATT))

    @info "rf_method=1 EISMINT2: ATT extrema" minimum(ATT) maximum(ATT) expected
    @test all(isfinite, ATT)
    @test isapprox(maximum(ATT), expected; rtol = 1e-12)
    @test isapprox(minimum(ATT), expected; rtol = 1e-12)

    # Sanity: the EISMINT2 prefactors differ from Greve & Blatter by
    # ~10% in the high-T branch — make sure the variant switch is real
    # and not silently picking the same kernel.
    expected_gb = _expected_att_gb(T_ice, T_pmp, enh, T0)
    @test !isapprox(expected, expected_gb; rtol = 1e-3)
end

@testset "mat: rf_method=1 wiring (water-content scaling)" begin
    T_ice = 263.0
    T_pmp = 273.15
    omega = 0.01   # 1% water content
    enh   = 1.0e6
    y = _setup_rf1_model(; T_ice = T_ice, T_pmp = T_pmp, omega = omega,
                          use_eismint2 = false, with_water = true,
                          enh_uniform  = enh,
                          alias = "mat_rf1_water")

    Yelmo.mat_step!(y, 0.0)

    T0       = y.c.T0
    base_att = _expected_att_gb(T_ice, T_pmp, enh, T0)
    expected = base_att * (1.0 + _WATER_FACTOR * omega)
    ATT      = Array(interior(y.mat.ATT))

    @info "rf_method=1 GB+water: ATT extrema" minimum(ATT) maximum(ATT) expected base_att
    @test all(isfinite, ATT)
    @test isapprox(maximum(ATT), expected; rtol = 1e-12)
    @test isapprox(minimum(ATT), expected; rtol = 1e-12)

    # Water scaling must actually move the value off the unscaled base.
    @test !isapprox(expected, base_att; rtol = 1e-3)
end

@testset "mat: rf_method=1 wiring (depth-average with non-uniform T)" begin
    # Set T_ice with a vertical gradient so ATT varies in z, then
    # check ATT_bar matches an independent call to `depth_average!`
    # on the same fields. `enh_uniform = 1e6` keeps ATT above the
    # `TOL_UNDERFLOW = 1e-15` clip in `vert_int_trapz_boundary!`.
    p = _rf1_params(; use_eismint2 = false, with_water = false,
                      enh_uniform   = 1.0e6)
    y = YelmoModel(FIXTURE_PATH, 25000.0;
                   alias  = "mat_rf1_zvar",
                   p      = p,
                   strict = false)

    T_pmp = 273.15
    fill!(interior(y.thrm.T_pmp), T_pmp)
    fill!(interior(y.thrm.omega), 0.0)

    # Vertical T_ice ramp: 240 K at base → 268 K at surface.
    Ti      = interior(y.thrm.T_ice)
    Nx, Ny, Nz = size(Ti)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        f         = (k - 1) / max(Nz - 1, 1)
        Ti[i, j, k] = 240.0 + f * (268.0 - 240.0)
    end

    Yelmo.mat_step!(y, 0.0)

    T0      = y.c.T0
    enh     = 1.0e6
    ATT     = Array(interior(y.mat.ATT))
    ATT_b   = Array(interior(y.mat.ATT_b))[:, :, 1]
    ATT_s   = Array(interior(y.mat.ATT_s))[:, :, 1]
    ATT_bar = Array(interior(y.mat.ATT_bar))[:, :, 1]

    @test ATT_b == ATT[:, :, 1]
    @test ATT_s == ATT[:, :, Nz]

    # Per-cell ATT must match the GB formula on the prescribed T column.
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        exp_val = _expected_att_gb(Ti[i, j, k], T_pmp, enh, T0)
        @test isapprox(ATT[i, j, k], exp_val; rtol = 1e-12)
    end

    # Independent depth-average of the same fields. Using the same
    # kernel `mat_step!` calls, but invoked here directly so the test
    # catches argument-order or argument-identity bugs in the wiring.
    ref_bar = similar(y.mat.ATT_bar)
    fill!(interior(ref_bar), 0.0)
    zeta_aa = Float64.(collect(znodes(y.gt, Center())))
    Yelmo.depth_average!(ref_bar, y.mat.ATT, y.mat.ATT_b, y.mat.ATT_s,
                         zeta_aa)
    ref_bar_arr = Array(interior(ref_bar))[:, :, 1]
    @test isapprox(ATT_bar, ref_bar_arr; rtol = 1e-12, atol = 0.0)

    # Sanity: ATT_bar lies between min and max of the column ATT
    # (depth-average of a positive non-uniform field).
    @test minimum(ATT) ≤ minimum(ATT_bar)
    @test maximum(ATT_bar) ≤ maximum(ATT)
end
