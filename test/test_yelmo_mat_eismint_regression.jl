## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# mat 1.6 single-step regression of `mat_step!` against the Fortran
# YelmoMirror EISMINT-1 moving-margin fixture at t = 25 kyr.
#
# The test loads the fixture into a `YelmoModel`, snapshots the
# Mirror-written `mat` fields, and runs ONE `mat_step!` against the
# loaded `dyn` / `tpo` state. Each output field is compared cell-by-
# cell to the Mirror reference using a relative L∞ metric with
# per-field tolerances calibrated to observed Yelmo.jl deviation:
#
#   - `ATT`, `ATT_bar` — exact-up-to-NetCDF-Float32 (rf_method = 0
#                       with rf_const = 1e-16 matches Mirror's
#                       namelist; both fields are a single constant).
#   - `enh`, `enh_bar` — exact (Mirror uses enh_shear = enh_stream =
#                       enh_shlf = 1.0, so enh = 1.0 everywhere).
#   - `visc`           — Glen viscosity from `dyn.strn_de` and ATT;
#                       tight (~1e-6) since both sides see the same
#                       inputs after restart load.
#   - `visc_bar`       — depth-average via Yelmo.jl's constant-extrap
#                       Center stagger vs Fortran's full-aa-grid
#                       trapezoidal. Observed deviation ~4e-5.
#   - `visc_int`       — `visc_bar · H_ice`; observed ~1% relative
#                       L∞ at the margin (H_ice gradient amplifies
#                       the trapz convention difference).
#
# Stress fields (`strs2D_*`) are NOT compared against Mirror in this
# test. The two implementations write different stress values to a
# single restart slot due to a one-step lag: Mirror computes
# `strs2D = 2 · visc_bar_{n-1} · strn2D_{n-1}` AT THE START of step
# n, BEFORE updating `visc_bar` to `visc_bar_n`. The fixture stores
# `visc_bar_n` (post-update). Loading that visc_bar into Yelmo.jl
# and running mat_step! produces a self-consistent stress
# `2 · visc_bar_n · strn2D_n`, which is offset from Mirror's
# `2 · visc_bar_{n-1} · strn2D_{n-1}` by step-to-step variation
# (~10-17% rel L∞ on a transient run). This is a fixture-shape
# limitation, not a physics divergence — the stress formula itself
# is unit-tested against analytical inputs in
# `test/test_yelmo_mat_stress.jl`. An internal-consistency check
# below verifies that `strs2D_txx == 2 · visc_bar_loaded · dxx`
# holds bit-exactly when `visc_bar_loaded` is the value Yelmo.jl
# saw at stress-compute time.
#
# The test isolates `mat_step!` from `dyn_step!` divergence: by
# loading the Mirror state before running mat, both Yelmo.jl and the
# reference start from the same `dyn.strn` / `tpo.{H_ice, f_grnd,
# f_ice}`, so the only deviation is in the mat physics itself.

using Test
using Statistics
using NCDatasets
using Yelmo
using Yelmo.YelmoModelPar: YelmoModelParameters, ymat_params, ydyn_params,
                           ytill_params, yneff_params, ytopo_params
using Oceananigans.Fields: interior

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "benchmarks", "fixtures"))
const FIXTURE_PATH = joinpath(FIXTURES_DIR, "eismint_moving_t25000.nc")


# Yelmo.jl-side parameters matching the EISMINT-moving Fortran namelist
# (`test/benchmarks/specs/yelmo_EISMINT_moving.nml`):
#
#   - rf_method = 0, rf_const = 1e-16
#   - n_glen = 3.0, visc_min = 1e3
#   - enh_method = "shear3D" with enh_shear = enh_stream = enh_shlf = 1.0
#
# `mat_step!` honours these to build ATT, enh, visc, visc_bar,
# visc_int, and strs2D_* from the loaded Mirror state.
function _mat_regression_params()
    return YelmoModelParameters("mat_eismint_regression";
        ydyn = ydyn_params(solver = "sia",
                           uz_method = 3,
                           visc_method = 1,
                           eps_0 = 1e-6),
        yneff = yneff_params(method = 0, const_ = 1.0),
        ytill = ytill_params(method = -1),
        ytopo = ytopo_params(solver = "expl"),
        ymat  = ymat_params(
            rf_method  = 0,
            rf_const   = 1e-16,
            n_glen     = 3.0,
            visc_min   = 1e3,
            de_max     = 0.5,
            enh_method = "shear3D",
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
    )
end


# Relative L∞ on field interiors. Same helper used by the lockstep test.
function _rel_linf_inner(a, b)
    @assert size(a) == size(b)
    diff = maximum(abs.(a .- b))
    ref  = maximum(abs.(b))
    return ref > 0 ? diff / ref : diff
end


# Snapshot a single 2D / 3D mat field from the fixture as a plain
# `Array{Float64}` matching the model's interior shape.
function _read_fixture_field(ds, name)
    haskey(ds, name) || return nothing
    raw = ds[name][:, :, :]
    # Drop the trailing singleton time dim if present.
    return Array{Float64}(raw)
end


@testset "mat 1.6: single-step mat_step! regression vs EISMINT-moving fixture" begin
    isfile(FIXTURE_PATH) || error(
        "mat regression: EISMINT-moving t=25000 fixture missing at $FIXTURE_PATH. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl eismint_moving --overwrite` first.")

    p = _mat_regression_params()
    y = YelmoModel(FIXTURE_PATH, 25000.0;
                   alias = "mat_regression",
                   p     = p,
                   strict = false)

    # Snapshot Mirror's mat fields BEFORE running mat_step!. These
    # are the references for the comparison.
    ref = NCDataset(FIXTURE_PATH, "r") do ds
        (
            ATT      = _read_fixture_field(ds, "ATT"),
            ATT_bar  = _read_fixture_field(ds, "ATT_bar"),
            enh      = _read_fixture_field(ds, "enh"),
            enh_bar  = _read_fixture_field(ds, "enh_bar"),
            visc     = _read_fixture_field(ds, "visc"),
            visc_bar = _read_fixture_field(ds, "visc_bar"),
            visc_int = _read_fixture_field(ds, "visc_int"),
        )
    end

    # Snapshot the loaded `visc_bar` and `strn2D_*` BEFORE mat_step!
    # so the internal-consistency stress check below has access to
    # the inputs `mat_step!` actually saw at stress-compute time.
    snap_visc_bar  = Array(interior(y.mat.visc_bar))[:, :, 1]
    snap_dxx       = Array(interior(y.dyn.strn2D_dxx))[:, :, 1]
    snap_dyy       = Array(interior(y.dyn.strn2D_dyy))[:, :, 1]
    snap_dxy       = Array(interior(y.dyn.strn2D_dxy))[:, :, 1]

    # Run mat_step!. Reads y.dyn.strn*, y.tpo.{H_ice, f_grnd, f_ice};
    # writes y.mat.{ATT, enh, visc, visc_bar, visc_int, strs2D_*,
    # enh_bar, ATT_bar}.
    Yelmo.mat_step!(y, 0.0)

    # ---------------- ATT (rf_method = 0 → constant) ----------------
    # NetCDF stores fields as Float32 by default; round-trip from
    # the rf_const = 1e-16 Float64 through Float32 produces a small
    # (~1e-8) deviation that's unavoidable.
    if ref.ATT !== nothing
        rel = _rel_linf_inner(Array(interior(y.mat.ATT)), ref.ATT)
        @info "mat regression: ATT" rel
        @test rel ≤ 1e-7
    end
    if ref.ATT_bar !== nothing
        rel = _rel_linf_inner(Array(interior(y.mat.ATT_bar)), ref.ATT_bar)
        @info "mat regression: ATT_bar" rel
        @test rel ≤ 1e-7
    end

    # ---------------- enh / enh_bar (all enh_* = 1 → 1.0 everywhere) -
    if ref.enh !== nothing
        rel = _rel_linf_inner(Array(interior(y.mat.enh)), ref.enh)
        @info "mat regression: enh" rel
        @test rel == 0.0
    end
    if ref.enh_bar !== nothing
        rel = _rel_linf_inner(Array(interior(y.mat.enh_bar)), ref.enh_bar)
        @info "mat regression: enh_bar" rel
        @test rel == 0.0
    end

    # ---------------- visc (Glen — function of dyn.strn_de + ATT) ---
    # Glen formula is exact pointwise; deviation comes from float
    # rounding only since both sides see the same de and ATT.
    if ref.visc !== nothing
        rel = _rel_linf_inner(Array(interior(y.mat.visc)), ref.visc)
        @info "mat regression: visc" rel
        @test rel ≤ 1e-5
    end

    # ---------------- visc_bar (depth average) ----------------------
    # Yelmo.jl uses constant-extrap Center stagger; Fortran uses
    # full-aa-grid trapezoidal (zeta_aa includes endpoints). Observed
    # ~4e-5 rel L∞ on EISMINT at t=25 kyr.
    if ref.visc_bar !== nothing
        rel = _rel_linf_inner(Array(interior(y.mat.visc_bar)), ref.visc_bar)
        @info "mat regression: visc_bar" rel
        @test rel ≤ 1e-3
    end

    # ---------------- visc_int (depth-integrated · H_ice) -----------
    # Same trapezoidal-convention deviation as visc_bar but amplified
    # by H_ice gradient at the dome margin. Observed ~1.25%.
    if ref.visc_int !== nothing
        rel = _rel_linf_inner(Array(interior(y.mat.visc_int)), ref.visc_int)
        @info "mat regression: visc_int" rel
        @test rel ≤ 0.05
    end

    # ---------------- Stress tensor internal consistency ------------
    # Verify that Yelmo.jl's strs2D fields equal the formula
    # `2 · visc_bar_loaded · strn2D_loaded` bit-exactly. This catches
    # any kernel bug in `calc_stress_tensor_2D!` (loop bounds, sign
    # flips, etc.) without comparing against Mirror's step-lagged
    # stress. (Stress vs Mirror is fundamentally non-comparable from
    # a single-step fixture — see file-header comment.)
    Txx = Array(interior(y.mat.strs2D_txx))[:, :, 1]
    Tyy = Array(interior(y.mat.strs2D_tyy))[:, :, 1]
    Txy = Array(interior(y.mat.strs2D_txy))[:, :, 1]
    @test maximum(abs.(Txx .- 2.0 .* snap_visc_bar .* snap_dxx)) ≤ 1e-12 * maximum(abs.(Txx))
    @test maximum(abs.(Tyy .- 2.0 .* snap_visc_bar .* snap_dyy)) ≤ 1e-12 * maximum(abs.(Tyy))
    @test maximum(abs.(Txy .- 2.0 .* snap_visc_bar .* snap_dxy)) ≤ 1e-12 * maximum(abs.(Txy))
end
