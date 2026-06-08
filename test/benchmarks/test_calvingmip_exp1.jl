## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# CalvingMIP Exp1 (circular domain, velocity-equil calving at r = 750 km)
# — standalone Yelmo.jl benchmark test.
#
# WHAT THIS TEST EXERCISES:
#
#   - CalvingMIPBenchmark struct construction + bed geometry.
#   - `calvmip_exp1!` calving law via `YelmoHooks.calv_flt` — the hook
#     pattern that keeps experiment-specific laws outside Yelmo.jl core.
#   - Full topo + dyn (SSA) + calving (LSF) pipeline on a circular
#     marine domain with SMB-driven ice growth from a cold start.
#   - Smoke: the model runs 1000 yr without error, no NaN, ice grows.
#   - Regression: ice-covered cell count at t=1000 matches the committed
#     YelmoMirror fixture within ±5%.
#
# EXPERIMENT SETUP:
#
#   Domain:    circular, x,y ∈ [−800, 800] km, dx = 25 km → 64 × 64.
#   Bed:       z_bed(r) = 900 − 2900·r²/R0²  (R0 = 800 km, bowl).
#   IC:        H_ice = 0, lsf = +1 (all ocean).
#   Forcing:   smb = 0.3 m/yr, T_srf = 223.15 K, Q_geo = 42 mW/m².
#   Calving:   calvmip_exp1! hook: cr = −u everywhere, zeroed inside
#              r < 750 km except at the 750 km boundary faces.
#
# FIXTURE:
#
#   fixtures/calvingmip_exp1_t1000.nc — committed YelmoMirror reference.
#   Regenerate: julia --project=test test/benchmarks/regenerate.jl \
#                    calvingmip_exp1 --overwrite

using Test
using NCDatasets

include("harness.jl")
using .YelmoBenchmarkHarness

using Yelmo
using Oceananigans: interior

const _SPEC = CalvingMIPBenchmark(:exp1; dx_km=25.0)

# Count ice-covered cells (H_ice > 0) from a YelmoModel.
_ice_cell_count(y) = count(h -> h > 0.0, interior(y.tpo.H_ice))

# Build `YelmoParameters` from the per-experiment Fortran namelist.
function _calvingmip_params(b::CalvingMIPBenchmark)
    return YelmoParameters(YelmoBenchmarkHarness.calvingmip_namelist_path(b),
                                 "calvingmip_$(lowercase(string(b.exp)))")
end

# -----------------------------------------------------------------------
# Bed geometry unit test
# -----------------------------------------------------------------------

@testset "benchmarks: calvmip_bed_circular geometry" begin
    # At r = 0 (centre): z_bed = Bc = 900 m
    @test calvmip_bed_circular(0.0, 0.0) ≈ 900.0

    # At r = R0 = 800 km (rim): z_bed = Bl = −2000 m
    @test calvmip_bed_circular(800e3, 0.0) ≈ -2000.0
    @test calvmip_bed_circular(0.0, 800e3) ≈ -2000.0

    # Parabolic: z_bed(r) = 900 − 2900 · (r/800e3)²
    r = 400e3
    expected = 900.0 - 2900.0 * (r / 800e3)^2
    @test calvmip_bed_circular(r, 0.0) ≈ expected atol=1e-6
    @test calvmip_bed_circular(0.0, r) ≈ expected atol=1e-6
end

# -----------------------------------------------------------------------
# Analytical IC test (t = 0)
# -----------------------------------------------------------------------

@testset "benchmarks: CalvingMIPBenchmark analytical IC (t=0)" begin
    b = _SPEC
    s = state(b, 0.0)

    @test s.xc ≈ b.xc
    @test s.yc ≈ b.yc
    @test all(iszero, s.H_ice)
    @test all(l -> l == 1.0, s.lsf)

    # Bed at domain centre ≈ 900 m (r ≈ 0).
    Nx = length(b.xc)
    Ny = length(b.yc)
    ic = Nx ÷ 2 + 1
    jc = Ny ÷ 2 + 1
    @test s.z_bed[ic, jc] ≈ calvmip_bed_circular(b.xc[ic], b.yc[jc]) atol=1e-6

    @test all(==(b.smb_const),   s.smb_ref)
    @test all(==(b.T_srf_const), s.T_srf)
    @test all(==(b.Q_geo_const), s.Q_geo)
end

# -----------------------------------------------------------------------
# Regression test against YelmoMirror fixture (t = 1000)
# -----------------------------------------------------------------------

const _FIXTURE_T = 1000.0

@testset "benchmarks: CalvingMIPBenchmark exp1 regression (t=1000)" begin
    b = _SPEC
    path = joinpath(@__DIR__, "fixtures",
                    "calvingmip_exp1_t$(Int(_FIXTURE_T)).nc")

    if !isfile(path)
        @warn "CalvingMIP exp1 fixture missing at $path — skipping regression. " *
              "Run `julia --project=test test/benchmarks/regenerate.jl " *
              "calvingmip_exp1 --overwrite` to create it."
        @test_skip true
    else
        s_ref = state(b, _FIXTURE_T)
        n_ref = count(h -> h > 0.0, s_ref.H_ice)

        p = _calvingmip_params(b)

        # In-memory model from the analytical IC (H_ice=0, lsf=+1).
        y = YelmoModel(b, 0.0; p=p, boundaries=:bounded)

        # Attach the exp1 calving hook (captures xc/yc from b).
        xc = b.xc; yc = b.yc
        y.hooks.calv_flt = (cx, cy, ux, uy, Hi, fi, lsf, t) ->
            calvmip_exp1!(cx, cy, ux, uy, Hi, fi, lsf, t; xc=xc, yc=yc)

        init_state!(y, 0.0; thrm_method = "robin")

        @test all(isfinite, interior(y.tpo.H_ice))
        @test all(isfinite, interior(y.tpo.lsf))

        dt = 1.0
        for _ in 1:Int(_FIXTURE_T)
            step!(y, dt)
        end

        n_final = _ice_cell_count(y)

        # Smoke: ice has grown from the cold start.
        @test n_final > 0
        @test all(isfinite, interior(y.tpo.H_ice))
        @test all(isfinite, interior(y.tpo.lsf))

        # Regression: ±5% of fixture ice cell count.
        @test abs(n_final - n_ref) / max(n_ref, 1) < 0.05
    end
end
