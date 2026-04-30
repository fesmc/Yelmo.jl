## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Smoke test for the benchmark scaffolding.
#
# Loads the BUELER-B fixture (analytical Halfar at t=1000) via
# YelmoModel and asserts:
#
#   1. The file exists and loads without error.
#   2. The grid size and time match the spec (31×31, t=1000 yr).
#   3. The loaded H_ice matches the analytical Halfar at t=1000 to
#      machine precision (no SIA solver is run — fixture IS the
#      analytical solution; the only sources of error are NetCDF
#      Float64 round-trip and the YelmoModel coordinate
#      m → km → m round-trip).
#   4. No NaN / Inf in any loaded `tpo`/`bnd` field.
#
# In milestone 3c, this same fixture supports a strict lockstep test:
# load the t=0 IC, run YelmoModel's SIA solver to t=1000, compare
# against the t=1000 fixture (which is the analytical reference).
# That test gates the SIA solver port.

using Test
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

include("specs/bueler_b_smoke.jl")

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

@testset "benchmarks: BUELER-B analytical fixture round-trip" begin
    fixture_path = joinpath(FIXTURES_DIR, "bueler_b_smoke__t1000.nc")
    @assert isfile(fixture_path) "Smoke fixture missing: $fixture_path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl` first."

    y = load_analytical_fixture(BUELER_B_SMOKE_SPEC; fixtures_dir=FIXTURES_DIR)

    @test y isa AbstractYelmoModel
    @test y.time == BUELER_B_SMOKE_SPEC.output_times[end]

    H = interior(y.tpo.H_ice)
    @test size(H, 1) == _BUELER_B_NX
    @test size(H, 2) == _BUELER_B_NX

    # Compare against analytical Halfar at t=1000 cell-by-cell. The
    # only difference between the loaded H_ice and the analytical
    # value should come from NetCDF Float64 round-trip — i.e.
    # bit-for-bit identical (zero error).
    H_ref = zeros(_BUELER_B_NX, _BUELER_B_NX)
    smb_throwaway = zeros(_BUELER_B_NX, _BUELER_B_NX)
    xc_m, yc_m = _bueler_b_axes()
    bueler_test_BC!(H_ref, smb_throwaway, xc_m, yc_m, 1000.0;
                    R0     = _BUELER_B_R0_KM,
                    H0     = _BUELER_B_H0,
                    lambda = _BUELER_B_LAMBDA,
                    n      = _BUELER_B_N,
                    A      = _BUELER_B_A,
                    rho_ice = _BUELER_B_RHO_ICE,
                    g      = _BUELER_B_G)

    @test maximum(abs.(H[:, :, 1] .- H_ref)) < 1e-9

    # Sanity: dome geometry is non-trivial and physical.
    @test maximum(H) > 100.0
    @test minimum(H) >= 0.0

    # Halfar is radially symmetric — H[i, j] should equal H[Nx+1-i, Ny+1-j]
    # to within machine precision. The fixture inherits this symmetry
    # from the analytical formula.
    H_rot  = reverse(H[:, :, 1])
    asymm  = maximum(abs.(H_rot .- H[:, :, 1])) /
             max(maximum(H[:, :, 1]), 1.0)
    @test asymm < 1e-12

    # No NaN / Inf in any loaded field across tpo and bnd. Most fields
    # are zero (left at default allocation by `strict=false` since the
    # fixture only sets H_ice / smb_ref / z_bed); these still need to
    # be free of NaN.
    for grp in (:tpo, :bnd)
        nt = getfield(y, grp)
        for k in keys(nt)
            arr = interior(getfield(nt, k))
            @test !any(isnan, arr)
            @test !any(isinf, arr)
        end
    end
end
