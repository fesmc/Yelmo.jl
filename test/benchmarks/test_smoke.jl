## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Smoke test for the benchmark scaffolding. Loads the BUELER-B fixture
# via YelmoModel and asserts:
#
#   1. The file exists and loads without error.
#   2. The grid size matches the spec (31×31 cells).
#   3. The loaded H_ice is a non-trivial dome (positive somewhere,
#      max > 100 m, symmetric about the centre to ~5%).
#   4. No NaN / Inf in any loaded `tpo`/`bnd` field.
#
# Does NOT validate physics against the analytical Halfar solution
# at end_time — that's a milestone-3c concern when the SIA solver is
# in place. Here we just prove the scaffolding (synthetic-grid
# construction, restart-write, fixture-load) round-trips cleanly.

using Test
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

include("specs/bueler_b_smoke.jl")

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

@testset "benchmarks: BUELER-B smoke fixture loads" begin
    fixture_path = joinpath(FIXTURES_DIR, "bueler_b_smoke__t1000.nc")
    @assert isfile(fixture_path) "Smoke fixture missing: $fixture_path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl` first."

    y = load_fixture(BUELER_B_SMOKE_SPEC; fixtures_dir=FIXTURES_DIR)

    @test y isa AbstractYelmoModel
    @test y.time == BUELER_B_SMOKE_SPEC.end_time

    H = interior(y.tpo.H_ice)
    @test size(H, 1) == _BUELER_B_NX
    @test size(H, 2) == _BUELER_B_NX

    @test maximum(H) > 100.0
    @test minimum(H) >= 0.0

    # Halfar evolution is radially symmetric; the dome should be
    # symmetric about the centre to within numerical tolerance after
    # 1000 yr of decay. Compare H[i, j] with H[Nx+1-i, Ny+1-j]
    # (rotation by π); allow 5% relative error.
    Nx, Ny = size(H, 1), size(H, 2)
    H_rot  = reverse(H[:, :, 1])
    H_orig = H[:, :, 1]
    asymm  = maximum(abs.(H_rot .- H_orig)) /
             max(maximum(H_orig), 1.0)
    @test asymm < 0.05

    # No NaN / Inf in any loaded field across tpo and bnd.
    for grp in (:tpo, :bnd)
        nt = getfield(y, grp)
        for k in keys(nt)
            arr = interior(getfield(nt, k))
            @test !any(isnan, arr)
            @test !any(isinf, arr)
        end
    end
end
