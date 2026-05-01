## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Smoke test for the benchmark scaffolding.
#
# Loads the BUELER-B fixture (analytical Halfar at t=1000) two ways:
#
#   1. **File-based** — read the committed NetCDF restart via the
#      existing `YelmoModel(restart_file, time; …)` constructor. This
#      is the "round-trip-through-disk" path the lockstep tests use.
#   2. **In-memory** — build the same `YelmoModel` directly from a
#      `BuelerBenchmark(:B; dx_km=50.0)` via `YelmoModel(b, t)`,
#      skipping NetCDF entirely. This is the path the SIA convergence
#      test (Commit 4) will use to construct the t=0 IC.
#
# The smoke test asserts:
#
#   * Both loads succeed.
#   * Both produce identical H_ice / smb_ref / z_bed (to 1e-12).
#   * The loaded H_ice matches the analytical Halfar formula at
#     t=1000 to machine precision (the only sources of error are
#     NetCDF Float64 round-trip and the YelmoModel coordinate
#     m → km → m round-trip on the file path).
#   * Dome geometry is non-trivial and physical, radially symmetric,
#     no NaN / Inf.
#
# In milestone 3c Commit 4, the t=0 IC will be passed through
# `YelmoModel(b, 0.0)`, the SIA solver run to t=1000, and the result
# compared against `state(b, 1000.0)` — that test gates the SIA
# solver port.

using Test
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

const _SPEC = BuelerBenchmark(:B; dx_km=50.0)
const _NX   = length(_SPEC.xc)   # 31

@testset "benchmarks: BUELER-B fixture round-trip (file vs in-memory)" begin
    fixture_path = joinpath(FIXTURES_DIR, "bueler_b_t1000.nc")
    @assert isfile(fixture_path) "Smoke fixture missing: $fixture_path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl` first."

    # 1. File-based load.
    y_file = YelmoModel(fixture_path, 1000.0;
                        alias  = "bueler_b_load",
                        groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                        strict = false)

    # 2. In-memory load — skip NetCDF entirely.
    y_mem = YelmoModel(_SPEC, 1000.0)

    @test y_file isa AbstractYelmoModel
    @test y_mem  isa AbstractYelmoModel
    @test y_file.time == 1000.0
    @test y_mem.time  == 1000.0

    H_file   = interior(y_file.tpo.H_ice)
    H_mem    = interior(y_mem.tpo.H_ice)
    zb_file  = interior(y_file.bnd.z_bed)
    zb_mem   = interior(y_mem.bnd.z_bed)
    smb_file = interior(y_file.bnd.smb_ref)
    smb_mem  = interior(y_mem.bnd.smb_ref)

    # Shape: 31×31×1 (Flat z-axis collapses to a singleton 3rd dim).
    @test size(H_file, 1) == _NX
    @test size(H_file, 2) == _NX
    @test size(H_mem,  1) == _NX
    @test size(H_mem,  2) == _NX

    # 3. File-based and in-memory loads must agree to machine
    #    precision on every state field.
    @test maximum(abs.(H_file   .- H_mem))   < 1e-12
    @test maximum(abs.(zb_file  .- zb_mem))  < 1e-12
    @test maximum(abs.(smb_file .- smb_mem)) < 1e-12

    # 4. Loaded H_ice matches the analytical Halfar formula at t=1000.
    H_ref = zeros(_NX, _NX)
    smb_throwaway = zeros(_NX, _NX)
    bueler_test_BC!(H_ref, smb_throwaway, _SPEC.xc, _SPEC.yc, 1000.0;
                    R0      = _SPEC.R0_km,
                    H0      = _SPEC.H0,
                    lambda  = _SPEC.lambda,
                    n       = _SPEC.n,
                    A       = _SPEC.A,
                    rho_ice = _SPEC.rho_ice,
                    g       = _SPEC.g)

    @test maximum(abs.(H_file[:, :, 1] .- H_ref)) < 1e-9
    @test maximum(abs.(H_mem[:,  :, 1] .- H_ref)) < 1e-12

    # 5. Dome geometry is non-trivial and physical (apply to both).
    for H in (H_file, H_mem)
        @test maximum(H) > 100.0
        @test minimum(H) >= 0.0
    end

    # 6. Halfar is radially symmetric — H[i, j] == H[Nx+1-i, Ny+1-j]
    #    to within machine precision. The fixture inherits this
    #    symmetry from the analytical formula.
    for H in (H_file, H_mem)
        H_rot  = reverse(H[:, :, 1])
        asymm  = maximum(abs.(H_rot .- H[:, :, 1])) /
                 max(maximum(H[:, :, 1]), 1.0)
        @test asymm < 1e-12
    end

    # 7. No NaN / Inf in any loaded field across tpo and bnd. Fields
    #    not explicitly set by the fixture / state are at their
    #    default-allocated values; these still need to be free of
    #    NaN.
    for y in (y_file, y_mem)
        for grp in (:tpo, :bnd)
            nt = getfield(y, grp)
            for k in keys(nt)
                arr = interior(getfield(nt, k))
                @test !any(isnan, arr)
                @test !any(isinf, arr)
            end
        end
    end

    # 8. analytical_velocity returns finite face-staggered Halfar
    #    velocities (Commit 4). Spot-check shape and finiteness; the
    #    closed-form math is verified in `test_sia.jl`.
    ux_ref, uy_ref = analytical_velocity(_SPEC, 1000.0)
    @test size(ux_ref) == (_NX + 1, _NX)
    @test size(uy_ref) == (_NX, _NX + 1)
    @test all(isfinite, ux_ref)
    @test all(isfinite, uy_ref)
end
