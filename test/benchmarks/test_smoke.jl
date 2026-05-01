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

# ----------------------------------------------------------------------
# TroughBenchmark construction smoke (no fixture round-trip — that lives
# in `test_trough.jl` once the YelmoMirror-produced fixture is committed).
# ----------------------------------------------------------------------

@testset "benchmarks: TroughBenchmark(:F17) construction" begin
    # 8-km variant for the regression fixture (smaller — fits in <1 MB).
    b8 = TroughBenchmark(:F17; dx_km=8.0)
    @test b8.variant === :F17
    @test b8.dx_km   == 8.0
    @test length(b8.xc) == 88                  # int(700/8) + 1
    @test length(b8.yc) == 21                  # int(160/8) + 1
    @test b8.xc[1]   == 0.0                    # Fortran origin in x
    @test b8.xc[end] == 696_000.0              # (Nx-1)·dx in metres
    @test b8.yc[1]   == -80_000.0              # -ly/2
    @test b8.yc[end] ==  80_000.0              # +ly/2
    @test b8.namelist_path |> isabspath        # resolved path
    @test endswith(b8.namelist_path, "specs/yelmo_TROUGH.nml")

    # 4-km variant for higher-fidelity local runs.
    b4 = TroughBenchmark(:F17; dx_km=4.0)
    @test length(b4.xc) == 176                 # int(700/4) + 1
    @test length(b4.yc) == 41                  # int(160/4) + 1

    # Fields default to TROUGH-F17.nml values.
    @test b8.fc_km   == 16.0
    @test b8.dc_m    == 500.0
    @test b8.wc_km   == 24.0
    @test b8.x_cf_km == 640.0

    # Variant validation.
    @test_throws ErrorException TroughBenchmark(:F18; dx_km=8.0)
end

# ----------------------------------------------------------------------
# HOMCBenchmark construction smoke. The 180° anti-symmetry SSA test
# lives in `test_hom_c.jl` (post-dyn_step!). Here we just verify the
# struct constructor + axis bounds + state shape.
# ----------------------------------------------------------------------

@testset "benchmarks: HOMCBenchmark(:C, L=80) construction" begin
    b = HOMCBenchmark(:C; L_km=80.0, dx_km=2.0)
    @test b.variant === :C
    @test b.L_km    == 80.0
    @test b.dx_km   == 2.0
    @test length(b.xc) == 40                  # L/dx = 40
    @test length(b.yc) == 40
    # Cell centres at (0.5·dx, 1.5·dx, …, (Nx - 0.5)·dx) in metres.
    @test b.xc[1]   ≈ 1000.0                 # 0.5 · 2 km
    @test b.xc[end] ≈ 79_000.0               # (40 - 0.5) · 2 km
    @test b.yc[1]   ≈ 1000.0
    @test b.yc[end] ≈ 79_000.0

    # Material defaults match the Yelmo Fortran namelist.
    @test b.H        ≈ 1000.0
    @test b.A_glen   ≈ 1e-16
    @test b.n_glen   ≈ 3.0
    @test b.beta0    ≈ 1000.0
    @test b.beta_amp ≈ 1000.0   # Yelmo Fortran convention (NOT 0.9)
    @test b.alpha_rad ≈ 0.1 * π / 180.0

    # state(b, 0) shape and values.
    s = state(b, 0.0)
    @test size(s.H_ice) == (40, 40)
    @test all(s.H_ice .== 1000.0)
    # z_bed = -x · tan α - H. Linear in x, constant in y.
    @test s.z_bed[1, 1]    ≈ -b.xc[1]   * tan(b.alpha_rad) - b.H
    @test s.z_bed[end, 1]  ≈ -b.xc[end] * tan(b.alpha_rad) - b.H
    @test all(abs.(diff(s.z_bed; dims=2)) .< 1e-10)   # constant in y

    # Variant validation.
    @test_throws ErrorException HOMCBenchmark(:A; L_km=80.0, dx_km=2.0)
    # Non-integer Nx rejected.
    @test_throws ErrorException HOMCBenchmark(:C; L_km=80.0, dx_km=3.0)
end
