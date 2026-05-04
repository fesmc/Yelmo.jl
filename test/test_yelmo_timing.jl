## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Smoke test for the per-section timing scaffold.
#
# Verifies (1) that toggling `timing` is functionally inert — runs with
# `timing=true` and `timing=false` produce bit-for-bit identical state,
# (2) that the expected sections are recorded, (3) that `reset_timings!`
# clears the timer, and (4) that `print_timings` doesn't error on
# either an enabled-with-data or disabled timer.

using Test
using Yelmo
using Oceananigans: interior

include("benchmarks/helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params,
                           yelmo_params

# Same EISMINT-1 moving setup as test/benchmarks/test_eismint_moving.jl,
# parameterised on the timing flag.
function _eismint_moving_params(; timing::Bool)
    return YelmoModelParameters("eismint_moving";
        yelmo = yelmo_params(
            dt_method     = 2,
            pc_method     = "HEUN",
            pc_controller = "PI42",
            pc_tol        = 5.0,
            pc_eps        = 1.0,
            pc_n_redo     = 5,
            dt_min        = 0.01,
            cfl_max       = 0.5,
            timing        = timing,
        ),
        ydyn = ydyn_params(solver="sia", uz_method=3, visc_method=1,
                           eps_0=1e-6, taud_lim=2e5),
        ytopo = ytopo_params(solver="expl", use_bmb=false),
        yneff = yneff_params(method=0, const_=1.0),
        ytill = ytill_params(method=-1),
        ymat  = ymat_params(n_glen=3.0, rf_const=1e-16, visc_min=1e3,
                            de_max=0.5, enh_method="shear3D",
                            enh_shear=1.0, enh_stream=1.0, enh_shlf=1.0),
    )
end

function _build_eismint_moving(b, p)
    y = YelmoModel(b, 0.0; p=p, boundaries=:bounded)
    fill!(interior(y.mat.ATT), p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), 0.0)
    fill!(interior(y.dyn.N_eff),  1.0)
    return y
end

@testset "timing scaffold" begin

    @testset "default off" begin
        # `timing` defaults to false on yelmo_params.
        b = EISMINT1MovingBenchmark()
        p = _eismint_moving_params(; timing=false)
        @test p.yelmo.timing == false

        y = _build_eismint_moving(b, p)
        @test y.timer isa YelmoTimer
        @test y.timer.enabled == false
        @test isempty(y.timer.counts)

        # Stepping doesn't populate the timer when disabled.
        Yelmo.step!(y, 100.0)
        @test isempty(y.timer.counts)

        # `print_timings` is safe to call on a disabled timer.
        io = IOBuffer()
        print_timings(y; io=io)
        @test occursin("Timing is disabled", String(take!(io)))
    end

    @testset "enabled records all expected sections" begin
        b = EISMINT1MovingBenchmark()
        p_off = _eismint_moving_params(; timing=false)
        p_on  = _eismint_moving_params(; timing=true)

        y_off = _build_eismint_moving(b, p_off)
        y_on  = _build_eismint_moving(b, p_on)

        # Run for a few outer steps under both flags.
        for _ in 1:3
            Yelmo.step!(y_off, 100.0)
            Yelmo.step!(y_on,  100.0)
        end

        # Functionally inert: bit-for-bit identical state.
        @test interior(y_on.tpo.H_ice) == interior(y_off.tpo.H_ice)

        # All instrumented sections recorded.
        expected = [:topo, :dyn, :mat,
                    :pc_predictor, :pc_corrector,
                    :dyn_sia, :dyn_jacobian_uxy, :dyn_uz,
                    :dyn_jacobian_uz, :dyn_strain]
        for s in expected
            @test haskey(y_on.timer.counts, s)
            @test y_on.timer.counts[s] > 0
            @test y_on.timer.total_ns[s] > 0
            @test y_on.timer.max_ns[s]   > 0
        end

        # The dyn sub-section sample count must equal the dyn count
        # (each dyn_step! enters every wrapped sub-kernel exactly once).
        @test y_on.timer.counts[:dyn_jacobian_uxy] == y_on.timer.counts[:dyn]
        @test y_on.timer.counts[:dyn_sia]          == y_on.timer.counts[:dyn]

        # Each PC outer step pairs one predictor and one corrector.
        @test y_on.timer.counts[:pc_predictor] == y_on.timer.counts[:pc_corrector]

        # `print_timings` produces non-empty output that mentions the
        # top-level sections.
        io = IOBuffer()
        print_timings(y_on; io=io)
        out = String(take!(io))
        @test occursin("dyn", out)
        @test occursin("topo", out)
        @test occursin("mat", out)
        @test occursin("total", out)
    end

    @testset "reset_timings! clears state" begin
        b = EISMINT1MovingBenchmark()
        p = _eismint_moving_params(; timing=true)
        y = _build_eismint_moving(b, p)

        Yelmo.step!(y, 100.0)
        @test !isempty(y.timer.counts)

        reset_timings!(y)
        @test isempty(y.timer.counts)
        @test isempty(y.timer.total_ns)
        @test isempty(y.timer.max_ns)
        @test y.timer.enabled == true   # `enabled` is preserved
    end
end
