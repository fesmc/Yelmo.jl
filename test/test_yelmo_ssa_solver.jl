## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3d / PR-B Commit 1 — `Solver` / `SSASolver` type unit tests.
#
# Verifies:
#   - `SSASolver()` constructs with documented default values.
#   - kwargs can override individual fields.
#   - `Solver` is the abstract supertype.
#   - `YdynParams()` (and `YelmoParameters("…").ydyn`) include a
#     default `SSASolver` instance.
#   - The deprecated `ssa_lis_opt` field has been removed from
#     `YdynParams`.

using Test
using Yelmo
using Yelmo.YelmoPar: YdynParams, ydyn_params, YelmoParameters

@testset "SSASolver: default field values" begin
    s = SSASolver()
    @test s.method          === :energy_quadratic   # Fortran ssa_solver = "energy"
    @test s.linear_method   === :auto
    @test s.precond         === :jacobi
    @test s.smoother        === :gauss_seidel
    @test s.rtol            == 1e-6
    @test s.itmax           == 200
end

@testset "SSASolver: kwarg overrides" begin
    s = SSASolver(method = :energy_quadratic, linear_method = :gmres,
                   precond = :amg_sa, smoother = :jacobi,
                   rtol = 1e-8, itmax = 500)
    @test s.method          === :energy_quadratic
    @test s.linear_method   === :gmres
    @test s.precond         === :amg_sa
    @test s.smoother        === :jacobi
    @test s.rtol            == 1e-8
    @test s.itmax           == 500
end

@testset "SSASolver: method/linear_method validation + auto resolution" begin
    @test_throws ErrorException SSASolver(method = :bogus)
    @test_throws ErrorException SSASolver(linear_method = :bogus)
    # Auto resolution
    @test resolve_linear_method(SSASolver(method = :residual))         === :bicgstab
    @test resolve_linear_method(SSASolver(method = :energy_quadratic)) === :cg
    # Explicit override returns unchanged
    @test resolve_linear_method(SSASolver(linear_method = :bicgstab))  === :bicgstab
    @test resolve_linear_method(SSASolver(method = :residual, linear_method = :cg)) === :cg
    @test resolve_linear_method(SSASolver(method = :energy_quadratic,
                                          linear_method = :bicgstab))  === :bicgstab
end

@testset "SSASolver: precond field defaults and overrides" begin
    @test SSASolver().precond === :jacobi   # locked-in default
    @test SSASolver(precond = :none).precond   === :none
    @test SSASolver(precond = :jacobi).precond === :jacobi
    @test SSASolver(precond = :amg_sa).precond === :amg_sa
    @test SSASolver(precond = :amg_rs).precond === :amg_rs
end

@testset "SSASolver: subtype of Solver" begin
    @test SSASolver <: Solver
    s = SSASolver()
    @test s isa Solver
end

@testset "YdynParams: default ssa_solver field is SSASolver()" begin
    yd = YdynParams()
    @test yd.ssa_solver isa SSASolver
    @test yd.ssa_solver == SSASolver()
end

@testset "ydyn_params(): default ssa_solver" begin
    yd = ydyn_params()
    @test yd.ssa_solver isa SSASolver
end

@testset "YdynParams: ssa_solver kwarg override" begin
    custom = SSASolver(rtol = 5e-3)
    yd = YdynParams(ssa_solver = custom)
    @test yd.ssa_solver === custom
    @test yd.ssa_solver.rtol == 5e-3
end

@testset "YdynParams: Picard settings are the Fortran ssa_iter_* keys" begin
    yd = YdynParams()
    @test (yd.ssa_iter_max, yd.ssa_iter_rel, yd.ssa_iter_conv) == (20, 0.7, 1e-2)
    @test !any(startswith(string(f), "picard") for f in fieldnames(SSASolver))
end

@testset "YdynParams: Lis options are the Fortran keys (unused by YelmoModel)" begin
    @test !(:ssa_lis_opt in fieldnames(YdynParams))
    @test :ssa_lis_opt_residual in fieldnames(YdynParams)
    @test :ssa_solver in fieldnames(YdynParams)
end

@testset "YelmoParameters: ydyn includes default SSASolver" begin
    p = YelmoParameters("test")
    @test p.ydyn.ssa_solver isa SSASolver
    @test p.ydyn.ssa_solver == SSASolver()
end
