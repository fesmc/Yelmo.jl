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
#   - `YdynParams()` (and `YelmoModelParameters("…").ydyn`) include a
#     default `SSASolver` instance.
#   - The deprecated `ssa_lis_opt` field has been removed from
#     `YdynParams`.

using Test
using Yelmo
using Yelmo.YelmoModelPar: YdynParams, ydyn_params, YelmoModelParameters

@testset "SSASolver: default field values" begin
    s = SSASolver()
    @test s.method          === :bicgstab
    @test s.smoother        === :gauss_seidel
    @test s.rtol            == 1e-6
    @test s.itmax           == 200
    @test s.picard_tol      == 1e-2
    @test s.picard_relax    == 0.7
    @test s.picard_iter_max == 50
end

@testset "SSASolver: kwarg overrides" begin
    s = SSASolver(method = :gmres, smoother = :jacobi,
                   rtol = 1e-8, itmax = 500,
                   picard_tol = 5e-3, picard_relax = 0.5,
                   picard_iter_max = 100)
    @test s.method          === :gmres
    @test s.smoother        === :jacobi
    @test s.rtol            == 1e-8
    @test s.itmax           == 500
    @test s.picard_tol      == 5e-3
    @test s.picard_relax    == 0.5
    @test s.picard_iter_max == 100
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
    custom = SSASolver(picard_tol = 5e-3)
    yd = YdynParams(ssa_solver = custom)
    @test yd.ssa_solver === custom
    @test yd.ssa_solver.picard_tol == 5e-3
end

@testset "YdynParams: ssa_lis_opt removed" begin
    @test !(:ssa_lis_opt in fieldnames(YdynParams))
    @test :ssa_solver in fieldnames(YdynParams)
end

@testset "YelmoModelParameters: ydyn includes default SSASolver" begin
    p = YelmoModelParameters("test")
    @test p.ydyn.ssa_solver isa SSASolver
    @test p.ydyn.ssa_solver == SSASolver()
end
