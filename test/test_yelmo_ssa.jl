## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 3d / PR-B integration tests for the SSA driver and the
# Krylov+AMG inner linear solve.
#
# Layered test set:
#
#   1. `_solve_ssa_linear!` — Krylov BiCGStab + AMG preconditioner on
#      synthetic SPD and non-SPD systems. Verifies the wrapper plumbing
#      and the AMG preconditioner setup.
#   2. Picard helpers — `picard_relax_visc!`, `picard_relax_vel!`,
#      `picard_calc_convergence_l2`, `picard_calc_convergence_l1rel_matrix!`,
#      `set_inactive_margins!` on hand-derived inputs (added in
#      commit 3).
#   3. Plug-flow analytical SSA — single-cell residual `ū = τ_d / β`
#      via `dyn_step!` with `solver = "ssa"` (added in commit 4).
#   4. Hybrid sanity — verify `ux_bar(hybrid) == ux_bar(sia) + ux_bar(ssa)`
#      to floating-point precision (added in commit 5).
#   5. Schoof slab convergence — added in commit 5.

using Test
using Yelmo
using Yelmo.YelmoModelDyn: _solve_ssa_linear!
using Oceananigans
using Oceananigans: interior
using SparseArrays
using LinearAlgebra
using Krylov: BicgstabWorkspace

_bounded_2d(Nx, Ny; dx=1.0) = RectilinearGrid(size=(Nx, Ny),
                                                x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                                topology=(Bounded, Bounded, Flat))

# ======================================================================
# Test set 1 — `_solve_ssa_linear!` Krylov+AMG plumbing
# ======================================================================

# Build a minimal scratch struct — only needs the Krylov workspace +
# AMG cache fields used by `_solve_ssa_linear!`.
function _build_solver_scratch(N_rows)
    return (
        ssa_solver_workspace = BicgstabWorkspace(N_rows, N_rows, Vector{Float64}),
        ssa_amg_cache        = Ref{Any}(nothing),
    )
end

@testset "_solve_ssa_linear!: SPD tridiagonal converges" begin
    n = 50
    A = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.0, n-1))
    b = ones(n)

    scratch = _build_solver_scratch(n)
    ssa = SSASolver(rtol = 1e-10, itmax = 100)
    x = _solve_ssa_linear!(scratch, A, b, ssa)

    @test scratch.ssa_solver_workspace.stats.solved == true
    @test norm(A * x .- b) / norm(b) < 1e-8
    @test scratch.ssa_amg_cache[] !== nothing
end

@testset "_solve_ssa_linear!: non-symmetric tridiagonal converges" begin
    # Asymmetric off-diagonals: -1 below, -2 above. Still diagonally
    # dominant (4 > 1+2 = 3), so BiCGStab + SA should converge.
    n = 50
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n
        push!(I, i); push!(J, i); push!(V, 4.0)
        if i > 1
            push!(I, i); push!(J, i - 1); push!(V, -1.0)
        end
        if i < n
            push!(I, i); push!(J, i + 1); push!(V, -2.0)
        end
    end
    A = sparse(I, J, V, n, n)
    @test A != A'   # confirm non-symmetric
    b = ones(n)

    scratch = _build_solver_scratch(n)
    ssa = SSASolver(rtol = 1e-10, itmax = 100)
    x = _solve_ssa_linear!(scratch, A, b, ssa)

    @test scratch.ssa_solver_workspace.stats.solved == true
    @test norm(A * x .- b) / norm(b) < 1e-8
end

@testset "_solve_ssa_linear!: unsupported smoother errors clearly" begin
    # `:jacobi` smoother is documented-but-not-yet-wired (see
    # `_amg_smoother` docstring). Verify the error path surfaces a
    # clear message rather than a cryptic upstream MethodError.
    n = 10
    A = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.0, n-1))
    b = ones(n)
    scratch = _build_solver_scratch(n)
    ssa = SSASolver(rtol = 1e-6, itmax = 50, smoother = :jacobi)
    @test_throws ErrorException _solve_ssa_linear!(scratch, A, b, ssa)
end
