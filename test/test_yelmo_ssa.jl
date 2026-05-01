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
using Yelmo.YelmoModelDyn: _solve_ssa_linear!,
                           picard_relax_visc!, picard_relax_vel!,
                           picard_calc_convergence_l2,
                           picard_calc_convergence_l1rel_matrix!,
                           set_inactive_margins!, calc_basal_stress!
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
# ======================================================================
# Test set 2 — Picard helpers
# ======================================================================

@testset "picard_relax_visc!: log-space mixing" begin
    g3 = RectilinearGrid(size=(3, 3, 2), x=(0.0, 3.0), y=(0.0, 3.0),
                         z=(0.0, 1.0),
                         topology=(Bounded, Bounded, Bounded))
    v_now = CenterField(g3); fill!(interior(v_now), 100.0)
    v_nm1 = CenterField(g3); fill!(interior(v_nm1), 1.0)

    # rel = 0.7 → log-space mixing.
    picard_relax_visc!(v_now, v_nm1, 0.7)
    expected = exp(0.3 * log(1.0) + 0.7 * log(100.0))
    @test all(interior(v_now) .≈ expected)

    # rel = 0 → returns previous viscosity.
    fill!(interior(v_now), 1000.0)
    fill!(interior(v_nm1), 50.0)
    picard_relax_visc!(v_now, v_nm1, 0.0)
    @test all(interior(v_now) .≈ 50.0)

    # rel = 1 → returns new viscosity.
    fill!(interior(v_now), 1000.0)
    fill!(interior(v_nm1), 50.0)
    picard_relax_visc!(v_now, v_nm1, 1.0)
    @test all(interior(v_now) .≈ 1000.0)
end

@testset "picard_relax_vel!: linear mixing" begin
    g = _bounded_2d(5, 5)
    ux_n = XFaceField(g);    fill!(interior(ux_n), 100.0)
    uy_n = YFaceField(g);    fill!(interior(uy_n), 200.0)
    ux_nm1 = XFaceField(g);  fill!(interior(ux_nm1), 0.0)
    uy_nm1 = YFaceField(g);  fill!(interior(uy_nm1), 0.0)

    picard_relax_vel!(ux_n, uy_n, ux_nm1, uy_nm1, 0.5)
    @test all(interior(ux_n) .≈ 50.0)
    @test all(interior(uy_n) .≈ 100.0)
end

@testset "picard_calc_convergence_l2: all-zero → resid=0" begin
    Nx, Ny = 5, 5
    ux  = zeros(Nx + 1, Ny, 1)
    uy  = zeros(Nx, Ny + 1, 1)
    uxp = zeros(Nx + 1, Ny, 1)
    uyp = zeros(Nx, Ny + 1, 1)
    @test picard_calc_convergence_l2(ux, uxp, uy, uyp) == 0.0
end

@testset "picard_calc_convergence_l2: identical fields above tol → resid=0" begin
    Nx, Ny = 5, 5
    ux  = fill(1.0, Nx + 1, Ny, 1)
    uxp = copy(ux)
    uy  = fill(1.0, Nx, Ny + 1, 1)
    uyp = copy(uy)
    @test picard_calc_convergence_l2(ux, uxp, uy, uyp) ≈ 0.0
end

@testset "picard_calc_convergence_l2: doubled velocity → resid≈1" begin
    Nx, Ny = 5, 5
    ux  = fill(2.0, Nx + 1, Ny, 1)
    uxp = fill(1.0, Nx + 1, Ny, 1)
    uy  = fill(0.0, Nx, Ny + 1, 1)
    uyp = fill(0.0, Nx, Ny + 1, 1)
    # res1 = sum((2 - 1)^2) over Nx+1=6 cells × Ny=5 + zeros
    #      = 30
    # res2 = sum(1^2) = 30
    # resid = 30 / (30 + 1e-10) ≈ 1
    r = picard_calc_convergence_l2(ux, uxp, uy, uyp)
    @test r ≈ 1.0 atol = 1e-9
end

@testset "picard_calc_convergence_l1rel_matrix!: tolerance gate" begin
    Nx, Ny = 3, 3
    ux  = fill(0.001, Nx + 1, Ny, 1)  # below vel_tol=1e-2
    uxp = zeros(Nx + 1, Ny, 1)
    uy  = fill(0.1, Nx, Ny + 1, 1)
    uyp = fill(0.05, Nx, Ny + 1, 1)
    err_x = zeros(Nx + 1, Ny, 1)
    err_y = zeros(Nx, Ny + 1, 1)
    picard_calc_convergence_l1rel_matrix!(err_x, err_y, ux, uy, uxp, uyp)
    @test all(err_x .== 0.0)   # gated out by vel_tol
    expected = 2.0 * abs(0.1 - 0.05) / abs(0.1 + 0.05 + 1e-5)
    @test all(err_y .≈ expected)
end

@testset "set_inactive_margins!: zero out partial-margin faces" begin
    # Mirrors Fortran `set_inactive_margins`
    # (velocity_general.f90:1675). Predicate: face ux(i, j) (between
    # cell (i, j) and (ip1, j)) is zeroed iff one side is partial-or-no
    # ice (f_ice < 1) AND the other side is fully ice-free (f_ice == 0).
    # If both sides are full ice (f=1), or both partial, or one full
    # and other partial-with-some-ice, the face stays.
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)
    ux = XFaceField(g);  fill!(interior(ux), 1.0)
    uy = YFaceField(g);  fill!(interior(uy), 1.0)
    f_ice = CenterField(g); fill!(interior(f_ice), 1.0)

    # All cells iced → no zeroing.
    set_inactive_margins!(ux, uy, f_ice)
    @test all(interior(ux) .== 1.0)
    @test all(interior(uy) .== 1.0)

    # Make cell (3, 3) ice-free. Cells around it have full ice
    # (f_ice=1). Predicate fails on every face (one side is f=1, not
    # f<1 nor f==0 with neighbour partial). Margins remain non-zero.
    fill!(interior(ux), 1.0)
    fill!(interior(uy), 1.0)
    interior(f_ice)[3, 3, 1] = 0.0
    set_inactive_margins!(ux, uy, f_ice)
    @test all(interior(ux) .== 1.0)
    @test all(interior(uy) .== 1.0)

    # Make cells (3, 3) and (4, 3) BOTH partial+ice-free (f=0 and f=0.5).
    # Face between them: predicate "f(3,3) < 1 AND f(4,3) == 0" is
    # false (4,3 is 0.5). predicate "f(3,3) == 0 AND f(4,3) < 1" is
    # true (0==0, 0.5<1) → zero ux at slot [4, 3, 1].
    fill!(interior(ux), 1.0)
    fill!(interior(uy), 1.0)
    interior(f_ice)[3, 3, 1] = 0.0
    interior(f_ice)[4, 3, 1] = 0.5
    set_inactive_margins!(ux, uy, f_ice)
    @test interior(ux)[4, 3, 1] == 0.0
    # Face between (4, 3) and (5, 3): f(4,3)=0.5<1 and f(5,3)=1; no
    # full-empty side, so face NOT zeroed.
    @test interior(ux)[5, 3, 1] == 1.0
end

@testset "calc_basal_stress!: tau = beta · u with underflow clip" begin
    g = _bounded_2d(3, 3)
    ux = XFaceField(g);  fill!(interior(ux), 100.0)  # m/yr
    uy = YFaceField(g);  fill!(interior(uy), 50.0)
    bx = XFaceField(g);  fill!(interior(bx), 1e3)    # Pa·yr/m
    by = YFaceField(g);  fill!(interior(by), 2e3)
    tx = XFaceField(g)
    ty = YFaceField(g)
    calc_basal_stress!(tx, ty, bx, by, ux, uy)
    @test all(interior(tx) .≈ 1e5)
    @test all(interior(ty) .≈ 1e5)

    # Underflow: tiny u → tiny tau → clipped to 0.
    fill!(interior(ux), 1e-9)
    calc_basal_stress!(tx, ty, bx, by, ux, uy)
    @test maximum(abs.(interior(tx))) == 0.0
end
