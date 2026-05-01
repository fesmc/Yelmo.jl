# ----------------------------------------------------------------------
# Solver type hierarchy for the Yelmo.jl dynamics module.
#
# `Solver` is the abstract supertype for all dynamic-velocity solver
# configurations. Concrete solver types carry the full set of knobs
# needed to configure both the inner linear solve (Krylov method,
# preconditioner, tolerances) and the outer non-linear iteration
# (Picard relaxation, convergence tolerance, max iterations).
#
# Currently defined:
#
#   - `SSASolver` — knobs for the SSA Picard iteration + Krylov+AMG
#     inner linear solve. Used by the `solver = "ssa"` and
#     `solver = "hybrid"` dispatch branches in `dyn_step!`.
#
# Future expansion: `DIVASolver`, `L1L2Solver`, etc. will extend
# `Solver` with their respective configuration parameters when those
# milestones land. The shared base type lets `YdynParams` carry a
# polymorphic solver field without changing API every time a new
# solver type lands.
#
# Locked-in design decisions (PR-B / milestone 3d):
#
#   - Default Krylov method for SSA: BiCGStab (`:bicgstab`). The SSA
#     matrix is non-symmetric (cross-coupling between ux and uy plus
#     boundary asymmetries) so symmetric-matrix solvers like CG / MINRES
#     are inappropriate.
#   - Default smoother for AMG: Gauss-Seidel (`:gauss_seidel`). Robust
#     on non-symmetric M-matrix-like systems.
#   - Default Picard tolerance: `1e-2`. Matches Fortran's
#     `YdynParams.ssa_iter_conv` default and the MISMIP+ namelist.
#   - Default Picard relaxation: `0.7`. Matches Fortran's
#     `YdynParams.ssa_iter_rel` default.
#
# This file is included directly by `Yelmo.jl` at the top level so that
# both `YelmoModelPar` (which carries an `SSASolver` field on
# `YdynParams`) and `YelmoModelDyn` (which uses the field to drive the
# Picard loop) can share one common type definition.
# ----------------------------------------------------------------------

module YelmoSolvers

export Solver, SSASolver

"""
    abstract type Solver end

Abstract supertype for dynamic-velocity solver configurations. Each
concrete subtype packages the full set of knobs needed to drive the
corresponding solver branch in `dyn_step!`.
"""
abstract type Solver end

"""
    SSASolver(; method = :bicgstab, precond = :jacobi,
                smoother = :gauss_seidel,
                rtol = 1e-6, itmax = 200,
                picard_tol = 1e-2, picard_relax = 0.7,
                picard_iter_max = 50)

Configuration object for the Shallow-Shelf Approximation (SSA) solver.
Carries both the inner linear solve parameters (Krylov method,
preconditioner choice, linear tolerances) and the outer Picard
iteration parameters (relaxation, convergence tolerance, max
iterations).

Used by `dyn_step!` when `y.p.ydyn.solver ∈ ("ssa", "hybrid")`.

Fields:

  - `method::Symbol` — Krylov solver method. Currently supports
    `:bicgstab` (default — robust on non-symmetric matrices like the
    SSA stiffness matrix). Forward placeholders: `:gmres`, `:cg`.
  - `precond::Symbol` — preconditioner for the Krylov solve. Default
    `:jacobi` (matches Fortran Yelmo's SLAB-S06 `ssa_lis_opt = "-i
    bicgsafe -p jacobi …"`). Supported values:
      * `:none` — no preconditioner. Useful for debug / verification.
      * `:jacobi` — diagonal scaling, `M = Diagonal(1 ./ diag(A))`.
        DEFAULT. Cheap and works well on the standard SSA matrix
        (non-symmetric, negative diagonals, moderate viscosity
        contrasts).
      * `:amg_sa` — `smoothed_aggregation` algebraic multigrid. Only
        appropriate for SPD-like systems (NOT the standard SSA — it
        causes BiCGStab to diverge on the SLAB-S06 problem). Kept
        opt-in for future SPD reformulations.
      * `:amg_rs` — `ruge_stuben` algebraic multigrid. Alternative
        AMG variant; less robust than smoothed-aggregation on SPD
        problems but sometimes works on weakly non-symmetric ones.
  - `smoother::Symbol` — AMG smoother (only consulted when `precond
    ∈ (:amg_sa, :amg_rs)`; ignored for `:none` / `:jacobi`).
    Currently supports `:gauss_seidel` (default — robust on
    non-symmetric systems). `:jacobi` would require a per-level
    workspace closure that the current AMG wrapper does not provide
    and is not yet wired through.
  - `rtol::Float64` — relative tolerance for the linear solve. The
    Krylov iteration stops when `‖r‖ ≤ rtol · ‖b‖`. Default `1e-6`.
  - `itmax::Int` — maximum Krylov iterations per linear solve before
    soft-warning on non-convergence. Default `200`.
  - `picard_tol::Float64` — outer Picard convergence tolerance on the
    L2-relative velocity change. Mirrors Fortran's
    `YdynParams.ssa_iter_conv` default. Default `1e-2`.
  - `picard_relax::Float64` — Picard relaxation parameter (mixing
    weight for the new velocity solution against the previous). Mirrors
    Fortran's `YdynParams.ssa_iter_rel`. Default `0.7`.
  - `picard_iter_max::Int` — maximum Picard outer iterations before
    soft-warning on non-convergence. Default `50`.

Mirrors Fortran's `velocity_ssa.f90:60-335 calc_velocity_ssa` outer
loop (Picard iteration with constant relaxation per Q5 / option (a)).
The Lis-syntax `ssa_lis_opt` namelist string is replaced by the
explicit Julia-native fields above.
"""
Base.@kwdef struct SSASolver <: Solver
    method::Symbol         = :bicgstab
    precond::Symbol        = :jacobi
    smoother::Symbol       = :gauss_seidel
    rtol::Float64          = 1e-6
    itmax::Int             = 200
    picard_tol::Float64    = 1e-2
    picard_relax::Float64  = 0.7
    picard_iter_max::Int   = 50
end

end # module YelmoSolvers
