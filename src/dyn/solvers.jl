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
#   - Default assembly `method` for SSA: `:residual` — assemble the
#     Jacobian of the strong-form SSA momentum residual directly.
#     Faithful to the Fortran port. The resulting matrix is non-
#     symmetric in general (BC handling, mask-based calving-front
#     treatment) so the inner Krylov method defaults to BiCGStab.
#   - Alternative `method = :energy_quadratic`: assemble the Hessian of
#     the discrete viscous-energy functional E[u] (ν, β, H frozen per
#     Picard iteration). Symmetric positive-definite by construction;
#     pairs naturally with CG. See `velocity_ssa_energy.jl` and the
#     derivation in `IceSheetStencils.jl/examples/energy_functional_demo.jl`.
#   - `method = :energy_nonlinear` is reserved for the future fully-
#     nonlinear minimisation (η baked into E via Glen's law, Newton/
#     L-BFGS replacing the Picard loop). Currently a stub.
#   - Default Krylov `linear_method`: `:auto` — `:bicgstab` for
#     `method = :residual`, `:cg` for `method = :energy_quadratic`.
#   - Default smoother for AMG: Gauss-Seidel (`:gauss_seidel`). Robust
#     on non-symmetric M-matrix-like systems.
#   - Default Picard tolerance: `1e-2`. Matches Fortran's
#     `YdynParams.ssa_iter_conv` default and the MISMIP+ namelist.
#   - Default Picard relaxation: `0.7`. Matches Fortran's
#     `YdynParams.ssa_iter_rel` default.
#
# This file is included directly by `Yelmo.jl` at the top level so that
# both `YelmoPar` (which carries an `SSASolver` field on
# `YdynParams`) and `YelmoModelDyn` (which uses the field to drive the
# Picard loop) can share one common type definition.
# ----------------------------------------------------------------------

module YelmoSolvers

export Solver, SSASolver, resolve_linear_method

"""
    abstract type Solver end

Abstract supertype for dynamic-velocity solver configurations. Each
concrete subtype packages the full set of knobs needed to drive the
corresponding solver branch in `dyn_step!`.
"""
abstract type Solver end

"""
    SSASolver(; method = :residual, linear_method = :auto,
                precond = :jacobi, smoother = :gauss_seidel,
                rtol = 1e-6, itmax = 200,
                picard_tol = 1e-2, picard_relax = 0.7,
                picard_iter_max = 50)

Configuration object for the Shallow-Shelf Approximation (SSA) solver.
Carries the choice of assembly formulation (`method`), the inner
linear-solve parameters (Krylov method, preconditioner, tolerances),
and the outer Picard iteration parameters (relaxation, convergence
tolerance, max iterations).

Used by `dyn_step!` when `y.p.ydyn.solver ∈ ("ssa", "hybrid")`.

Fields:

  - `method::Symbol` — assembly formulation for the SSA stiffness
    matrix. Default `:residual`. Supported values:
      * `:residual` — DEFAULT. Assemble `A` as the Jacobian of the
        strong-form SSA momentum residual, faithful to the Fortran
        port (`solver_ssa_ac.f90`). The matrix is non-symmetric in
        general (BC handling, mask-based calving fronts), so
        `linear_method = :auto` resolves to `:bicgstab`.
      * `:energy_quadratic` — assemble `A` as the Hessian of the
        discrete viscous-energy functional `E[u]` with η, β, H frozen
        per Picard iteration. Symmetric positive-definite by
        construction; `linear_method = :auto` resolves to `:cg`.
        Calving-front traction is encoded as a linear boundary-work
        term (RHS only); Dirichlet BCs are encoded as a quadratic
        penalty (`κ`). See `velocity_ssa_energy.jl` and the
        derivation in `IceSheetStencils.jl` for details.
      * `:energy_nonlinear` — RESERVED. Future fully-nonlinear
        minimisation of `E[u]` with η baked in via Glen's law
        (Newton/L-BFGS replaces the Picard loop). Currently raises
        an error at the dispatch site.
  - `linear_method::Symbol` — Krylov solver for the inner linear
    solve. Default `:auto` — resolves to `:bicgstab` for
    `method = :residual` and `:cg` for `method = :energy_quadratic`.
    Explicit values: `:bicgstab`, `:cg`. (`:gmres` is a forward
    placeholder, not yet wired through.) Choosing `:cg` for the
    `:residual` formulation is unsafe — its matrix is non-symmetric.
  - `precond::Symbol` — preconditioner for the Krylov solve. Default
    `:jacobi` (matches Fortran Yelmo's SLAB-S06 `ssa_lis_opt = "-i
    bicgsafe -p jacobi …"`). Supported values:
      * `:none` — no preconditioner. Useful for debug / verification.
      * `:jacobi` — diagonal scaling, `M = Diagonal(1 ./ diag(A))`.
        DEFAULT. Cheap and works well on the standard SSA matrix
        (non-symmetric, negative diagonals, moderate viscosity
        contrasts).
      * `:amg_sa` — `smoothed_aggregation` algebraic multigrid. Only
        appropriate for SPD-like systems (NOT the `:residual`
        formulation — it causes BiCGStab to diverge on the SLAB-S06
        problem). Suitable for `method = :energy_quadratic`.
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
    method::Symbol         = :residual
    linear_method::Symbol  = :auto
    precond::Symbol        = :jacobi
    smoother::Symbol       = :gauss_seidel
    rtol::Float64          = 1e-6
    itmax::Int             = 200
    picard_tol::Float64    = 1e-2
    picard_relax::Float64  = 0.7
    picard_iter_max::Int   = 50

    function SSASolver(method, linear_method, precond, smoother,
                       rtol, itmax, picard_tol, picard_relax, picard_iter_max)
        method ∈ (:residual, :energy_quadratic, :energy_nonlinear) || error(
            "SSASolver: method=$(method) not recognized. Expected one of " *
            ":residual, :energy_quadratic, :energy_nonlinear.")
        linear_method ∈ (:auto, :bicgstab, :cg, :gmres) || error(
            "SSASolver: linear_method=$(linear_method) not recognized. " *
            "Expected one of :auto, :bicgstab, :cg, :gmres.")
        new(method, linear_method, precond, smoother,
            rtol, itmax, picard_tol, picard_relax, picard_iter_max)
    end
end

"""
    resolve_linear_method(ssa::SSASolver) -> Symbol

Return the concrete Krylov method to use for the inner linear solve,
resolving `linear_method = :auto` based on `method`:

  - `method = :residual`         → `:bicgstab`
  - `method = :energy_quadratic` → `:cg`
  - `method = :energy_nonlinear` → `:cg` (reserved; dispatch will
                                          error before reaching here)

If `linear_method` is not `:auto`, returns it unchanged.
"""
function resolve_linear_method(ssa::SSASolver)
    if ssa.linear_method !== :auto
        return ssa.linear_method
    end
    if ssa.method === :residual
        return :bicgstab
    elseif ssa.method === :energy_quadratic
        return :cg
    elseif ssa.method === :energy_nonlinear
        return :cg
    else
        error("resolve_linear_method: unhandled method=$(ssa.method).")
    end
end

end # module YelmoSolvers
