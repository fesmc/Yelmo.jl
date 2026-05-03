# ----------------------------------------------------------------------
# Generic 2D tracer advection on a regular RectilinearGrid.
#
# `advect_tracer!` is the dispatch entry point. It selects between two
# schemes via the `scheme` keyword:
#
#   `:upwind_explicit` — Forward Euler in time, first-order upwind
#       (`Oceananigans.Advection.UpwindBiased(order=1)`) in space.
#       CFL-limited; substeps internally to honour `cfl_safety`.
#       This is the `expl-upwind` solver in Fortran Yelmo.
#
#   `:upwind_implicit` — Backward Euler in time, first-order upwind
#       in space, solved as a 5-point sparse linear system per outer
#       step. CFL-unconstrained. Mirrors Fortran Yelmo's `impl-lis`
#       (solver_advection.f90:86–102), with `WOVI = 1.0` (pure
#       backward Euler). Requires a persistent `ImplicitAdvectionCache`
#       passed via the `cache` keyword. The cache holds the sparse
#       operator, RHS / solution buffers, the GMRES workspace, and
#       the (lazily-refactored) ILU0 preconditioner. Allocate once at
#       model setup with `init_advection_cache(grid)`; the operator is
#       refreshed once per outer step (`update_advection_operator!`)
#       and reused for every tracer advected with the same velocity /
#       timestep / mask (`solve_advection!`).
#
# Used in `topo_step!` for both ice thickness `H_ice` (advected at
# the depth-averaged ice velocity `(ux_bar, uy_bar)`) and the
# level-set function `lsf` (advected at `u_bar + cr` — see
# `lsf_update!`).
# ----------------------------------------------------------------------

import Oceananigans
using Oceananigans.Advection: div_Uc, UpwindBiased
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: topology, Periodic, Bounded
using SparseArrays: SparseMatrixCSC, sparse
using Krylov: GmresWorkspace, gmres!
using KrylovPreconditioners: KrylovPreconditioners as KP

export advect_tracer!, advect_tracer_upwind_explicit!,
       advect_tracer_upwind_implicit!,
       ImplicitAdvectionCache, init_advection_cache,
       update_advection_matrix!, update_advection_operator!,
       solve_advection!

const _DEFAULT_OCN_UPWIND = UpwindBiased(order=1)

"""
    ImplicitAdvectionCache

Persistent state for the implicit upwind tracer-advection solver.

One cache is shared across every tracer advected at the same velocity
field, timestep, and ice-mask within a given outer step (currently
`H_ice` and `lsf`). The discrete operator depends on `(ux, uy, dt,
mask)` only — not on the tracer — so a single refresh per outer step
amortises across all solves.

Lifecycle inside one outer step:

  1. `update_advection_operator!(cache, ux, uy, dt; mask=...)` — once.
     Refills `A.nzval` in place using the upwind-flux switch and the
     boundary-condition rules; refactors `P`. Sparsity pattern is
     fixed at construction.
  2. `solve_advection!(cache, c)` — once per tracer. Packs the current
     tracer values plus any explicit source into `b`, calls
     `bicgstab!(work, A, b; M=P, …)` into preallocated `x`, unpacks
     back into `c`. Zero allocation per call.

Fields:

  - `A`     — 5-point CSC sparse operator. Sparsity pattern static.
  - `b`     — RHS buffer, length `Nx*Ny`.
  - `x`     — solution buffer, length `Nx*Ny`.
  - `work`  — `Krylov.GmresWorkspace` (memory=20 restart length),
              reused across timesteps and tracers. We use GMRES
              rather than BiCGSTAB because the
              `KrylovPreconditioners` ILU + BiCGSTAB combination
              hits an `αₖ == 0` breakdown on certain CFL=1.0
              configurations of our 5-point M-matrix; GMRES + the
              same ILU converges in 1–2 iterations on the same
              matrix.
  - `P`     — `Ref{Any}` holding the ILU0 preconditioner. Lazy: first
              `update_advection_operator!` call factorises;
              subsequent calls refactor in place if the
              `KrylovPreconditioners` API allows, otherwise rebuild.
              The `Ref{Any}` indirection mirrors `ssa_amg_cache` and
              keeps the concrete preconditioner type out of the
              struct signature (preserves precompilation).
  - `Nx`, `Ny` — grid sizes (sanity checks at solve time).
"""
struct ImplicitAdvectionCache
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    work::GmresWorkspace
    P::Base.RefValue{Any}
    # `nz_*` index maps: position in `A.nzval` for the (row=cell(i,j),
    # col=stencil_neighbour) entry. `0` means the neighbour does not
    # exist (Bounded-axis edge with no wraparound). Built once at
    # `init_advection_cache` time so `update_advection_operator!` can
    # write coefficients in place without (row, col) → nzval lookup.
    nz_self::Matrix{Int}
    nz_east::Matrix{Int}
    nz_west::Matrix{Int}
    nz_north::Matrix{Int}
    nz_south::Matrix{Int}
    periodic_x::Bool
    periodic_y::Bool
    Nx::Int
    Ny::Int
end

"""
    init_advection_cache(grid::RectilinearGrid) -> ImplicitAdvectionCache

Allocate the persistent cache for the implicit advection solver,
sized for `grid` and with the 5-point stencil sparsity pattern of
`A` set up at construction time. Subsequent `update_advection_operator!`
and `solve_advection!` calls do not allocate (modulo the
preconditioner refactorisation, which may allocate inside
`KrylovPreconditioners`).

Honors `topology(grid)` for the x and y axes: `Periodic` axes contribute
wraparound stencil entries, `Bounded` axes drop off-grid neighbours.
3rd-axis topology is ignored (2D operator).
"""
function init_advection_cache(grid)
    Nx = size(grid, 1)
    Ny = size(grid, 2)
    topo = topology(grid)
    periodic_x = topo[1] === Periodic
    periodic_y = topo[2] === Periodic
    A = _build_5pt_sparsity(Nx, Ny, topo[1], topo[2])

    nz_self  = Matrix{Int}(undef, Nx, Ny)
    nz_east  = zeros(Int, Nx, Ny)
    nz_west  = zeros(Int, Nx, Ny)
    nz_north = zeros(Int, Nx, Ny)
    nz_south = zeros(Int, Nx, Ny)
    _fill_nzval_index_maps!(nz_self, nz_east, nz_west, nz_north, nz_south,
                            A, Nx, Ny, periodic_x, periodic_y)

    N = Nx * Ny
    return ImplicitAdvectionCache(
        A,
        Vector{Float64}(undef, N),
        Vector{Float64}(undef, N),
        GmresWorkspace(N, N, Vector{Float64}; memory = 20),
        Ref{Any}(nothing),
        nz_self, nz_east, nz_west, nz_north, nz_south,
        periodic_x, periodic_y,
        Nx, Ny,
    )
end

# Locate (row, col) in `A.nzval`. `A` is CSC, so we scan the column.
# Returns 0 if the entry is not in the sparsity pattern.
@inline function _find_nzval_index(A::SparseMatrixCSC, row::Int, col::Int)
    rng = A.colptr[col]:(A.colptr[col + 1] - 1)
    @inbounds for k in rng
        A.rowval[k] == row && return k
    end
    return 0
end

# Fill the per-cell stencil index lookups so `update_advection_operator!`
# can write coefficients into `A.nzval` in O(1) per entry.
function _fill_nzval_index_maps!(nz_self, nz_east, nz_west, nz_north, nz_south,
                                 A::SparseMatrixCSC, Nx::Int, Ny::Int,
                                 periodic_x::Bool, periodic_y::Bool)
    for j in 1:Ny, i in 1:Nx
        n = _row_index(i, j, Nx)
        nz_self[i, j] = _find_nzval_index(A, n, n)

        # E neighbour
        i_e = i < Nx ? i + 1 : (periodic_x ? 1 : 0)
        if i_e != 0
            nz_east[i, j] = _find_nzval_index(A, n, _row_index(i_e, j, Nx))
        end

        # W neighbour
        i_w = i > 1 ? i - 1 : (periodic_x ? Nx : 0)
        if i_w != 0
            nz_west[i, j] = _find_nzval_index(A, n, _row_index(i_w, j, Nx))
        end

        # N neighbour
        j_n = j < Ny ? j + 1 : (periodic_y ? 1 : 0)
        if j_n != 0
            nz_north[i, j] = _find_nzval_index(A, n, _row_index(i, j_n, Nx))
        end

        # S neighbour
        j_s = j > 1 ? j - 1 : (periodic_y ? Ny : 0)
        if j_s != 0
            nz_south[i, j] = _find_nzval_index(A, n, _row_index(i, j_s, Nx))
        end
    end
    return nothing
end

# Lexicographic flat index: i,j → row in [1, Nx*Ny].
@inline _row_index(i::Int, j::Int, Nx::Int) = i + (j - 1) * Nx

# Build the 5-point upwind stencil sparsity pattern for an Nx×Ny grid
# with the given x / y topologies (Oceananigans `Periodic` or `Bounded`).
# All `nzval` entries are zero; `update_advection_operator!` overwrites
# them in place each outer step.
#
# At each cell we always include the diagonal entry plus the four
# stencil neighbours (E/W/N/S) where they exist. Off-grid neighbours
# under `Bounded` are dropped (no entry); under `Periodic` they wrap.
# Mask changes (cells joining or leaving the ice domain) flip nzval
# only — they do not change which neighbour entries are present.
function _build_5pt_sparsity(Nx::Int, Ny::Int, tx, ty)
    periodic_x = tx === Periodic
    periodic_y = ty === Periodic

    # Worst-case nnz: 5 entries per cell. We allocate that, then
    # only emit the entries that actually exist.
    n_max = 5 * Nx * Ny
    I = Vector{Int}(undef, n_max)
    J = Vector{Int}(undef, n_max)
    V = Vector{Float64}(undef, n_max)
    k = 0

    @inline function emit!(row::Int, col::Int)
        k += 1
        I[k] = row
        J[k] = col
        V[k] = 0.0
    end

    for j in 1:Ny, i in 1:Nx
        row = _row_index(i, j, Nx)
        emit!(row, row)  # diagonal

        # East neighbour (i+1, j)
        if i < Nx
            emit!(row, _row_index(i + 1, j, Nx))
        elseif periodic_x
            emit!(row, _row_index(1, j, Nx))
        end

        # West neighbour (i-1, j)
        if i > 1
            emit!(row, _row_index(i - 1, j, Nx))
        elseif periodic_x
            emit!(row, _row_index(Nx, j, Nx))
        end

        # North neighbour (i, j+1)
        if j < Ny
            emit!(row, _row_index(i, j + 1, Nx))
        elseif periodic_y
            emit!(row, _row_index(i, 1, Nx))
        end

        # South neighbour (i, j-1)
        if j > 1
            emit!(row, _row_index(i, j - 1, Nx))
        elseif periodic_y
            emit!(row, _row_index(i, Ny, Nx))
        end
    end

    resize!(I, k); resize!(J, k); resize!(V, k)
    N = Nx * Ny
    return sparse(I, J, V, N, N)
end

"""
    advect_tracer!(c, ux, uy, dt;
                   scheme = :upwind_explicit, cache = nothing,
                   cfl_safety, fill_velocity_halos) -> c

Advance a 2D cell-centred tracer `c` (CenterField) by `dt` years.

`scheme` selects the solver:

  - `:upwind_explicit` (default) — forward Euler with first-order
    upwind in space; CFL-limited internal substepping. See
    `advect_tracer_upwind_explicit!`. `cache` is ignored.
  - `:upwind_implicit` — backward Euler with first-order upwind,
    solved as a single linear system per outer step. CFL-unconstrained.
    Requires `cache::ImplicitAdvectionCache`. See
    `advect_tracer_upwind_implicit!`.

Velocities `ux` (XFaceField) and `uy` (YFaceField) are held fixed
during the call.

`cfl_safety` (default 0.1, matching Yelmo Fortran's `ytopo.cfl_max`)
applies to `:upwind_explicit` only; `:upwind_implicit` ignores it.
"""
function advect_tracer!(c, ux, uy, dt::Real;
                        scheme::Symbol = :upwind_explicit,
                        cache = nothing,
                        source = nothing,
                        mask = nothing,
                        cfl_safety::Real = 0.1,
                        fill_velocity_halos::Bool = true,
                        atol::Real = 1e-12,
                        rtol::Real = 1e-12,
                        itmax::Int = 1000)
    if scheme === :upwind_explicit
        advect_tracer_upwind_explicit!(c, ux, uy, dt;
            cfl_safety = cfl_safety,
            fill_velocity_halos = fill_velocity_halos)
    elseif scheme === :upwind_implicit
        cache_inst = _resolve_implicit_cache(cache, c.grid)
        advect_tracer_upwind_implicit!(c, ux, uy, dt, cache_inst;
            source = source, mask = mask,
            fill_velocity_halos = fill_velocity_halos,
            atol = atol, rtol = rtol, itmax = itmax)
    else
        error("advect_tracer!: unknown scheme $(scheme). " *
              "Supported: :upwind_explicit, :upwind_implicit.")
    end
    return c
end

# Resolve the `cache` argument of `advect_tracer!` for the implicit
# scheme. Accepts an `ImplicitAdvectionCache` directly, or a
# `Ref{Any}` that holds one (lazily allocating on first use). The
# `Ref{Any}` form is what `y.tpo.scratch.adv_cache` carries —
# YelmoCore can't construct an `ImplicitAdvectionCache` eagerly because
# the type lives in a later-included module.
_resolve_implicit_cache(cache::ImplicitAdvectionCache, grid) = cache
function _resolve_implicit_cache(cache::Base.RefValue, grid)
    cache[] === nothing && (cache[] = init_advection_cache(grid))
    return cache[]::ImplicitAdvectionCache
end
_resolve_implicit_cache(::Nothing, grid) = error(
    "advect_tracer!: scheme=:upwind_implicit requires `cache=...` " *
    "(an `ImplicitAdvectionCache`, typically `y.tpo.scratch.adv_cache`).")

"""
    advect_tracer_upwind_explicit!(c, ux, uy, dt;
                                   cfl_safety, fill_velocity_halos) -> c

Forward Euler + first-order upwind tracer advection. Substeps
internally with `dt_sub ≤ cfl_safety * min(dx/|u|_max, dy/|v|_max)`
until `dt` is reached. Mirrors Fortran Yelmo's `expl-upwind`.
"""
function advect_tracer_upwind_explicit!(c, ux, uy, dt::Real;
                                        cfl_safety::Real = 0.1,
                                        fill_velocity_halos::Bool = true)
    grid = c.grid
    U = (u=ux, v=uy, w=Oceananigans.Fields.ZeroField())

    if fill_velocity_halos
        fill_halo_regions!(ux)
        fill_halo_regions!(uy)
    end

    tend = similar(interior(c))
    elapsed = 0.0
    while elapsed < dt
        cfl_dt = _cfl_dt(grid, ux, uy, cfl_safety)
        dt_sub = min(dt - elapsed, cfl_dt)

        fill_halo_regions!(c)
        @inbounds for k in axes(tend, 3), j in axes(tend, 2), i in axes(tend, 1)
            tend[i, j, k] = -div_Uc(i, j, k, grid, _DEFAULT_OCN_UPWIND, U, c)
        end
        interior(c) .+= dt_sub .* tend

        elapsed += dt_sub
    end

    fill_halo_regions!(c)
    return c
end

"""
    advect_tracer_upwind_implicit!(c, ux, uy, dt, cache;
                                   source, mask, fill_velocity_halos,
                                   atol, rtol, itmax) -> c

Backward-Euler + first-order upwind tracer advection. Refreshes the
operator stored in `cache` for the current `(ux, uy, dt, mask)`,
then solves `A · c^{n+1} = c^n + dt · source` for `c` via GMRES
preconditioned with the cached ILU-style factorisation.

CFL-unconstrained — the discretisation is stable for arbitrary
`dt`, though accuracy still degrades at large CFL.

This wrapper does both refresh + solve. For multi-tracer use where
several tracers share the same `(ux, uy, dt, mask)`, call
`update_advection_operator!` once and `solve_advection!` per
tracer to avoid redundant refactorisations of the preconditioner.

`fill_velocity_halos` is accepted for parity with the explicit
path; the implicit assembly does not consult halo cells (Bounded
edges are handled by Dirichlet rows in the matrix), so the flag
is currently a no-op.
"""
function advect_tracer_upwind_implicit!(c, ux, uy, dt::Real,
                                        cache::ImplicitAdvectionCache;
                                        source = nothing,
                                        mask = nothing,
                                        fill_velocity_halos::Bool = true,
                                        atol::Real = 1e-12,
                                        rtol::Real = 1e-12,
                                        itmax::Int = 1000)
    grid = c.grid
    dx = _dx(grid)
    dy = _dy(grid)

    update_advection_operator!(cache, ux, uy, dt, dx, dy; mask = mask)
    solve_advection!(cache, c, dt;
                     source = source, atol = atol, rtol = rtol, itmax = itmax)

    return c
end

"""
    update_advection_matrix!(cache, ux, uy, dt, dx, dy; mask) -> cache

Refresh `cache.A.nzval` in place for the current `(ux, uy, dt,
mask)`. Sparsity pattern is unchanged.

This is the zero-allocation hot loop: matrix coefficients only, no
preconditioner refactorisation. The companion `update_advection_operator!`
wraps this with a fresh ILU factorisation; use that for the
once-per-outer-step refresh, and use this directly only when
benchmarking the assembly kernel or when the preconditioner is
managed externally.

Discretisation matches Fortran Yelmo's `impl-lis`
(solver_advection.f90:143–531) with `WOVI = 1.0` (pure backward
Euler):

  - Velocities `ux::XFaceField`, `uy::YFaceField` give cell-face
    values. Per Yelmo.jl / Oceananigans staggering:
      ux[i,j]   = west face of cell (i,j)  (= `ux_1` in Fortran)
      ux[i+1,j] = east face of cell (i,j)  (= `ux_2`)
      uy[i,j]   = south face of cell (i,j) (= `uy_1`)
      uy[i,j+1] = north face of cell (i,j) (= `uy_2`)
    For Periodic axes the +1 neighbour wraps via `mod1`.

  - Upwind switch on each face: positive face velocity ⇒ tracer
    value comes from the upstream cell; negative ⇒ from the
    downstream cell. The implicit form folds this into the matrix:
    diagonal accumulates outflow contributions; off-diagonals carry
    inflow contributions from upstream neighbours.

  - Boundary cells:
      • Periodic axis  → wraparound stencil (interior assembly).
      • Bounded edge   → Dirichlet zero (matrix row = δᵢⱼ, RHS=0).

  - Mask handling (`mask` is an `Int` matrix of size `(Nx, Ny)` or
    `nothing`):
      mask[i,j] ==  0 → Dirichlet zero.
      mask[i,j] == -1 → Dirichlet to the current `cache.b` value
                        (caller-provided prescribed value).
                        NOTE: Fortran sets b=0 here, which zeroes
                        the prescribed value at solve time. We treat
                        that as a Fortran bug and let the caller
                        prescribe via the b vector instead.
      otherwise (default 1) → dynamic / interior assembly.
    `nothing` ⇒ all cells are dynamic.

This step does NOT touch `cache.b` or `cache.P[]`. The RHS is
filled per tracer in `solve_advection!`. The preconditioner is
(re)factored in `solve_advection!` (or a future explicit
preconditioner-refresh helper).

`dx`, `dy` are the cell sizes in metres.
"""
function update_advection_matrix!(cache::ImplicitAdvectionCache,
                                  ux, uy, dt::Real, dx::Real, dy::Real;
                                  mask = nothing)
    # Outer wrapper that extracts concrete `interior` arrays then
    # dispatches to a type-specialised inner kernel. Splitting like
    # this is the standard Julia idiom for fully zero-allocation hot
    # loops over Oceananigans `Field`s: the compiler specialises
    # `_update_advection_operator_kernel!` to the concrete `SubArray`
    # eltype/parent so scalar `Ux[i,j,1]` reads do not box.
    _update_advection_operator_kernel!(cache, interior(ux), interior(uy),
                                       Float64(dt), Float64(dx), Float64(dy),
                                       mask)
    return cache
end

"""
    update_advection_operator!(cache, ux, uy, dt, dx, dy; mask) -> cache

Refresh both the matrix `cache.A` and the preconditioner `cache.P[]`
for the current `(ux, uy, dt, mask)`. Once-per-outer-step call;
amortises across every `solve_advection!` invocation in the step.

The matrix update goes through `update_advection_matrix!`
(zero-allocation hot loop). The preconditioner refactor allocates a
fresh `KrylovPreconditioners.ILUFactorization` (CPU `ilu` has no
in-place `update!`); the cost is O(nnz), not per-cell.

`τ` is the Crout drop-tolerance. We use `1e-12` rather than the
ostensibly-cleaner `0.0` because `KrylovPreconditioners.ilu` with
exactly zero produces a broken factorisation that triggers
BiCGSTAB breakdown on iteration 2 (αₖ == 0, observed during PR-3
implementation). At `τ = 1e-12` the factorisation is effectively
the same — no realistic ILU entry is below 1e-12 — and BiCGSTAB
converges in 1–2 iterations on our 5-point M-matrix. For a
future GPU port, swap this call for `KP.kp_ilu0(cache.A_gpu)` —
same preconditioner role, different backend.
"""
function update_advection_operator!(cache::ImplicitAdvectionCache,
                                    ux, uy, dt::Real, dx::Real, dy::Real;
                                    mask = nothing)
    update_advection_matrix!(cache, ux, uy, dt, dx, dy; mask = mask)
    cache.P[] = KP.ilu(cache.A; τ = 1e-12)
    return cache
end

function _update_advection_operator_kernel!(cache::ImplicitAdvectionCache,
                                            Ux::AbstractArray{Float64, 3},
                                            Uy::AbstractArray{Float64, 3},
                                            dt::Float64, dx::Float64, dy::Float64,
                                            mask)
    Nx, Ny = cache.Nx, cache.Ny
    nzval = cache.A.nzval

    # CFL factors. Each upwind contribution is `c{xy} * face_velocity`.
    cx = dt / dx
    cy = dt / dy

    # Sparsity is static — only nzval changes. Zero everything first
    # so off-diagonal entries that disappear under the upwind switch
    # (e.g. a wind-direction flip) do not retain a stale value.
    fill!(nzval, 0.0)

    periodic_x = cache.periodic_x
    periodic_y = cache.periodic_y
    nz_self  = cache.nz_self
    nz_east  = cache.nz_east
    nz_west  = cache.nz_west
    nz_north = cache.nz_north
    nz_south = cache.nz_south

    @inbounds for j in 1:Ny, i in 1:Nx
        # Resolve mask. 0 / -1 short-circuit to a Dirichlet identity row.
        if mask !== nothing
            m = mask[i, j]
            if m == 0 || m == -1
                nzval[nz_self[i, j]] = 1.0
                continue
            end
        end

        # Bounded-edge cells: Dirichlet zero. The east/west/north/south
        # neighbour entries from `_build_5pt_sparsity` exist (when they
        # exist at all) but stay at 0 from the `fill!` above.
        on_x_edge = !periodic_x && (i == 1 || i == Nx)
        on_y_edge = !periodic_y && (j == 1 || j == Ny)
        if on_x_edge || on_y_edge
            nzval[nz_self[i, j]] = 1.0
            continue
        end

        # Interior assembly. Resolve face indices for E and N neighbours
        # (Periodic wraps to 1; Bounded uses i+1 / j+1 — and we know we
        # are not on an edge by the check above).
        i_eface = periodic_x && i == Nx ? 1 : i + 1
        j_nface = periodic_y && j == Ny ? 1 : j + 1

        ux_w = Ux[i,       j, 1]   # west face  (Fortran ux_1)
        ux_e = Ux[i_eface, j, 1]   # east face  (Fortran ux_2)
        uy_s = Uy[i, j,       1]   # south face (Fortran uy_1)
        uy_n = Uy[i, j_nface, 1]   # north face (Fortran uy_2)

        # Diagonal: 1 (mass term) + sum of outflow contributions.
        # An "outflow" face is one where the upwind value comes from
        # cell (i,j) itself: ux_w<0 (west pulls out), ux_e>0 (east
        # pushes out), uy_s<0, uy_n>0.
        a_diag = 1.0
        if ux_w < 0.0; a_diag -= cx * ux_w; end          # = +cx*|ux_w|
        if ux_e > 0.0; a_diag += cx * ux_e; end
        if uy_s < 0.0; a_diag -= cy * uy_s; end
        if uy_n > 0.0; a_diag += cy * uy_n; end
        nzval[nz_self[i, j]] = a_diag

        # Off-diagonals: inflow contributions from upstream neighbours.
        # Negative entries (the upstream tracer enters this cell with a
        # negative coefficient on the LHS).
        if ux_w > 0.0
            nzval[nz_west[i, j]]  = -cx * ux_w
        end
        if uy_s > 0.0
            nzval[nz_south[i, j]] = -cy * uy_s
        end
        if ux_e < 0.0
            nzval[nz_east[i, j]]  =  cx * ux_e    # ux_e<0 ⇒ negative
        end
        if uy_n < 0.0
            nzval[nz_north[i, j]] =  cy * uy_n
        end
    end

    return nothing
end

"""
    solve_advection!(cache, c, dt; source, atol, rtol, itmax) -> c

Solve `A · c^{n+1} = c^n + dt · source` for the tracer `c` using
the operator already refreshed in `cache` by an earlier
`update_advection_operator!` call. The matrix `A` is shared across
every tracer advected at the same `(ux, uy, dt, mask)`; this
function only touches the per-tracer `b`, `x`, and the field `c`.

`source` is an optional `CenterField` (or compatible array) holding
the explicit source/sink term for `c` in the same units as
`∂c/∂t`. Pass `nothing` for pure advection (no source).

Tolerances default to the Fortran `impl-lis` defaults
(`-tol 1.0e-12 -maxiter 1000`). The initial guess is the current
`c` value (passed in via `cache.x`).

Returns `c` modified in place, plus solver `stats` are stored in
`cache.work` (GMRES workspace).
"""
function solve_advection!(cache::ImplicitAdvectionCache, c, dt::Real;
                          source = nothing,
                          atol::Real = 1e-12,
                          rtol::Real = 1e-12,
                          itmax::Int = 1000)
    cache.P[] === nothing &&
        error("solve_advection!: preconditioner is not yet built. " *
              "Call `update_advection_operator!` before `solve_advection!`.")

    # Pack RHS and initial guess. Concrete-typed inner kernel keeps
    # the loop zero-alloc (same trick as `update_advection_operator!`).
    src_array = source === nothing ? nothing : interior(source)
    _pack_rhs_kernel!(cache.b, cache.x, interior(c), src_array, Float64(dt),
                      cache.Nx, cache.Ny)

    # In-place GMRES. Initial guess `cache.x` (= the current `c`).
    # We use GMRES rather than BiCGSTAB because BiCGSTAB combined with
    # the KrylovPreconditioners ILU produces an `αₖ == 0` breakdown
    # on certain CFL=1.0 configurations of our 5-point M-matrix
    # (observed during PR-3). GMRES + ILU converges in 1–2 iterations
    # on the same problems, so the swap is both more robust and
    # faster. Memory=20 is the GMRES restart length; our solves
    # converge well below that, so it never matters.
    gmres!(cache.work, cache.A, cache.b, cache.x;
           M = cache.P[], ldiv = true,
           atol = atol, rtol = rtol, itmax = itmax,
           history = false)

    # Unpack solution back into `c`. `cache.work.x` holds the latest
    # iterate after `bicgstab!` returns.
    _unpack_solution_kernel!(interior(c), cache.work.x, cache.Nx, cache.Ny)

    return c
end

# Pack `(c_n + dt·source)` into the RHS vector `b` in lexicographic
# order, and seed the initial guess `x0` with the current tracer
# values. Both buffers use the same `_row_index(i,j,Nx)` flattening
# as the matrix assembly.
function _pack_rhs_kernel!(b::Vector{Float64}, x0::Vector{Float64},
                           C::AbstractArray{Float64, 3},
                           ::Nothing, dt::Float64,
                           Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        n = _row_index(i, j, Nx)
        c_now = C[i, j, 1]
        b[n]  = c_now
        x0[n] = c_now
    end
    return nothing
end

function _pack_rhs_kernel!(b::Vector{Float64}, x0::Vector{Float64},
                           C::AbstractArray{Float64, 3},
                           S::AbstractArray{Float64, 3}, dt::Float64,
                           Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        n = _row_index(i, j, Nx)
        c_now = C[i, j, 1]
        b[n]  = c_now + dt * S[i, j, 1]
        x0[n] = c_now
    end
    return nothing
end

function _unpack_solution_kernel!(C::AbstractArray{Float64, 3},
                                  x::Vector{Float64},
                                  Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        n = _row_index(i, j, Nx)
        C[i, j, 1] = x[n]
    end
    return nothing
end

# Largest dt for which `cfl_safety * (|u|·dt/dx + |v|·dt/dy) ≤ 1` over
# the whole grid. Returns `Inf` if velocity is identically zero (then
# only the outer `dt` constrains the loop).
function _cfl_dt(grid, ux, uy, cfl_safety::Real)
    dx = _dx(grid)
    dy = _dy(grid)

    umax = maximum(abs, interior(ux))
    vmax = maximum(abs, interior(uy))

    inv_dt = umax / dx + vmax / dy
    inv_dt > 0 || return Inf
    return cfl_safety / inv_dt
end

# Cell sizes from a regularly-spaced RectilinearGrid. We don't yet
# support stretched grids in the advection kernel; flag explicitly.
function _dx(grid::RectilinearGrid)
    Δx = grid.Δxᶜᵃᵃ
    Δx isa Number || error("advect_tracer! requires a uniform x-spacing for now (got $(typeof(Δx))).")
    return abs(Δx)
end

function _dy(grid::RectilinearGrid)
    Δy = grid.Δyᵃᶜᵃ
    Δy isa Number || error("advect_tracer! requires a uniform y-spacing for now (got $(typeof(Δy))).")
    return abs(Δy)
end
