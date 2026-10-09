# ----------------------------------------------------------------------
# Shallow-Shelf Approximation (SSA) momentum kernels.
#
# This file lands SSA assembly-side helpers in milestone 3d / PR-A.2:
#
#   - `set_ssa_masks!` — port of `solver_ssa_ac.f90:854 set_ssa_masks`.
#     Walks each ac-staggered face position and sets the SSA solver
#     mask value:
#       0 = inactive (Dirichlet u = 0)
#       1 = active grounded (shelfy-stream)
#       2 = active floating (shelf)
#       3 = lateral BC at calving front
#       4 = deactivated lateral BC (treated as inner SSA)
#
#   - `_assemble_ssa_matrix!` — port of
#     `solver_ssa_ac.f90:240-826 linear_solver_matrix_ssa_ac_csr_2D`.
#     Populates the COO triplet buffers (`I_idx`, `J_idx`, `vals`) and
#     RHS vector `b_vec` in `dyn.scratch.ssa_*`. Block-row layout:
#
#         row 2k-1 → ux equation at cell k = (j-1)*Nx + i
#         row 2k   → uy equation at cell k
#
# The driver (`calc_velocity_ssa!`) and the actual Krylov+AMG solve
# arrive in milestone 3d / PR-B. PR-A.2 stops at filling the COO
# buffers — no SparseMatrixCSC assembly, no linear solve, no
# `solver=="ssa"` dispatch wiring.
#
# Index conventions (Yelmo.jl XFace / YFace face stagger):
#
#   - Fortran `ux(i, j)` lives at `Ux[i+1, j, 1]` of an `XFaceField`
#     under `Bounded` x. Under `Periodic` x the slot is `mod1(i+1, Nx)`
#     via `_ip1_modular`.
#   - Fortran `uy(i, j)` lives at `Uy[i, j+1, 1]` of a `YFaceField`
#     under `Bounded` y. Under `Periodic` y via `_jp1_modular`.
#   - `ssa_mask_acx` / `ssa_mask_acy` are stored as Float64 on
#     XFace/YFace fields (Yelmo.jl convention; Fortran integers).
#     Cast to Int at comparison sites.
#
# Periodic-x / -y handling: the matrix-assembly kernel uses explicit
# wrap on (im1, ip1, jm1, jp1) for the column-index arithmetic (the
# matrix indices `ij2n` must address actual interior cells, not halo
# cells). For neighbour reads of halo-fill-able fields, callers fill
# halos before invocation and the kernel reads the wrapped values
# through the standard halo path.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.Grids: topology, Bounded, Periodic, AbstractTopology
using Oceananigans.BoundaryConditions: fill_halo_regions!

using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: norm, Diagonal, diag
using Krylov: bicgstab!, cg!
using AlgebraicMultigrid: smoothed_aggregation, ruge_stuben, aspreconditioner,
                          GaussSeidel, Jacobi
using NCDatasets: NCDataset, defDim, defVar
using Base.Threads: @threads

export set_ssa_masks!, _assemble_ssa_matrix!,
       _solve_ssa_linear!, calc_velocity_ssa!,
       picard_relax_visc!, picard_relax_vel!,
       picard_calc_convergence_l2, picard_calc_convergence_l1rel_matrix!,
       set_inactive_margins!, calc_basal_stress!,
       dump_ssa_assembly

"""
    set_ssa_masks!(ssa_mask_acx, ssa_mask_acy, mask_frnt, f_ice, f_grnd;
                   lateral_bc::AbstractString, use_ssa::Bool = true)
        -> (ssa_mask_acx, ssa_mask_acy)

Set the SSA solver masks on the faces. Fortran face `(i, j)` is written to
`ssa_mask_acx[i+1, j, 1]` (and `ssa_mask_acy[i, j+1, 1]`):

  - 0 = no SSA (velocity zero), also a wall where floating ice meets
        ice-free land.
  - 1 = grounded ice or grounding line (shelfy-stream).
  - 2 = floating ice (shelf).
  - 3 = ice front with the lateral boundary condition.
  - 4 = ice front treated as inner SSA (half drag in the assembler).

A face is active when either neighbour is fully covered (`f_ice == 1`; the
model passes `f_ice_dyn`). Front faces (a front cell `mask_frnt > 0` next
to an ice-free cell `mask_frnt < 0`) are set by `_front_face_mask` from
`lateral_bc` ("none", "floating"/"float", "marine", "all"). With
`use_ssa = false` all faces are 0.

Port of `solver_ssa_ac.f90:set_ssa_masks` (yelmo dev).
"""
function set_ssa_masks!(ssa_mask_acx, ssa_mask_acy, mask_frnt, f_ice, f_grnd;
                        lateral_bc::AbstractString, use_ssa::Bool = true)
    lateral_bc in ("none", "floating", "float", "marine", "all") ||
        error("set_ssa_masks!: lateral_bc = \"$lateral_bc\" not recognized.")

    Mx = interior(ssa_mask_acx)
    My = interior(ssa_mask_acy)
    fill!(Mx, 0.0)
    fill!(My, 0.0)
    use_ssa || return ssa_mask_acx, ssa_mask_acy

    MF = interior(mask_frnt)
    Fi = interior(f_ice)
    Fg = interior(f_grnd)
    Nx, Ny = size(Fi, 1), size(Fi, 2)
    Tx = topology(ssa_mask_acx.grid, 1)
    Ty = topology(ssa_mask_acy.grid, 2)
    lat = _lateral_bc_code(lateral_bc)

    _set_ssa_masks_kernel!(Mx, My, MF, Fi, Fg, lat, Tx, Ty, Nx, Ny)
    return ssa_mask_acx, ssa_mask_acy
end

# `lateral_bc` as an integer for the kernel: 0 none, 1 floating, 2 marine, 3 all.
_lateral_bc_code(lateral_bc::AbstractString) =
    lateral_bc == "none"   ? 0 :
    lateral_bc == "marine" ? 2 :
    lateral_bc == "all"    ? 3 : 1

function _set_ssa_masks_kernel!(Mx, My, MF, Fi, Fg, lat::Int,
                                ::Type{Tx}, ::Type{Ty}, Nx::Int, Ny::Int
                               ) where {Tx<:AbstractTopology, Ty<:AbstractTopology}
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1  = _neighbor_ip1(i, Nx, Tx)
        jp1  = _neighbor_jp1(j, Ny, Ty)
        ip1f = _ip1_modular(i, Nx, Tx)
        jp1f = _jp1_modular(j, Ny, Ty)

        # x-direction: face between (i, j) and (ip1, j).
        if Fi[i, j, 1] == 1.0 || Fi[ip1, j, 1] == 1.0
            Mx[ip1f, j, 1] = (Fg[i, j, 1] > 0.0 || Fg[ip1, j, 1] > 0.0) ? 1.0 : 2.0
        end
        m = _front_face_mask(MF[i, j, 1], MF[ip1, j, 1], lat)
        m >= 0 && (Mx[ip1f, j, 1] = Float64(m))

        # y-direction: face between (i, j) and (i, jp1).
        if Fi[i, j, 1] == 1.0 || Fi[i, jp1, 1] == 1.0
            My[i, jp1f, 1] = (Fg[i, j, 1] > 0.0 || Fg[i, jp1, 1] > 0.0) ? 1.0 : 2.0
        end
        m = _front_face_mask(MF[i, j, 1], MF[i, jp1, 1], lat)
        m >= 0 && (My[i, jp1f, 1] = Float64(m))
    end
    return nothing
end

# SSA mask of the face between two cells with ice-front codes `a` and `b`
# (`MASK_FRNT_*`): -1 not a front face, 0 wall, 3 lateral BC, 4 front
# treated as inner SSA. A front is only floating or marine across a face
# whose ice-free side is ocean: across ice-free land, floating ice ends at a
# wall and grounded ice is a front grounded above sea level, whatever its
# bed. Port of `solver_ssa_ac.f90:front_face_mask` (yelmo dev).
@inline function _front_face_mask(a, b, lat::Int)
    if a > 0 && b < 0
        code_ice, code_free = a, b
    elseif a < 0 && b > 0
        code_ice, code_free = b, a
    else
        return -1
    end
    if code_free == MASK_FRNT_ICE_FREE_LAND
        code_ice == MASK_FRNT_FLOAT && return 0
        code_ice = MASK_FRNT_GRND
    end
    lat == 0 && return 4
    lat == 1 && return code_ice == MASK_FRNT_FLOAT ? 3 : 4
    lat == 2 && return (code_ice == MASK_FRNT_FLOAT || code_ice == MASK_FRNT_MARINE) ? 3 : 4
    return 3
end

# ----------------------------------------------------------------------
# _assemble_ssa_matrix!
#
# Faithful 1:1 port of `solver_ssa_ac.f90:240-826
# linear_solver_matrix_ssa_ac_csr_2D`. Populates the COO triplet
# buffers + RHS in `dyn.scratch`. The actual sparse-matrix
# construction and Krylov solve land in PR-B.
#
# Block-row interleaving (matches Fortran):
#
#     row(ux at cell (i,j)) = 2 * ((j-1)*Nx + i) - 1
#     row(uy at cell (i,j)) = 2 * ((j-1)*Nx + i)
#     col(ux at cell (i,j)) = 2 * ((j-1)*Nx + i) - 1
#     col(uy at cell (i,j)) = 2 * ((j-1)*Nx + i)
#
# Boundary-condition palette (Fortran lines 156-212):
#
#     boundaries = :MISMIP3D     → (free-slip, periodic, no-slip, periodic)
#     boundaries = :TROUGH       → (free-slip, periodic, no-slip, periodic)
#     boundaries = :periodic     → (periodic, periodic, periodic, periodic)
#     boundaries = :periodic_x   → (periodic, free-slip, periodic, free-slip)
#     boundaries = :periodic_y   → (free-slip, periodic, free-slip, periodic)
#     boundaries = :infinite     → (free-slip, free-slip, free-slip, free-slip)
#     boundaries = :mask         → (free-slip, free-slip, free-slip, free-slip)
#     boundaries = :zeros        → (no-slip, no-slip, no-slip, no-slip)
#
#   bcs[1] = right (i = Nx)
#   bcs[2] = top (j = Ny)
#   bcs[3] = left (i = 1)
#   bcs[4] = bottom (j = 1)
#
# PR-A.2 supports topology pairs (Bounded, Bounded, Flat) and
# (Bounded, Periodic, Flat). The (Periodic, *) pairs are deferred to
# a later PR; the kernel asserts on entry and rejects them.
#
# Halo handling: caller pre-fills halos on the inputs. Inside the
# kernel, all reads on `(im1, ip1, jm1, jp1)` go through the wrapped
# integer indices (NOT halo reads) — required because the COO
# column-index arithmetic must address actual interior cells, not halo
# cells. The `mask_frnt` field is read at neighbouring (i+1, j+1)
# diagonal positions for the lateral-BC detection (Fortran lines
# 1028-1043 / 1080-1095 inside set_ssa_masks); this kernel does not
# do diagonal reads itself, so `fill_corner_halos!` is not required.
# ----------------------------------------------------------------------

# Convert a `boundaries` symbol to the 4-tuple of edge-BC symbols.
# Per Fortran `solver_ssa_ac.f90:156-212`. The 4-tuple is
# (right, top, left, bottom) matching Fortran's `bcs(1:4)`.
function _ssa_resolve_bcs(boundaries::Symbol)
    if boundaries == :MISMIP3D || boundaries == :TROUGH
        return (:free_slip, :periodic, :no_slip, :periodic)
    elseif boundaries == :periodic
        return (:periodic, :periodic, :periodic, :periodic)
    elseif boundaries == :periodic_x
        return (:periodic, :free_slip, :periodic, :free_slip)
    elseif boundaries == :periodic_y
        return (:free_slip, :periodic, :free_slip, :periodic)
    elseif boundaries == :infinite || boundaries == :mask
        return (:free_slip, :free_slip, :free_slip, :free_slip)
    elseif boundaries == :zeros
        return (:no_slip, :no_slip, :no_slip, :no_slip)
    else
        error("_ssa_resolve_bcs: unknown boundaries :$(boundaries).")
    end
end

# Linear cell index. Fortran `lgs%ij2n(i, j) = (j-1)*Nx + i`.
@inline _ij2n(i::Int, j::Int, Nx::Int) = (j - 1) * Nx + i
# Block-interleaved row/column indices (Fortran convention).
@inline _row_ux(i::Int, j::Int, Nx::Int) = 2 * _ij2n(i, j, Nx) - 1
@inline _row_uy(i::Int, j::Int, Nx::Int) = 2 * _ij2n(i, j, Nx)

# Append a single (row, col, val) triplet to the COO buffers and
# bump the counter. Inlined to keep the kernel readable.
@inline function _push_coo!(I_idx::Vector{Int}, J_idx::Vector{Int},
                            vals::Vector{Float64}, k::Int,
                            row::Int, col::Int, val::Float64)
    I_idx[k] = row
    J_idx[k] = col
    vals[k]  = val
    return k
end

# CSC cache for the SSA stiffness matrix.
#
# Within one `dyn_step!` Picard loop the (row, col) structure of the
# assembled SSA matrix is invariant — only the values change as
# viscosity / beta refresh. On `picard_iter == 1` we build the CSC
# from the COO triplets via `sparse(...)` (allocating colptr / rowval
# / nzval) and compute a permutation `coo_to_csc[k]` that maps each
# COO triplet index `k` to its position in `A.nzval` (summed across
# duplicates). On subsequent Picard iters we reset `A.nzval .= 0`
# and accumulate `A.nzval[coo_to_csc[k]] += vals[k]`, reusing the
# same `A` object — zero-allocation refresh.
#
# Across `dyn_step!` calls the SSA mask can change (grounding line
# moves), so the cache is rebuilt on every `iter == 1`. Pre-PR cost
# was one `sparse(...)` call per Picard iter; post-PR is one
# `sparse(...)` per `dyn_step!`.
function _build_or_refresh_ssa_csc!(scratch,
                                     I_idx::Vector{Int},
                                     J_idx::Vector{Int},
                                     vals::Vector{Float64},
                                     nnz::Int, N::Int,
                                     picard_iter::Int)
    if picard_iter == 1 || scratch.ssa_csc[] === nothing
        I_view = view(I_idx, 1:nnz)
        J_view = view(J_idx, 1:nnz)
        V_view = view(vals,  1:nnz)
        A = sparse(I_view, J_view, V_view, N, N)
        # Build COO → CSC permutation: for each k, find the index in
        # A.nzval corresponding to (I_idx[k], J_idx[k]). `A.rowval` is
        # sorted within each column, so a binary search inside the
        # column slice is O(log nnz_per_col).
        @inbounds for k in 1:nnz
            i, j = I_idx[k], J_idx[k]
            col_start = A.colptr[j]
            col_end   = A.colptr[j+1] - 1
            pos = searchsortedfirst(view(A.rowval, col_start:col_end), i)
            scratch.ssa_coo_to_csc[k] = col_start + pos - 1
        end
        scratch.ssa_csc[] = A
        return A
    else
        A = scratch.ssa_csc[]::SparseMatrixCSC{Float64,Int}
        nzval = A.nzval
        fill!(nzval, 0.0)
        @inbounds for k in 1:nnz
            nzval[scratch.ssa_coo_to_csc[k]] += vals[k]
        end
        return A
    end
end

"""
    _assemble_ssa_matrix!(I_idx, J_idx, vals, b_vec, nnz_ref,
                          ux_b, uy_b,
                          beta_acx, beta_acy,
                          visc_eff_int, visc_ab,
                          ssa_mask_acx, ssa_mask_acy, mask_frnt,
                          H_ice, f_ice,
                          taud_acx, taud_acy,
                          taul_int_acx, taul_int_acy,
                          dx::Real, dy::Real;
                          boundaries::Symbol=:zeros,
                          lateral_bc::AbstractString="floating")

Faithful port of `solver_ssa_ac.f90:240-826
linear_solver_matrix_ssa_ac_csr_2D`. Populates the COO triplet
buffers `(I_idx, J_idx, vals)` and the RHS `b_vec` for the SSA
linear system. Updates `nnz_ref[]` to the actual number of non-zeros
written.

Inputs:

  - `ux_b`, `uy_b` — current (Picard) velocity, XFace / YFace 2D.
  - `beta_acx`, `beta_acy` — basal friction on faces (XFace / YFace).
  - `visc_eff_int` — depth-integrated viscosity at aa cells (CenterField);
    Fortran `N_aa(i, j)` ↔ `interior(visc_eff_int)[i, j, 1]`.
  - `visc_ab` — corner-staggered viscosity from `stagger_visc_aa_ab!`,
    `Field((Face(), Face(), Center()), g)`. Fortran `N_ab(i, j)` ↔
    `interior(visc_ab)[i+1, j+1, 1]`.
  - `ssa_mask_acx`, `ssa_mask_acy` — SSA mask, XFace / YFace 2D.
  - `mask_frnt`, `H_ice`, `f_ice`, `taud_acx`, `taud_acy`,
    `taul_int_acx`, `taul_int_acy` — geometry / forcing fields.
  - `dx`, `dy` — grid spacing in metres.
  - `beta_acx`, `beta_acy` are the friction of the matrix, with
    `beta_min` at grounded faces already set (`set_beta_min_grounded!`).
  - `boundaries` — Symbol selecting the edge-BC palette.
  - `lateral_bc` — accepted for signature parity (used by the caller
    when pre-computing the masks).

Block-row layout: row `2k - 1` is the ux equation at cell
`k = (j-1)*Nx + i`, row `2k` is the uy equation at cell k.

Returns the kernel arguments unchanged (in-place mutation of the
COO buffers, RHS, and `nnz_ref`).

Topology: `(Bounded, Bounded, Flat)`, `(Bounded, Periodic, Flat)`,
`(Periodic, Bounded, Flat)`, and `(Periodic, Periodic, Flat)` are
supported. Other combinations (e.g. anything other than `Flat` for the
third axis) error. The matrix-assembly kernel itself is topology-clean
under both axes — face-slot reads go through `_ip1_modular` /
`_jp1_modular` and all boundary-row branches guard with
`bcs[k] !== :periodic`, so under fully-periodic boundaries the kernel
falls through to the inner-SSA stencil at i==1 / i==Nx / j==1 / j==Ny
and the wrapped-int neighbour math (`im1=Nx`, `ip1=1`) gives the
periodic-wrap intent. Periodic-x support enables the ISMIP-HOM-C
benchmark.
"""
function _assemble_ssa_matrix!(I_idx::Vector{Int},
                               J_idx::Vector{Int},
                               vals::Vector{Float64},
                               b_vec::Vector{Float64},
                               nnz_ref::Ref{Int},
                               ux_b, uy_b,
                               beta_acx, beta_acy,
                               visc_eff_int, visc_ab,
                               ssa_mask_acx, ssa_mask_acy, mask_frnt,
                               H_ice, f_ice,
                               taud_acx, taud_acy,
                               taul_int_acx, taul_int_acy,
                               dx::Real, dy::Real;
                               boundaries::Symbol=:zeros,
                               lateral_bc::AbstractString="floating")
    # Wrapper: do halo fills + lift Field views to plain SubArrays +
    # look up topology, then dispatch to the parametric kernel below.
    # Same wrapper-+-parametric-kernel template as the dyn 3D series
    # and the DIVA viscosity / helpers PRs (#45 / #47 / #48 / #49 /
    # #50 / #51 / #52 / #53). The kernel sees concrete `Float64`
    # scalars and topology subtypes as `Type` parameters, which lets
    # the per-row `_ip1_modular` / `_jp1_modular` calls fold at
    # compile time and the inner Field-read loops become alloc-free.

    # ---- Topology checks. ----
    # Only Bounded / Periodic supported on each horizontal axis.
    Tx_top = topology(visc_eff_int.grid, 1)
    Ty_top = topology(visc_eff_int.grid, 2)
    (Tx_top === Bounded || Tx_top === Periodic) || error(
        "_assemble_ssa_matrix!: x-topology must be Bounded or Periodic " *
        "(got $(Tx_top)).")
    (Ty_top === Bounded || Ty_top === Periodic) || error(
        "_assemble_ssa_matrix!: y-topology must be Bounded or Periodic " *
        "(got $(Ty_top)).")

    # Fill halos on every input the kernel reads with i±1 / j±1 stencils.
    fill_halo_regions!(visc_eff_int)
    fill_halo_regions!(visc_ab)
    fill_halo_regions!(beta_acx)
    fill_halo_regions!(beta_acy)
    fill_halo_regions!(ssa_mask_acx)
    fill_halo_regions!(ssa_mask_acy)
    fill_halo_regions!(mask_frnt)
    fill_halo_regions!(H_ice)
    fill_halo_regions!(f_ice)
    fill_halo_regions!(taud_acx)
    fill_halo_regions!(taud_acy)
    fill_halo_regions!(taul_int_acx)
    fill_halo_regions!(taul_int_acy)
    fill_halo_regions!(ux_b)
    fill_halo_regions!(uy_b)

    Ux  = interior(ux_b)
    Uy  = interior(uy_b)
    Bx  = interior(beta_acx)
    By  = interior(beta_acy)
    Naa = interior(visc_eff_int)
    Nab = interior(visc_ab)
    Mx  = interior(ssa_mask_acx)
    My  = interior(ssa_mask_acy)
    MF  = interior(mask_frnt)
    Hi  = interior(H_ice)
    Fi  = interior(f_ice)
    Tdx = interior(taud_acx)
    Tdy = interior(taud_acy)
    Tlx = interior(taul_int_acx)
    Tly = interior(taul_int_acy)

    Nx = size(Hi, 1)
    Ny = size(Hi, 2)

    return _assemble_ssa_matrix_kernel!(
        I_idx, J_idx, vals, b_vec, nnz_ref,
        Ux, Uy, Bx, By, Naa, Nab, Mx, My, MF, Hi, Fi, Tdx, Tdy, Tlx, Tly,
        Float64(dx), Float64(dy),
        Tx_top, Ty_top, Nx, Ny;
        boundaries = boundaries, lateral_bc = lateral_bc)
end

# Compute kernel — concrete-typed scalars, parametric topology, plain
# arrays. ParallelStencil-shape: all per-row work over flat arrays.
function _assemble_ssa_matrix_kernel!(I_idx::Vector{Int},
                                       J_idx::Vector{Int},
                                       vals::Vector{Float64},
                                       b_vec::Vector{Float64},
                                       nnz_ref::Ref{Int},
                                       Ux, Uy, Bx, By, Naa, Nab, Mx, My, MF,
                                       Hi, Fi, Tdx, Tdy, Tlx, Tly,
                                       dx::Float64, dy::Float64,
                                       ::Type{Tx_top}, ::Type{Ty_top},
                                       Nx::Int, Ny::Int;
                                       boundaries::Symbol=:zeros,
                                       lateral_bc::AbstractString="floating",
        ) where {Tx_top<:AbstractTopology, Ty_top<:AbstractTopology}

    bcs = _ssa_resolve_bcs(boundaries)

    # Fortran `inv_dx*N` factors (lines 221-227). We compute the
    # exact same combinations to keep the per-row arithmetic
    # bit-equivalent to the reference.
    inv_dx     = 1.0 / dx
    inv_dxdx   = 1.0 / (dx * dx)
    inv_dy     = 1.0 / dy
    inv_dydy   = 1.0 / (dy * dy)
    inv_dxdy   = 1.0 / (dx * dy)

    # COO write counter (mirrors Fortran `k`).
    k = 0

    # Helpers to read field values via Fortran cell indices (i, j).
    # Map to Yelmo.jl staggered storage:
    #   ux(i, j)        ↔ Ux[i+1, j, 1]                (slot wraps under Periodic-x)
    #   uy(i, j)        ↔ Uy[i, j+1, 1]                (slot wraps under Periodic-y)
    #   beta_acx(i, j)  ↔ Bx[i+1, j, 1]
    #   beta_acy(i, j)  ↔ By[i, j+1, 1]
    #   N_aa(i, j)      ↔ Naa[i, j, 1]                 (Center)
    #   N_ab(i, j)      ↔ Nab[i+1, j+1, 1]             (Face/Face/Center)
    #   ssa_mask_acx    ↔ Mx[i+1, j, 1]
    #   ssa_mask_acy    ↔ My[i, j+1, 1]
    #   mask_frnt(i, j) ↔ MF[i, j, 1]
    #   H_ice(i, j)     ↔ Hi[i, j, 1]
    #   f_ice(i, j)     ↔ Fi[i, j, 1]
    #   taud_acx(i, j)  ↔ Tdx[i+1, j, 1]
    #   taud_acy(i, j)  ↔ Tdy[i, j+1, 1]
    #   taul_int_acx    ↔ Tlx[i+1, j, 1]
    #   taul_int_acy    ↔ Tly[i, j+1, 1]
    #
    # The (im1, ip1, jm1, jp1) wrap below mirrors Fortran lines 249-257.
    # Independent of grid topology — Fortran's matrix assembly assumes
    # periodic wrap unconditionally and then applies the boundary-row
    # special case based on `bcs(1:4)`. We do the same.

    # Iterate over Fortran cell coordinates (1-based). The Fortran loop
    # is `do n = 1, lgs%nmax-1, 2` which maps `(n+1)/2 → cell index k`,
    # incrementing in (i, j) row-major (j outer, i inner). Match it.
    @inbounds for j in 1:Ny, i in 1:Nx
        # Periodic-wrap neighbour indices (Fortran 249-257). These are
        # used for the column-index arithmetic; the actual access is
        # not field-halo, so the wrap must be explicit here.
        im1 = i - 1
        if im1 == 0;     im1 = Nx;  end
        ip1 = i + 1
        if ip1 == Nx + 1; ip1 = 1;  end
        jm1 = j - 1
        if jm1 == 0;     jm1 = Ny;  end
        jp1 = j + 1
        if jp1 == Ny + 1; jp1 = 1;  end

        # ip1f / jp1f are the storage slots for "the +1 face" — match
        # the staggered-storage convention (slot i+1 under Bounded,
        # mod1(i+1, Nx) under Periodic). Used for ux/uy/beta/Mx/My/Tdx/Tdy/Tlx/Tly
        # reads at the (i, j) face position itself.
        ip1f_i = _ip1_modular(i, Nx, Tx_top)
        jp1f_j = _jp1_modular(j, Ny, Ty_top)

        # ===========================================================
        # ----- Equations for ux at cell (i, j) -----
        # Fortran lines 259-543.
        # ===========================================================
        nr = _row_ux(i, j, Nx)

        # Read the int-valued mask (Float64 storage; cast at site).
        ssa_mask_x = Int(Mx[ip1f_i, j, 1])

        if ssa_mask_x == 0
            # Fortran lines 265-273. Dirichlet u = 0.
            k += 1
            _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
            b_vec[nr] = 0.0

        elseif ssa_mask_x == -1
            # Fortran lines 275-285. Prescribed velocity.
            ux_now = Ux[ip1f_i, j, 1]
            k += 1
            _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
            b_vec[nr] = ux_now

        elseif i == 1 && bcs[3] !== :periodic
            # Fortran lines 287-314. Left boundary (i == 1).
            if bcs[3] === :free_slip
                # ux(1, j) - ux(2, j) = 0 → ux(1) = ux(2).
                nc1 = _row_ux(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_ux(ip1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else  # no-slip
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif i == Nx && bcs[1] !== :periodic
            # Fortran lines 316-343. Right boundary.
            if bcs[1] === :free_slip
                nc1 = _row_ux(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_ux(Nx - 1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif j == 1 && bcs[4] !== :periodic
            # Fortran lines 345-373. Lower boundary.
            if bcs[4] === :free_slip
                nc1 = _row_ux(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_ux(i, jp1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif j == Ny && bcs[2] !== :periodic
            # Fortran lines 375-403. Upper boundary.
            if bcs[2] === :free_slip
                nc1 = _row_ux(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_ux(i, Ny - 1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif ssa_mask_x == 3
            # Fortran lines 404-473. Lateral BC at calving front.
            if Fi[i, j, 1] == 1.0 && Fi[ip1, j, 1] < 1.0
                # === Case 1: ice-free to the right === (Fortran 407-439)
                N_aa_now = Naa[i, j, 1]

                nc = _row_ux(im1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -4.0 * inv_dx * N_aa_now)

                nc = _row_uy(i, jm1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -2.0 * inv_dy * N_aa_now)

                nc = _row_ux(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    4.0 * inv_dx * N_aa_now)

                nc = _row_uy(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    2.0 * inv_dy * N_aa_now)

                b_vec[nr] = Tlx[ip1f_i, j, 1]
            else
                # === Case 2: ice-free to the left === (Fortran 440-472)
                N_aa_now = Naa[ip1, j, 1]

                nc = _row_ux(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -4.0 * inv_dx * N_aa_now)

                nc = _row_uy(ip1, jm1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -2.0 * inv_dy * N_aa_now)

                nc = _row_ux(ip1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    4.0 * inv_dx * N_aa_now)

                nc = _row_uy(ip1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    2.0 * inv_dy * N_aa_now)

                b_vec[nr] = Tlx[ip1f_i, j, 1]
            end

        else
            # Fortran lines 475-540. === Inner SSA solution. ===
            beta_now = Bx[ip1f_i, j, 1]

            # Index helpers for `N_ab` reads. Fortran `N_ab(i, j)` ↔
            # `Nab[i+1, j+1, 1]`. With wrapped (im1, jm1) this becomes
            # `Nab[im1+1, j+1, 1]` etc. — but `im1+1` under wrap maps
            # to `i` (when i == 1, im1 = Nx → im1+1 = Nx+1 which is
            # out of bounds under Bounded-x; PR-A.2 only supports
            # Bounded-x so we use the standard slot. For safety we
            # use `_ab_idx` helpers below.
            # Under Bounded-x: i+1 ∈ 2..Nx+1, im1+1 ∈ 1..Nx, both in
            # bounds of the Nab array shape (Nx+1, Ny+1, 1).
            # Under Periodic-y: j+1 = jp1f_j (Ny if j == Ny → wraps),
            # jm1+1 = j (in bounds when j ≥ 1).
            #
            # For the inner-stencil reads at `(i, j)`, `(i, jm1)`,
            # use slot `i+1, j+1` (always in bounds under Bounded-x)
            # and slot `i+1, jp1f_j` for jp1 reads.
            # Under Bounded-x, slot `i+1` ∈ 2..Nx+1 is in bounds for
            # the (Nx+1, Ny+1)-shaped `Nab` interior. Under Periodic-y
            # the Y-Face dim has shape Ny, so slot `j+1` must wrap via
            # `_jp1_modular(j, Ny, Ty_top)`. Use the wrap helpers
            # uniformly.
            ip1f_x = _ip1_modular(i, Nx, Tx_top)
            jp1f_y = _jp1_modular(j, Ny, Ty_top)
            jm1_y  = _jp1_modular(jm1, Ny, Ty_top)   # = jm1 + 1 with wrap
            im1_x  = _ip1_modular(im1, Nx, Tx_top)   # = im1 + 1 with wrap

            Nab_ij   = Nab[ip1f_x, jp1f_y, 1]
            Nab_ijm1 = Nab[ip1f_x, jm1_y,  1]
            Nab_im1j = Nab[im1_x,  jp1f_y, 1]

            # -- vx terms (Fortran 481-508). --
            nc = _row_ux(i, j, Nx)
            v  = -4.0 * inv_dxdx * (Naa[ip1, j, 1] + Naa[i, j, 1]) -
                  1.0 * inv_dydy * (Nab_ij + Nab_ijm1) -
                  beta_now
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_ux(ip1, j, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                4.0 * inv_dxdx * Naa[ip1, j, 1])

            nc = _row_ux(im1, j, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                4.0 * inv_dxdx * Naa[i, j, 1])

            nc = _row_ux(i, jp1, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                1.0 * inv_dydy * Nab_ij)

            nc = _row_ux(i, jm1, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                1.0 * inv_dydy * Nab_ijm1)

            # -- vy terms (Fortran 510-534). --
            nc = _row_uy(i, j, Nx)
            v = -2.0 * inv_dxdy * Naa[i, j, 1] -
                 1.0 * inv_dxdy * Nab_ij
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_uy(ip1, j, Nx)
            v =  2.0 * inv_dxdy * Naa[ip1, j, 1] +
                 1.0 * inv_dxdy * Nab_ij
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_uy(ip1, jm1, Nx)
            v = -2.0 * inv_dxdy * Naa[ip1, j, 1] -
                 1.0 * inv_dxdy * Nab_ijm1
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_uy(i, jm1, Nx)
            v =  2.0 * inv_dxdy * Naa[i, j, 1] +
                 1.0 * inv_dxdy * Nab_ijm1
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            b_vec[nr] = Tdx[ip1f_i, j, 1]
        end

        # ===========================================================
        # ----- Equations for uy at cell (i, j) -----
        # Fortran lines 546-824.
        # ===========================================================
        nr = _row_uy(i, j, Nx)
        ssa_mask_y = Int(My[i, jp1f_j, 1])

        if ssa_mask_y == 0
            k += 1
            _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
            b_vec[nr] = 0.0

        elseif ssa_mask_y == -1
            uy_now = Uy[i, jp1f_j, 1]
            k += 1
            _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
            b_vec[nr] = uy_now

        elseif j == 1 && bcs[4] !== :periodic
            # Fortran lines 571-598.
            if bcs[4] === :free_slip
                nc1 = _row_uy(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_uy(i, jp1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif j == Ny && bcs[2] !== :periodic
            # Fortran lines 600-627.
            if bcs[2] === :free_slip
                nc1 = _row_uy(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_uy(i, Ny - 1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif i == 1 && bcs[3] !== :periodic
            # Fortran lines 629-656.
            if bcs[3] === :free_slip
                nc1 = _row_uy(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_uy(ip1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif i == Nx && bcs[1] !== :periodic
            # Fortran lines 658-685.
            if bcs[1] === :free_slip
                nc1 = _row_uy(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc1, 1.0)
                nc2 = _row_uy(Nx - 1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc2, -1.0)
                b_vec[nr] = 0.0
            else
                k += 1
                _push_coo!(I_idx, J_idx, vals, k, nr, nr, 1.0)
                b_vec[nr] = 0.0
            end

        elseif ssa_mask_y == 3
            # Fortran lines 687-756.
            if Fi[i, j, 1] == 1.0 && Fi[i, jp1, 1] < 1.0
                # Case 1: ice-free to the top (Fortran 690-722).
                N_aa_now = Naa[i, j, 1]

                nc = _row_ux(im1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -2.0 * inv_dx * N_aa_now)

                nc = _row_uy(i, jm1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -4.0 * inv_dy * N_aa_now)

                nc = _row_ux(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    2.0 * inv_dx * N_aa_now)

                nc = _row_uy(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    4.0 * inv_dy * N_aa_now)

                b_vec[nr] = Tly[i, jp1f_j, 1]
            else
                # Case 2: ice-free to the bottom (Fortran 723-755).
                N_aa_now = Naa[i, jp1, 1]

                nc = _row_ux(im1, jp1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -2.0 * inv_dx * N_aa_now)

                nc = _row_uy(i, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                   -4.0 * inv_dy * N_aa_now)

                nc = _row_ux(i, jp1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    2.0 * inv_dx * N_aa_now)

                nc = _row_uy(i, jp1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                    4.0 * inv_dy * N_aa_now)

                b_vec[nr] = Tly[i, jp1f_j, 1]
            end

        else
            # Fortran lines 758-822. === Inner SSA solution (uy). ===
            beta_now = By[i, jp1f_j, 1]

            ip1f_x = _ip1_modular(i, Nx, Tx_top)
            jp1f_y = _jp1_modular(j, Ny, Ty_top)
            im1_x  = _ip1_modular(im1, Nx, Tx_top)

            Nab_ij   = Nab[ip1f_x, jp1f_y, 1]
            Nab_im1j = Nab[im1_x,  jp1f_y, 1]

            # -- vy terms (Fortran 764-791). --
            nc = _row_uy(i, j, Nx)
            v = -4.0 * inv_dydy * (Naa[i, jp1, 1] + Naa[i, j, 1]) -
                 1.0 * inv_dxdx * (Nab_ij + Nab_im1j) -
                 beta_now
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_uy(i, jp1, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                4.0 * inv_dydy * Naa[i, jp1, 1])

            nc = _row_uy(i, jm1, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                4.0 * inv_dydy * Naa[i, j, 1])

            nc = _row_uy(ip1, j, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                1.0 * inv_dxdx * Nab_ij)

            nc = _row_uy(im1, j, Nx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc,
                                1.0 * inv_dxdx * Nab_im1j)

            # -- vx terms (Fortran 793-817). --
            nc = _row_ux(i, j, Nx)
            v = -2.0 * inv_dxdy * Naa[i, j, 1] -
                 1.0 * inv_dxdy * Nab_ij
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_ux(i, jp1, Nx)
            v =  2.0 * inv_dxdy * Naa[i, jp1, 1] +
                 1.0 * inv_dxdy * Nab_ij
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_ux(im1, jp1, Nx)
            v = -2.0 * inv_dxdy * Naa[i, jp1, 1] -
                 1.0 * inv_dxdy * Nab_im1j
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            nc = _row_ux(im1, j, Nx)
            v =  2.0 * inv_dxdy * Naa[i, j, 1] +
                 1.0 * inv_dxdy * Nab_im1j
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc, v)

            b_vec[nr] = Tdy[i, jp1f_j, 1]
        end
    end

    nnz_ref[] = k
    return nothing
end

# ----------------------------------------------------------------------
# Linear-solve wrapper (PR-B commit 2; preconditioner refactor).
#
# Runs a Krylov solver (currently BiCGStab) over the assembled SSA
# system `A x = b`. `A` is the COO-built `SparseMatrixCSC` and `b` is
# the assembled RHS.
#
# Preconditioner is selected by `SSASolver.precond`:
#   :none    → no preconditioner (pure Krylov).
#   :jacobi  → diagonal scaling, `M = Diagonal(1 ./ diag(A))`. DEFAULT.
#              Matches Fortran Yelmo's SLAB-S06 namelist
#              `ssa_lis_opt = "-i bicgsafe -p jacobi …"`.
#   :amg_sa  → AlgebraicMultigrid.smoothed_aggregation(A). Only
#              appropriate for SPD-like systems — the standard SSA
#              matrix is non-symmetric with negative diagonals and
#              BiCGStab+SA diverges on it. Kept opt-in for future
#              SPD reformulations.
#   :amg_rs  → AlgebraicMultigrid.ruge_stuben(A). Alternative AMG.
#
# AlgebraicMultigrid's `aspreconditioner` returns a `Preconditioner`
# object whose `ldiv!` overload approximates `A^{-1} y`. We pass
# `ldiv=true` to Krylov so it dispatches to `ldiv!` rather than `mul!`.
# For Jacobi (`Diagonal{Float64}`), `ldiv=false` (default) is correct:
# Krylov calls `mul!(y, M, x)` which evaluates `D * x` = the inverse
# diagonal scaling.
#
# Soft-warn on non-convergence: matches Fortran's Lis behavior of
# logging non-convergence and proceeding (the outer Picard iteration
# is the safety net).
# ----------------------------------------------------------------------

# AMG smoother helper — only consulted when precond ∈ (:amg_sa, :amg_rs).
# AlgebraicMultigrid.jl's `Jacobi` smoother needs a per-level workspace
# at construction time that the SA / RS drivers do not currently provide
# through a closure-style hook, so we soft-fail on `:jacobi` for AMG and
# steer callers to `:gauss_seidel`.
function _amg_smoother(sym::Symbol)
    if sym === :gauss_seidel
        return GaussSeidel()
    elseif sym === :jacobi
        error("_amg_smoother: :jacobi smoother not yet supported for AMG. " *
              "AlgebraicMultigrid.jl's Jacobi smoother needs a per-level " *
              "workspace at construction time which the current wrapper " *
              "doesn't provide. Use :gauss_seidel for AMG, or set " *
              "SSASolver(precond = :jacobi) for diagonal-scaling " *
              "preconditioning of the Krylov solver itself.")
    else
        error("_amg_smoother: smoother=$(sym) not supported. " *
              "Choose :gauss_seidel (default).")
    end
end

# Build the preconditioner per `ssa.precond`. Returns `(M, ldiv_flag)`
# where `M` is the preconditioner (or `nothing` for `:none`) and
# `ldiv_flag` tells Krylov whether to dispatch via `ldiv!` (true, for
# AMG) or `mul!` (false, for Jacobi). For `:none`, `M === nothing`
# and the caller skips the `M` kwarg entirely.
function _build_ssa_precond(scratch,
                            A::SparseMatrixCSC{Float64,Int},
                            ssa::SSASolver)
    if ssa.precond === :none
        scratch.ssa_amg_cache[] = nothing
        return nothing, false
    elseif ssa.precond === :jacobi
        # Refresh the cached `d_inv` buffer in-place (was a fresh
        # `Vector{Float64}` allocation each Picard iter before).
        d_inv = scratch.ssa_jacobi_d_inv
        N_rows = size(A, 1)
        length(d_inv) == N_rows || error(
            "_build_ssa_precond: scratch.ssa_jacobi_d_inv has length " *
            "$(length(d_inv)), expected $(N_rows).")
        @inbounds for k in 1:N_rows
            d_inv[k] = 1.0 / A[k, k]
        end
        any(!isfinite, d_inv) && error("_build_ssa_precond: Jacobi " *
              "preconditioner has non-finite diagonal entries (singular " *
              "row?). Check the SSA mask handling and matrix assembly.")
        scratch.ssa_amg_cache[] = nothing
        return Diagonal(d_inv), false
    elseif ssa.precond === :amg_sa
        smoother = _amg_smoother(ssa.smoother)
        ml = smoothed_aggregation(A; presmoother = smoother,
                                      postsmoother = smoother)
        scratch.ssa_amg_cache[] = ml  # keep alive for diagnostics
        return aspreconditioner(ml), true
    elseif ssa.precond === :amg_rs
        smoother = _amg_smoother(ssa.smoother)
        ml = ruge_stuben(A; presmoother = smoother,
                            postsmoother = smoother)
        scratch.ssa_amg_cache[] = ml
        return aspreconditioner(ml), true
    else
        error("_build_ssa_precond: precond=$(ssa.precond) not recognized. " *
              "Expected :none, :jacobi, :amg_sa, or :amg_rs.")
    end
end

"""
    _solve_ssa_linear!(x_dest, scratch, A::SparseMatrixCSC{Float64,Int},
                       b::Vector{Float64}, ssa::SSASolver) -> x_dest

Solve `A x = b` for the SSA linear system using the configured Krylov
method + preconditioner (per `ssa.precond`). Writes the solution into
the caller-supplied `x_dest::Vector{Float64}` (must have `length ==
length(b)`) and returns the same vector. Uses the Krylov workspace in
`scratch.ssa_solver_workspace` and the cached Jacobi `d_inv` buffer in
`scratch.ssa_jacobi_d_inv` so per-call allocation is minimal.

Soft-warns (does not error) on non-convergence — Fortran-faithful;
the outer Picard loop is the safety net for poor inner convergence.
"""
function _solve_ssa_linear!(x_dest::Vector{Float64},
                            scratch,
                            A::SparseMatrixCSC{Float64,Int},
                            b::Vector{Float64},
                            ssa::SSASolver)
    length(x_dest) == length(b) || error(
        "_solve_ssa_linear!: x_dest length $(length(x_dest)) ≠ b length $(length(b))")

    # Build (or rebuild) the preconditioner. The matrix coefficients
    # change every Picard iteration so we cannot cache across calls;
    # for `:jacobi` the d_inv vector is refreshed in-place into
    # `scratch.ssa_jacobi_d_inv`.
    M, ldiv_flag = _build_ssa_precond(scratch, A, ssa)

    # Resolve `linear_method = :auto` against the chosen assembly
    # `method` (`:residual` → `:bicgstab`, `:energy_quadratic` → `:cg`).
    linmeth = resolve_linear_method(ssa)
    if linmeth === :bicgstab
        workspace = scratch.ssa_solver_workspace
        if M === nothing
            bicgstab!(workspace, A, b;
                      rtol = ssa.rtol, itmax = ssa.itmax,
                      history = false)
        else
            bicgstab!(workspace, A, b;
                      M = M, ldiv = ldiv_flag,
                      rtol = ssa.rtol, itmax = ssa.itmax,
                      history = false)
        end
        # Copy the workspace solution into the caller-supplied buffer
        # (in-place — was `copy(workspace.x)` returning a fresh Vector).
        copyto!(x_dest, workspace.x)
        _count_ssa_lin_solve!(scratch, workspace.stats)
        if !workspace.stats.solved
            res = norm(A * x_dest .- b)
            @warn "SSA BiCGStab did not converge" precond=ssa.precond niter=workspace.stats.niter residual=res rtol=ssa.rtol itmax=ssa.itmax
        end
        return x_dest
    elseif linmeth === :cg
        # CG requires SPD `A`. Only safe for `method = :energy_quadratic`
        # (Hessian of the discrete viscous-energy functional). The
        # `:residual` formulation produces a non-symmetric system and
        # must use `:bicgstab` — `resolve_linear_method` enforces this
        # for `linear_method = :auto`; an explicit override here is the
        # caller's responsibility.
        workspace = scratch.ssa_cg_workspace
        if M === nothing
            cg!(workspace, A, b;
                rtol = ssa.rtol, itmax = ssa.itmax,
                history = false)
        else
            cg!(workspace, A, b;
                M = M, ldiv = ldiv_flag,
                rtol = ssa.rtol, itmax = ssa.itmax,
                history = false)
        end
        copyto!(x_dest, workspace.x)
        _count_ssa_lin_solve!(scratch, workspace.stats)
        if !workspace.stats.solved
            res = norm(A * x_dest .- b)
            @warn "SSA CG did not converge" precond=ssa.precond niter=workspace.stats.niter residual=res rtol=ssa.rtol itmax=ssa.itmax
        end
        return x_dest
    else
        error("_solve_ssa_linear!: linear_method=$(linmeth) not yet implemented. " *
              "Currently only :bicgstab and :cg are supported.")
    end
end

# Linear solver iterations and failures (breakdown or iteration limit) of
# a velocity solve, summed over its Picard iterations (Fortran
# `ssa_lin_iter`, `ssa_lin_fail`; reset by `_reset_ssa_lin_counts!`).
function _count_ssa_lin_solve!(scratch, stats)
    scratch.ssa_lin_iter[] += stats.niter
    stats.solved || (scratch.ssa_lin_fail[] += 1)
    return nothing
end

function _reset_ssa_lin_counts!(scratch)
    scratch.ssa_lin_iter[] = 0
    scratch.ssa_lin_fail[] = 0
    return nothing
end

"""
    count_vel_lim_faces(ux, uy, ssa_mask_acx, ssa_mask_acy, u_max) -> Int

Number of free ssa faces (`ssa_mask ≥ 1`) where the velocity limit acts:
a component at the clip value `u_max` (Fortran `count_vel_lim_faces`,
`ssa_vel_lim_method = "clip"`; the "drag" limit is not ported).
"""
function count_vel_lim_faces(ux, uy, ssa_mask_acx, ssa_mask_acy, u_max::Float64)
    Ux, Uy = interior(ux), interior(uy)
    Mx, My = interior(ssa_mask_acx), interior(ssa_mask_acy)
    Nx, Ny = size(ux.grid, 1), size(ux.grid, 2)
    Tx, Ty = topology(ux.grid, 1), topology(uy.grid, 2)
    n_lim = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        ie = _ip1_modular(i, Nx, Tx)     # east face of cell i
        jn = _jp1_modular(j, Ny, Ty)     # north face of cell j
        (Mx[ie, j, 1] >= 1 && abs(Ux[ie, j, 1]) >= u_max) && (n_lim += 1)
        (My[i, jn, 1] >= 1 && abs(Uy[i, jn, 1]) >= u_max) && (n_lim += 1)
    end
    return n_lim
end

# ----------------------------------------------------------------------
# Picard helpers (PR-B commit 3).
#
# Faithful ports of the helpers from
# /Users/alrobi001/models/yelmo/src/physics/velocity_general.f90:
#
#   - picard_relax_visc      (line 2107)
#   - picard_relax_vel       (line 2088)
#   - picard_calc_convergence_l2 (line 1917, norm_method=1)
#   - picard_calc_convergence_l1rel_matrix (line 1883)
#   - set_inactive_margins   (line 1675)
#
# These are small standalone functions used by the SSA driver
# (`calc_velocity_ssa!`) — they do not touch any Yelmo-specific state.
# ----------------------------------------------------------------------

const _SSA_PICARD_DU_REG  = 1e-10        # divide-by-zero floor (Fortran 1957)
const _SSA_PICARD_VEL_TOL = 1e-5         # m/yr, vel mask threshold (Fortran 1958)
const _SSA_PICARD_TOL_UF  = 1e-15        # underflow drop (Fortran TOL_UNDERFLOW)

"""
    picard_relax_visc!(visc_eff, visc_eff_nm1, rel) -> visc_eff

Apply Picard relaxation to the 3D viscosity field in **log space**
(matches Fortran `picard_relax_visc`, line 2117 — `visc =
exp((1-rel)·log(visc_prev) + rel·log(visc))`). Operates in-place on
`visc_eff`. The log-space mixing keeps positivity strictly: a
linearly-relaxed viscosity could underflow to 0 or below if the
adjacent step rejected wildly, which destabilizes the next iteration.

Both fields must be `Field`-like with `interior` views of identical
shape. Mirrors `velocity_general.f90:2107 picard_relax_visc`.
"""
function picard_relax_visc!(visc_eff, visc_eff_nm1, rel::Real)
    V    = interior(visc_eff)
    Vnm1 = interior(visc_eff_nm1)
    size(V) == size(Vnm1) || error(
        "picard_relax_visc!: shape mismatch ($(size(V)) vs $(size(Vnm1))).")
    _picard_relax_visc_kernel!(V, Vnm1, Float64(rel))
    return visc_eff
end

# Compute kernel — explicit (i, j, k) iteration with concrete `Float64`
# rel. The original `eachindex(V)` over a SubArray with non-trivial
# strides (Oceananigans' `interior(...)` stride tuple) yields
# `CartesianIndex{3}` per iteration, allocating one heap object each
# time; the explicit loop avoids that.
function _picard_relax_visc_kernel!(V, Vnm1, rel::Float64)
    rm = 1.0 - rel
    Nx = size(V, 1); Ny = size(V, 2); Nz = size(V, 3)
    # Each (i, j, k) writes to its own slot — no shared writes.
    # `@threads` over the outermost (Nz) axis: viscosity arrays are
    # vertically-3D so `Nz` is typically 11–41, comparable to the
    # number of available cores. Inner (j, i) stay sequential per
    # thread so the hot loop body keeps tight SIMD.
    @threads for k in 1:Nz
        @inbounds for j in 1:Ny, i in 1:Nx
            v_prev = Vnm1[i, j, k]
            v_now  = V[i, j, k]
            # Guard against log(0): if either is zero, fall back to
            # linear mixing (preserves the Fortran behavior where
            # visc_min floor ensures positive values, but be defensive
            # in case the caller passed an unfilled array on the
            # first iteration).
            if v_prev > 0.0 && v_now > 0.0
                V[i, j, k] = exp(rm * log(v_prev) + rel * log(v_now))
            else
                V[i, j, k] = rm * v_prev + rel * v_now
            end
        end
    end
    return nothing
end

"""
    picard_relax_vel!(ux_n, uy_n, ux_nm1, uy_nm1, rel) -> (ux_n, uy_n)

Apply linear Picard relaxation to the face-staggered velocity fields:

    ux_n .= ux_nm1 + rel · (ux_n - ux_nm1)
    uy_n .= uy_nm1 + rel · (uy_n - uy_nm1)

In-place on `ux_n` and `uy_n`. Mirrors `velocity_general.f90:2088
picard_relax_vel` (elemental subroutine).
"""
function picard_relax_vel!(ux_n, uy_n, ux_nm1, uy_nm1, rel::Real)
    Ux    = interior(ux_n)
    Uy    = interior(uy_n)
    Uxnm1 = interior(ux_nm1)
    Uynm1 = interior(uy_nm1)
    _picard_relax_vel_kernel!(Ux, Uy, Uxnm1, Uynm1, Float64(rel))
    return ux_n, uy_n
end

# Compute kernel — same eachindex-vs-CartesianIndex fix as
# `picard_relax_visc!` above. Explicit (i, j, k) iteration avoids the
# heap-allocated `CartesianIndex{3}` per loop step on Oceananigans
# `interior(...)` SubArrays.
function _picard_relax_vel_kernel!(Ux, Uy, Uxnm1, Uynm1, rel::Float64)
    # Velocity arrays are 2D-flavored (Nz=1) so thread on the
    # row axis `j`, not on `k`.
    Nxx, Nxy, Nxz = size(Ux, 1), size(Ux, 2), size(Ux, 3)
    @threads for j in 1:Nxy
        @inbounds for k in 1:Nxz, i in 1:Nxx
            Ux[i, j, k] = Uxnm1[i, j, k] + rel * (Ux[i, j, k] - Uxnm1[i, j, k])
        end
    end
    Nyx, Nyy, Nyz = size(Uy, 1), size(Uy, 2), size(Uy, 3)
    @threads for j in 1:Nyy
        @inbounds for k in 1:Nyz, i in 1:Nyx
            Uy[i, j, k] = Uynm1[i, j, k] + rel * (Uy[i, j, k] - Uynm1[i, j, k])
        end
    end
    return nothing
end

"""
    picard_calc_convergence_l2(ux, ux_nm1, uy, uy_nm1, mask_acx, mask_acy) -> Float64

Relative L2 change of the face velocity between two Picard iterations,
`sqrt(Σ(u − u_prev)²) / (sqrt(Σ u_prev²) + 1e-10)` over the faces whose
momentum equation is solved (`mask > 0`) and with `|u| > 1e-5` m/yr;
0 if there are none (Fortran `picard_calc_convergence_l2`,
`norm_method = 1`, velocity_general.f90:2120). Errors if the velocity of
a solved face or the residual is not finite (Fortran stops there too).

`ux`, `uy` are the XFace / YFace velocity fields, `mask_acx`, `mask_acy`
the SSA masks; each face is visited once (Fortran face `i` = east face
of cell `i`).
"""
function picard_calc_convergence_l2(ux, ux_nm1, uy, uy_nm1, mask_acx, mask_acy)
    Ux, Uxp, Mx = interior(ux), interior(ux_nm1), interior(mask_acx)
    Uy, Uyp, My = interior(uy), interior(uy_nm1), interior(mask_acy)
    Nx, Ny = size(ux.grid, 1), size(ux.grid, 2)
    Tx, Ty = topology(ux.grid, 1), topology(uy.grid, 2)
    res1 = 0.0
    res2 = 0.0
    n_check = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        ie = _ip1_modular(i, Nx, Tx)     # east face of cell i
        jn = _jp1_modular(j, Ny, Ty)     # north face of cell j
        for (u, up, m) in ((Ux[ie, j, 1], Uxp[ie, j, 1], Mx[ie, j, 1]),
                           (Uy[i, jn, 1], Uyp[i, jn, 1], My[i, jn, 1]))
            (abs(u) > _SSA_PICARD_VEL_TOL && m > 0) || continue
            n_check += 1
            d = u - up
            abs(d) < _SSA_PICARD_TOL_UF && (d = 0.0)
            abs(up) < _SSA_PICARD_TOL_UF && (up = 0.0)
            res1 += d * d
            res2 += up * up
        end
    end
    resid = n_check > 0 ? sqrt(res1) / (sqrt(res2) + _SSA_PICARD_DU_REG) : 0.0
    _check_finite_velocity(resid, ux, uy, mask_acx, mask_acy)
    return resid
end

# Stop if the solution is not finite: NaN/Inf on a solved face (mask > 0)
# or in the residual (Fortran velocity_general.f90:2272-2304).
function _check_finite_velocity(resid::Float64, ux, uy, mask_acx, mask_acy)
    Ux, Mx = interior(ux), interior(mask_acx)
    Uy, My = interior(uy), interior(mask_acy)
    bad_x = findfirst(k -> Mx[k] > 0 && !isfinite(Ux[k]), CartesianIndices(Ux))
    bad_y = findfirst(k -> My[k] > 0 && !isfinite(Uy[k]), CartesianIndices(Uy))
    (isfinite(resid) && bad_x === nothing && bad_y === nothing) && return nothing
    error("SSA Picard: velocity solution is not finite (residual = $(resid); " *
          "first non-finite x-face: $(bad_x === nothing ? "none" : Tuple(bad_x)), " *
          "y-face: $(bad_y === nothing ? "none" : Tuple(bad_y))).")
end

# Clip each velocity component to [-u_max, u_max] (Fortran `ssa_vel_clip`,
# `ssa_vel_lim_method = "clip"`). A non-finite value is left as is for
# `_check_finite_velocity`.
function ssa_vel_clip!(ux, uy, u_max::Float64)
    _clip_kernel!(interior(ux), u_max)
    _clip_kernel!(interior(uy), u_max)
    return ux, uy
end

function _clip_kernel!(U::AbstractArray{Float64,3}, u_max::Float64)
    @inbounds for k in axes(U, 3), j in axes(U, 2), i in axes(U, 1)
        U[i, j, k] = clamp(U[i, j, k], -u_max, u_max)
    end
    return U
end

"""
    picard_calc_convergence_l1rel_matrix!(err_x, err_y,
                                          ux, uy, ux_nm1, uy_nm1)
        -> (err_x, err_y)

Per-cell L1 relative error matrix (Fortran
`picard_calc_convergence_l1rel_matrix`, line 1883):

    err_x = 2·|ux - ux_prev| / |ux + ux_prev + tol|   (where |ux| > vel_tol)
          = 0                                          (otherwise)
    err_y = analogous

Used as a diagnostic per-face residual field. In-place on `err_x` /
`err_y` (interior arrays).
"""
function picard_calc_convergence_l1rel_matrix!(err_x::AbstractArray,
                                               err_y::AbstractArray,
                                               ux::AbstractArray,
                                               uy::AbstractArray,
                                               ux_nm1::AbstractArray,
                                               uy_nm1::AbstractArray)
    tol = 1e-5
    vel_tol = 1e-2   # Fortran ssa_vel_tolerance (line 1896)
    Nxx, Nxy, Nxz = size(err_x, 1), size(err_x, 2), size(err_x, 3)
    @inbounds for k in 1:Nxz, j in 1:Nxy, i in 1:Nxx
        u    = ux[i, j, k]
        unm1 = ux_nm1[i, j, k]
        if abs(u) > vel_tol
            err_x[i, j, k] = 2.0 * abs(u - unm1) / abs(u + unm1 + tol)
        else
            err_x[i, j, k] = 0.0
        end
    end
    Nyx, Nyy, Nyz = size(err_y, 1), size(err_y, 2), size(err_y, 3)
    @inbounds for k in 1:Nyz, j in 1:Nyy, i in 1:Nyx
        u    = uy[i, j, k]
        unm1 = uy_nm1[i, j, k]
        if abs(u) > vel_tol
            err_y[i, j, k] = 2.0 * abs(u - unm1) / abs(u + unm1 + tol)
        else
            err_y[i, j, k] = 0.0
        end
    end
    return err_x, err_y
end

"""
    set_inactive_margins!(ux_b, uy_b, f_ice) -> (ux_b, uy_b)

Zero out velocity at faces touching ice-free cells (matches Fortran
`set_inactive_margins`, velocity_general.f90:1675). Specifically:

  - For face-x at `(i, j)` (between cells `(i, j)` and `(i+1, j)`):
    if `f_ice(i, j) < 1` AND `f_ice(i+1, j) == 0`, set `ux(i, j) = 0`.
    The other direction is symmetric.
  - For face-y at `(i, j)`: analogous with `(i, j+1)`.

Operates on `interior` views. `ux_b` is XFace, `uy_b` is YFace, `f_ice`
is Center; standard Yelmo.jl staggering convention.
"""
function set_inactive_margins!(ux_b, uy_b, f_ice)
    Ux = interior(ux_b)
    Uy = interior(uy_b)
    Fi = interior(f_ice)

    Nx = size(Fi, 1)
    Ny = size(Fi, 2)
    Tx_top = topology(ux_b.grid, 1)
    Ty_top = topology(uy_b.grid, 2)

    # Each cell (i, j) writes only to its own +1 face slot,
    # `Ux[ip1f, j]` and `Uy[i, jp1f]`. Different cells map to
    # different slots, so threading on `j` is race-free.
    @threads for j in 1:Ny
        @inbounds for i in 1:Nx
            ip1 = i == Nx ? (Tx_top === Periodic ? 1 : Nx) : i + 1
            jp1 = j == Ny ? (Ty_top === Periodic ? 1 : Ny) : j + 1
            ip1f = _ip1_modular(i, Nx, Tx_top)
            jp1f = _jp1_modular(j, Ny, Ty_top)

            # x-face between (i, j) and (ip1, j): at slot [ip1f, j].
            if (Fi[i, j, 1] < 1.0 && Fi[ip1, j, 1] == 0.0) ||
               (Fi[i, j, 1] == 0.0 && Fi[ip1, j, 1] < 1.0)
                Ux[ip1f, j, 1] = 0.0
            end
            # y-face between (i, j) and (i, jp1): at slot [i, jp1f].
            if (Fi[i, j, 1] < 1.0 && Fi[i, jp1, 1] == 0.0) ||
               (Fi[i, j, 1] == 0.0 && Fi[i, jp1, 1] < 1.0)
                Uy[i, jp1f, 1] = 0.0
            end
        end
    end
    return ux_b, uy_b
end

"""
    calc_basal_stress!(taub_acx, taub_acy, beta_acx, beta_acy, ux_b, uy_b)
        -> (taub_acx, taub_acy)

Diagnose basal stress as `tau_b = beta · u_b` on each face. Underflow
clip below `1e-5` Pa (matches Fortran `calc_basal_stress`, velocity_ssa.f90:679).
"""
function calc_basal_stress!(taub_acx, taub_acy, beta_acx, beta_acy, ux_b, uy_b)
    Tx = interior(taub_acx)
    Ty = interior(taub_acy)
    Bx = interior(beta_acx)
    By = interior(beta_acy)
    Ux = interior(ux_b)
    Uy = interior(uy_b)
    tol = 1e-5
    # Pure per-cell: write to `Tx[i, j, k]` only. Thread on `j`
    # (the row axis is typically large enough; the velocity arrays
    # are 2D so threading on `k` would give Nz=1 task).
    Nxx, Nxy, Nxz = size(Tx, 1), size(Tx, 2), size(Tx, 3)
    @threads for j in 1:Nxy
        @inbounds for k in 1:Nxz, i in 1:Nxx
            v = Bx[i, j, k] * Ux[i, j, k]
            Tx[i, j, k] = abs(v) < tol ? 0.0 : v
        end
    end
    Nyx, Nyy, Nyz = size(Ty, 1), size(Ty, 2), size(Ty, 3)
    @threads for j in 1:Nyy
        @inbounds for k in 1:Nyz, i in 1:Nyx
            v = By[i, j, k] * Uy[i, j, k]
            Ty[i, j, k] = abs(v) < tol ? 0.0 : v
        end
    end
    return taub_acx, taub_acy
end

# ----------------------------------------------------------------------
# calc_velocity_ssa! — SSA Picard driver (PR-B commit 4)
#
# Faithful port of /Users/alrobi001/models/yelmo/src/physics/
# velocity_ssa.f90:60-335 calc_velocity_ssa.
#
# Outer Picard iteration:
#
#   1. Snapshot previous (visc_eff, ux_b, uy_b) for relaxation +
#      convergence check.
#   2. Update 3D effective viscosity from current ux_b/uy_b. Two
#      paths per `visc_method`:
#        - visc_method == 0: constant viscosity (uses ydyn.visc_const).
#        - visc_method == 1: gauss-quadrature node-stencil (calc_visc_eff_3D_nodes).
#        - visc_method == 2: aa-only stencil (calc_visc_eff_3D_aa).
#   3. Apply log-space Picard relaxation to viscosity.
#   4. Update depth-integrated viscosity visc_eff_int.
#   5. Update beta (basal drag) from current ux_b/uy_b + c_bed.
#   6. Stagger beta to face-staggered beta_acx/beta_acy; the matrix and
#      taub use a copy with beta_min at grounded faces (set_beta_min_grounded!).
#   7. Stagger viscosity to ab-corner (visc_ab cache).
#   8. Assemble SSA matrix → COO triplets + RHS in dyn.scratch.
#   9. Build SparseMatrixCSC and run BiCGStab+AMG.
#  10. Unpack solution back into ux_b, uy_b face slots.
#  11. Clip to ssa_vel_max; linear Picard velocity relaxation (also on
#      the first iteration, towards the previous call's solution).
#  12. Compute L2 residual; record in scratch; check convergence.
#
# Inactive margins are closed once, before the loop (Fortran).
#
# After loop: compute basal stress tau_b = beta · u_b.
#
# The Fortran driver also has an adaptive corr_theta block that's
# disabled (`if (.FALSE.)`); we mirror that omission and use a
# constant relaxation parameter `p_ydyn.ssa_iter_rel`.
# ----------------------------------------------------------------------

"""
    calc_velocity_ssa!(y::YelmoModel) -> y

Run the SSA Picard iteration on `y`. Updates `y.dyn.ux_b`, `y.dyn.uy_b`,
`y.dyn.taub_acx`, `y.dyn.taub_acy`, `y.dyn.visc_eff`,
`y.dyn.visc_eff_int`, `y.dyn.beta`, `y.dyn.beta_acx`, `y.dyn.beta_acy`,
plus diagnostic scratch fields (`scratch.ssa_residuals`,
`scratch.ssa_iter_now`, `scratch.ssa_picard_*_nm1`).

Reads the linear-solver knobs from `y.p.ydyn.ssa_solver` (`SSASolver`)
and the Picard settings from `ydyn.ssa_iter_*`. Other inputs come from the prior pre-solver kinematic
calls in `dyn_step!` (driving stress, lateral BC stress, masks, c_bed).

Mirrors `velocity_ssa.f90:60 calc_velocity_ssa`. Uses the
`(Bounded, Bounded, Flat)` and `(Bounded, Periodic, Flat)` topologies
supported by `_assemble_ssa_matrix!` (PR-A.2).
"""
function calc_velocity_ssa!(y)
    p_ydyn = y.p.ydyn
    p_ymat = y.p.ymat
    ssa    = p_ydyn.ssa_solver

    Nx = size(y.g, 1)
    Ny = size(y.g, 2)
    Nz = size(interior(y.dyn.visc_eff), 3)

    dx_g = y.g.Δxᶜᵃᵃ
    dy_g = y.g.Δyᵃᶜᵃ
    dx = abs(Float64(dx_g isa Number ? dx_g : error("calc_velocity_ssa!: stretched x-grid not supported.")))
    dy = abs(Float64(dy_g isa Number ? dy_g : error("calc_velocity_ssa!: stretched y-grid not supported.")))

    sc = y.dyn.scratch

    # 1. Compute SSA masks for this dyn step.
    set_ssa_masks!(y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy,
                   y.tpo.mask_frnt, y.tpo.f_ice_dyn, y.tpo.f_grnd;
                   lateral_bc = p_ydyn.ssa_lat_bc)

    # 2. Snapshot initial state for the post-iter convergence check.
    interior(sc.ssa_picard_ux_b_nm1) .= interior(y.dyn.ux_b)
    interior(sc.ssa_picard_uy_b_nm1) .= interior(y.dyn.uy_b)
    interior(sc.ssa_picard_visc_eff_nm1) .= interior(y.dyn.visc_eff)

    # Reset error-diagnostic scratches (Fortran lines 152-153).
    fill!(interior(y.dyn.ssa_err_acx), 1.0)
    fill!(interior(y.dyn.ssa_err_acy), 1.0)

    # Vertical zeta_aa for visc_eff (Center-staggered). Used by
    # calc_visc_eff_3D_*.
    zeta_c = znodes(y.gt, Center())

    # Pre-step margin setting (Fortran line 160).
    set_inactive_margins!(y.dyn.ux_b, y.dyn.uy_b, y.tpo.f_ice_dyn)

    converged = false
    iter_now = 0
    n_resid_max = length(sc.ssa_residuals)
    _reset_ssa_lin_counts!(sc)

    for iter in 1:p_ydyn.ssa_iter_max
        iter_now = iter

        # Snapshot n minus 1 state before computing new viscosity / vel.
        interior(sc.ssa_picard_visc_eff_nm1) .= interior(y.dyn.visc_eff)
        interior(sc.ssa_picard_ux_b_nm1)     .= interior(y.dyn.ux_b)
        interior(sc.ssa_picard_uy_b_nm1)     .= interior(y.dyn.uy_b)

        # ---- Step 1: viscosity update (Fortran lines 182-209). ----
        if p_ydyn.visc_method == 0
            fill!(interior(y.dyn.visc_eff), Float64(p_ydyn.visc_const))
        elseif p_ydyn.visc_method == 1
            calc_visc_eff_3D_nodes!(y.dyn.visc_eff, y.dyn.ux_b, y.dyn.uy_b,
                                    y.mat.ATT, y.tpo.H_ice_dyn, y.tpo.f_ice_dyn,
                                    zeta_c, dx, dy,
                                    p_ymat.n_glen, p_ydyn.eps_0)
        elseif p_ydyn.visc_method == 2
            calc_visc_eff_3D_aa!(y.dyn.visc_eff, y.dyn.ux_b, y.dyn.uy_b,
                                 y.mat.ATT, y.tpo.H_ice_dyn, y.tpo.f_ice_dyn,
                                 zeta_c, dx, dy,
                                 p_ymat.n_glen, p_ydyn.eps_0)
        else
            error("calc_velocity_ssa!: visc_method=$(p_ydyn.visc_method) not supported.")
        end

        # ---- Step 2: log-space Picard relaxation on viscosity. ----
        # Every iteration, the first one towards the viscosity of the
        # previous call (Fortran velocity_ssa.f90:236-248); none for a
        # constant viscosity.
        if p_ydyn.visc_method != 0
            picard_relax_visc!(y.dyn.visc_eff, sc.ssa_picard_visc_eff_nm1,
                               p_ydyn.ssa_iter_rel)
        end

        # ---- Step 3: depth-integrated viscosity. ----
        # Boundary visc fields under the Option C convention: the
        # 3D `visc_eff` Center stagger does NOT include the bed
        # (zeta = 0) or surface (zeta = 1) endpoints. Approximate the
        # boundary visc by the nearest-Center value — exact for
        # `visc_method = 0` (constant `visc_const` fills all centres,
        # so the endpoints inherit the same value) and for isothermal
        # uniform-ATT cases under `visc_method = 1, 2`; approximate for
        # temperature-dependent ATT. Matches the SIA convention for
        # ATT_bed / ATT_surf in `calc_velocity_sia!`. Revisit when
        # therm wires temperature-dependent ATT (milestone 3g).
        @views interior(sc.ssa_visc_eff_b)[:, :, 1] .=
            interior(y.dyn.visc_eff)[:, :, 1]
        @views interior(sc.ssa_visc_eff_s)[:, :, 1] .=
            interior(y.dyn.visc_eff)[:, :, end]
        calc_visc_eff_int!(y.dyn.visc_eff_int, y.dyn.visc_eff,
                           sc.ssa_visc_eff_b, sc.ssa_visc_eff_s,
                           y.tpo.H_ice_dyn, y.tpo.f_ice_dyn, zeta_c)

        # ---- Step 4: beta on aa-cells (uses current ux_b/uy_b/c_bed). ----
        calc_beta!(y.dyn.beta, y.dyn.c_bed, y.dyn.ux_b, y.dyn.uy_b,
                   y.tpo.H_ice_dyn, y.tpo.f_ice_dyn, y.tpo.H_grnd, y.tpo.f_grnd,
                   y.bnd.z_bed, y.bnd.z_sl;
                   beta_method  = p_ydyn.beta_method,
                   beta_const   = p_ydyn.beta_const,
                   beta_q       = p_ydyn.beta_q,
                   beta_u0      = p_ydyn.beta_u0,
                   beta_gl_scale = p_ydyn.beta_gl_scale,
                   beta_gl_f    = p_ydyn.beta_gl_f,
                   H_grnd_lim   = p_ydyn.H_grnd_lim,
                   beta_min     = p_ydyn.beta_min,
                   rho_ice      = y.c.rho_ice, rho_sw = y.c.rho_sw)

        # ---- Step 5: stagger beta to faces. ----
        stagger_beta!(y.dyn.beta_acx, y.dyn.beta_acy, y.dyn.beta,
                      y.tpo.H_ice_dyn, y.tpo.f_ice_dyn, y.dyn.ux_b, y.dyn.uy_b,
                      y.tpo.f_grnd, y.tpo.f_grnd_acx, y.tpo.f_grnd_acy;
                      beta_gl_stag = p_ydyn.beta_gl_stag,
                      beta_min     = p_ydyn.beta_min)

        # ---- Step 5b: friction of the matrix (and of taub). ----
        interior(sc.ssa_beta_acx) .= interior(y.dyn.beta_acx)
        interior(sc.ssa_beta_acy) .= interior(y.dyn.beta_acy)
        set_beta_min_grounded!(sc.ssa_beta_acx, sc.ssa_beta_acy,
                               y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy, p_ydyn.beta_min)

        # ---- Step 6: stagger viscosity to ab-corner cache. ----
        stagger_visc_aa_ab!(sc.ssa_n_aa_ab, y.dyn.visc_eff_int,
                            y.tpo.H_ice_dyn, y.tpo.f_ice_dyn)

        # ---- Step 7: assemble SSA matrix into COO buffers + RHS. ----
        # Dispatch on `ssa.method`:
        #   :residual         — Jacobian of the strong-form SSA momentum
        #                       residual (Fortran-faithful).
        #   :energy_quadratic — Hessian of the discrete viscous-energy
        #                       functional with η, β, H frozen
        #                       (symmetric positive-definite). Assembly
        #                       lives in `velocity_ssa_energy.jl`.
        #   :energy_nonlinear — RESERVED. Future fully-nonlinear
        #                       minimisation; not yet implemented.
        if ssa.method === :residual
            _assemble_ssa_matrix!(
                sc.ssa_I_idx, sc.ssa_J_idx, sc.ssa_vals,
                sc.ssa_b_vec, sc.ssa_nnz,
                y.dyn.ux_b, y.dyn.uy_b,
                sc.ssa_beta_acx, sc.ssa_beta_acy,
                y.dyn.visc_eff_int, sc.ssa_n_aa_ab,
                y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy, y.tpo.mask_frnt,
                y.tpo.H_ice_dyn, y.tpo.f_ice_dyn,
                y.dyn.taud_acx, y.dyn.taud_acy,
                y.dyn.taul_int_acx, y.dyn.taul_int_acy,
                dx, dy;
                boundaries = domain_boundaries(y.p),
                lateral_bc = p_ydyn.ssa_lat_bc,
            )
        elseif ssa.method === :energy_quadratic
            _assemble_ssa_matrix_energy!(
                sc.ssa_I_idx, sc.ssa_J_idx, sc.ssa_vals,
                sc.ssa_b_vec, sc.ssa_nnz,
                y.dyn.ux_b, y.dyn.uy_b,
                sc.ssa_beta_acx, sc.ssa_beta_acy,
                y.dyn.visc_eff_int, sc.ssa_n_aa_ab,
                y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy, y.tpo.mask_frnt,
                y.tpo.H_ice_dyn, y.tpo.f_ice_dyn,
                y.dyn.taud_acx, y.dyn.taud_acy,
                y.dyn.taul_int_acx, y.dyn.taul_int_acy,
                dx, dy;
                boundaries = domain_boundaries(y.p),
                lateral_bc = p_ydyn.ssa_lat_bc,
            )
        elseif ssa.method === :energy_nonlinear
            error("calc_velocity_ssa!: method = :energy_nonlinear is reserved " *
                  "for the future fully-nonlinear energy-minimisation solver " *
                  "(η baked into E[u] via Glen's law, Newton/L-BFGS replacing " *
                  "the Picard loop). Not yet implemented.")
        else
            error("calc_velocity_ssa!: unrecognised method=$(ssa.method). " *
                  "Expected :residual, :energy_quadratic, or :energy_nonlinear.")
        end

        # ---- Step 8: build (or refresh) sparse CSC, solve. ----
        # On `iter == 1` builds A from COO via `sparse(...)` and caches
        # the COO→CSC permutation in `sc.ssa_coo_to_csc`. On subsequent
        # iters refreshes `A.nzval` in-place via the cached permutation.
        nnz_now = sc.ssa_nnz[]
        N_rows  = 2 * Nx * Ny
        A = _build_or_refresh_ssa_csc!(sc,
                                       sc.ssa_I_idx, sc.ssa_J_idx, sc.ssa_vals,
                                       nnz_now, N_rows, iter)
        x = _solve_ssa_linear!(sc.ssa_x_vec, sc, A, sc.ssa_b_vec, ssa)

        # ---- Step 9: unpack x → ux_b, uy_b face slots. ----
        Tx_top = topology(y.dyn.ux_b.grid, 1)
        Ty_top = topology(y.dyn.uy_b.grid, 2)
        Ux = interior(y.dyn.ux_b)
        Uy = interior(y.dyn.uy_b)
        @inbounds for j in 1:Ny, i in 1:Nx
            row_ux = _row_ux(i, j, Nx)
            row_uy = _row_uy(i, j, Nx)
            ip1f = _ip1_modular(i, Nx, Tx_top)
            jp1f = _jp1_modular(j, Ny, Ty_top)
            Ux[ip1f, j, 1] = x[row_ux]
            Uy[i, jp1f, 1] = x[row_uy]
        end
        # Replicate the leading face slot under Bounded for readers that
        # use slot-1 (matches calc_velocity_sia! convention).
        if Tx_top === Bounded
            @views Ux[1, :, :] .= Ux[2, :, :]
        end
        if Ty_top === Bounded
            @views Uy[:, 1, :] .= Uy[:, 2, :]
        end

        # ---- Step 9b: velocity limit (Fortran `ssa_vel_clip`). ----
        # Only "clip" is ported (`with_ported_options`); a non-finite
        # solution stops the run in the convergence check below.
        ssa_vel_clip!(y.dyn.ux_b, y.dyn.uy_b, Float64(p_ydyn.ssa_vel_max))

        # ---- Step 10: linear Picard velocity relaxation. ----
        # Every iteration, the first one towards the velocity of the
        # previous call (Fortran velocity_ssa.f90:361).
        picard_relax_vel!(y.dyn.ux_b, y.dyn.uy_b,
                          sc.ssa_picard_ux_b_nm1, sc.ssa_picard_uy_b_nm1,
                          p_ydyn.ssa_iter_rel)

        # ---- Step 12: convergence check (L2 relative residual). ----
        l2_resid = picard_calc_convergence_l2(y.dyn.ux_b, sc.ssa_picard_ux_b_nm1,
                                              y.dyn.uy_b, sc.ssa_picard_uy_b_nm1,
                                              y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy)
        if iter ≤ n_resid_max
            sc.ssa_residuals[iter] = l2_resid
        end

        # ---- L1-rel diagnostic matrix (Fortran line 297). ----
        picard_calc_convergence_l1rel_matrix!(
            interior(y.dyn.ssa_err_acx), interior(y.dyn.ssa_err_acy),
            interior(y.dyn.ux_b), interior(y.dyn.uy_b),
            interior(sc.ssa_picard_ux_b_nm1), interior(sc.ssa_picard_uy_b_nm1))

        if l2_resid < p_ydyn.ssa_iter_conv
            converged = true
            break
        end
    end

    sc.ssa_iter_now[] = iter_now
    sc.ssa_lim_n[] = count_vel_lim_faces(y.dyn.ux_b, y.dyn.uy_b, y.dyn.ssa_mask_acx,
                                         y.dyn.ssa_mask_acy, Float64(p_ydyn.ssa_vel_max))

    if !converged
        @warn "SSA Picard did not converge" iter = iter_now resid = (iter_now > 0 && iter_now <= n_resid_max ? sc.ssa_residuals[iter_now] : NaN) tol = p_ydyn.ssa_iter_conv
    end

    # Post-loop: basal stress with the friction of the matrix (Fortran
    # velocity_ssa.f90:403).
    calc_basal_stress!(y.dyn.taub_acx, y.dyn.taub_acy,
                       sc.ssa_beta_acx, sc.ssa_beta_acy,
                       y.dyn.ux_b, y.dyn.uy_b)

    return y
end

# ----------------------------------------------------------------------
# Diagnostic: dump assembled SSA stiffness matrix + RHS to NetCDF.
#
# Reads the most recently assembled COO triplets from
# `y.dyn.scratch.ssa_I_idx`/`ssa_J_idx`/`ssa_vals`/`ssa_nnz` and the RHS
# from `ssa_b_vec`. Also performs the same `sparse(I, J, V)` conversion
# that `_solve_ssa_linear!`'s caller does and writes the resulting CSC
# arrays — useful for spotting duplicate-entry summing (COO `nnz` vs
# CSC `nnz` mismatch) or other surprises in the COO → CSC step.
#
# Snapshot timing: by the time `dyn_step!` returns, the COO buffers
# reflect the LAST Picard iteration's assembly. To dump a specific
# iteration, call this from inside the Picard loop or wire a flag.
#
# Diagnostic-only — not part of the production API. The NetCDF schema
# is intentionally minimal and may change.
# ----------------------------------------------------------------------
"""
    dump_ssa_assembly(y; path::AbstractString = "ssa_assembly.nc") -> path

Write the most recently assembled SSA stiffness matrix `A` (in both
COO and CSC form) plus RHS vector `b` to NetCDF for offline inspection.

Reads from `y.dyn.scratch.ssa_I_idx`/`ssa_J_idx`/`ssa_vals`/`ssa_nnz`
and `ssa_b_vec`. The `sparse(I, J, V, nrows, nrows)` conversion mirrors
what `calc_velocity_ssa!` does immediately before calling
`_solve_ssa_linear!`.

NetCDF schema:
  - dim `nnz_coo` — number of COO non-zeros (= `ssa_nnz[]`).
  - dim `nrows`   — `2 * Nx * Ny` (matrix dimension).
  - dim `csc_nz`  — number of CSC non-zeros (after de-dup + sort).
  - dim `csc_colp`— `nrows + 1` (CSC `colptr` length).
  - var `I` (Int64, len `nnz_coo`) — COO row indices.
  - var `J` (Int64, len `nnz_coo`) — COO column indices.
  - var `V` (Float64, len `nnz_coo`) — COO values.
  - var `b` (Float64, len `nrows`) — RHS vector.
  - var `csc_colptr`, `csc_rowval`, `csc_nzval` — CSC view of `A`.
  - attrs: `Nx`, `Ny`, `nnz_coo`, `nnz_csc`.

Diagnostic-only — not part of the production API.
"""
function dump_ssa_assembly(y; path::AbstractString = "ssa_assembly.nc")
    sc  = y.dyn.scratch
    nnz = sc.ssa_nnz[]
    nnz > 0 || error("dump_ssa_assembly: ssa_nnz[] == 0 — call after `_assemble_ssa_matrix!`.")

    I_buf = sc.ssa_I_idx[1:nnz]
    J_buf = sc.ssa_J_idx[1:nnz]
    V_buf = sc.ssa_vals[1:nnz]
    b_buf = copy(sc.ssa_b_vec)

    Nx = size(y.g, 1)
    Ny = size(y.g, 2)
    nrows = 2 * Nx * Ny

    # CSC conversion — matches `calc_velocity_ssa!` exactly.
    A = sparse(I_buf, J_buf, V_buf, nrows, nrows)

    isfile(path) && rm(path)
    NCDataset(path, "c") do ds
        defDim(ds, "nnz_coo", nnz)
        defDim(ds, "nrows",   nrows)
        defDim(ds, "csc_nz",  length(A.nzval))
        defDim(ds, "csc_colp", length(A.colptr))

        defVar(ds, "I", I_buf, ("nnz_coo",))
        defVar(ds, "J", J_buf, ("nnz_coo",))
        defVar(ds, "V", V_buf, ("nnz_coo",))
        defVar(ds, "b", b_buf, ("nrows",))
        defVar(ds, "csc_colptr", A.colptr, ("csc_colp",))
        defVar(ds, "csc_rowval", A.rowval, ("csc_nz",))
        defVar(ds, "csc_nzval",  A.nzval,  ("csc_nz",))

        ds.attrib["Nx"]      = Nx
        ds.attrib["Ny"]      = Ny
        ds.attrib["nnz_coo"] = nnz
        ds.attrib["nnz_csc"] = length(A.nzval)
    end
    return path
end
