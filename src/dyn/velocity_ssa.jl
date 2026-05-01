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
using Oceananigans.Grids: topology, Bounded, Periodic
using Oceananigans.BoundaryConditions: fill_halo_regions!

using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: norm, Diagonal, diag
using Krylov: bicgstab!
using AlgebraicMultigrid: smoothed_aggregation, ruge_stuben, aspreconditioner,
                          GaussSeidel, Jacobi
using NCDatasets: NCDataset, defDim, defVar

export set_ssa_masks!, _assemble_ssa_matrix!,
       _solve_ssa_linear!, calc_velocity_ssa!,
       picard_relax_visc!, picard_relax_vel!,
       picard_calc_convergence_l2, picard_calc_convergence_l1rel_matrix!,
       set_inactive_margins!, calc_basal_stress!,
       dump_ssa_assembly

# Integer constants for `mask_frnt`. Mirror the Fortran values from
# `solver_ssa_ac.f90:899-903`. `val_disabled = 5` is internal to
# `set_ssa_masks` and used to mark a front position that should NOT
# trigger lateral-BC mask=3.
const _SSA_MASK_FRNT_ICE_FREE  = -1
const _SSA_MASK_FRNT_FLT       = 1
const _SSA_MASK_FRNT_MARINE    = 2
const _SSA_MASK_FRNT_GRND      = 3
const _SSA_MASK_FRNT_DISABLED  = 5

"""
    set_ssa_masks!(ssa_mask_acx, ssa_mask_acy,
                   mask_frnt, H_ice, f_ice, f_grnd, z_base, z_sl,
                   dx::Real;
                   use_ssa::Bool=true,
                   lateral_bc::AbstractString="floating")
        -> (ssa_mask_acx, ssa_mask_acy)

Set the SSA solver masks per Fortran convention. Mask values written
to `ssa_mask_acx[i+1, j, 1]` (Fortran `ssa_mask_acx(i, j)`) and
`ssa_mask_acy[i, j+1, 1]` (Fortran `ssa_mask_acy(i, j)`):

  - 0 = SSA inactive at this face (Dirichlet u = 0).
  - 1 = active grounded / grounding-line (shelfy-stream).
  - 2 = active floating (shelf).
  - 3 = lateral boundary condition (calving front).
  - 4 = deactivated lateral boundary; treated as inner SSA.

If `use_ssa == false`, all faces stay 0 (everything Dirichlet).

`lateral_bc` selects which fronts trigger the BC:
  - "none"               : no fronts.
  - "floating"|"float"|"slab"|"slab-ext" : only floating fronts.
  - "marine"             : floating + grounded-marine.
  - "all"                : all fronts.

`z_base` and `z_sl` are accepted for Fortran signature parity but
not actually consumed by the current Fortran logic (the ice-base
slope check is gated on a parameter that defaults off). They are
kept in the signature for forward compatibility.

Port of `solver_ssa_ac.f90:854 set_ssa_masks` (lines 854-1104).
"""
function set_ssa_masks!(ssa_mask_acx, ssa_mask_acy,
                        mask_frnt, H_ice, f_ice, f_grnd, z_base, z_sl,
                        dx::Real;
                        use_ssa::Bool=true,
                        lateral_bc::AbstractString="floating")

    Mx = interior(ssa_mask_acx)
    My = interior(ssa_mask_acy)
    MF = interior(mask_frnt)
    Hi = interior(H_ice)
    Fi = interior(f_ice)
    Fg = interior(f_grnd)

    Nx = size(Hi, 1)
    Ny = size(Hi, 2)

    Tx_top = topology(ssa_mask_acx.grid, 1)
    Ty_top = topology(ssa_mask_acy.grid, 2)

    # Initialise to all zero (Fortran lines 911-912).
    fill!(Mx, 0.0)
    fill!(My, 0.0)

    use_ssa || return ssa_mask_acx, ssa_mask_acy

    # ---- Step 1: build mask_frnt_dyn from mask_frnt + lateral_bc ----
    # Fortran lines 916-978. We allocate a small Int matrix here
    # (size Nx*Ny is tiny relative to the matrix we'll assemble).
    mask_frnt_dyn = Matrix{Int}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        mask_frnt_dyn[i, j] = Int(MF[i, j, 1])
    end

    if lateral_bc == "none"
        # Disable all front detection — treat all ice as inner SSA.
        @inbounds for j in 1:Ny, i in 1:Nx
            if mask_frnt_dyn[i, j] > 0
                mask_frnt_dyn[i, j] = _SSA_MASK_FRNT_DISABLED
            end
        end
    elseif lateral_bc == "floating" || lateral_bc == "float" ||
           lateral_bc == "slab"     || lateral_bc == "slab-ext"
        # Only floating fronts trigger BC; disable grounded + marine.
        @inbounds for j in 1:Ny, i in 1:Nx
            v = mask_frnt_dyn[i, j]
            if v == _SSA_MASK_FRNT_GRND || v == _SSA_MASK_FRNT_MARINE
                mask_frnt_dyn[i, j] = _SSA_MASK_FRNT_DISABLED
            end
        end
    elseif lateral_bc == "marine"
        # Disable only purely-grounded above-sea-level fronts.
        @inbounds for j in 1:Ny, i in 1:Nx
            if mask_frnt_dyn[i, j] == _SSA_MASK_FRNT_GRND
                mask_frnt_dyn[i, j] = _SSA_MASK_FRNT_DISABLED
            end
        end
    elseif lateral_bc == "all"
        # No-op — all fronts already accurately diagnosed.
    else
        error("set_ssa_masks!: lateral_bc=\"$lateral_bc\" not recognized.")
    end

    # ---- Step 2: define ssa solver masks (Fortran lines 983-1098). ----
    @inbounds for j in 1:Ny, i in 1:Nx
        # Fortran neighbour indices clamp at domain edges (Fortran lines
        # 987-990) — independent of grid topology in the Fortran source.
        # We mirror the Fortran clamp convention here. Periodic wrap is
        # only used in the matrix-assembly kernel; for set_ssa_masks
        # the mask choice at the boundary follows the Fortran clamp.
        im1 = max(i - 1, 1)
        ip1 = min(i + 1, Nx)
        jm1 = max(j - 1, 1)
        jp1 = min(j + 1, Ny)

        # ===== x-direction (acx face, between cell (i, j) and (ip1, j)) =====
        # Fortran lines 995-1025.
        if Fi[i, j, 1] == 1.0 || Fi[ip1, j, 1] == 1.0
            # ac-node borders an ice-covered cell.
            if Fg[i, j, 1] > 0.0 || Fg[ip1, j, 1] > 0.0
                mval = 1   # shelfy-stream (grounded or GL)
            else
                mval = 2   # shelf
            end

            # SPECIAL CASE: floating ice next to ice-free land
            # (Fortran lines 1009-1023).
            if mval == 2
                if Fg[i, j, 1] == 0.0 &&
                   (Fg[ip1, j, 1] > 0.0 && Hi[ip1, j, 1] == 0.0)
                    mval = 0
                elseif (Fg[i, j, 1] > 0.0 && Hi[i, j, 1] == 0.0) &&
                       Fg[ip1, j, 1] == 0.0
                    mval = 0
                end
            end

            # Write to the X-face slot for cell (i, j).
            ip1f = _ip1_modular(i, Nx, Tx_top)
            Mx[ip1f, j, 1] = Float64(mval)
        end

        # Overwrite for lateral BC (Fortran lines 1028-1034).
        let mfd_ij  = mask_frnt_dyn[i, j],
            mfd_ip1 = mask_frnt_dyn[ip1, j]
            if (mfd_ij > 0 && mfd_ip1 < 0) ||
               (mfd_ij < 0 && mfd_ip1 > 0)
                ip1f = _ip1_modular(i, Nx, Tx_top)
                Mx[ip1f, j, 1] = 3.0
            end
            # Deactivated lateral BC (Fortran lines 1037-1043).
            if (mfd_ij == _SSA_MASK_FRNT_DISABLED && mfd_ip1 < 0) ||
               (mfd_ij < 0 && mfd_ip1 == _SSA_MASK_FRNT_DISABLED)
                ip1f = _ip1_modular(i, Nx, Tx_top)
                Mx[ip1f, j, 1] = 4.0
            end
        end

        # ===== y-direction (acy face, between cell (i, j) and (i, jp1)) =====
        # Fortran lines 1047-1077.
        if Fi[i, j, 1] == 1.0 || Fi[i, jp1, 1] == 1.0
            if Fg[i, j, 1] > 0.0 || Fg[i, jp1, 1] > 0.0
                mval = 1
            else
                mval = 2
            end

            if mval == 2
                if Fg[i, j, 1] == 0.0 &&
                   (Fg[i, jp1, 1] > 0.0 && Hi[i, jp1, 1] == 0.0)
                    mval = 0
                elseif (Fg[i, j, 1] > 0.0 && Hi[i, j, 1] == 0.0) &&
                       Fg[i, jp1, 1] == 0.0
                    mval = 0
                end
            end

            jp1f = _jp1_modular(j, Ny, Ty_top)
            My[i, jp1f, 1] = Float64(mval)
        end

        # Overwrite for lateral BC (Fortran lines 1080-1095).
        let mfd_ij  = mask_frnt_dyn[i, j],
            mfd_jp1 = mask_frnt_dyn[i, jp1]
            if (mfd_ij > 0 && mfd_jp1 < 0) ||
               (mfd_ij < 0 && mfd_jp1 > 0)
                jp1f = _jp1_modular(j, Ny, Ty_top)
                My[i, jp1f, 1] = 3.0
            end
            if (mfd_ij == _SSA_MASK_FRNT_DISABLED && mfd_jp1 < 0) ||
               (mfd_ij < 0 && mfd_jp1 == _SSA_MASK_FRNT_DISABLED)
                jp1f = _jp1_modular(j, Ny, Ty_top)
                My[i, jp1f, 1] = 4.0
            end
        end
    end

    return ssa_mask_acx, ssa_mask_acy
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
#     boundaries = :zeros        → (no-slip, no-slip, no-slip, no-slip)
#     default                    → (no-slip, no-slip, no-slip, no-slip)
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
    elseif boundaries == :periodic_y || boundaries == :bounded_y_periodic ||
           boundaries == :y_periodic
        return (:free_slip, :periodic, :free_slip, :periodic)
    elseif boundaries == :infinite
        return (:free_slip, :free_slip, :free_slip, :free_slip)
    elseif boundaries == :zeros || boundaries == :bounded || boundaries == :no_slip
        return (:no_slip, :no_slip, :no_slip, :no_slip)
    else
        # Match the Fortran DEFAULT branch (no-slip).
        return (:no_slip, :no_slip, :no_slip, :no_slip)
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

"""
    _assemble_ssa_matrix!(I_idx, J_idx, vals, b_vec, nnz_ref,
                          ux_b, uy_b,
                          beta_acx, beta_acy,
                          visc_eff_int, visc_ab,
                          ssa_mask_acx, ssa_mask_acy, mask_frnt,
                          H_ice, f_ice,
                          taud_acx, taud_acy,
                          taul_int_acx, taul_int_acy,
                          dx::Real, dy::Real, beta_min::Real;
                          boundaries::Symbol=:bounded,
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
  - `beta_min` — minimum allowed beta for grounded ice
    (Fortran lines 479, 762).
  - `boundaries` — Symbol selecting the edge-BC palette.
  - `lateral_bc` — accepted for signature parity (used by the caller
    when pre-computing the masks).

Block-row layout: row `2k - 1` is the ux equation at cell
`k = (j-1)*Nx + i`, row `2k` is the uy equation at cell k.

Returns the kernel arguments unchanged (in-place mutation of the
COO buffers, RHS, and `nnz_ref`).

Topology: only `(Bounded, Bounded, Flat)` and `(Bounded, Periodic, Flat)`
are supported in PR-A.2. Other combinations error.
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
                               dx::Real, dy::Real, beta_min::Real;
                               boundaries::Symbol=:bounded,
                               lateral_bc::AbstractString="floating")

    # ---- Topology checks (PR-A.2 scope). ----
    Tx_top = topology(visc_eff_int.grid, 1)
    Ty_top = topology(visc_eff_int.grid, 2)
    Tx_top === Bounded || error(
        "_assemble_ssa_matrix!: x-topology must be Bounded for PR-A.2 (got $(Tx_top)).")
    (Ty_top === Bounded || Ty_top === Periodic) || error(
        "_assemble_ssa_matrix!: y-topology must be Bounded or Periodic " *
        "for PR-A.2 (got $(Ty_top)).")

    # Fill halos on every input the kernel reads with i±1 / j±1 stencils.
    # Note: the kernel does NOT use halo reads for the column-index
    # arithmetic — it computes `im1`, `ip1`, etc. as wrapped/clamped
    # integers. The halo fills here are only for fields read through
    # the standard halo path (none in PR-A.2 — kept here as a hook
    # for symmetry with the SIA wrapper). Each input is filled to keep
    # callers honest about boundary state.
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
            if ssa_mask_x == 1 && Bx[ip1f_i, j, 1] == 0.0
                beta_now = beta_min
            end

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
            if ssa_mask_y == 1 && By[i, jp1f_j, 1] == 0.0
                beta_now = beta_min
            end

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
        d_inv = 1.0 ./ diag(A)
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
    _solve_ssa_linear!(scratch, A::SparseMatrixCSC{Float64,Int},
                       b::Vector{Float64}, ssa::SSASolver) -> Vector{Float64}

Solve `A x = b` for the SSA linear system using the configured Krylov
method + preconditioner (per `ssa.precond`). Returns `x` (a fresh
`Vector{Float64}` sized to match `b`). Uses the Krylov workspace
stored in `scratch.ssa_solver_workspace` so per-call allocation is
minimal.

Soft-warns (does not error) on non-convergence — Fortran-faithful;
the outer Picard loop is the safety net for poor inner convergence.
"""
function _solve_ssa_linear!(scratch,
                            A::SparseMatrixCSC{Float64,Int},
                            b::Vector{Float64},
                            ssa::SSASolver)
    # Build (or rebuild) the preconditioner. The matrix coefficients
    # change every Picard iteration so we cannot cache across calls.
    M, ldiv_flag = _build_ssa_precond(scratch, A, ssa)

    # Run the configured Krylov method. Currently only :bicgstab is
    # implemented; future expansion to :gmres / :cg goes here.
    if ssa.method === :bicgstab
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
        x = copy(workspace.x)
        if !workspace.stats.solved
            res = norm(A * x .- b)
            @warn "SSA BiCGStab did not converge" precond=ssa.precond niter=workspace.stats.niter residual=res rtol=ssa.rtol itmax=ssa.itmax
        end
        return x
    else
        error("_solve_ssa_linear!: method=$(ssa.method) not yet implemented. " *
              "Currently only :bicgstab is supported.")
    end
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
    r  = Float64(rel)
    rm = 1.0 - r
    @inbounds @simd for k in eachindex(V)
        v_prev = Vnm1[k]
        v_now  = V[k]
        # Guard against log(0): if either is zero, fall back to linear
        # mixing (preserves the Fortran behavior where visc_min floor
        # ensures positive values, but be defensive in case the caller
        # passed an unfilled array on the first iteration).
        if v_prev > 0.0 && v_now > 0.0
            V[k] = exp(rm * log(v_prev) + r * log(v_now))
        else
            V[k] = rm * v_prev + r * v_now
        end
    end
    return visc_eff
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
    r = Float64(rel)
    @inbounds @simd for k in eachindex(Ux)
        Ux[k] = Uxnm1[k] + r * (Ux[k] - Uxnm1[k])
    end
    @inbounds @simd for k in eachindex(Uy)
        Uy[k] = Uynm1[k] + r * (Uy[k] - Uynm1[k])
    end
    return ux_n, uy_n
end

"""
    picard_calc_convergence_l2(ux, ux_nm1, uy, uy_nm1) -> Float64

Relative L2 residual norm of the velocity change between successive
Picard iterations. Mirrors `velocity_general.f90:1917
picard_calc_convergence_l2` with `norm_method = 1` (the only branch
actually used by the Fortran driver):

    res1 = sum((u - u_prev)²) over points where |u| > vel_tol
    res2 = sum((u_prev)²)     over the same points
    resid = res1 / (res2 + du_reg)

If no points pass the velocity-tolerance mask, returns 0.0 (matches
Fortran).

Inputs are interior arrays (Yelmo.jl `interior(field)` slabs) of
shape `(Nx_face, Ny, 1)` — sums over all entries. The Fortran
`mask_acx > 0` predicate is enforced by passing only the active-SSA
indices; for our wrapper we instead include all face cells (the
inactive-mask cells have `u = 0` from the Dirichlet rows so they
fail the `vel_tol` threshold automatically).
"""
function picard_calc_convergence_l2(ux::AbstractArray, ux_nm1::AbstractArray,
                                    uy::AbstractArray, uy_nm1::AbstractArray)
    res1 = 0.0
    res2 = 0.0
    @inbounds @simd for k in eachindex(ux)
        if abs(ux[k]) > _SSA_PICARD_VEL_TOL
            tx = ux[k] - ux_nm1[k]
            abs(tx) < _SSA_PICARD_TOL_UF && (tx = 0.0)
            res1 += tx * tx
            tp = ux_nm1[k]
            abs(tp) < _SSA_PICARD_TOL_UF && (tp = 0.0)
            res2 += tp * tp
        end
    end
    @inbounds @simd for k in eachindex(uy)
        if abs(uy[k]) > _SSA_PICARD_VEL_TOL
            ty = uy[k] - uy_nm1[k]
            abs(ty) < _SSA_PICARD_TOL_UF && (ty = 0.0)
            res1 += ty * ty
            tp = uy_nm1[k]
            abs(tp) < _SSA_PICARD_TOL_UF && (tp = 0.0)
            res2 += tp * tp
        end
    end
    return res1 / (res2 + _SSA_PICARD_DU_REG)
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
    @inbounds @simd for k in eachindex(err_x)
        if abs(ux[k]) > vel_tol
            err_x[k] = 2.0 * abs(ux[k] - ux_nm1[k]) /
                       abs(ux[k] + ux_nm1[k] + tol)
        else
            err_x[k] = 0.0
        end
    end
    @inbounds @simd for k in eachindex(err_y)
        if abs(uy[k]) > vel_tol
            err_y[k] = 2.0 * abs(uy[k] - uy_nm1[k]) /
                       abs(uy[k] + uy_nm1[k] + tol)
        else
            err_y[k] = 0.0
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

    @inbounds for j in 1:Ny, i in 1:Nx
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
    @inbounds @simd for k in eachindex(Tx)
        v = Bx[k] * Ux[k]
        Tx[k] = abs(v) < tol ? 0.0 : v
    end
    @inbounds @simd for k in eachindex(Ty)
        v = By[k] * Uy[k]
        Ty[k] = abs(v) < tol ? 0.0 : v
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
#   6. Stagger beta to face-staggered beta_acx/beta_acy.
#   7. Stagger viscosity to ab-corner (visc_ab cache).
#   8. Assemble SSA matrix → COO triplets + RHS in dyn.scratch.
#   9. Build SparseMatrixCSC and run BiCGStab+AMG.
#  10. Unpack solution back into ux_b, uy_b face slots.
#  11. Apply linear Picard velocity relaxation.
#  12. Set inactive margins (zero velocity at fully-empty faces).
#  13. Compute L2 residual; record in scratch; check convergence.
#
# After loop: compute basal stress tau_b = beta · u_b.
#
# The Fortran driver also has an adaptive corr_theta block that's
# disabled (`if (.FALSE.)`); we mirror that omission and use a
# constant relaxation parameter `ssa.picard_relax`.
# ----------------------------------------------------------------------

"""
    calc_velocity_ssa!(y::YelmoModel) -> y

Run the SSA Picard iteration on `y`. Updates `y.dyn.ux_b`, `y.dyn.uy_b`,
`y.dyn.taub_acx`, `y.dyn.taub_acy`, `y.dyn.visc_eff`,
`y.dyn.visc_eff_int`, `y.dyn.beta`, `y.dyn.beta_acx`, `y.dyn.beta_acy`,
plus diagnostic scratch fields (`scratch.ssa_residuals`,
`scratch.ssa_iter_now`, `scratch.ssa_picard_*_nm1`).

Reads from `y.p.ydyn.ssa_solver` (the `SSASolver` object) for all
solver knobs. Other inputs come from the prior pre-solver kinematic
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
                   y.tpo.mask_frnt, y.tpo.H_ice, y.tpo.f_ice,
                   y.tpo.f_grnd, y.bnd.z_bed, y.bnd.z_sl, dx;
                   use_ssa = true,
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
    set_inactive_margins!(y.dyn.ux_b, y.dyn.uy_b, y.tpo.f_ice)

    converged = false
    iter_now = 0
    n_resid_max = length(sc.ssa_residuals)

    for iter in 1:ssa.picard_iter_max
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
                                    y.mat.ATT, y.tpo.H_ice, y.tpo.f_ice,
                                    zeta_c, dx, dy,
                                    p_ymat.n_glen, p_ydyn.eps_0)
        elseif p_ydyn.visc_method == 2
            calc_visc_eff_3D_aa!(y.dyn.visc_eff, y.dyn.ux_b, y.dyn.uy_b,
                                 y.mat.ATT, y.tpo.H_ice, y.tpo.f_ice,
                                 zeta_c, dx, dy,
                                 p_ymat.n_glen, p_ydyn.eps_0)
        else
            error("calc_velocity_ssa!: visc_method=$(p_ydyn.visc_method) not supported.")
        end

        # ---- Step 2: log-space Picard relaxation on viscosity. ----
        # Fortran line 212. Skip on the very first iteration since the
        # nm1 snapshot equals the n value (pre-iteration state).
        if iter > 1
            picard_relax_visc!(y.dyn.visc_eff, sc.ssa_picard_visc_eff_nm1,
                               ssa.picard_relax)
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
                           y.tpo.H_ice, y.tpo.f_ice, zeta_c)

        # ---- Step 4: beta on aa-cells (uses current ux_b/uy_b/c_bed). ----
        calc_beta!(y.dyn.beta, y.dyn.c_bed, y.dyn.ux_b, y.dyn.uy_b,
                   y.tpo.H_ice, y.tpo.f_ice, y.tpo.H_grnd, y.tpo.f_grnd,
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
                      y.tpo.H_ice, y.tpo.f_ice, y.dyn.ux_b, y.dyn.uy_b,
                      y.tpo.f_grnd, y.tpo.f_grnd_acx, y.tpo.f_grnd_acy;
                      beta_gl_stag = p_ydyn.beta_gl_stag,
                      beta_min     = p_ydyn.beta_min)

        # ---- Step 6: stagger viscosity to ab-corner cache. ----
        stagger_visc_aa_ab!(sc.ssa_n_aa_ab, y.dyn.visc_eff_int,
                            y.tpo.H_ice, y.tpo.f_ice)

        # ---- Step 7: assemble SSA matrix into COO buffers + RHS. ----
        _assemble_ssa_matrix!(
            sc.ssa_I_idx, sc.ssa_J_idx, sc.ssa_vals,
            sc.ssa_b_vec, sc.ssa_nnz,
            y.dyn.ux_b, y.dyn.uy_b,
            y.dyn.beta_acx, y.dyn.beta_acy,
            y.dyn.visc_eff_int, sc.ssa_n_aa_ab,
            y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy, y.tpo.mask_frnt,
            y.tpo.H_ice, y.tpo.f_ice,
            y.dyn.taud_acx, y.dyn.taud_acy,
            y.dyn.taul_int_acx, y.dyn.taul_int_acy,
            dx, dy, p_ydyn.beta_min;
            boundaries = _ssa_boundaries_symbol(y),
            lateral_bc = p_ydyn.ssa_lat_bc,
        )

        # ---- Step 8: build sparse matrix, solve. ----
        nnz = sc.ssa_nnz[]
        I_view = view(sc.ssa_I_idx, 1:nnz)
        J_view = view(sc.ssa_J_idx, 1:nnz)
        V_view = view(sc.ssa_vals,  1:nnz)
        N_rows = 2 * Nx * Ny
        A = sparse(I_view, J_view, V_view, N_rows, N_rows)
        x = _solve_ssa_linear!(sc, A, sc.ssa_b_vec, ssa)

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

        # ---- Step 10: linear Picard velocity relaxation. ----
        if iter > 1
            picard_relax_vel!(y.dyn.ux_b, y.dyn.uy_b,
                              sc.ssa_picard_ux_b_nm1, sc.ssa_picard_uy_b_nm1,
                              ssa.picard_relax)
        end

        # ---- Step 11: zero out fully-empty-margin face velocities. ----
        set_inactive_margins!(y.dyn.ux_b, y.dyn.uy_b, y.tpo.f_ice)

        # ---- Step 12: convergence check (L2 relative residual). ----
        l2_resid = picard_calc_convergence_l2(
            interior(y.dyn.ux_b), interior(sc.ssa_picard_ux_b_nm1),
            interior(y.dyn.uy_b), interior(sc.ssa_picard_uy_b_nm1))
        if iter ≤ n_resid_max
            sc.ssa_residuals[iter] = l2_resid
        end

        # ---- L1-rel diagnostic matrix (Fortran line 297). ----
        picard_calc_convergence_l1rel_matrix!(
            interior(y.dyn.ssa_err_acx), interior(y.dyn.ssa_err_acy),
            interior(y.dyn.ux_b), interior(y.dyn.uy_b),
            interior(sc.ssa_picard_ux_b_nm1), interior(sc.ssa_picard_uy_b_nm1))

        if l2_resid < ssa.picard_tol
            converged = true
            break
        end
    end

    sc.ssa_iter_now[] = iter_now

    if !converged
        @warn "SSA Picard did not converge" iter = iter_now resid = (iter_now > 0 && iter_now <= n_resid_max ? sc.ssa_residuals[iter_now] : NaN) tol = ssa.picard_tol
    end

    # Post-loop: basal stress (Fortran line 325).
    calc_basal_stress!(y.dyn.taub_acx, y.dyn.taub_acy,
                       y.dyn.beta_acx, y.dyn.beta_acy,
                       y.dyn.ux_b, y.dyn.uy_b)

    return y
end

# Resolve the boundaries Symbol used by `_assemble_ssa_matrix!` from
# the model's grid topology. We don't have direct access to the
# `boundaries` namelist value; map from grid topology pair instead.
function _ssa_boundaries_symbol(y)
    Tx = topology(y.g, 1)
    Ty = topology(y.g, 2)
    if Tx === Bounded && Ty === Bounded
        return :bounded
    elseif Tx === Bounded && Ty === Periodic
        return :periodic_y
    elseif Tx === Periodic && Ty === Bounded
        return :periodic_x
    elseif Tx === Periodic && Ty === Periodic
        return :periodic
    else
        error("_ssa_boundaries_symbol: unsupported topology pair ($(Tx), $(Ty)).")
    end
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
