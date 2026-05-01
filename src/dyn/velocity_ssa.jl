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
using LinearAlgebra: norm
using Krylov: bicgstab!
using AlgebraicMultigrid: smoothed_aggregation, aspreconditioner,
                          GaussSeidel, Jacobi

export set_ssa_masks!, _assemble_ssa_matrix!,
       _solve_ssa_linear!

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
# Linear-solve wrapper (PR-B commit 2).
#
# Runs Krylov.jl BiCGStab with an AlgebraicMultigrid.jl smoothed-
# aggregation preconditioner over the assembled SSA system `A x = b`.
# `A` is the COO-built `SparseMatrixCSC` and `b` is the assembled RHS.
#
# AMG configuration: per the locked-in SSASolver.smoother knob:
#   :gauss_seidel → smoothed_aggregation(A; presmoother=GaussSeidel(),
#                                         postsmoother=GaussSeidel())
#   :jacobi       → smoothed_aggregation(A; presmoother=Jacobi(),
#                                         postsmoother=Jacobi())
#
# AlgebraicMultigrid's `aspreconditioner` returns a `Preconditioner`
# object whose `ldiv!` overload approximates `A^{-1} y`. We pass
# `ldiv=true` to Krylov so it dispatches to `ldiv!` rather than `mul!`.
#
# Soft-warn on non-convergence: matches Fortran's Lis behavior of
# logging non-convergence and proceeding (the outer Picard iteration
# is the safety net).
# ----------------------------------------------------------------------

# Choose the AMG smoother per the SSASolver knob. Currently only
# :gauss_seidel is wired — AlgebraicMultigrid.jl's `Jacobi`
# constructor requires an explicit workspace vector matching the
# level's matrix size at construction time, which doesn't fit the
# closure-style API the SA driver expects. Future work: wrap a
# closure that lazily constructs Jacobi(x; iter=1) with the level's
# residual vector. Soft-fail with a clear error message for now —
# matches the "no autonomous shortcuts" guidance from CLAUDE.md.
function _amg_smoother(sym::Symbol)
    if sym === :gauss_seidel
        return GaussSeidel()
    elseif sym === :jacobi
        error("_amg_smoother: :jacobi smoother not yet supported. " *
              "AlgebraicMultigrid.jl's Jacobi smoother needs a per-level " *
              "workspace at construction time which the current wrapper " *
              "doesn't provide. Use :gauss_seidel for now.")
    else
        error("_amg_smoother: smoother=$(sym) not supported. " *
              "Choose :gauss_seidel (default).")
    end
end

"""
    _solve_ssa_linear!(scratch, A::SparseMatrixCSC{Float64,Int},
                       b::Vector{Float64}, ssa::SSASolver) -> Vector{Float64}

Solve `A x = b` for the SSA linear system using the configured Krylov
method + AMG preconditioner. Returns `x` (a fresh `Vector{Float64}`
sized to match `b`). Uses the Krylov workspace stored in
`scratch.ssa_solver_workspace` so per-call allocation is minimal.

Soft-warns (does not error) on non-convergence — Fortran-faithful;
the outer Picard loop is the safety net for poor inner convergence.
"""
function _solve_ssa_linear!(scratch,
                            A::SparseMatrixCSC{Float64,Int},
                            b::Vector{Float64},
                            ssa::SSASolver)
    # Build (or rebuild) the AMG preconditioner. The matrix coefficients
    # change every Picard iteration so we cannot cache across calls.
    smoother = _amg_smoother(ssa.smoother)
    ml = smoothed_aggregation(A; presmoother = smoother,
                                  postsmoother = smoother)
    M  = aspreconditioner(ml)
    scratch.ssa_amg_cache[] = ml  # keep alive for diagnostics

    # Run BiCGStab. The AMG `Preconditioner` overloads `ldiv!`, so pass
    # ldiv=true to make Krylov call `ldiv!(y, M, x)` rather than `mul!`.
    # Currently only :bicgstab is implemented; future expansion to
    # :gmres / :cg goes here.
    if ssa.method === :bicgstab
        workspace = scratch.ssa_solver_workspace
        bicgstab!(workspace, A, b;
                  M = M, ldiv = true,
                  rtol = ssa.rtol, itmax = ssa.itmax,
                  history = false)
        x = copy(workspace.x)
        if !workspace.stats.solved
            res = norm(A * x .- b)
            @warn "SSA BiCGStab did not converge" niter=workspace.stats.niter residual=res rtol=ssa.rtol itmax=ssa.itmax
        end
        return x
    else
        error("_solve_ssa_linear!: method=$(ssa.method) not yet implemented. " *
              "Currently only :bicgstab is supported.")
    end
end

