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

export set_ssa_masks!, _assemble_ssa_matrix!

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
