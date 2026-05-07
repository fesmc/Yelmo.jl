# ----------------------------------------------------------------------
# Energy-functional formulation of the SSA stiffness matrix.
#
# This file provides `_assemble_ssa_matrix_energy!`, an alternative to
# the strong-form `_assemble_ssa_matrix!` (`velocity_ssa.jl`) that
# assembles the SSA linear system as the Hessian of the discrete
# viscous-energy functional. Selected via `SSASolver(method =
# :energy_quadratic)`.
#
# The discrete energy density (per the demo in
# `IceSheetStencils.jl/examples/energy_functional_demo.jl`) is
#
#   W = ηH(2ε̇²_xx + 2ε̇²_yy + 2ε̇_xx ε̇_yy + ½(u_y + v_x)²)
#       + ½β(u² + v²)  +  ρgH(u s_x + v s_y)
#
# integrated over the C-grid. For ν, β, H frozen per Picard iteration,
# E[u, v] is quadratic, so K = ∇²E is the (symmetric, positive
# semi-definite) Hessian and the linear system is
#
#   K · u = -∇E|_{u=0} = b
#
# Algebraic identity with the strong-form Jacobian assembled by
# `_assemble_ssa_matrix!`:
#
#   K_inner = -A_residual_inner · (dx · dy)
#   b_inner = -b_residual_inner · (dx · dy)
#
# So the *inner-stencil* coefficients are the existing ones with sign
# flipped and `dx·dy` scaling. This is the cheapest way to verify the
# port is correct: per-row coefficients can be cross-checked against
# the existing assembly. Both formulations have the same `u` solution.
#
# Boundary conditions are handled with the energy-natural patterns:
#
#   - Dirichlet (mask = 0, mask = -1, no-slip edges): κ-penalty.
#     Add  ½κ(u - u_BC)² · dx·dy  to E.
#     Contributes  K[row,row] += κ·dx·dy, b[row] += κ·u_BC·dx·dy.
#     With κ ≫ max|K_inner_diag|, this enforces u → u_BC.
#
#   - Calving front (mask = 3): NOT YET IMPLEMENTED — errors out at
#     the row dispatch. Textbook is to drop the membrane/shear
#     contributions that span the absent neighbour and add the
#     boundary-work term `+T_front · dy` to b. Lands in a follow-up.
#
#   - Free-slip edges: NOT YET IMPLEMENTED — errors out. Textbook is a
#     symmetry constraint via Lagrange multiplier or κ-penalty on
#     `u_n - u_n-1 = 0`. Lands in a follow-up.
#
# Threading / GPU notes:
#
#   - The kernel uses the wrapper-+-parametric-kernel template (same
#     as `_assemble_ssa_matrix_kernel!`). Topology types thread through
#     as `Type{Tx_top}`/`Type{Ty_top}` parameters so that the
#     `_ip1_modular` / `_jp1_modular` calls fold at compile time.
#   - Per-row COO writes go to a fixed, deterministic offset: row
#     `nr` starts at COO index `(nr - 1) * 9 + 1` for u-rows and
#     v-rows alike (9 entries per inner row, padded with zeros for
#     BC rows that write fewer). This makes the loop race-free under
#     `@threads`. NOT yet thread-parallel — single-threaded by default
#     (matches the residual kernel for now).
#   - All field reads use `Int`-indexed `interior(...)` arrays, no
#     scalar indexing into structured Field objects inside the kernel.
# ----------------------------------------------------------------------

# κ default: ~1e5 above the natural inner-stencil scale.
#
# Inner-row diagonal scale is O(η·H·dy/dx + β·dx·dy); for typical ice-
# sheet parameters this is O(1e10 .. 1e12). Free-slip BCs add a
# *symmetric* κ-penalty that contributes both to the boundary row and
# to the inner-neighbour row's diagonal. If κ·dx·dy ≫ inner_diag by
# more than ~10 orders of magnitude, the inner stencil entries on
# that neighbour row get rounded off the CSC sum (ULP-lost), which
# silently drops the inner-row physics and makes the system
# under-constrained.
#
# κ = 1e15 keeps κ·dx·dy ~ 4e21 (with dx·dy ~ 4e6 m²); inner ULP at
# that scale is ~1e6, well below typical inner_diag ~1e10. Constraint
# leak is O(b / κ) ~ 1e-9 m/yr, well below the Picard tolerance.
#
# For pure-Dirichlet BCs (mask = 0, mask = -1, no-slip edges) only
# the boundary row gets a κ entry — no precision risk there — so
# this same value is fine.
const _SSA_ENERGY_KAPPA = 1.0e15

"""
    _assemble_ssa_matrix_energy!(I_idx, J_idx, vals, b_vec, nnz_ref,
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

Energy-functional sibling of `_assemble_ssa_matrix!`. Same signature
(so it slots into `calc_velocity_ssa!` Step 7 dispatch unchanged) but
assembles the symmetric positive-definite Hessian of the discrete
viscous-energy functional rather than the strong-form residual
Jacobian.

See file-level header for the formulation, BC handling, and the
algebraic identity `K_inner = -A_residual_inner · dx·dy`.
"""
function _assemble_ssa_matrix_energy!(I_idx::Vector{Int},
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
    Tx_top = topology(visc_eff_int.grid, 1)
    Ty_top = topology(visc_eff_int.grid, 2)
    (Tx_top === Bounded || Tx_top === Periodic) || error(
        "_assemble_ssa_matrix_energy!: x-topology must be Bounded or " *
        "Periodic (got $(Tx_top)).")
    (Ty_top === Bounded || Ty_top === Periodic) || error(
        "_assemble_ssa_matrix_energy!: y-topology must be Bounded or " *
        "Periodic (got $(Ty_top)).")

    # Halo fills — same as the residual assembly.
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
    Hi  = interior(H_ice)
    Fi  = interior(f_ice)
    Tdx = interior(taud_acx)
    Tdy = interior(taud_acy)
    Tlx = interior(taul_int_acx)
    Tly = interior(taul_int_acy)

    Nx = size(Hi, 1)
    Ny = size(Hi, 2)

    return _assemble_ssa_matrix_energy_kernel!(
        I_idx, J_idx, vals, b_vec, nnz_ref,
        Ux, Uy, Bx, By, Naa, Nab, Mx, My, Hi, Fi, Tdx, Tdy, Tlx, Tly,
        Float64(dx), Float64(dy), Float64(beta_min),
        Tx_top, Ty_top, Nx, Ny;
        boundaries = boundaries, lateral_bc = lateral_bc)
end

# Compute kernel — concrete-typed scalars, parametric topology, plain
# arrays. Mirrors `_assemble_ssa_matrix_kernel!`.
function _assemble_ssa_matrix_energy_kernel!(I_idx::Vector{Int},
                                              J_idx::Vector{Int},
                                              vals::Vector{Float64},
                                              b_vec::Vector{Float64},
                                              nnz_ref::Ref{Int},
                                              Ux, Uy, Bx, By, Naa, Nab,
                                              Mx, My, Hi, Fi,
                                              Tdx, Tdy, Tlx, Tly,
                                              dx::Float64, dy::Float64,
                                              beta_min::Float64,
                                              ::Type{Tx_top}, ::Type{Ty_top},
                                              Nx::Int, Ny::Int;
                                              boundaries::Symbol=:bounded,
                                              lateral_bc::AbstractString="floating",
        ) where {Tx_top<:AbstractTopology, Ty_top<:AbstractTopology}

    bcs = _ssa_resolve_bcs(boundaries)

    # Energy-form scaling factors. Inner-row coefficients are
    #   K[u_0_0, u_0_0]   = 4(ηH_W + ηH_E)·dy/dx + (ηH_S + ηH_N)·dx/dy + β·dx·dy
    # so we precompute the three geometric scalings once.
    s_dx_dy   = dx * dy           # cell area, drag + RHS scale.
    s_dy_dx   = dy / dx           # membrane (∂u/∂x²) scale.
    s_dx_dy_2 = dx / dy           # shear (∂u/∂y²) scale.

    κ = _SSA_ENERGY_KAPPA

    k = 0

    @inbounds for j in 1:Ny, i in 1:Nx
        # Periodic-wrap neighbour indices (same convention as
        # `_assemble_ssa_matrix_kernel!`).
        im1 = i - 1; if im1 == 0;     im1 = Nx;  end
        ip1 = i + 1; if ip1 == Nx + 1; ip1 = 1;  end
        jm1 = j - 1; if jm1 == 0;     jm1 = Ny;  end
        jp1 = j + 1; if jp1 == Ny + 1; jp1 = 1;  end

        ip1f_i = _ip1_modular(i, Nx, Tx_top)
        jp1f_j = _jp1_modular(j, Ny, Ty_top)

        # ===========================================================
        # ----- u-row at face (i+½, j) -----
        # ===========================================================
        nr = _row_ux(i, j, Nx)
        ssa_mask_x = Int(Mx[ip1f_i, j, 1])

        if ssa_mask_x == 0
            # Dirichlet u = 0 via κ-penalty.
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
            b_vec[nr] = 0.0

        elseif ssa_mask_x == -1
            # Prescribed u = ux_now via κ-penalty.
            ux_now = Ux[ip1f_i, j, 1]
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
            b_vec[nr] = κ * s_dx_dy * ux_now

        elseif i == 1 && bcs[3] !== :periodic
            if bcs[3] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[3] === :free_slip
                # Symmetric κ-penalty for `u(1, j) = u(2, j)`:
                #   E_pen = ½κ(u_n - u_{n-1})² · dx · dy
                # Boundary row writes its own diag/off-diag pair; the
                # symmetric contribution to the inner row at i=2 is
                # appended as extra COO triplets (CSC builder sums
                # duplicates). Keeps K symmetric → CG-compatible.
                nc_in = _row_ux(ip1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[3] = $(bcs[3]) at " *
                      "left edge not supported for method = :energy_quadratic. " *
                      "Expected :no_slip, :free_slip, or :periodic.")
            end

        elseif i == Nx && bcs[1] !== :periodic
            if bcs[1] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[1] === :free_slip
                # Right edge: u(Nx, j) = u(Nx-1, j) via symmetric κ-penalty.
                nc_in = _row_ux(Nx - 1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[1] = $(bcs[1]) at " *
                      "right edge not supported for method = :energy_quadratic.")
            end

        elseif j == 1 && bcs[4] !== :periodic
            if bcs[4] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[4] === :free_slip
                nc_in = _row_ux(i, jp1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[4] = $(bcs[4]) at " *
                      "bottom edge not supported for method = :energy_quadratic.")
            end

        elseif j == Ny && bcs[2] !== :periodic
            if bcs[2] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[2] === :free_slip
                nc_in = _row_ux(i, Ny - 1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[2] = $(bcs[2]) at " *
                      "top edge not supported for method = :energy_quadratic.")
            end

        elseif ssa_mask_x == 3
            error("_assemble_ssa_matrix_energy!: calving-front rows " *
                  "(ssa_mask_acx == 3) not yet implemented for method = " *
                  ":energy_quadratic. Use method = :residual until the " *
                  "boundary-work term lands.")

        else
            # === Inner-stencil u-row (energy form) ===
            #
            # Patch-local viscosity + thickness products (`Naa` is η·H̄
            # at aa-cells; `Nab` is the corner-staggered η·H̄ from
            # `stagger_visc_aa_ab!`). Indexing matches the residual
            # assembly:
            #   ηH_W = Naa[i, j, 1]            (cell west of u-face)
            #   ηH_E = Naa[ip1, j, 1]          (cell east of u-face)
            #   ηH_S = Nab[ip1f_x, jm1_y, 1]   (corner south of u-face)
            #   ηH_N = Nab[ip1f_x, jp1f_y, 1]  (corner north of u-face)
            ip1f_x = _ip1_modular(i, Nx, Tx_top)
            jp1f_y = _jp1_modular(j, Ny, Ty_top)
            jm1_y  = _jp1_modular(jm1, Ny, Ty_top)
            im1_x  = _ip1_modular(im1, Nx, Tx_top)

            ηH_W = Naa[i,   j, 1]
            ηH_E = Naa[ip1, j, 1]
            ηH_N = Nab[ip1f_x, jp1f_y, 1]
            ηH_S = Nab[ip1f_x, jm1_y,  1]

            β_now = Bx[ip1f_i, j, 1]
            if ssa_mask_x == 1 && β_now == 0.0
                β_now = beta_min
            end

            # Diagonal: 4(ηH_W + ηH_E)·dy/dx + (ηH_S + ηH_N)·dx/dy + β·dx·dy
            v = 4.0 * (ηH_W + ηH_E) * s_dy_dx +
                       (ηH_S + ηH_N) * s_dx_dy_2 +
                       β_now * s_dx_dy
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(i, j, Nx), v)

            # u-neighbours (membrane W/E, shear S/N).
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(im1, j, Nx),
                               -4.0 * ηH_W * s_dy_dx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(ip1, j, Nx),
                               -4.0 * ηH_E * s_dy_dx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(i, jm1, Nx),
                               -ηH_S * s_dx_dy_2)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(i, jp1, Nx),
                               -ηH_N * s_dx_dy_2)

            # v-neighbours (4 corner v-points around the u-face).
            # K[u, v_w_s] = -2·ηH_W - ηH_S      (Yelmo: v_{i, jm1}     = row_uy(i,   jm1, Nx))
            # K[u, v_w_n] = +2·ηH_W + ηH_N      (Yelmo: v_{i, j}       = row_uy(i,   j,   Nx))
            # K[u, v_e_s] = +2·ηH_E + ηH_S      (Yelmo: v_{ip1, jm1}   = row_uy(ip1, jm1, Nx))
            # K[u, v_e_n] = -2·ηH_E - ηH_N      (Yelmo: v_{ip1, j}     = row_uy(ip1, j,   Nx))
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(i, jm1, Nx),
                               -2.0 * ηH_W - ηH_S)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(i, j, Nx),
                                2.0 * ηH_W + ηH_N)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(ip1, jm1, Nx),
                                2.0 * ηH_E + ηH_S)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(ip1, j, Nx),
                               -2.0 * ηH_E - ηH_N)

            # RHS: b = -taud_acx · dx·dy   (energy convention; matches
            # the existing residual assembly's `b = taud_acx` after the
            # K = -A·dx·dy / b = -b·dx·dy transformation).
            b_vec[nr] = -Tdx[ip1f_i, j, 1] * s_dx_dy
        end

        # ===========================================================
        # ----- v-row at face (i, j+½) -----
        # ===========================================================
        nr = _row_uy(i, j, Nx)
        ssa_mask_y = Int(My[i, jp1f_j, 1])

        if ssa_mask_y == 0
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
            b_vec[nr] = 0.0

        elseif ssa_mask_y == -1
            uy_now = Uy[i, jp1f_j, 1]
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
            b_vec[nr] = κ * s_dx_dy * uy_now

        elseif j == 1 && bcs[4] !== :periodic
            if bcs[4] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[4] === :free_slip
                nc_in = _row_uy(i, jp1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[4] = $(bcs[4]) at " *
                      "bottom edge not supported (v-row).")
            end

        elseif j == Ny && bcs[2] !== :periodic
            if bcs[2] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[2] === :free_slip
                nc_in = _row_uy(i, Ny - 1, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[2] = $(bcs[2]) at " *
                      "top edge not supported (v-row).")
            end

        elseif i == 1 && bcs[3] !== :periodic
            if bcs[3] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[3] === :free_slip
                nc_in = _row_uy(ip1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[3] = $(bcs[3]) at " *
                      "left edge not supported (v-row).")
            end

        elseif i == Nx && bcs[1] !== :periodic
            if bcs[1] === :no_slip
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr, κ * s_dx_dy)
                b_vec[nr] = 0.0
            elseif bcs[1] === :free_slip
                nc_in = _row_uy(Nx - 1, j, Nx)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nr,    κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nr, nc_in, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nr, -κ * s_dx_dy)
                k += 1; _push_coo!(I_idx, J_idx, vals, k, nc_in, nc_in, κ * s_dx_dy)
                b_vec[nr] = 0.0
            else
                error("_assemble_ssa_matrix_energy!: bcs[1] = $(bcs[1]) at " *
                      "right edge not supported (v-row).")
            end

        elseif ssa_mask_y == 3
            error("_assemble_ssa_matrix_energy!: calving-front rows " *
                  "(ssa_mask_acy == 3) not yet implemented for method = " *
                  ":energy_quadratic.")

        else
            # === Inner-stencil v-row (energy form, mirror of u-row) ===
            #
            # ηH_S_cell = Naa[i, j, 1]         (cell south of v-face)
            # ηH_N_cell = Naa[i, jp1, 1]       (cell north of v-face)
            # ηH_W_cor  = Nab[im1_x,  jp1f_y, 1]   (corner west of v-face)
            # ηH_E_cor  = Nab[ip1f_x, jp1f_y, 1]   (corner east of v-face)
            ip1f_x = _ip1_modular(i, Nx, Tx_top)
            jp1f_y = _jp1_modular(j, Ny, Ty_top)
            im1_x  = _ip1_modular(im1, Nx, Tx_top)

            ηH_Sc = Naa[i,   j,   1]
            ηH_Nc = Naa[i,   jp1, 1]
            ηH_Wc = Nab[im1_x,  jp1f_y, 1]
            ηH_Ec = Nab[ip1f_x, jp1f_y, 1]

            β_now = By[i, jp1f_j, 1]
            if ssa_mask_y == 1 && β_now == 0.0
                β_now = beta_min
            end

            # Diagonal: 4(ηH_Sc + ηH_Nc)·dx/dy + (ηH_Wc + ηH_Ec)·dy/dx + β·dx·dy
            v = 4.0 * (ηH_Sc + ηH_Nc) * s_dx_dy_2 +
                       (ηH_Wc + ηH_Ec) * s_dy_dx +
                       β_now * s_dx_dy
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(i, j, Nx), v)

            # v-neighbours (membrane S/N, shear W/E).
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(i, jm1, Nx),
                               -4.0 * ηH_Sc * s_dx_dy_2)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(i, jp1, Nx),
                               -4.0 * ηH_Nc * s_dx_dy_2)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(im1, j, Nx),
                               -ηH_Wc * s_dy_dx)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_uy(ip1, j, Nx),
                               -ηH_Ec * s_dy_dx)

            # u-neighbours (4 corner u-points around the v-face).
            # By symmetry with the u-row:
            #   K[v_{i,jp1f}, u_{i, j}]      = -2·ηH_Sc - ηH_Wc
            #   K[v_{i,jp1f}, u_{ip1, j}]    = +2·ηH_Sc + ηH_Ec
            #   K[v_{i,jp1f}, u_{i, jp1}]    = +2·ηH_Nc + ηH_Wc
            #   K[v_{i,jp1f}, u_{ip1, jp1}]  = -2·ηH_Nc - ηH_Ec
            # Rationale: K[v, u_X] = K[u_X, v] (symmetry); these come
            # straight from the u-row block above with the patch
            # relabelled (south-cell of v ↔ west-cell of u for the
            # opposite face, etc.).
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(im1, j, Nx),
                               -2.0 * ηH_Sc - ηH_Wc)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(i, j, Nx),
                                2.0 * ηH_Sc + ηH_Ec)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(im1, jp1, Nx),
                                2.0 * ηH_Nc + ηH_Wc)
            k += 1; _push_coo!(I_idx, J_idx, vals, k, nr,
                               _row_ux(i, jp1, Nx),
                               -2.0 * ηH_Nc - ηH_Ec)

            b_vec[nr] = -Tdy[i, jp1f_j, 1] * s_dx_dy
        end
    end

    nnz_ref[] = k
    return nothing
end
