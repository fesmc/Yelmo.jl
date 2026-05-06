# ----------------------------------------------------------------------
# Horizontal advection precompute for the implicit thermo solver.
#
# Direct port of Fortran `calc_advec_horizontal_3D` +
# `calc_advec_horizontal_column` (`physics/thermodynamics.f90:254-463`).
# Pre-computes a 3D `advecxy` field that the column solver consumes
# explicitly in its RHS so each column solve is independent.
#
# Per-cell scheme is 1st-order upwind, with a "convergent" branch
# averaging upwind+downwind contributions when ux(i-1)>0 and
# ux(i)<0 (or analogous for y). Boundary cells (i = 1, j = 1, i = Nx,
# j = Ny) are zeroed; Fortran's `set_boundaries_3D_aa` post-pass also
# zeros them, so we just leave them at the initial fill.
#
# Topology (Bounded vs Periodic) controlled via the dyn-borrowed
# helpers; the kernel is parametric in the topology types so neighbour
# index lookups fold at compile time.
# ----------------------------------------------------------------------

"""
    calc_advec_horizontal_3D!(advecxy_field, var_field, ux_field, uy_field, dx)
        -> advecxy_field

Fill `advecxy_field` (3D, K/yr) with the 1st-order upwind horizontal
advection of `var_field` (e.g., `T_ice` for the temp solver, `enth`
for the enth solver). Boundary (i = 1, j = 1, i = Nx, j = Ny) cells
are zeroed.
"""
function calc_advec_horizontal_3D!(advecxy_field, var_field,
                                   ux_field, uy_field, dx::Real)
    adv_d = advecxy_field.data
    var_d = var_field.data
    ux_d  = ux_field.data
    uy_d  = uy_field.data
    Tx    = topology(advecxy_field.grid, 1)
    Ty    = topology(advecxy_field.grid, 2)
    Nx    = advecxy_field.grid.Nx
    Ny    = advecxy_field.grid.Ny
    Nz    = advecxy_field.grid.Nz
    return _calc_advec_horizontal_3D_kernel!(adv_d, var_d, ux_d, uy_d,
                                             Float64(dx),
                                             Tx, Ty, Nx, Ny, Nz)
end

function _calc_advec_horizontal_3D_kernel!(adv, var, ux, uy,
                                           dx::Float64,
                                           ::Type{Tx_top}, ::Type{Ty_top},
                                           Nx::Int, Ny::Int, Nz::Int,
                                           ) where {Tx_top<:AbstractTopology,
                                                    Ty_top<:AbstractTopology}
    dx_inv = 1.0 / dx

    # Zero everything first (the boundary loop in Fortran's set_boundaries_3D_aa
    # zeros boundary cells; here we zero everywhere then fill the interior).
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        adv[i, j, k] = 0.0
    end

    @inbounds for j in 2:(Ny - 1), i in 2:(Nx - 1)
        # Topology-aware neighbour indices.
        im1 = _neighbor_im1(i, Nx, Tx_top)
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jm1 = _neighbor_jm1(j, Ny, Ty_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        ux_im1_y = _ip1_modular(im1, Nx, Tx_top)
        ux_i_y   = _ip1_modular(i,   Nx, Tx_top)
        uy_jm1_y = _jp1_modular(jm1, Ny, Ty_top)
        uy_j_y   = _jp1_modular(j,   Ny, Ty_top)

        for k in 1:Nz
            # Yelmo XFaceField: Fortran `ux(i, j, k)` is the eastern face
            # of cell (i, j) → Yelmo index `ux[ux_i_y, j, k]`.
            ux_e = ux[ux_i_y,   j, k]   # Fortran ux(i, j, k)
            ux_w = ux[ux_im1_y, j, k]   # Fortran ux(im1, j, k)
            uy_n = uy[i, uy_j_y,   k]   # Fortran uy(i, j, k)
            uy_s = uy[i, uy_jm1_y, k]   # Fortran uy(i, jm1, k)
            ux_aa = 0.5 * (ux_e + ux_w)
            uy_aa = 0.5 * (uy_n + uy_s)

            # --- x-advection ---
            local advecx::Float64
            if ux_w > 0.0 && ux_e < 0.0 && i >= 3 && i <= Nx - 2
                advx_left  = dx_inv * ux_w * (-(var[im1, j, k] - var[i, j, k]))
                advx_right = dx_inv * ux_e * (var[ip1, j, k]  - var[i, j, k])
                advecx = 0.5 * (advx_left + advx_right)
            elseif ux_aa > 0.0 && i >= 3
                advecx = dx_inv * ux_w * (-(var[im1, j, k] - var[i, j, k]))
            elseif ux_aa > 0.0 && i == 2
                advecx = dx_inv * ux_w * (-(var[im1, j, k] - var[i, j, k]))
            elseif ux_aa < 0.0 && i <= Nx - 2
                advecx = dx_inv * ux_e * (var[ip1, j, k] - var[i, j, k])
            elseif ux_aa < 0.0 && i == Nx - 1
                advecx = dx_inv * ux_e * (var[ip1, j, k] - var[i, j, k])
            else
                advecx = 0.0
            end

            # --- y-advection ---
            local advecy::Float64
            if uy_s > 0.0 && uy_n < 0.0 && j >= 3 && j <= Ny - 2
                advy_left  = dx_inv * uy_s * (-(var[i, jm1, k] - var[i, j, k]))
                advy_right = dx_inv * uy_n * (var[i, jp1, k]  - var[i, j, k])
                advecy = 0.5 * (advy_left + advy_right)
            elseif uy_aa > 0.0 && j >= 3
                advecy = dx_inv * uy_s * (-(var[i, jm1, k] - var[i, j, k]))
            elseif uy_aa > 0.0 && j == 2
                advecy = dx_inv * uy_s * (-(var[i, jm1, k] - var[i, j, k]))
            elseif uy_aa < 0.0 && j <= Ny - 2
                advecy = dx_inv * uy_n * (var[i, jp1, k] - var[i, j, k])
            elseif uy_aa < 0.0 && j == Ny - 1
                advecy = dx_inv * uy_n * (var[i, jp1, k] - var[i, j, k])
            else
                advecy = 0.0
            end

            adv[i, j, k] = advecx + advecy
        end
    end
    return nothing
end
