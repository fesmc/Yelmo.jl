# ----------------------------------------------------------------------
# Basal frictional heating Q_b (mW m^-2).
#
# Two implementations matching the Fortran `qb_method` choices:
#
#   - `qb_method = 1` ("aa")    : `calc_basal_heating_simplestagger!` —
#       single aa-node value per cell, computed from the average of
#       the four surrounding ac-x / ac-y face stresses and velocities.
#       Boundary handling is Fortran-style clamped (`im1 = max(i-1,1)`).
#   - `qb_method = 2` ("nodes") : `calc_basal_heating_nodes!` —
#       2-point Gauss–Legendre quadrature on the cell's four ab-corners.
#       Default and best choice per the Fortran header.
#
# Both forms produce `Q_b` in [mW m^-2] (multiplied by `1e3 / sec_year`
# from the natural [J a-1 m-2] units), zero outside ice-covered cells,
# clamped to ≥ 0.
#
# Forward-Euler weighting (beta1 = 1, beta2 = 0) — fully overwritten
# per call. Same Fortran convention as `strain_heating.jl`.
#
# `calc_basal_heating_nodes!` reuses the dyn module's
# `gq2d_nodes_2pt` / `gq2d_interp_to_node` helpers and the
# topology-aware neighbour helpers in `dyn/topology_helpers.jl`. These
# logically belong in a shared `src/utils/` module — moving them is
# left for a follow-up cleanup.
# ----------------------------------------------------------------------

"""
    calc_basal_heating_simplestagger!(Q_b_field, ux_b_field, uy_b_field,
                                      taub_acx_field, taub_acy_field,
                                      sec_year) -> Q_b_field

Simple-stagger basal frictional heating (qb_method = 1). Direct port
of Fortran `calc_basal_heating_simplestagger`
(`thermodynamics.f90:794`).

Per-cell formula:

    uxy_aa  = sqrt( (0.5*(ux(i,j) + ux(im1, j)))^2 +
                    (0.5*(uy(i,j) + uy(i, jm1)))^2 )
    taub_aa = sqrt( (0.5*(taub_acx(i,j) + taub_acx(im1, j)))^2 +
                    (0.5*(taub_acy(i,j) + taub_acy(i, jm1)))^2 )
    Q_b     = uxy_aa * taub_aa * 1e3 / sec_year     [mW m^-2]

Boundary cells use Fortran-style clamping (`im1 = max(i-1, 1)`).
Negatives clamped to zero.
"""
function calc_basal_heating_simplestagger!(Q_b_field,
                                           ux_b_field, uy_b_field,
                                           taub_acx_field, taub_acy_field,
                                           sec_year::Real)
    Q_d  = Q_b_field.data
    ux_d = ux_b_field.data
    uy_d = uy_b_field.data
    tx_d = taub_acx_field.data
    ty_d = taub_acy_field.data
    Nx   = Q_b_field.grid.Nx
    Ny   = Q_b_field.grid.Ny
    return _calc_basal_heating_simplestagger_kernel!(
        Q_d, ux_d, uy_d, tx_d, ty_d, Float64(sec_year), Nx, Ny)
end

function _calc_basal_heating_simplestagger_kernel!(Q, ux, uy, tx, ty,
                                                    sec_year::Float64,
                                                    Nx::Int, Ny::Int)
    inv_sec_year_mW = 1e3 / sec_year
    @inbounds for j in 1:Ny, i in 1:Nx
        # Fortran clamps boundary indices.
        im1 = max(i - 1, 1)
        jm1 = max(j - 1, 1)
        # Yelmo XFaceField/YFaceField shift: Fortran `ux(i, j)` (face
        # east of cell (i, j)) is `ux[i+1, j, 1]` in Yelmo. Fortran
        # `ux(im1, j)` is `ux[im1+1, j, 1]` = `ux[i, j, 1]`.
        ux_i   = ux[i + 1, j,     1]
        ux_im1 = ux[im1 + 1, j,   1]
        uy_j   = uy[i,     j + 1, 1]
        uy_jm1 = uy[i,     jm1 + 1, 1]

        tx_i   = tx[i + 1, j,     1]
        tx_im1 = tx[im1 + 1, j,   1]
        ty_j   = ty[i,     j + 1, 1]
        ty_jm1 = ty[i,     jm1 + 1, 1]

        uxy_aa  = sqrt((0.5 * (ux_i + ux_im1))^2 +
                       (0.5 * (uy_j + uy_jm1))^2)
        taub_aa = sqrt((0.5 * (tx_i + tx_im1))^2 +
                       (0.5 * (ty_j + ty_jm1))^2)

        Q_now = abs(uxy_aa * taub_aa) * inv_sec_year_mW
        Q[i, j, 1] = Q_now < 0.0 ? 0.0 : Q_now
    end
    return nothing
end

"""
    calc_basal_heating_nodes!(Q_b_field, ux_b_field, uy_b_field,
                              taub_acx_field, taub_acy_field, f_ice_field,
                              sec_year) -> Q_b_field

Gauss–Legendre 2-point quadrature basal frictional heating
(qb_method = 2, default). Direct port of Fortran
`calc_basal_heating_nodes` (`thermodynamics.f90:706`).

Per ice-covered cell:

  1. Interpolate `ux_b` (acx-staggered), `uy_b` (acy-staggered),
     `taub_acx`, `taub_acy` from their face stagger to the four
     ab-corners of cell (i, j) (Fortran `gq2D_to_nodes_acx/acy`),
     averaging two adjacent ac-stagger values per corner.
  2. Map the four corner values to the four GQ nodes via the
     bilinear shape functions (`gq2d_interp_to_node`).
  3. At each node compute `Qbn = |sqrt(ux²+uy²) * sqrt(taux²+tauy²)|`.
  4. Weighted-average back to the aa-cell-centre and convert to
     mW m^-2 via `* 1e3 / sec_year`.

Cells with `f_ice < 1` are zeroed (Fortran else branch). Topology
(periodic / bounded) is honoured via the `_neighbor_*` helpers
imported from `..YelmoModelDyn`.
"""
function calc_basal_heating_nodes!(Q_b_field,
                                   ux_b_field, uy_b_field,
                                   taub_acx_field, taub_acy_field,
                                   f_ice_field,
                                   sec_year::Real)
    Q_d   = Q_b_field.data
    ux_d  = ux_b_field.data
    uy_d  = uy_b_field.data
    tx_d  = taub_acx_field.data
    ty_d  = taub_acy_field.data
    fi_d  = f_ice_field.data
    Tx    = topology(Q_b_field.grid, 1)
    Ty    = topology(Q_b_field.grid, 2)
    Nx    = Q_b_field.grid.Nx
    Ny    = Q_b_field.grid.Ny
    return _calc_basal_heating_nodes_kernel!(
        Q_d, ux_d, uy_d, tx_d, ty_d, fi_d, Float64(sec_year),
        Tx, Ty, Nx, Ny)
end

function _calc_basal_heating_nodes_kernel!(Q, ux, uy, tx, ty, fi,
                                           sec_year::Float64,
                                           ::Type{Tx_top}, ::Type{Ty_top},
                                           Nx::Int, Ny::Int,
                                           ) where {Tx_top<:AbstractTopology,
                                                    Ty_top<:AbstractTopology}
    xr, yr, wt, wt_tot = gq2d_nodes_2pt()
    inv_sec_year_mW = 1e3 / sec_year

    @inbounds for j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] != 1.0
            Q[i, j, 1] = 0.0
            continue
        end

        # Topology-aware neighbours (matches dyn basal_dragging
        # convention).
        im1 = _neighbor_im1(i, Nx, Tx_top)
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jm1 = _neighbor_jm1(j, Ny, Ty_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)

        # Yelmo face-array indices for "Fortran ux(k, j)".
        ux_im1_y = _ip1_modular(im1, Nx, Tx_top)
        ux_i_y   = _ip1_modular(i,   Nx, Tx_top)
        ux_ip1_y = _ip1_modular(ip1, Nx, Tx_top)
        uy_jm1_y = _jp1_modular(jm1, Ny, Ty_top)
        uy_j_y   = _jp1_modular(j,   Ny, Ty_top)
        uy_jp1_y = _jp1_modular(jp1, Ny, Ty_top)

        # Stagger ux_b (acx) → 4 ab-corners around cell (i, j) by
        # averaging two adjacent acx values per corner. Same shape as
        # `_calc_beta_aa_power_plastic!` in dyn/basal_dragging.jl.
        v_ab_ux = (
            0.5 * (ux[ux_im1_y, jm1, 1] + ux[ux_im1_y, j,   1]),  # SW
            0.5 * (ux[ux_i_y,   jm1, 1] + ux[ux_i_y,   j,   1]),  # SE
            0.5 * (ux[ux_i_y,   j,   1] + ux[ux_i_y,   jp1, 1]),  # NE
            0.5 * (ux[ux_im1_y, j,   1] + ux[ux_im1_y, jp1, 1]),  # NW
        )
        v_ab_uy = (
            0.5 * (uy[im1, uy_jm1_y, 1] + uy[i,   uy_jm1_y, 1]),  # SW
            0.5 * (uy[i,   uy_jm1_y, 1] + uy[ip1, uy_jm1_y, 1]),  # SE
            0.5 * (uy[i,   uy_j_y,   1] + uy[ip1, uy_j_y,   1]),  # NE
            0.5 * (uy[im1, uy_j_y,   1] + uy[i,   uy_j_y,   1]),  # NW
        )
        v_ab_tx = (
            0.5 * (tx[ux_im1_y, jm1, 1] + tx[ux_im1_y, j,   1]),
            0.5 * (tx[ux_i_y,   jm1, 1] + tx[ux_i_y,   j,   1]),
            0.5 * (tx[ux_i_y,   j,   1] + tx[ux_i_y,   jp1, 1]),
            0.5 * (tx[ux_im1_y, j,   1] + tx[ux_im1_y, jp1, 1]),
        )
        v_ab_ty = (
            0.5 * (ty[im1, uy_jm1_y, 1] + ty[i,   uy_jm1_y, 1]),
            0.5 * (ty[i,   uy_jm1_y, 1] + ty[ip1, uy_jm1_y, 1]),
            0.5 * (ty[i,   uy_j_y,   1] + ty[ip1, uy_j_y,   1]),
            0.5 * (ty[im1, uy_j_y,   1] + ty[i,   uy_j_y,   1]),
        )

        # Suppress unused warning under Bounded topology where ip1/jp1
        # may collapse.
        _ = ux_ip1_y; _ = uy_jp1_y

        # Map ab-corner values to the four 2-point GQ nodes.
        uxn = ntuple(p -> gq2d_interp_to_node(v_ab_ux, xr[p], yr[p]), 4)
        uyn = ntuple(p -> gq2d_interp_to_node(v_ab_uy, xr[p], yr[p]), 4)
        txn = ntuple(p -> gq2d_interp_to_node(v_ab_tx, xr[p], yr[p]), 4)
        tyn = ntuple(p -> gq2d_interp_to_node(v_ab_ty, xr[p], yr[p]), 4)

        # Per-node Qb and weighted average.
        Qb_aa = 0.0
        for p in 1:4
            qbp   = abs(sqrt(uxn[p]^2 + uyn[p]^2) *
                        sqrt(txn[p]^2 + tyn[p]^2))
            Qb_aa += qbp * wt[p]
        end
        Qb_aa /= wt_tot
        Qb_aa = Qb_aa < 0.0 ? 0.0 : Qb_aa
        Q[i, j, 1] = Qb_aa * inv_sec_year_mW
    end
    return nothing
end
