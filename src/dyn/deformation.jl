# ----------------------------------------------------------------------
# Deformation kernels.
#
# Strain-rate tensor helpers and the velocity-Jacobian routines. The
# Yelmo.jl `dyn` module owns all kinematic / deformation quantities
# derived from the velocity field — strain rate (`strn`, `strn2D`),
# velocity Jacobian (`jvel`), vertical velocity (`uz`, `uz_star`).
# This deviates from Fortran Yelmo where these live in `mat`; we
# follow data flow rather than the Fortran organisation. The 3D
# viscosity / Glen flow law / rate factor stay in `mat` (when ported).
#
# Routines in this file:
#
#   - `_calc_strain_rate_horizontal_2D!` — depth-averaged strain-rate
#     derivatives (`dudx, dudy, dvdx, dvdy`) at aa-cells. Used by the
#     SSA `calc_visc_eff_3D_nodes!` driver in `viscosity.jl` for its
#     Gauss-quadrature evaluation. Port of `deformation.f90:1667
#     calc_strain_rate_horizontal_2D`.
#
#   - `calc_jacobian_vel_3D_uxyterms!` — first phase of the velocity
#     Jacobian; populates `jvel.{dxx, dxy, dxz, dyx, dyy, dyz}` from
#     `(ux, uy)` plus the bed/surface gradients used for the
#     sigma-coordinate correction. Does NOT touch `jvel.dz*` (those are
#     written in `_uzterms`, deferred to milestone 3h commit C3).
#     Port of `deformation.f90:508 calc_jacobian_vel_3D_uxyterms`.
#
# Staggering convention (matches Fortran "all tensor components live
# at the same location as the velocity component they differentiate"):
#
#   - `jvel.dxx`, `jvel.dxy`, `jvel.dxz` live at acx faces (where ux
#     lives). In Yelmo.jl they are stored as CenterField with slot
#     `[i, j, k]` representing "the value at acx of cell `i`" — the
#     same convention the topo module uses for `dzsdx` / `dzbdx`.
#     This avoids an XFaceField + topology-dispatched-write footprint
#     for fields that are only ever read element-wise by the strain
#     rate Gauss-quadrature consumer.
#   - `jvel.dyx`, `jvel.dyy`, `jvel.dyz` similarly live at acy and are
#     CenterField with slot `[i, j, k]` = acy of cell `j`.
#   - `jvel.dzx`, `jvel.dzy`, `jvel.dzz` live at z-faces, aa-cells
#     horizontally — ZFaceField (already in the schema).
#
# The `(ux, uy)` reads in this routine use `_ip1_modular` /
# `_jp1_modular` because those ARE XFaceField / YFaceField and follow
# the slot-`i+1`-under-Bounded face convention.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.Grids: topology, Bounded, Periodic, AbstractTopology

export calc_jacobian_vel_3D_uxyterms!,
       calc_jacobian_vel_3D_uzterms!,
       calc_strain_rate_tensor_jac_quad3D!

# ----------------------------------------------------------------------
# 2D horizontal strain-rate helper (port of deformation.f90:1667
# `calc_strain_rate_horizontal_2D`). Used by the SSA quadrature
# viscosity driver. Output is on aa-cells (Nx × Ny matrices).
#
# Centered second-order centered differences in the interior, with
# upstream/downstream second-order one-sided fixes at full-ice cells
# adjacent to partial-ice neighbours. Cross-term margin corrections
# (dudy / dvdx) are intentionally omitted — Fortran line 1760-1761:
# "do not treat cross terms as symmetry breaks down. Better to keep
# it clean."
#
# Margin-guard semantics under Periodic (`im1 > 1` / `ip1 < Nx`-style
# tests for one-cell-from-boundary detection) are not reworked here —
# HOM-C is full-ice and doesn't exercise this branch. Revisit when
# calving-driven margins land.
# ----------------------------------------------------------------------
function _calc_strain_rate_horizontal_2D!(dudx::AbstractArray, dudy::AbstractArray,
                                          dvdx::AbstractArray, dvdy::AbstractArray,
                                          ux_int::AbstractArray, uy_int::AbstractArray,
                                          fi_int::AbstractArray,
                                          dx::Float64, dy::Float64,
                                          Tx::Type{<:AbstractTopology},
                                          Ty::Type{<:AbstractTopology})
    Nx, Ny = size(dudx, 1), size(dudx, 2)

    fill!(dudx, 0.0); fill!(dudy, 0.0)
    fill!(dvdx, 0.0); fill!(dvdy, 0.0)

    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = _neighbor_im1(i, Nx, Tx)
        ip1 = _neighbor_ip1(i, Nx, Tx)
        jm1 = _neighbor_jm1(j, Ny, Ty)
        jp1 = _neighbor_jp1(j, Ny, Ty)

        ux_i   = _ip1_modular(i,   Nx, Tx)
        ux_im1 = _ip1_modular(im1, Nx, Tx)
        ux_ip1 = _ip1_modular(ip1, Nx, Tx)
        uy_j   = _jp1_modular(j,   Ny, Ty)
        uy_jm1 = _jp1_modular(jm1, Ny, Ty)
        uy_jp1 = _jp1_modular(jp1, Ny, Ty)

        # Centered, second-order.
        dudx[i, j] = (ux_int[ux_ip1, j, 1] - ux_int[ux_im1, j, 1]) / (2 * dx)
        dudy[i, j] = (ux_int[ux_i,   jp1, 1] - ux_int[ux_i, jm1, 1]) / (2 * dy)
        dvdx[i, j] = (uy_int[ip1, uy_j, 1]   - uy_int[im1, uy_j, 1]) / (2 * dx)
        dvdy[i, j] = (uy_int[i,   uy_jp1, 1] - uy_int[i,   uy_jm1, 1]) / (2 * dy)

        # dudx margin (Fortran line 1721-1739).
        if fi_int[i, j, 1] == 1.0 && fi_int[ip1, j, 1] < 1.0
            if fi_int[im1, j, 1] == 1.0 && im1 > 1
                im2 = im1 - 1
                ux_im2 = _ip1_modular(im2, Nx, Tx)
                dudx[i, j] = (1 * ux_int[ux_im2, j, 1] - 4 * ux_int[ux_im1, j, 1] +
                              3 * ux_int[ux_i,   j, 1]) / (2 * dx)
            else
                dudx[i, j] = (ux_int[ux_i, j, 1] - ux_int[ux_im1, j, 1]) / dx
            end
        elseif fi_int[i, j, 1] < 1.0 && fi_int[ip1, j, 1] == 1.0
            if ip1 < Nx
                ip2 = ip1 + 1
                if fi_int[ip2, j, 1] == 1.0
                    ux_ip2 = _ip1_modular(ip2, Nx, Tx)
                    dudx[i, j] = -(1 * ux_int[ux_ip2, j, 1] - 4 * ux_int[ux_ip1, j, 1] +
                                   3 * ux_int[ux_i,   j, 1]) / (2 * dx)
                else
                    dudx[i, j] = (ux_int[ux_ip1, j, 1] - ux_int[ux_i, j, 1]) / dx
                end
            else
                dudx[i, j] = (ux_int[ux_ip1, j, 1] - ux_int[ux_i, j, 1]) / dx
            end
        end

        # dvdy margin (Fortran line 1742-1758).
        if fi_int[i, j, 1] == 1.0 && fi_int[i, jp1, 1] < 1.0
            if fi_int[i, jm1, 1] == 1.0 && jm1 > 1
                jm2 = jm1 - 1
                uy_jm2 = _jp1_modular(jm2, Ny, Ty)
                dvdy[i, j] = (1 * uy_int[i, uy_jm2, 1] - 4 * uy_int[i, uy_jm1, 1] +
                              3 * uy_int[i, uy_j,   1]) / (2 * dy)
            else
                dvdy[i, j] = (uy_int[i, uy_j, 1] - uy_int[i, uy_jm1, 1]) / dy
            end
        elseif fi_int[i, j, 1] < 1.0 && fi_int[i, jp1, 1] == 1.0
            if jp1 < Ny
                jp2 = jp1 + 1
                if fi_int[i, jp2, 1] == 1.0
                    uy_jp2 = _jp1_modular(jp2, Ny, Ty)
                    dvdy[i, j] = -(1 * uy_int[i, uy_jp2, 1] - 4 * uy_int[i, uy_jp1, 1] +
                                   3 * uy_int[i, uy_j,   1]) / (2 * dy)
                else
                    dvdy[i, j] = (uy_int[i, uy_jp1, 1] - uy_int[i, uy_j, 1]) / dy
                end
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------------
# Velocity Jacobian — xy-row terms.
# ----------------------------------------------------------------------
"""
    calc_jacobian_vel_3D_uxyterms!(
        jvel_dxx, jvel_dxy, jvel_dxz,
        jvel_dyx, jvel_dyy, jvel_dyz,
        ux, uy,
        H_ice, f_ice,
        dzsdx, dzsdy, dzbdx, dzbdy,
        zeta_aa, dx, dy
    ) -> nothing

Populate the xy-row Jacobian components from horizontal velocity
fields. Writes:

  - `dxx`, `dxy`, `dxz` — at acx of cell `i` (slot `[i, j, k]` of the
    CenterField).
  - `dyx`, `dyy`, `dyz` — at acy of cell `j` (slot `[i, j, k]`).

`dx*` / `dy*` slots `[i, j, k]` read as "value at acx (resp. acy) of
cell `i` (resp. `j`)" — the same convention `dzsdx` / `dzbdx` use.

Two phases:

  1. Vertical derivatives `dxz`, `dyz` from the 3-point Lagrange
     formula for unequal layer spacing (Singh & Bhadauria), with
     forward/centered/backward stencils at the bottom/interior/top
     respectively. `H_now_ac{x,y}` is a margin-aware face-staggered
     ice thickness.
  2. Horizontal derivatives `dxx`, `dxy`, `dyx`, `dyy` via centered
     second-order FD, with one-sided second-order fixes at ice-front
     cells. Sigma-coordinate transform corrections
     `c_{x,y} = -((1-ζ)·dzbd{x,y} + ζ·dzsd{x,y})` are then folded in:

         dxx += c_x      · dxz
         dxy += c_y_acx  · dxz
         dyx += c_x_acy  · dyz
         dyy += c_y      · dyz

     where `c_x_acy` / `c_y_acx` are the cross-stagger 4-cell averages
     of the bed/surface gradients to the partner face.

Fortran's `if (.FALSE.) ... else` branch (simple upwind FD) and the
`corr_grad_lim` clamping branch are dead code; only the active
branches are ported. The Fortran `uz` and `f_grnd` parameters are
unused in the routine body and are dropped from this signature.
Cross-term margin fixes for `dxy` / `dyx` are intentionally omitted
(Fortran line 775-777).

Port of `yelmo/src/physics/deformation.f90:508 calc_jacobian_vel_3D_uxyterms`.
"""
function calc_jacobian_vel_3D_uxyterms!(
        jvel_dxx, jvel_dxy, jvel_dxz,
        jvel_dyx, jvel_dyy, jvel_dyz,
        ux, uy,
        H_ice, f_ice,
        dzsdx, dzsdy, dzbdx, dzbdy,
        zeta_aa::AbstractVector{<:Real},
        dx::Real, dy::Real,
    )

    Dxx = interior(jvel_dxx); Dxy = interior(jvel_dxy); Dxz = interior(jvel_dxz)
    Dyx = interior(jvel_dyx); Dyy = interior(jvel_dyy); Dyz = interior(jvel_dyz)
    UX  = interior(ux);  UY  = interior(uy)
    H   = interior(H_ice);  fi = interior(f_ice)
    DZSX = interior(dzsdx); DZSY = interior(dzsdy)
    DZBX = interior(dzbdx); DZBY = interior(dzbdy)

    Nx, Ny = size(H, 1), size(H, 2)
    Nz     = length(zeta_aa)
    Nz == size(Dxx, 3) || error(
        "calc_jacobian_vel_3D_uxyterms!: Nz mismatch — zeta_aa has length $(Nz) " *
        "but jvel_dxx has 3rd-dim size $(size(Dxx, 3))")

    Tx = topology(jvel_dxx.grid, 1)
    Ty = topology(jvel_dxx.grid, 2)

    fill!(Dxx, 0.0); fill!(Dxy, 0.0); fill!(Dxz, 0.0)
    fill!(Dyx, 0.0); fill!(Dyy, 0.0); fill!(Dyz, 0.0)

    dx_f = Float64(dx); dy_f = Float64(dy)

    # ===== Step 1: vertical derivatives `dxz`, `dyz` ========================
    # Fortran lines 587-702. 3-point Lagrange for unequal vertical layers.
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1 = _neighbor_ip1(i, Nx, Tx)
        jp1 = _neighbor_jp1(j, Ny, Ty)

        # Margin-aware staggered ice thickness at acx and acy.
        H_now_acx =
            if fi[i, j, 1] == 1.0 && fi[ip1, j, 1] < 1.0
                H[i, j, 1]
            elseif fi[i, j, 1] < 1.0 && fi[ip1, j, 1] == 1.0
                H[ip1, j, 1]
            else
                0.5 * (H[i, j, 1] + H[ip1, j, 1])
            end

        H_now_acy =
            if fi[i, j, 1] == 1.0 && fi[i, jp1, 1] < 1.0
                H[i, j, 1]
            elseif fi[i, j, 1] < 1.0 && fi[i, jp1, 1] == 1.0
                H[i, jp1, 1]
            else
                0.5 * (H[i, j, 1] + H[i, jp1, 1])
            end

        ux_i = _ip1_modular(i, Nx, Tx)
        uy_j = _jp1_modular(j, Ny, Ty)

        if H_now_acx > 0.0
            # k=1: forward 3-point Lagrange (Fortran lines 658-661).
            k = 1
            h1 = H_now_acx * (zeta_aa[k+1] - zeta_aa[k])
            h2 = H_now_acx * (zeta_aa[k+2] - zeta_aa[k+1])
            Dxz[i, j, k] = -(2*h1 + h2)/(h1*(h1 + h2)) * UX[ux_i, j, k]   +
                            (h1 + h2)/(h1*h2)         * UX[ux_i, j, k+1] -
                            h1/(h2*(h1 + h2))         * UX[ux_i, j, k+2]
            # 2..Nz-1: centered 3-point Lagrange (Fortran lines 664-668).
            for k in 2:Nz-1
                h1 = H_now_acx * (zeta_aa[k]   - zeta_aa[k-1])
                h2 = H_now_acx * (zeta_aa[k+1] - zeta_aa[k])
                Dxz[i, j, k] = -h2/(h1*(h1 + h2))    * UX[ux_i, j, k-1] -
                                (h1 - h2)/(h1*h2)   * UX[ux_i, j, k]   +
                                h1/(h2*(h1 + h2))    * UX[ux_i, j, k+1]
            end
            # k=Nz: backward 3-point Lagrange (Fortran lines 670-674).
            k = Nz
            h1 = H_now_acx * (zeta_aa[k-1] - zeta_aa[k-2])
            h2 = H_now_acx * (zeta_aa[k]   - zeta_aa[k-1])
            Dxz[i, j, k] = h2/(h1*(h1 + h2))      * UX[ux_i, j, k-2] -
                            (h1 + h2)/(h1*h2)    * UX[ux_i, j, k-1] +
                            (h1 + 2*h2)/(h2*(h1 + h2)) * UX[ux_i, j, k]
        end

        if H_now_acy > 0.0
            k = 1
            h1 = H_now_acy * (zeta_aa[k+1] - zeta_aa[k])
            h2 = H_now_acy * (zeta_aa[k+2] - zeta_aa[k+1])
            Dyz[i, j, k] = -(2*h1 + h2)/(h1*(h1 + h2)) * UY[i, uy_j, k]   +
                            (h1 + h2)/(h1*h2)         * UY[i, uy_j, k+1] -
                            h1/(h2*(h1 + h2))         * UY[i, uy_j, k+2]
            for k in 2:Nz-1
                h1 = H_now_acy * (zeta_aa[k]   - zeta_aa[k-1])
                h2 = H_now_acy * (zeta_aa[k+1] - zeta_aa[k])
                Dyz[i, j, k] = -h2/(h1*(h1 + h2))    * UY[i, uy_j, k-1] -
                                (h1 - h2)/(h1*h2)   * UY[i, uy_j, k]   +
                                h1/(h2*(h1 + h2))    * UY[i, uy_j, k+1]
            end
            k = Nz
            h1 = H_now_acy * (zeta_aa[k-1] - zeta_aa[k-2])
            h2 = H_now_acy * (zeta_aa[k]   - zeta_aa[k-1])
            Dyz[i, j, k] = h2/(h1*(h1 + h2))      * UY[i, uy_j, k-2] -
                            (h1 + h2)/(h1*h2)    * UY[i, uy_j, k-1] +
                            (h1 + 2*h2)/(h2*(h1 + h2)) * UY[i, uy_j, k]
        end
    end

    # ===== Step 2: horizontal derivatives + sigma corrections ===============
    # Fortran lines 704-833.
    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = _neighbor_im1(i, Nx, Tx); ip1 = _neighbor_ip1(i, Nx, Tx)
        jm1 = _neighbor_jm1(j, Ny, Ty); jp1 = _neighbor_jp1(j, Ny, Ty)

        ux_i   = _ip1_modular(i,   Nx, Tx)
        ux_im1 = _ip1_modular(im1, Nx, Tx)
        ux_ip1 = _ip1_modular(ip1, Nx, Tx)
        uy_j   = _jp1_modular(j,   Ny, Ty)
        uy_jm1 = _jp1_modular(jm1, Ny, Ty)
        uy_jp1 = _jp1_modular(jp1, Ny, Ty)

        for k in 1:Nz
            zk = zeta_aa[k]

            # ---- Centered second-order FD (Fortran lines 719-723) ----
            Dxx[i, j, k] = (UX[ux_ip1, j, k] - UX[ux_im1, j, k]) / (2 * dx_f)
            Dxy[i, j, k] = (UX[ux_i, jp1, k] - UX[ux_i, jm1, k]) / (2 * dy_f)
            Dyx[i, j, k] = (UY[ip1, uy_j, k] - UY[im1, uy_j, k]) / (2 * dx_f)
            Dyy[i, j, k] = (UY[i, uy_jp1, k] - UY[i, uy_jm1, k]) / (2 * dy_f)

            # ---- One-sided margin fixes for dxx (Fortran lines 728-750) ----
            if fi[i, j, 1] == 1.0 && fi[ip1, j, 1] < 1.0
                if im1 > 1 && fi[im1, j, 1] == 1.0
                    im2 = im1 - 1
                    ux_im2 = _ip1_modular(im2, Nx, Tx)
                    Dxx[i, j, k] = (1 * UX[ux_im2, j, k] - 4 * UX[ux_im1, j, k] +
                                    3 * UX[ux_i, j, k]) / (2 * dx_f)
                else
                    Dxx[i, j, k] = (UX[ux_i, j, k] - UX[ux_im1, j, k]) / dx_f
                end
            elseif fi[i, j, 1] < 1.0 && fi[ip1, j, 1] == 1.0
                if ip1 < Nx
                    ip2 = ip1 + 1
                    if fi[ip2, j, 1] == 1.0
                        ux_ip2 = _ip1_modular(ip2, Nx, Tx)
                        Dxx[i, j, k] = -(1 * UX[ux_ip2, j, k] - 4 * UX[ux_ip1, j, k] +
                                         3 * UX[ux_i, j, k]) / (2 * dx_f)
                    else
                        Dxx[i, j, k] = (UX[ux_ip1, j, k] - UX[ux_i, j, k]) / dx_f
                    end
                else
                    Dxx[i, j, k] = (UX[ux_ip1, j, k] - UX[ux_i, j, k]) / dx_f
                end
            end

            # ---- One-sided margin fixes for dyy (Fortran lines 753-773) ----
            # Note: Fortran has a quirk on line 757 where the test is
            # `f_ice(i,jm1) == 1.0` rather than `f_ice(i,jm2) == 1.0` —
            # likely a typo, but porting faithfully (the `else` branch
            # is harmless when triggered).
            if fi[i, j, 1] == 1.0 && fi[i, jp1, 1] < 1.0
                if jm1 > 1 && fi[i, jm1, 1] == 1.0
                    jm2 = jm1 - 1
                    uy_jm2 = _jp1_modular(jm2, Ny, Ty)
                    Dyy[i, j, k] = (1 * UY[i, uy_jm2, k] - 4 * UY[i, uy_jm1, k] +
                                    3 * UY[i, uy_j, k]) / (2 * dy_f)
                else
                    Dyy[i, j, k] = (UY[i, uy_j, k] - UY[i, uy_jm1, k]) / dy_f
                end
            elseif fi[i, j, 1] < 1.0 && fi[i, jp1, 1] == 1.0
                if jp1 < Ny
                    jp2 = jp1 + 1
                    if fi[i, jp2, 1] == 1.0
                        uy_jp2 = _jp1_modular(jp2, Ny, Ty)
                        Dyy[i, j, k] = -(1 * UY[i, uy_jp2, k] - 4 * UY[i, uy_jp1, k] +
                                         3 * UY[i, uy_j, k]) / (2 * dy_f)
                    else
                        Dyy[i, j, k] = (UY[i, uy_jp1, k] - UY[i, uy_j, k]) / dy_f
                    end
                end
                # NB: Fortran has no else here — Dyy keeps the centered value.
            end

            # ---- Sigma-coordinate transform corrections ----
            # Greve & Blatter (2009) Eqs. 5.131-5.132. The H_ice factor is
            # absorbed by the fact that `dxz` is already in z-units (m/yr per
            # m), not zeta-units — see Fortran comment lines 787-790.
            #
            # `c_x` at acx of cell i: read dzbdx, dzsdx (CenterField stored
            # at acx-stagger) at slot [i, j, 1].
            c_x = -((1.0 - zk) * DZBX[i, j, 1] + zk * DZSX[i, j, 1])
            c_y = -((1.0 - zk) * DZBY[i, j, 1] + zk * DZSY[i, j, 1])

            # Cross-corrections — `c_x` averaged to acy and `c_y` to acx.
            # Fortran lines 796-802.
            dzbdx_acy = 0.25 * (DZBX[i, j, 1] + DZBX[i, jp1, 1] +
                                DZBX[im1, j, 1] + DZBX[im1, jp1, 1])
            dzsdx_acy = 0.25 * (DZSX[i, j, 1] + DZSX[i, jp1, 1] +
                                DZSX[im1, j, 1] + DZSX[im1, jp1, 1])
            c_x_acy = -((1.0 - zk) * dzbdx_acy + zk * dzsdx_acy)

            dzbdy_acx = 0.25 * (DZBY[i, j, 1] + DZBY[ip1, j, 1] +
                                DZBY[i, jm1, 1] + DZBY[ip1, jm1, 1])
            dzsdy_acx = 0.25 * (DZSY[i, j, 1] + DZSY[ip1, j, 1] +
                                DZSY[i, jm1, 1] + DZSY[ip1, jm1, 1])
            c_y_acx = -((1.0 - zk) * dzbdy_acx + zk * dzsdy_acx)

            # Apply corrections (Fortran lines 823-827). dxz / dyz live at
            # the same horizontal stagger as dxx / dyy here (CenterField
            # slot [i, j, k]), so no slot remap.
            Dxx[i, j, k] += c_x     * Dxz[i, j, k]
            Dxy[i, j, k] += c_y_acx * Dxz[i, j, k]
            Dyx[i, j, k] += c_x_acy * Dyz[i, j, k]
            Dyy[i, j, k] += c_y     * Dyz[i, j, k]
        end
    end

    return nothing
end

# ----------------------------------------------------------------------
# Velocity Jacobian — z-row terms.
# ----------------------------------------------------------------------
"""
    calc_jacobian_vel_3D_uzterms!(
        jvel_dzx, jvel_dzy, jvel_dzz,
        uz,
        H_ice, f_ice,
        dzsdx, dzsdy, dzbdx, dzbdy,
        zeta_ac, dx, dy
    ) -> nothing

Populate the z-row Jacobian components from the vertical-velocity
field. Writes:

  - `dzz` — `∂uz/∂z` at aa-cells horizontally, `zeta_ac` faces
    vertically. ZFaceField slot `[i, j, k]` is the value at aa-cell
    `(i, j)` and zeta_ac level k.
  - `dzx`, `dzy` — `∂uz/∂x`, `∂uz/∂y` at the same stagger.

Two phases:

  1. `dzz` via 3-point Lagrange formula for unequal layer spacing
     (Singh & Bhadauria), with forward stencil at the bed face,
     centered in the interior, and a simple downwind formula
     `(uz[k] − uz[k−1]) / (H·Δζ_ac)` at the surface (the Fortran's
     3-point downwind formula was found "broken" — the active code
     uses the simple downwind, see `deformation.f90:983-984`).
  2. `dzx`, `dzy` via centered second-order FD of `uz` horizontally
     with one-sided second-order fixes at ice-front cells. Sigma-
     coordinate transform corrections use the aa-cell averages
     `dzbdx_aa = ½·(dzbdx[i, j] + dzbdx[im1, j])` (the two acx faces
     around aa-cell `i`) and analogously for the other gradients,
     evaluated at `zeta_ac[k]` rather than `zeta_aa[k]`. Then:

         dzx += c_x · dzz
         dzy += c_y · dzz

This routine should be called AFTER `calc_uz_3D_jac!` has populated
`uz` for the current iteration.

The Fortran `ux`, `uy`, `f_grnd`, `boundaries` parameters are unused
in the routine body and dropped. `dz*` margin one-sided corrections
follow the "ice front" pattern (full ice cell adjacent to a partial
neighbour), with both-sides-partial yielding `dzx = 0` (Fortran line
1022) — *unlike* the uxy-terms `dxx` margin which always uses one of
the available sides.

Port of `yelmo/src/physics/deformation.f90:843 calc_jacobian_vel_3D_uzterms`.
"""
function calc_jacobian_vel_3D_uzterms!(
        jvel_dzx, jvel_dzy, jvel_dzz,
        uz,
        H_ice, f_ice,
        dzsdx, dzsdy, dzbdx, dzbdy,
        zeta_ac::AbstractVector{<:Real},
        dx::Real, dy::Real,
    )

    Dzx = interior(jvel_dzx); Dzy = interior(jvel_dzy); Dzz = interior(jvel_dzz)
    UZ  = interior(uz)
    H   = interior(H_ice);   fi   = interior(f_ice)
    DZSX = interior(dzsdx);  DZSY = interior(dzsdy)
    DZBX = interior(dzbdx);  DZBY = interior(dzbdy)

    Nx, Ny  = size(H, 1), size(H, 2)
    Nz_ac   = length(zeta_ac)
    Nz_ac == size(Dzz, 3) || error(
        "calc_jacobian_vel_3D_uzterms!: zeta_ac length $(Nz_ac) ≠ jvel_dzz Nz $(size(Dzz, 3))")

    Tx = topology(jvel_dzz.grid, 1)
    Ty = topology(jvel_dzz.grid, 2)

    fill!(Dzx, 0.0); fill!(Dzy, 0.0); fill!(Dzz, 0.0)

    dx_f = Float64(dx); dy_f = Float64(dy)

    # ===== Step 1: vertical derivative `dzz` (Fortran lines 918-991) =====
    @inbounds for j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] == 1.0
            H_now = H[i, j, 1]

            # Bottom face (k=1): forward 3-point Lagrange.
            k = 1
            h1 = H_now * (zeta_ac[k+1] - zeta_ac[k])
            h2 = H_now * (zeta_ac[k+2] - zeta_ac[k+1])
            Dzz[i, j, k] = -(2*h1 + h2)/(h1*(h1 + h2)) * UZ[i, j, k]   +
                            (h1 + h2)/(h1*h2)         * UZ[i, j, k+1] -
                            h1/(h2*(h1 + h2))         * UZ[i, j, k+2]

            # Interior faces (k=2..Nz_ac-1): centered 3-point Lagrange.
            for k in 2:Nz_ac-1
                h1 = H_now * (zeta_ac[k]   - zeta_ac[k-1])
                h2 = H_now * (zeta_ac[k+1] - zeta_ac[k])
                Dzz[i, j, k] = -h2/(h1*(h1 + h2))    * UZ[i, j, k-1] -
                                (h1 - h2)/(h1*h2)   * UZ[i, j, k]   +
                                h1/(h2*(h1 + h2))    * UZ[i, j, k+1]
            end

            # Top face (k=Nz_ac): simple downwind FD. Fortran's 3-point
            # downwind formula was found broken (deformation.f90:977-984
            # — formula commented out, replaced with the simple FD).
            k = Nz_ac
            Dzz[i, j, k] = (UZ[i, j, k] - UZ[i, j, k-1]) /
                           (H_now * (zeta_ac[k] - zeta_ac[k-1]))
        end
    end

    # ===== Step 2: horizontal derivatives + sigma corrections ============
    # Fortran lines 995-1097.
    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = _neighbor_im1(i, Nx, Tx); ip1 = _neighbor_ip1(i, Nx, Tx)
        jm1 = _neighbor_jm1(j, Ny, Ty); jp1 = _neighbor_jp1(j, Ny, Ty)

        if fi[i, j, 1] == 1.0
            for k in 1:Nz_ac
                # ---- Centered second-order FD (Fortran lines 1015-1016) ----
                Dzx[i, j, k] = (UZ[ip1, j, k] - UZ[im1, j, k]) / (2 * dx_f)
                Dzy[i, j, k] = (UZ[i, jp1, k] - UZ[i, jm1, k]) / (2 * dy_f)

                # ---- One-sided margin fixes for dzx (Fortran lines 1021-1045) ----
                if fi[ip1, j, 1] < 1.0 && fi[im1, j, 1] < 1.0
                    # Both x-neighbours partial — drop the gradient.
                    Dzx[i, j, k] = 0.0
                elseif fi[ip1, j, 1] < 1.0 && fi[im1, j, 1] == 1.0
                    if im1 > 1
                        im2 = im1 - 1
                        if fi[im2, j, 1] == 1.0
                            Dzx[i, j, k] = (UZ[im2, j, k] - 4 * UZ[im1, j, k] +
                                            3 * UZ[i, j, k]) / (2 * dx_f)
                        else
                            Dzx[i, j, k] = (UZ[i, j, k] - UZ[im1, j, k]) / dx_f
                        end
                    else
                        Dzx[i, j, k] = (UZ[i, j, k] - UZ[im1, j, k]) / dx_f
                    end
                elseif fi[ip1, j, 1] == 1.0 && fi[im1, j, 1] < 1.0
                    if ip1 < Nx
                        ip2 = ip1 + 1
                        if fi[ip2, j, 1] == 1.0
                            Dzx[i, j, k] = -(UZ[ip2, j, k] - 4 * UZ[ip1, j, k] +
                                             3 * UZ[i, j, k]) / (2 * dx_f)
                        else
                            Dzx[i, j, k] = (UZ[ip1, j, k] - UZ[i, j, k]) / dx_f
                        end
                    else
                        Dzx[i, j, k] = (UZ[ip1, j, k] - UZ[i, j, k]) / dx_f
                    end
                end

                # ---- One-sided margin fixes for dzy (Fortran lines 1048-1072) ----
                if fi[i, jp1, 1] < 1.0 && fi[i, jm1, 1] < 1.0
                    Dzy[i, j, k] = 0.0
                elseif fi[i, jp1, 1] < 1.0 && fi[i, jm1, 1] == 1.0
                    if jm1 > 1
                        jm2 = jm1 - 1
                        if fi[i, jm2, 1] == 1.0
                            Dzy[i, j, k] = (UZ[i, jm2, k] - 4 * UZ[i, jm1, k] +
                                            3 * UZ[i, j, k]) / (2 * dy_f)
                        else
                            Dzy[i, j, k] = (UZ[i, j, k] - UZ[i, jm1, k]) / dy_f
                        end
                    else
                        Dzy[i, j, k] = (UZ[i, j, k] - UZ[i, jm1, k]) / dy_f
                    end
                elseif fi[i, jp1, 1] == 1.0 && fi[i, jm1, 1] < 1.0
                    if jp1 < Ny
                        jp2 = jp1 + 1
                        if fi[i, jp2, 1] == 1.0
                            Dzy[i, j, k] = -(UZ[i, jp2, k] - 4 * UZ[i, jp1, k] +
                                             3 * UZ[i, j, k]) / (2 * dy_f)
                        else
                            Dzy[i, j, k] = (UZ[i, jp1, k] - UZ[i, j, k]) / dy_f
                        end
                    else
                        Dzy[i, j, k] = (UZ[i, jp1, k] - UZ[i, j, k]) / dy_f
                    end
                end

                # ---- Sigma-coordinate transform corrections ----
                # Fortran lines 1078-1089. dzbdx_aa averages the two
                # acx faces of cell i (slot [i,j,1] and [im1,j,1] under
                # the dzbdx CenterField storage convention).
                dzbdx_aa = 0.5 * (DZBX[i, j, 1] + DZBX[im1, j, 1])
                dzbdy_aa = 0.5 * (DZBY[i, j, 1] + DZBY[i, jm1, 1])
                dzsdx_aa = 0.5 * (DZSX[i, j, 1] + DZSX[im1, j, 1])
                dzsdy_aa = 0.5 * (DZSY[i, j, 1] + DZSY[i, jm1, 1])

                zk = zeta_ac[k]
                c_x = -((1.0 - zk) * dzbdx_aa + zk * dzsdx_aa)
                c_y = -((1.0 - zk) * dzbdy_aa + zk * dzsdy_aa)

                Dzx[i, j, k] += c_x * Dzz[i, j, k]
                Dzy[i, j, k] += c_y * Dzz[i, j, k]
            end
        end
    end

    return nothing
end

# ----------------------------------------------------------------------
# Strain-rate tensor from velocity Jacobian.
# ----------------------------------------------------------------------

# Average an acx-staggered (CenterField slot indexing) field's 4 corners
# around aa-cell (i, j) at layer k. Gauss-Legendre 2-point with uniform
# weights collapses to the 4-corner mean.
@inline function _avg_acx_to_aa(F::AbstractArray, k::Int,
                                i::Int, j::Int,
                                im1::Int, ip1::Int, jm1::Int, jp1::Int)
    sw = 0.5 * (F[im1, jm1, k] + F[im1, j,   k])
    se = 0.5 * (F[i,   jm1, k] + F[i,   j,   k])
    ne = 0.5 * (F[i,   j,   k] + F[i,   jp1, k])
    nw = 0.5 * (F[im1, j,   k] + F[im1, jp1, k])
    return 0.25 * (sw + se + ne + nw)
end

# Average an acy-staggered field's 4 corners around aa-cell (i, j).
@inline function _avg_acy_to_aa(F::AbstractArray, k::Int,
                                i::Int, j::Int,
                                im1::Int, ip1::Int, jm1::Int, jp1::Int)
    sw = 0.5 * (F[im1, jm1, k] + F[i,   jm1, k])
    se = 0.5 * (F[i,   jm1, k] + F[ip1, jm1, k])
    ne = 0.5 * (F[i,   j,   k] + F[ip1, j,   k])
    nw = 0.5 * (F[im1, j,   k] + F[i,   j,   k])
    return 0.25 * (sw + se + ne + nw)
end

# Average an aa-horizontally-staggered field's 4 corners around aa-cell
# (i, j) at layer k. Used after vertical interp of `jvel.dz{x,y}` from
# zeta_ac to zeta_aa.
@inline function _avg_aa_to_aa(F::AbstractArray, k::Int,
                               i::Int, j::Int,
                               im1::Int, ip1::Int, jm1::Int, jp1::Int)
    sw = 0.25 * (F[im1, jm1, k] + F[i, jm1, k] + F[im1, j, k] + F[i, j, k])
    se = 0.25 * (F[i,   jm1, k] + F[ip1, jm1, k] + F[i, j,   k] + F[ip1, j, k])
    ne = 0.25 * (F[i,   j,   k] + F[ip1, j,   k] + F[i, jp1, k] + F[ip1, jp1, k])
    nw = 0.25 * (F[im1, j,   k] + F[i,   j,   k] + F[im1, jp1, k] + F[i, jp1, k])
    return 0.25 * (sw + se + ne + nw)
end

"""
    calc_strain_rate_tensor_jac_quad3D!(
        strn_dxx, strn_dyy, strn_dxy, strn_dxz, strn_dyz,
        strn_de, strn_div, strn_f_shear,
        strn2D_dxx, strn2D_dyy, strn2D_dxy, strn2D_dxz, strn2D_dyz,
        strn2D_de, strn2D_div, strn2D_f_shear,
        jvel_dxx, jvel_dxy, jvel_dxz,
        jvel_dyx, jvel_dyy, jvel_dyz,
        jvel_dzx, jvel_dzy,
        f_ice, f_grnd,
        zeta_aa, de_max
    ) -> nothing

Compute the strain-rate tensor `strn` (3D) and its depth average
`strn2D` from the velocity Jacobian `jvel`. Symmetrises the tensor:

    dxx_strn = dxx_jvel
    dyy_strn = dyy_jvel
    dxy_strn = ½ (dxy_jvel + dyx_jvel)
    dxz_strn = ½ (dxz_jvel + dzx_jvel)
    dyz_strn = ½ (dyz_jvel + dzy_jvel)

Then derives:

    de       = √(dxx² + dyy² + dxx·dyy + dxy² + dxz² + dyz²)
              ↦ clamped to ≤ de_max
    div      = dxx + dyy
    f_shear  = √(dxz² + dyz²) / de   if de > 0  else  1.0
              ↦ 0 on floating ice (f_grnd == 0)
              ↦ clamped to [0, 1]

Strn fields land at aa-cells (horizontal and vertical zeta_aa centres),
matching the schema. Per-layer reads use a 4-corner Gauss-quadrature
average (gq2D-rendered) of the acx/acy-staggered jvel components. The
`jvel.dzx`/`jvel.dzy` z-row terms are at zeta_ac vertically and aa
horizontally; they are vertically interpolated to `zeta_aa[k]` via
`½·(dzx[i,j,k] + dzx[i,j,k+1])` before the horizontal corner average.

Faithfulness: this is a 2D-quadrature rendering of the Fortran
`_quad3D` routine — equivalent for layer-uniform velocities and
sub-percent divergent for stretched stratified runs (matches the
gq2D choice in `calc_uz_3D_jac!`). The Fortran 8-node gq3D port can
be added later if a benchmark surfaces a real divergence.

The depth-averaged `strn2D` fields use a uniform-spacing centre-only
trapezoidal-style integral (∑ strn[k] / Nz). For Yelmo's typical
near-uniform zeta this matches `calc_vertical_integrated_2D` to
trailing decimals.

The 2D principal-strain eigenvalues `eps_eig_1`/`eps_eig_2` from the
Fortran routine are not in the Yelmo.jl model schema and not computed
here. Re-add if a downstream consumer needs them.

Port of `yelmo/src/physics/deformation.f90:1341 calc_strain_rate_tensor_jac_quad3D`.
"""
function calc_strain_rate_tensor_jac_quad3D!(
        strn_dxx, strn_dyy, strn_dxy, strn_dxz, strn_dyz,
        strn_de,  strn_div, strn_f_shear,
        strn2D_dxx, strn2D_dyy, strn2D_dxy, strn2D_dxz, strn2D_dyz,
        strn2D_de,  strn2D_div, strn2D_f_shear,
        jvel_dxx, jvel_dxy, jvel_dxz,
        jvel_dyx, jvel_dyy, jvel_dyz,
        jvel_dzx, jvel_dzy,
        f_ice, f_grnd,
        zeta_aa::AbstractVector{<:Real},
        de_max::Real,
    )

    Sxx = interior(strn_dxx); Syy = interior(strn_dyy); Sxy = interior(strn_dxy)
    Sxz = interior(strn_dxz); Syz = interior(strn_dyz)
    Sde = interior(strn_de);  Sdv = interior(strn_div); Sfs = interior(strn_f_shear)

    Jxx = interior(jvel_dxx); Jxy = interior(jvel_dxy); Jxz = interior(jvel_dxz)
    Jyx = interior(jvel_dyx); Jyy = interior(jvel_dyy); Jyz = interior(jvel_dyz)
    Jzx = interior(jvel_dzx); Jzy = interior(jvel_dzy)

    fi = interior(f_ice);  fg = interior(f_grnd)

    Nx, Ny  = size(fi, 1), size(fi, 2)
    Nz_aa   = length(zeta_aa)

    Tx = topology(strn_dxx.grid, 1)
    Ty = topology(strn_dxx.grid, 2)

    # Initialise everything — ice-free cells stay at zero (Fortran skips
    # the assignment under `if (f_ice == 1)`, leaving the previous value;
    # for a fresh-allocation Field that's zero).
    fill!(Sxx, 0.0); fill!(Syy, 0.0); fill!(Sxy, 0.0)
    fill!(Sxz, 0.0); fill!(Syz, 0.0)
    fill!(Sde, 0.0); fill!(Sdv, 0.0); fill!(Sfs, 0.0)

    de_max_f = Float64(de_max)

    @inbounds for j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] != 1.0
            continue
        end

        im1 = _neighbor_im1(i, Nx, Tx); ip1 = _neighbor_ip1(i, Nx, Tx)
        jm1 = _neighbor_jm1(j, Ny, Ty); jp1 = _neighbor_jp1(j, Ny, Ty)

        for k in 1:Nz_aa
            # ----- Symmetrised aa-cell tensor components -----
            dxx_aa = _avg_acx_to_aa(Jxx, k, i, j, im1, ip1, jm1, jp1)
            dyy_aa = _avg_acy_to_aa(Jyy, k, i, j, im1, ip1, jm1, jp1)

            dxy_acx = _avg_acx_to_aa(Jxy, k, i, j, im1, ip1, jm1, jp1)
            dxy_acy = _avg_acy_to_aa(Jyx, k, i, j, im1, ip1, jm1, jp1)
            dxy_aa  = 0.5 * (dxy_acx + dxy_acy)

            # `jvel.dxz` is at acx, zeta_aa vertically (CenterField slot k).
            # `jvel.dzx` is at aa-horizontally, zeta_ac vertically (ZFace
            # slot k). Layer-centre `zeta_aa[k]` lies between zeta_ac[k]
            # and zeta_ac[k+1] in our scheme; interp = 0.5·(slot k + slot k+1).
            dxz_acx = _avg_acx_to_aa(Jxz, k, i, j, im1, ip1, jm1, jp1)
            # Vertically-interpolated dzx at zeta_aa[k] (storage in 4D temp):
            # do the vertical interp + horizontal aa-corner average inline.
            dzx_aa  = _aa_corner_avg_zface_at_zeta_aa(Jzx, k, i, j, im1, ip1, jm1, jp1)
            dxz_aa  = 0.5 * (dxz_acx + dzx_aa)

            dyz_acy = _avg_acy_to_aa(Jyz, k, i, j, im1, ip1, jm1, jp1)
            dzy_aa  = _aa_corner_avg_zface_at_zeta_aa(Jzy, k, i, j, im1, ip1, jm1, jp1)
            dyz_aa  = 0.5 * (dyz_acy + dzy_aa)

            # ----- Effective strain rate (Fortran lines 1478-1483) -----
            de_sq = dxx_aa^2 + dyy_aa^2 + dxx_aa * dyy_aa +
                    dxy_aa^2 + dxz_aa^2 + dyz_aa^2
            de = sqrt(max(de_sq, 0.0))
            if de > de_max_f
                de = de_max_f
            end

            # ----- Shear fraction (Fortran lines 1494-1511) -----
            if de > 0.0
                shear_sq = dxz_aa^2 + dyz_aa^2
                f_shear = sqrt(shear_sq) / de
            else
                f_shear = 1.0
            end
            if fg[i, j, 1] == 0.0
                f_shear = 0.0   # Floating ice — pure stretching.
            end
            f_shear = clamp(f_shear, 0.0, 1.0)

            # ----- Write outputs -----
            Sxx[i, j, k] = dxx_aa
            Syy[i, j, k] = dyy_aa
            Sxy[i, j, k] = dxy_aa
            Sxz[i, j, k] = dxz_aa
            Syz[i, j, k] = dyz_aa
            Sde[i, j, k] = de
            Sdv[i, j, k] = dxx_aa + dyy_aa
            Sfs[i, j, k] = f_shear
        end
    end

    # ----- Depth-averaged 2D tensor -----
    # Uniform-spacing centre-only mean — ∑ strn[k] / Nz_aa.
    _depth_avg!(strn2D_dxx,     Sxx, Nx, Ny, Nz_aa)
    _depth_avg!(strn2D_dyy,     Syy, Nx, Ny, Nz_aa)
    _depth_avg!(strn2D_dxy,     Sxy, Nx, Ny, Nz_aa)
    _depth_avg!(strn2D_dxz,     Sxz, Nx, Ny, Nz_aa)
    _depth_avg!(strn2D_dyz,     Syz, Nx, Ny, Nz_aa)
    _depth_avg!(strn2D_de,      Sde, Nx, Ny, Nz_aa)
    _depth_avg!(strn2D_div,     Sdv, Nx, Ny, Nz_aa)
    _depth_avg!(strn2D_f_shear, Sfs, Nx, Ny, Nz_aa)

    return nothing
end

# Vertical-then-horizontal corner average for a ZFace field at zeta_aa[k]:
# vertically interpolate to zeta_aa[k] = ½·(slot k + slot k+1), then take
# the 4-corner aa-average.
@inline function _aa_corner_avg_zface_at_zeta_aa(F::AbstractArray, k::Int,
                                                  i::Int, j::Int,
                                                  im1::Int, ip1::Int,
                                                  jm1::Int, jp1::Int)
    # Helper: value of F at zeta_aa[k] for slot (i, j).
    @inline _vinterp(ii, jj) = 0.5 * (F[ii, jj, k] + F[ii, jj, k + 1])

    sw = 0.25 * (_vinterp(im1, jm1) + _vinterp(i, jm1) + _vinterp(im1, j) + _vinterp(i, j))
    se = 0.25 * (_vinterp(i,   jm1) + _vinterp(ip1, jm1) + _vinterp(i, j)   + _vinterp(ip1, j))
    ne = 0.25 * (_vinterp(i,   j)   + _vinterp(ip1, j)   + _vinterp(i, jp1) + _vinterp(ip1, jp1))
    nw = 0.25 * (_vinterp(im1, j)   + _vinterp(i,   j)   + _vinterp(im1, jp1) + _vinterp(i, jp1))
    return 0.25 * (sw + se + ne + nw)
end

@inline function _depth_avg!(out2D, in3D::AbstractArray,
                             Nx::Int, Ny::Int, Nz::Int)
    OUT = interior(out2D)
    @inbounds for j in 1:Ny, i in 1:Nx
        s = 0.0
        for k in 1:Nz
            s += in3D[i, j, k]
        end
        OUT[i, j, 1] = s / Nz
    end
    return out2D
end
