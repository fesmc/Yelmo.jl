# ----------------------------------------------------------------------
# Vertical velocity (uz) and thermodynamic-corrected uz_star.
#
# Port of Fortran `calc_uz_3D_jac` (velocity_general.f90:39). This is
# the production "uz_method = 3" routine that:
#
#   1. Computes the basal vertical velocity from the kinematic BC at
#      the moving lower surface (Greve & Blatter 2009 Eq. 5.31):
#
#          uz(z=z_b) = ∂z_b/∂t + u·∇z_b - a_b
#
#      where `a_b` is the basal accumulation rate. In Yelmo's sign
#      convention `bmb` is positive for accumulation (mass GAINED at
#      the base), so `−a_b = +bmb` and the Fortran formula reads
#
#          uz_b = dzbdt + uz_grid + f_bmb·bmb + ux·dzbdx + uy·dzbdy
#
#      The `+f_bmb·bmb` is correct under Yelmo conventions (cf. an
#      audit note saved 2026-05-04 — the Greve-Blatter expression uses
#      ablation-positive while Yelmo uses accumulation-positive).
#
#   2. Integrates the incompressibility relation
#
#          ∂uz/∂z = −(∂ux/∂x + ∂uy/∂y)
#
#      upward through the column on `zeta_ac` (face) levels:
#
#          uz(k) = uz(k-1) − H·Δζ_ac · (dudx + dvdy)|_{kmid}
#
#      where `kmid = k-1` is the layer center (`zeta_aa` index) and
#      `(dudx, dvdy)` are read from the precomputed velocity Jacobian
#      `jvel.dxx`, `jvel.dyy` (already sigma-corrected at strain-rate
#      time — see `calc_jacobian_vel_3D_uxyterms!`).
#
#   3. Reconstructs the thermodynamic-advection vertical velocity
#      `uz_star = uz·c_z + ux·c_x + uy·c_y + c_t` per layer face, where
#      `c_{x,y,t}` are the sigma-coordinate transform corrections
#      (Greve-Blatter Eqs. 5.131-2). `c_z = 1` and the H-inverse is
#      omitted: `uz_star` is in m/yr (same units as uz), not m/m
#      despite the misleading Fortran comment.
#
# Vertical staggering (Yelmo.jl convention):
#
#   - `uz`, `uz_star`: ZFaceField. Bed (zeta = 0) at interior slot 1;
#     surface (zeta = 1) at interior slot end (Nz_aa+1 = Nz_ac).
#   - `ux`, `uy`: XFace/YFaceField; layer-centered vertically (zeta_aa).
#   - `jvel.dxx`, `jvel.dyy`: CenterField with cell-center indexing for
#     acx/acy-located values (matching `dzsdx`/`dzbdx` convention).
#
# Faithfulness to Fortran:
#
#   - Quadrature: ports the `use_gq3D = .FALSE.` branch (2D Gauss
#     quadrature at the layer center). The Fortran production path
#     hardcodes `use_gq3D = .TRUE.` (8-node 3D quadrature spanning
#     kmid-1 to kmid+1 layers). For layer-uniform velocities the two
#     are identical; for stretched-grid stratified runs they may differ
#     by sub-percent. If a benchmark surfaces a real divergence, switch
#     to the 3D quadrature path then.
#   - The `uz_grid` term is hard-coded to 0 (Fortran line 227); the
#     formula remains commented out and labeled "untested".
#   - `f_grnd` and `boundaries` parameters from the Fortran signature
#     are unused in the body and dropped from the Julia signature.
#   - Underflow zero-out (`abs(uz) < TOL_UNDERFLOW`) and `uz_lim = 10
#     m/yr` symmetric clamp are preserved. The asymmetric `uz_min =
#     -10 m/yr` floor is also preserved despite being redundant under
#     the symmetric clamp (the floor is silently shadowed by the clamp
#     on the next line).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.Grids: topology, Bounded, Periodic

export calc_uz_3D_jac!, calc_uz_3D!, calc_uz_3D_aa!

const _UZ_MIN = -10.0   # m/yr  (Fortran line 125)
const _UZ_LIM =  10.0   # m/yr  (Fortran line 126)
const _TOL_UNDERFLOW = 1e-10   # matches yelmo_defs.f90 TOL_UNDERFLOW

# ----------------------------------------------------------------------
# Helpers — staggered-to-aa value via 4-corner Gauss-quadrature average.
# Mirrors the Fortran `gq2D_to_nodes_acx` + weighted-sum pattern used
# throughout calc_uz_3D_jac. For the 2-point Gauss rule with uniform
# weights, this reduces to the 4-corner average — but spelled out via
# `gq2d_interp_to_node` for parity with `viscosity.jl:330` / Fortran.
# ----------------------------------------------------------------------

# acx-staggered field stored in CenterField with cell-center indexing
# for acx-located values (e.g. dzsdx, dzbdx, jvel.dxx). Matching
# Fortran `gq2D_to_nodes_acx` corner recipe (Fortran line 405-409).
@inline function _gq2D_acx_to_aa(F::AbstractArray, k::Int,
                                 i::Int, j::Int,
                                 im1::Int, ip1::Int, jm1::Int, jp1::Int,
                                 xr, yr, wt, wt_tot)
    v_sw = 0.5 * (F[im1, jm1, k] + F[im1, j,   k])
    v_se = 0.5 * (F[i,   jm1, k] + F[i,   j,   k])
    v_ne = 0.5 * (F[i,   j,   k] + F[i,   jp1, k])
    v_nw = 0.5 * (F[im1, j,   k] + F[im1, jp1, k])
    v_ab = (v_sw, v_se, v_ne, v_nw)
    acc = 0.0
    @inbounds for p in 1:4
        acc += gq2d_interp_to_node(v_ab, xr[p], yr[p]) * wt[p]
    end
    return acc / wt_tot
end

# acx field on a 2D Field (k=1 always).
@inline _gq2D_acx_to_aa_2D(F, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot) =
    _gq2D_acx_to_aa(F, 1, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot)

# acy-staggered. Fortran `gq2D_to_nodes_acy` recipe (Fortran line 433-437).
@inline function _gq2D_acy_to_aa(F::AbstractArray, k::Int,
                                 i::Int, j::Int,
                                 im1::Int, ip1::Int, jm1::Int, jp1::Int,
                                 xr, yr, wt, wt_tot)
    v_sw = 0.5 * (F[im1, jm1, k] + F[i,   jm1, k])
    v_se = 0.5 * (F[i,   jm1, k] + F[ip1, jm1, k])
    v_ne = 0.5 * (F[i,   j,   k] + F[ip1, j,   k])
    v_nw = 0.5 * (F[im1, j,   k] + F[i,   j,   k])
    v_ab = (v_sw, v_se, v_ne, v_nw)
    acc = 0.0
    @inbounds for p in 1:4
        acc += gq2d_interp_to_node(v_ab, xr[p], yr[p]) * wt[p]
    end
    return acc / wt_tot
end

@inline _gq2D_acy_to_aa_2D(F, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot) =
    _gq2D_acy_to_aa(F, 1, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot)

# ux/uy are XFace/YFaceFields — slot indexing is `_ip1_modular(i, ...)`
# (= i+1 under Bounded-x). The Fortran corner recipe still applies, but
# slot indices need the +1 remap.
@inline function _gq2D_acx_to_aa_face(UX::AbstractArray, k::Int,
                                      i::Int, j::Int,
                                      im1::Int, ip1::Int, jm1::Int, jp1::Int,
                                      Nx::Int, Tx, xr, yr, wt, wt_tot)
    ux_i   = _ip1_modular(i,   Nx, Tx)
    ux_im1 = _ip1_modular(im1, Nx, Tx)
    v_sw = 0.5 * (UX[ux_im1, jm1, k] + UX[ux_im1, j,   k])
    v_se = 0.5 * (UX[ux_i,   jm1, k] + UX[ux_i,   j,   k])
    v_ne = 0.5 * (UX[ux_i,   j,   k] + UX[ux_i,   jp1, k])
    v_nw = 0.5 * (UX[ux_im1, j,   k] + UX[ux_im1, jp1, k])
    v_ab = (v_sw, v_se, v_ne, v_nw)
    acc = 0.0
    @inbounds for p in 1:4
        acc += gq2d_interp_to_node(v_ab, xr[p], yr[p]) * wt[p]
    end
    return acc / wt_tot
end

@inline function _gq2D_acy_to_aa_face(UY::AbstractArray, k::Int,
                                      i::Int, j::Int,
                                      im1::Int, ip1::Int, jm1::Int, jp1::Int,
                                      Ny::Int, Ty, xr, yr, wt, wt_tot)
    uy_j   = _jp1_modular(j,   Ny, Ty)
    uy_jm1 = _jp1_modular(jm1, Ny, Ty)
    v_sw = 0.5 * (UY[im1, uy_jm1, k] + UY[i,   uy_jm1, k])
    v_se = 0.5 * (UY[i,   uy_jm1, k] + UY[ip1, uy_jm1, k])
    v_ne = 0.5 * (UY[i,   uy_j,   k] + UY[ip1, uy_j,   k])
    v_nw = 0.5 * (UY[im1, uy_j,   k] + UY[i,   uy_j,   k])
    v_ab = (v_sw, v_se, v_ne, v_nw)
    acc = 0.0
    @inbounds for p in 1:4
        acc += gq2d_interp_to_node(v_ab, xr[p], yr[p]) * wt[p]
    end
    return acc / wt_tot
end

# ----------------------------------------------------------------------
# Underflow + symmetric clamp: matches Fortran `if (abs(.) < TOL) . = 0`
# followed by `call minmax(., uz_lim)`. The `uz_min = -10` floor (Fortran
# line 240) is silently shadowed by the symmetric `±10` clamp on the next
# line; we apply both for parity but note the redundancy.
# ----------------------------------------------------------------------
@inline function _uz_underflow_minmax!(arr::AbstractArray, idx...)
    v = arr[idx...]
    if abs(v) < _TOL_UNDERFLOW
        arr[idx...] = 0.0
        return nothing
    end
    if v < -_UZ_LIM
        arr[idx...] = -_UZ_LIM
    elseif v > _UZ_LIM
        arr[idx...] = _UZ_LIM
    end
    return nothing
end

# ----------------------------------------------------------------------
# `calc_uz_3D_jac!` — production routine (uz_method = 3).
# ----------------------------------------------------------------------
"""
    calc_uz_3D_jac!(
        uz, uz_star,
        ux, uy,
        jvel_dxx, jvel_dyy,
        H_ice, f_ice,
        smb, bmb, dHdt, dzsdt,
        dzsdx, dzsdy, dzbdx, dzbdy,
        zeta_aa, zeta_ac,
        dx, dy, use_bmb,
    ) -> nothing

Compute the vertical ice velocity `uz` (continuity) and the
thermodynamic-corrected `uz_star` (sigma-coordinate transform) on the
zeta_ac face levels. Iced cells use the kinematic BC at the bed plus
upward integration of the incompressibility relation; ice-free cells
get `uz = dzbdt − max(smb, 0)` per layer (Fortran line 370).

Parameters drop the Fortran `f_grnd` and `boundaries` (unused in
body); the `jvel` argument splits into `jvel_dxx`, `jvel_dyy` since
those are the only Jacobian components consumed.

Port of `velocity_general.f90:39 calc_uz_3D_jac` via the
`use_gq3D = .FALSE.` (2D quadrature) branch — see file docstring.
"""
function calc_uz_3D_jac!(
        uz, uz_star,
        ux, uy,
        jvel_dxx, jvel_dyy,
        H_ice, f_ice,
        smb, bmb, dHdt, dzsdt,
        dzsdx, dzsdy, dzbdx, dzbdy,
        zeta_aa::AbstractVector{<:Real},
        zeta_ac::AbstractVector{<:Real},
        dx::Real, dy::Real,
        use_bmb::Bool,
    )

    UZ      = interior(uz);        UZS     = interior(uz_star)
    UX      = interior(ux);        UY      = interior(uy)
    Dxx     = interior(jvel_dxx);  Dyy     = interior(jvel_dyy)
    H       = interior(H_ice);     fi      = interior(f_ice)
    SMB     = interior(smb);       BMB     = interior(bmb)
    DHDT    = interior(dHdt);      DZSDT   = interior(dzsdt)
    DZSX    = interior(dzsdx);     DZSY    = interior(dzsdy)
    DZBX    = interior(dzbdx);     DZBY    = interior(dzbdy)

    Nx, Ny  = size(H, 1), size(H, 2)
    Nz_aa   = length(zeta_aa)
    Nz_ac   = length(zeta_ac)

    Nz_aa == size(Dxx, 3) || error(
        "calc_uz_3D_jac!: zeta_aa length $(Nz_aa) ≠ jvel_dxx Nz $(size(Dxx, 3))")
    Nz_ac == size(UZ, 3) || error(
        "calc_uz_3D_jac!: zeta_ac length $(Nz_ac) ≠ uz Nz $(size(UZ, 3))")
    Nz_ac == Nz_aa + 1 || error(
        "calc_uz_3D_jac!: zeta_ac length $(Nz_ac) ≠ zeta_aa length + 1 $(Nz_aa+1)")

    Tx = topology(uz.grid, 1)
    Ty = topology(uz.grid, 2)

    fill!(UZ, 0.0)

    f_bmb = use_bmb ? 1.0 : 0.0

    xr, yr, wt, wt_tot = gq2d_nodes(2)

    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = _neighbor_im1(i, Nx, Tx); ip1 = _neighbor_ip1(i, Nx, Tx)
        jm1 = _neighbor_jm1(j, Ny, Ty); jp1 = _neighbor_jp1(j, Ny, Ty)

        # Diagnose dzbdt = dzsdt - dHdt regardless of ice cover (used
        # in both branches). Fortran lines 181-183.
        dzsdt_now = DZSDT[i, j, 1]
        dhdt_now  = DHDT[i, j, 1]
        dzbdt_now = dzsdt_now - dhdt_now

        if fi[i, j, 1] == 1.0
            # ===================== Ice-covered branch =====================

            H_now = H[i, j, 1]

            # Aa-cell-centered bed/surface gradients via 2D Gauss-quadrature
            # corner-then-bilinear-then-weighted-sum (Fortran lines 202-213).
            dzbdx_aa = _gq2D_acx_to_aa_2D(DZBX, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot)
            dzbdy_aa = _gq2D_acy_to_aa_2D(DZBY, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot)
            dzsdx_aa = _gq2D_acx_to_aa_2D(DZSX, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot)
            dzsdy_aa = _gq2D_acy_to_aa_2D(DZSY, i, j, im1, ip1, jm1, jp1, xr, yr, wt, wt_tot)

            # Aa-cell-centered (ux, uy) at the BASE layer (kmid=1)
            # (Fortran lines 216-220).
            ux_aa_b = _gq2D_acx_to_aa_face(UX, 1, i, j, im1, ip1, jm1, jp1,
                                           Nx, Tx, xr, yr, wt, wt_tot)
            uy_aa_b = _gq2D_acy_to_aa_face(UY, 1, i, j, im1, ip1, jm1, jp1,
                                           Ny, Ty, xr, yr, wt, wt_tot)

            # uz_grid hardcoded to 0 — Fortran line 227.
            uz_grid = 0.0

            # ----- Basal kinematic BC (Fortran line 234) -----
            # Greve-Blatter Eq. 5.31 with Yelmo's accumulation-positive bmb
            # convention (so `+f_bmb*bmb` corresponds to G&B's `-a_b` for
            # ablation-positive `a_b`).
            UZ[i, j, 1] = dzbdt_now + uz_grid + f_bmb * BMB[i, j, 1] +
                          ux_aa_b * dzbdx_aa + uy_aa_b * dzbdy_aa
            _uz_underflow_minmax!(UZ, i, j, 1)
            # _UZ_MIN floor — silently shadowed by the symmetric clamp
            # already applied above (kept for Fortran parity).
            if UZ[i, j, 1] < _UZ_MIN
                UZ[i, j, 1] = _UZ_MIN
            end
            # Re-clamp in case the floor pushed back into [-uz_lim, uz_lim].
            _uz_underflow_minmax!(UZ, i, j, 1)

            # ----- Vertical integration upward (Fortran lines 251-302) -----
            for k in 2:Nz_ac
                kmid = k - 1   # layer center (zeta_aa index)

                # Fortran's `use_gq3D = .FALSE.` branch — 2D quadrature
                # at the kmid layer. dudx/dvdy live at the same horizontal
                # stagger as dxx/dyy: CenterField slot [.,.,kmid].
                dudx_aa = _gq2D_acx_to_aa(Dxx, kmid, i, j, im1, ip1, jm1, jp1,
                                          xr, yr, wt, wt_tot)
                dvdy_aa = _gq2D_acy_to_aa(Dyy, kmid, i, j, im1, ip1, jm1, jp1,
                                          xr, yr, wt, wt_tot)

                # Greve-Blatter Eq. 5.95 (Fortran line 292).
                UZ[i, j, k] = UZ[i, j, k-1] -
                              H_now * (zeta_ac[k] - zeta_ac[k-1]) * (dudx_aa + dvdy_aa)
                _uz_underflow_minmax!(UZ, i, j, k)
            end

            # ----- uz_star reconstruction (Fortran lines 306-363) -----
            for k in 1:Nz_ac
                # Layer-uniform horizontal velocity at zeta_ac[k] —
                # average of layers `kup` and `kdn` straddling the face.
                # Boundary cases (Fortran lines 313-322):
                if k == 1
                    kup = 1; kdn = 1
                elseif k == Nz_ac
                    kup = Nz_aa; kdn = Nz_aa
                else
                    kup = k; kdn = k - 1
                end

                ux_up = _gq2D_acx_to_aa_face(UX, kup, i, j, im1, ip1, jm1, jp1,
                                             Nx, Tx, xr, yr, wt, wt_tot)
                ux_dn = _gq2D_acx_to_aa_face(UX, kdn, i, j, im1, ip1, jm1, jp1,
                                             Nx, Tx, xr, yr, wt, wt_tot)
                ux_aa = 0.5 * (ux_up + ux_dn)

                uy_up = _gq2D_acy_to_aa_face(UY, kup, i, j, im1, ip1, jm1, jp1,
                                             Ny, Ty, xr, yr, wt, wt_tot)
                uy_dn = _gq2D_acy_to_aa_face(UY, kdn, i, j, im1, ip1, jm1, jp1,
                                             Ny, Ty, xr, yr, wt, wt_tot)
                uy_aa = 0.5 * (uy_up + uy_dn)

                zeta_now = zeta_ac[k]

                # Sigma-coordinate transform corrections — Fortran lines
                # 348-351. H-inverse omitted (cf. lines 342-347): this is
                # the convention used for Yelmo's thermodynamic advection
                # which works in d/dz, not d/dζ. Output uz_star is in m/yr.
                c_x = -((1.0 - zeta_now) * dzbdx_aa  + zeta_now * dzsdx_aa)
                c_y = -((1.0 - zeta_now) * dzbdy_aa  + zeta_now * dzsdy_aa)
                c_t = -((1.0 - zeta_now) * dzbdt_now + zeta_now * dzsdt_now)
                # c_z = 1.0 — Fortran line 351.

                UZS[i, j, k] = ux_aa * c_x + uy_aa * c_y + UZ[i, j, k] + c_t
                _uz_underflow_minmax!(UZS, i, j, k)
            end

        else
            # ===================== Ice-free branch =====================
            # Fortran lines 365-378. Note this is `f_ice ≠ 1`, i.e. ice
            # absent or partial — NOT a floating-ice branch (real shelves
            # have `f_ice = 1` and go through the main branch above).
            for k in 1:Nz_ac
                v = dzbdt_now - max(SMB[i, j, 1], 0.0)
                UZ[i, j, k] = v
                _uz_underflow_minmax!(UZ, i, j, k)
                UZS[i, j, k] = UZ[i, j, k]
            end
        end
    end

    return nothing
end

# ----------------------------------------------------------------------
# uz_method = 1 ("uz_aa") — legacy aa-node formulation. NOT ported.
# uz_method = 2 ("uz_nodes") — intermediate formulation. NOT ported.
# ----------------------------------------------------------------------
function calc_uz_3D!(args...; kwargs...)
    error("calc_uz_3D!: uz_method = 2 (\"uz_nodes\") is not ported in Yelmo.jl. " *
          "Only uz_method = 3 (\"uz_jac\", `calc_uz_3D_jac!`) is supported. " *
          "Switch `ydyn.uz_method = 3` in the configuration.")
end

function calc_uz_3D_aa!(args...; kwargs...)
    error("calc_uz_3D_aa!: uz_method = 1 (\"uz_aa\") is not ported in Yelmo.jl. " *
          "Only uz_method = 3 (\"uz_jac\", `calc_uz_3D_jac!`) is supported. " *
          "Switch `ydyn.uz_method = 3` in the configuration.")
end
