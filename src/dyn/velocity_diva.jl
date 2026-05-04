# ----------------------------------------------------------------------
# DIVA (Depth-Integrated Viscosity Approximation) velocity solver.
#
# Faithful port of `yelmo/src/physics/velocity_diva.f90` (~1160 lines).
# References:
#   - Goldberg (2011), J. Glaciol. 57(201) — original DIVA derivation.
#   - Lipscomb et al. (2019, GMD) — CISM2 implementation, Eqs. 29-35.
#   - Arthern et al. (2015) — F2 integral formulation, Eq. 7.
#
# DIVA reduces the 3D Stokes equations to a 2D depth-integrated SSA-like
# system with an effective basal friction `beta_eff` that combines the
# usual basal sliding coefficient `beta` with the depth-integrated
# inverse viscosity F2:
#
#     F2(i,j)        = ∫_zb^zs (1 / η(i,j,z)) · ((s - z) / H)^2 dz       [m / (Pa·yr)]
#     beta_eff       = beta / (1 + beta · F2)        (sliding case)
#     beta_eff       = 1 / F2                        (no_slip case)
#
# The 2D system solves for the depth-averaged velocity `ux_bar / uy_bar`
# using exactly the same SSA matrix kernel (`_assemble_ssa_matrix!`) —
# the only difference vs `calc_velocity_ssa!` is that DIVA passes
# `beta_eff_acx / beta_eff_acy` in place of `beta_acx / beta_acy`.
# `visc_eff_int` is identical to SSA's (no DIVA-specific modification —
# F2 only modifies friction, not viscosity).
#
# After Picard convergence, the 3D velocity is reconstructed in closed
# form via the F1 integral:
#
#     F1(i,j,k)    = ∫_zb^z(k) (1 / η(i,j,ℓ)) · ((s - ℓ) / H) dℓ        [cumulative]
#     u_b(i,j)     = u_bar(i,j) − F2(i,j) · τ_b(i,j)                    (sliding)
#     u_b(i,j)     = 0                                                   (no_slip)
#     u(i,j,k)     = u_b(i,j) + F1(i,j,k) · τ_b(i,j)
#     u_i(i,j,k)   = u(i,j,k) − u_b(i,j)                                 (shearing component)
#
# The Picard outer loop is structurally identical to SSA's (snapshot →
# update viscosity → log-relax visc → recompute F2 / beta_eff → assemble
# matrix → solve → relax velocity → check convergence). Only the inputs
# to the matrix kernel differ.
# ----------------------------------------------------------------------

using SparseArrays: sparse
using Krylov: BicgstabWorkspace, bicgstab!
using Oceananigans.Fields: interior, XFaceField, YFaceField, CenterField, Field
using Oceananigans.Grids: Bounded, Periodic, Face, Center, znodes, topology, AbstractTopology
using Oceananigans.BoundaryConditions: fill_halo_regions!

# Pull SSA helpers we reuse verbatim.
using ..YelmoCore: AbstractYelmoModel, YelmoModel
import ..YelmoCore: dyn_step!  # extended in YelmoModelDyn.jl

# velocity_ssa.jl helpers we depend on. Same module, so no `using` needed —
# they are in scope here when this file is included after velocity_ssa.jl.

export calc_velocity_diva!,
       calc_F_integral_2D!, calc_F1_integral_3D!,
       calc_beta_eff!, stagger_beta_eff!,
       calc_vel_basal_diva!, calc_vel_horizontal_3D!


# ======================================================================
# F-integrals
# ======================================================================

"""
    calc_F_integral_2D!(F_int, visc_eff, H_ice, f_ice, zeta_aa; n=2.0)
        -> F_int

Compute the depth-integrated inverse-viscosity moment

    F_n(i,j) = ∫_0^1 (H_eff(i,j) / η(i,j,ζ)) · (1 − ζ)^n dζ

where `H_eff = H_ice / f_ice` for fully-ice-covered cells (`f_ice == 1`)
and `F_n = 0` at partial-ice / ice-free cells.

`visc_eff` is a 3D `CenterField` with `Nz = length(zeta_aa)` interior
layers (Option C: bed and surface NOT included). The integrand at the
bed (`ζ = 0`) and surface (`ζ = 1`) endpoints is approximated by the
nearest-Center value of `1/η`, matching the Option-C convention used by
`calc_visc_eff_int!`.

Trapezoidal integration over the `zeta_aa` mesh including bed/surface
endpoints. For DIVA, `n = 2` (Goldberg 2011 Eq. 41/42; Lipscomb 2019
Eq. 30). For the cumulative 3D `F1` see `calc_F1_integral_3D!`.

Port of Fortran `velocity_diva.f90:916-961 calc_F_integral`.
"""
function calc_F_integral_2D!(F_int, visc_eff, H_ice, f_ice,
                             zeta_aa::AbstractVector{<:Real};
                             n::Real = 2.0)
    # Wrapper: same template as the dyn 3D / DIVA viscosity series.
    # Lift Field views, materialise zeta as concrete `Vector{Float64}`,
    # dispatch into the typed-scalar kernel below.
    Fi = interior(F_int)
    V  = interior(visc_eff)
    H  = interior(H_ice)
    fi = interior(f_ice)

    Nx, Ny = size(Fi, 1), size(Fi, 2)
    Nz     = size(V, 3)
    Nz == length(zeta_aa) || error(
        "calc_F_integral_2D!: visc_eff Nz=$(Nz) but zeta_aa has length $(length(zeta_aa)).")

    zeta = collect(Float64, zeta_aa)
    _calc_F_integral_2D_kernel!(Fi, V, H, fi, zeta, Float64(n), Nx, Ny, Nz)
    return F_int
end

function _calc_F_integral_2D_kernel!(Fi, V, H, fi,
        zeta::Vector{Float64}, n_pow::Float64,
        Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] != 1.0
            Fi[i, j, 1] = 0.0
            continue
        end
        H_eff = H[i, j, 1]   # Note: f_ice == 1 here, so H_eff == H.

        # Trapezoidal integration over [0, 1] including bed/surface
        # endpoints. Bed endpoint (ζ=0) approximated with V[i,j,1]
        # (nearest Center); surface endpoint (ζ=1) with V[i,j,end].
        z_prev = 0.0
        eta_prev = V[i, j, 1]
        integ_prev = (H_eff / eta_prev) * (1.0 - z_prev) ^ n_pow

        acc = 0.0
        for k in 1:Nz
            z_k   = zeta[k]
            eta_k = V[i, j, k]
            integ_k = (H_eff / eta_k) * (1.0 - z_k) ^ n_pow
            acc += 0.5 * (integ_prev + integ_k) * (z_k - z_prev)
            z_prev = z_k
            integ_prev = integ_k
        end
        z_k = 1.0
        eta_k = V[i, j, Nz]
        integ_k = (H_eff / eta_k) * (1.0 - z_k) ^ n_pow         # = 0 for n > 0
        acc += 0.5 * (integ_prev + integ_k) * (z_k - z_prev)

        Fi[i, j, 1] = acc
    end
    return nothing
end


"""
    calc_F1_integral_3D!(F1, visc_eff, H_ice, f_ice, zeta_aa) -> F1

Cumulative inverse-viscosity moment integrated from the bed up to each
Center layer:

    F1(i,j,k) = ∫_0^ζ_k (H_eff(i,j) / η(i,j,ζ')) · (1 − ζ') dζ'

with the bed endpoint (ζ=0) approximated using nearest-Center η as in
[`calc_F_integral_2D!`](@ref). At ice-free / partial-ice cells F1 is
zero across all layers. Used by [`calc_vel_horizontal_3D!`](@ref) to
reconstruct the 3D horizontal velocity from `u_b` and the basal stress.

Port of Fortran `velocity_diva.f90:459` (the F1 sweep inside
`calc_vel_horizontal_3D`).
"""
function calc_F1_integral_3D!(F1, visc_eff, H_ice, f_ice,
                              zeta_aa::AbstractVector{<:Real})
    F = interior(F1)
    V = interior(visc_eff)
    H = interior(H_ice)
    fi = interior(f_ice)

    Nx, Ny = size(F, 1), size(F, 2)
    Nz     = size(V, 3)
    (size(F, 3) == Nz && length(zeta_aa) == Nz) || error(
        "calc_F1_integral_3D!: shape mismatch F1=$(size(F))  visc=$(size(V))  zeta=$(length(zeta_aa)).")

    zeta = collect(Float64, zeta_aa)
    _calc_F1_integral_3D_kernel!(F, V, H, fi, zeta, Nx, Ny, Nz)
    return F1
end

function _calc_F1_integral_3D_kernel!(F, V, H, fi,
        zeta::Vector{Float64}, Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] != 1.0
            for k in 1:Nz
                F[i, j, k] = 0.0
            end
            continue
        end
        H_eff = H[i, j, 1]

        # Bed endpoint integrand (ζ=0).
        z_prev = 0.0
        eta_prev = V[i, j, 1]
        integ_prev = (H_eff / eta_prev) * (1.0 - z_prev)   # n = 1

        acc = 0.0
        for k in 1:Nz
            z_k = zeta[k]
            eta_k = V[i, j, k]
            integ_k = (H_eff / eta_k) * (1.0 - z_k)
            acc += 0.5 * (integ_prev + integ_k) * (z_k - z_prev)
            F[i, j, k] = acc
            z_prev = z_k
            integ_prev = integ_k
        end
    end
    return nothing
end


# ======================================================================
# beta_eff
# ======================================================================

"""
    calc_beta_eff!(beta_eff, beta, F2; no_slip::Bool = false) -> beta_eff

Goldberg (2011) effective basal friction combining sliding-law `beta`
with the depth-integrated inverse-viscosity F2:

  - **no_slip = true**  (no basal sliding): `beta_eff = 1 / F2`
                                              (Goldberg 2011 Eq. 42)
  - **no_slip = false** (basal sliding):    `beta_eff = beta / (1 + beta · F2)`
                                              (Goldberg 2011 Eq. 41)

At ice-free / partial-ice cells F2 is 0; under sliding `beta_eff = beta`
falls back to the raw friction coefficient (no viscous coupling in the
column). Under no_slip with `F2 = 0` the formula would diverge — we
instead set `beta_eff = beta` as a fallback (the SSA mask zeroes those
cells anyway).

Port of Fortran `velocity_diva.f90:963-998 calc_beta_eff`.
"""
function calc_beta_eff!(beta_eff, beta, F2; no_slip::Bool = false)
    Be = interior(beta_eff)
    B  = interior(beta)
    F  = interior(F2)

    if no_slip
        @inbounds for k in eachindex(Be)
            f = F[k]
            Be[k] = f > 0.0 ? 1.0 / f : B[k]   # margin / ice-free fallback
        end
    else
        @inbounds for k in eachindex(Be)
            Be[k] = B[k] / (1.0 + B[k] * F[k])
        end
    end
    return beta_eff
end


"""
    stagger_beta_eff!(beta_eff_acx, beta_eff_acy, beta_eff,
                      H_ice, f_ice, ux_bar, uy_bar,
                      f_grnd, f_grnd_acx, f_grnd_acy;
                      beta_gl_stag, beta_min) -> (beta_eff_acx, beta_eff_acy)

Stagger `beta_eff` from aa-cells to the acx / acy face grid. Reuses the
existing [`stagger_beta!`](@ref) verbatim — DIVA shares the same
margin / grounding-line staggering logic as SSA; only the underlying
field being staggered differs.
"""
function stagger_beta_eff!(beta_eff_acx, beta_eff_acy, beta_eff,
                           H_ice, f_ice, ux_bar, uy_bar,
                           f_grnd, f_grnd_acx, f_grnd_acy;
                           beta_gl_stag::Int, beta_min::Real)
    return stagger_beta!(beta_eff_acx, beta_eff_acy, beta_eff,
                         H_ice, f_ice, ux_bar, uy_bar,
                         f_grnd, f_grnd_acx, f_grnd_acy;
                         beta_gl_stag = beta_gl_stag,
                         beta_min     = beta_min)
end


# ======================================================================
# Post-Picard reconstruction
# ======================================================================

"""
    calc_vel_basal_diva!(ux_b, uy_b, ux_bar, uy_bar,
                         taub_acx, taub_acy, F2, f_ice;
                         no_slip::Bool = false)
        -> (ux_b, uy_b)

Recover the basal velocity from the converged depth-averaged solution
via Goldberg (2011) Eq. 34:

    u_b = u_bar − F2_face · τ_b           (sliding)
    u_b = 0                                (no_slip)

where `F2_face` is `F2` staggered to the acx / acy face grid using a
margin-aware rule: when both adjacent cells are fully ice-covered take
their average; when one is partial / ice-free take the ice-covered
side; when both are ice-free `F2_face = 0` and `u_b = u_bar`.

Mirrors Fortran `velocity_diva.f90:1000-1065 calc_vel_basal`.
"""
function calc_vel_basal_diva!(ux_b, uy_b, ux_bar, uy_bar,
                              taub_acx, taub_acy, F2, f_ice;
                              no_slip::Bool = false)
    Uxb = interior(ux_b)
    Uyb = interior(uy_b)
    Uxbar = interior(ux_bar)
    Uybar = interior(uy_bar)

    if no_slip
        fill!(Uxb, 0.0)
        fill!(Uyb, 0.0)
        return ux_b, uy_b
    end

    Tx = interior(taub_acx)
    Ty = interior(taub_acy)
    Fc = interior(F2)
    Fi = interior(f_ice)

    Nx, Ny = size(Fc, 1), size(Fc, 2)
    Tx_top = topology(ux_b.grid, 1)
    Ty_top = topology(uy_b.grid, 2)

    @inbounds for j in 1:Ny, i in 1:Nx
        ip1  = _neighbor_ip1(i, Nx, Tx_top)
        jp1  = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)

        # x-face between cell (i, j) and (ip1, j) — slot [ip1f, j].
        F2_acx = _stagger_margin_face(Fc[i, j, 1], Fc[ip1, j, 1],
                                      Fi[i, j, 1], Fi[ip1, j, 1])
        Uxb[ip1f, j, 1] = Uxbar[ip1f, j, 1] - F2_acx * Tx[ip1f, j, 1]

        # y-face between cell (i, j) and (i, jp1) — slot [i, jp1f].
        F2_acy = _stagger_margin_face(Fc[i, j, 1], Fc[i, jp1, 1],
                                      Fi[i, j, 1], Fi[i, jp1, 1])
        Uyb[i, jp1f, 1] = Uybar[i, jp1f, 1] - F2_acy * Ty[i, jp1f, 1]
    end

    # Replicate the leading face slot under Bounded for downstream readers.
    if Tx_top === Bounded
        @views Uxb[1, :, :] .= Uxb[2, :, :]
    end
    if Ty_top === Bounded
        @views Uyb[:, 1, :] .= Uyb[:, 2, :]
    end

    return ux_b, uy_b
end

# Margin-aware face stagger: when both cells are fully iced take their
# average; when only one is iced take that side's value; when neither
# is iced return 0. Mirrors Fortran `calc_staggered_margin`
# (velocity_diva.f90:1128-1158).
@inline function _stagger_margin_face(v_left::Real, v_right::Real,
                                      f_left::Real, f_right::Real)
    if f_left == 1.0 && f_right == 1.0
        return 0.5 * (v_left + v_right)
    elseif f_left == 1.0
        return v_left
    elseif f_right == 1.0
        return v_right
    else
        return 0.0
    end
end


"""
    calc_vel_horizontal_3D!(ux, uy, ux_b, uy_b,
                            taub_acx, taub_acy, F1_3D, f_ice)
        -> (ux, uy)

Reconstruct the 3D horizontal velocity from the basal velocity and the
basal stress, using the cumulative F1 integral (Lipscomb 2019 Eq. 29):

    u(i,j,k) = u_b(i,j) + F1_face(i,j,k) · τ_b(i,j)

with the same margin-aware face stagger of F1 as in
`calc_vel_basal_diva!`.

Port of Fortran `velocity_diva.f90:414-491 calc_vel_horizontal_3D`.
"""
function calc_vel_horizontal_3D!(ux, uy, ux_b, uy_b,
                                 taub_acx, taub_acy, F1_3D, f_ice)
    # Wrapper: lift Field views, look up topology, dispatch into the
    # parametric kernel below. Same template as the dyn 3D series.
    Ux = interior(ux)
    Uy = interior(uy)
    Uxb = interior(ux_b)
    Uyb = interior(uy_b)
    Tx = interior(taub_acx)
    Ty = interior(taub_acy)
    F  = interior(F1_3D)
    Fi = interior(f_ice)

    Nx, Ny = size(Fi, 1), size(Fi, 2)
    Nz     = size(F, 3)
    Tx_top = topology(ux.grid, 1)
    Ty_top = topology(uy.grid, 2)

    _calc_vel_horizontal_3D_kernel!(Ux, Uy, Uxb, Uyb, Tx, Ty, F, Fi,
                                    Tx_top, Ty_top, Nx, Ny, Nz)
    return ux, uy
end

function _calc_vel_horizontal_3D_kernel!(Ux, Uy, Uxb, Uyb, Tx, Ty, F, Fi,
        ::Type{Tx_top}, ::Type{Ty_top}, Nx::Int, Ny::Int, Nz::Int,
    ) where {Tx_top<:AbstractTopology, Ty_top<:AbstractTopology}

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip1  = _neighbor_ip1(i, Nx, Tx_top)
        jp1  = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)

        F1_acx = _stagger_margin_face(F[i, j, k], F[ip1, j, k],
                                      Fi[i, j, 1], Fi[ip1, j, 1])
        Ux[ip1f, j, k] = Uxb[ip1f, j, 1] + F1_acx * Tx[ip1f, j, 1]

        F1_acy = _stagger_margin_face(F[i, j, k], F[i, jp1, k],
                                      Fi[i, j, 1], Fi[i, jp1, 1])
        Uy[i, jp1f, k] = Uyb[i, jp1f, 1] + F1_acy * Ty[i, jp1f, 1]
    end

    # Replicate leading face slots under Bounded.
    if Tx_top === Bounded
        @views Ux[1, :, :] .= Ux[2, :, :]
    end
    if Ty_top === Bounded
        @views Uy[:, 1, :] .= Uy[:, 2, :]
    end

    return nothing
end


# ======================================================================
# Main DIVA Picard driver
# ======================================================================

"""
    calc_velocity_diva!(y) -> y

Top-level DIVA driver. Iterates the SSA-like 2D depth-integrated
momentum balance with depth-integrated effective friction `beta_eff`
until the depth-averaged velocity converges, then reconstructs the
3D horizontal velocity field via the F1 closed-form integral.

Picard outer loop mirrors `calc_velocity_ssa!` step-for-step; the
only differences vs SSA are:

  - **Step 4b (NEW)**: compute `F2` integral.
  - **Step 4c (NEW)**: derive `beta_eff` from `beta` and `F2`.
  - **Step 5 (MODIFIED)**: stagger `beta_eff` to `(beta_eff_acx, beta_eff_acy)`
                          (in addition to `beta` → `beta_acx, beta_acy`).
  - **Step 7 (MODIFIED)**: pass `beta_eff_acx / beta_eff_acy` to
                          `_assemble_ssa_matrix!` instead of `beta_acx / beta_acy`,
                          and write Picard iterates into `ux_bar / uy_bar`
                          rather than `ux_b / uy_b` (DIVA's matrix solves
                          the depth-averaged momentum balance directly).
  - **Post-Picard**: compute basal stress from `beta_eff`, recover `ux_b`
                    via the F2 formula, then reconstruct the 3D `ux / uy`
                    via F1.

Port of Fortran `velocity_diva.f90:69-412 calc_velocity_diva`.
"""
function calc_velocity_diva!(y)
    p_ydyn = y.p.ydyn
    p_ymat = y.p.ymat
    ssa    = p_ydyn.ssa_solver

    Nx = size(y.g, 1)
    Ny = size(y.g, 2)
    Nz = size(interior(y.dyn.visc_eff), 3)

    dx_g = y.g.Δxᶜᵃᵃ
    dy_g = y.g.Δyᵃᶜᵃ
    dx = abs(Float64(dx_g isa Number ? dx_g : error("calc_velocity_diva!: stretched x-grid not supported.")))
    dy = abs(Float64(dy_g isa Number ? dy_g : error("calc_velocity_diva!: stretched y-grid not supported.")))

    sc = y.dyn.scratch
    no_slip = p_ydyn.no_slip

    # Step 1 — SSA masks.
    set_ssa_masks!(y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy,
                   y.tpo.mask_frnt, y.tpo.H_ice, y.tpo.f_ice,
                   y.tpo.f_grnd, y.bnd.z_bed, y.bnd.z_sl, dx;
                   use_ssa = true,
                   lateral_bc = p_ydyn.ssa_lat_bc)

    # Step 2 — snapshot for convergence check (DIVA uses ux_bar, not ux_b).
    interior(sc.diva_picard_ux_bar_nm1) .= interior(y.dyn.ux_bar)
    interior(sc.diva_picard_uy_bar_nm1) .= interior(y.dyn.uy_bar)
    interior(sc.ssa_picard_visc_eff_nm1) .= interior(y.dyn.visc_eff)

    fill!(interior(y.dyn.ssa_err_acx), 1.0)
    fill!(interior(y.dyn.ssa_err_acy), 1.0)

    zeta_c = znodes(y.gt, Center())
    set_inactive_margins!(y.dyn.ux_bar, y.dyn.uy_bar, y.tpo.f_ice)

    iter_now = 0
    n_resid_max = length(sc.ssa_residuals)

    for iter in 1:ssa.picard_iter_max
        iter_now = iter

        interior(sc.ssa_picard_visc_eff_nm1) .= interior(y.dyn.visc_eff)
        interior(sc.diva_picard_ux_bar_nm1)  .= interior(y.dyn.ux_bar)
        interior(sc.diva_picard_uy_bar_nm1)  .= interior(y.dyn.uy_bar)

        # Step 3 — viscosity from current depth-averaged velocity.
        # NOTE: Fortran's DIVA recomputes viscosity from a 3D velocity
        # built from ux_bar via vertical shear; here we feed `ux_bar`
        # directly to `calc_visc_eff_3D_*` since that helper interprets
        # its velocity argument as the depth-averaged horizontal
        # component for strain-rate purposes (matching how SSA uses
        # ux_b == ux_bar). The vertical-shear contribution is added by
        # the basal-stress / F-integral coupling — not re-injected
        # here. Follow-up: port Fortran's `calc_vertical_shear_3D` if
        # 3D-strain corrections become important for our benchmarks.
        if p_ydyn.visc_method == 0
            fill!(interior(y.dyn.visc_eff), Float64(p_ydyn.visc_const))
        elseif p_ydyn.visc_method == 1
            calc_visc_eff_3D_nodes!(y.dyn.visc_eff, y.dyn.ux_bar, y.dyn.uy_bar,
                                    y.mat.ATT, y.tpo.H_ice, y.tpo.f_ice,
                                    zeta_c, dx, dy,
                                    p_ymat.n_glen, p_ydyn.eps_0)
        elseif p_ydyn.visc_method == 2
            calc_visc_eff_3D_aa!(y.dyn.visc_eff, y.dyn.ux_bar, y.dyn.uy_bar,
                                 y.mat.ATT, y.tpo.H_ice, y.tpo.f_ice,
                                 zeta_c, dx, dy,
                                 p_ymat.n_glen, p_ydyn.eps_0)
        else
            error("calc_velocity_diva!: visc_method=$(p_ydyn.visc_method) not supported.")
        end

        # Step 3b — log-Picard relax visc.
        if iter > 1
            picard_relax_visc!(y.dyn.visc_eff, sc.ssa_picard_visc_eff_nm1,
                               ssa.picard_relax)
        end

        # Step 4 — depth-integrated viscosity (same as SSA).
        @views interior(sc.ssa_visc_eff_b)[:, :, 1] .=
            interior(y.dyn.visc_eff)[:, :, 1]
        @views interior(sc.ssa_visc_eff_s)[:, :, 1] .=
            interior(y.dyn.visc_eff)[:, :, end]
        calc_visc_eff_int!(y.dyn.visc_eff_int, y.dyn.visc_eff,
                           sc.ssa_visc_eff_b, sc.ssa_visc_eff_s,
                           y.tpo.H_ice, y.tpo.f_ice, zeta_c)

        # Step 4b — F2 integral (DIVA-specific).
        calc_F_integral_2D!(sc.diva_F2, y.dyn.visc_eff,
                            y.tpo.H_ice, y.tpo.f_ice, zeta_c; n = 2.0)

        # Step 5 — beta and beta_eff on aa-cells (uses CURRENT depth-averaged
        # velocity for the friction-law nonlinearity).
        calc_beta!(y.dyn.beta, y.dyn.c_bed, y.dyn.ux_bar, y.dyn.uy_bar,
                   y.tpo.H_ice, y.tpo.f_ice, y.tpo.H_grnd, y.tpo.f_grnd,
                   y.bnd.z_bed, y.bnd.z_sl;
                   beta_method   = p_ydyn.beta_method,
                   beta_const    = p_ydyn.beta_const,
                   beta_q        = p_ydyn.beta_q,
                   beta_u0       = p_ydyn.beta_u0,
                   beta_gl_scale = p_ydyn.beta_gl_scale,
                   beta_gl_f     = p_ydyn.beta_gl_f,
                   H_grnd_lim    = p_ydyn.H_grnd_lim,
                   beta_min      = p_ydyn.beta_min,
                   rho_ice       = y.c.rho_ice, rho_sw = y.c.rho_sw)

        # Step 5b — beta_eff = β / (1 + β·F2) [or 1/F2 for no_slip].
        calc_beta_eff!(sc.diva_beta_eff, y.dyn.beta, sc.diva_F2;
                       no_slip = no_slip)

        # Step 6a — stagger β to face grid (kept for diagnostics; the
        # matrix kernel reads `beta_eff_acx / beta_eff_acy` below).
        stagger_beta!(y.dyn.beta_acx, y.dyn.beta_acy, y.dyn.beta,
                      y.tpo.H_ice, y.tpo.f_ice, y.dyn.ux_bar, y.dyn.uy_bar,
                      y.tpo.f_grnd, y.tpo.f_grnd_acx, y.tpo.f_grnd_acy;
                      beta_gl_stag = p_ydyn.beta_gl_stag,
                      beta_min     = p_ydyn.beta_min)

        # Step 6b — stagger β_eff to face grid (DIVA-specific input
        # to the SSA matrix kernel).
        stagger_beta_eff!(sc.diva_beta_eff_acx, sc.diva_beta_eff_acy,
                          sc.diva_beta_eff,
                          y.tpo.H_ice, y.tpo.f_ice,
                          y.dyn.ux_bar, y.dyn.uy_bar,
                          y.tpo.f_grnd, y.tpo.f_grnd_acx, y.tpo.f_grnd_acy;
                          beta_gl_stag = p_ydyn.beta_gl_stag,
                          beta_min     = p_ydyn.beta_min)

        # Step 6c — corner-stagger viscosity (same as SSA).
        stagger_visc_aa_ab!(sc.ssa_n_aa_ab, y.dyn.visc_eff_int,
                            y.tpo.H_ice, y.tpo.f_ice)

        # Step 7 — assemble SSA matrix with β_eff in place of β.
        # The kernel and inputs are otherwise identical to SSA.
        _assemble_ssa_matrix!(
            sc.ssa_I_idx, sc.ssa_J_idx, sc.ssa_vals,
            sc.ssa_b_vec, sc.ssa_nnz,
            y.dyn.ux_bar, y.dyn.uy_bar,
            sc.diva_beta_eff_acx, sc.diva_beta_eff_acy,
            y.dyn.visc_eff_int, sc.ssa_n_aa_ab,
            y.dyn.ssa_mask_acx, y.dyn.ssa_mask_acy, y.tpo.mask_frnt,
            y.tpo.H_ice, y.tpo.f_ice,
            y.dyn.taud_acx, y.dyn.taud_acy,
            y.dyn.taul_int_acx, y.dyn.taul_int_acy,
            dx, dy, p_ydyn.beta_min;
            boundaries = _ssa_boundaries_symbol(y),
            lateral_bc = p_ydyn.ssa_lat_bc,
        )

        # Step 8 — sparse matrix + Krylov solve.
        nnz = sc.ssa_nnz[]
        I_view = view(sc.ssa_I_idx, 1:nnz)
        J_view = view(sc.ssa_J_idx, 1:nnz)
        V_view = view(sc.ssa_vals,  1:nnz)
        N_rows = 2 * Nx * Ny
        A = sparse(I_view, J_view, V_view, N_rows, N_rows)
        x = _solve_ssa_linear!(sc, A, sc.ssa_b_vec, ssa)

        # Step 9 — unpack into ux_bar / uy_bar (NOT ux_b / uy_b — the
        # DIVA matrix solves for the depth-averaged velocity directly).
        Tx_top = topology(y.dyn.ux_bar.grid, 1)
        Ty_top = topology(y.dyn.uy_bar.grid, 2)
        Uxbar = interior(y.dyn.ux_bar)
        Uybar = interior(y.dyn.uy_bar)
        @inbounds for j in 1:Ny, i in 1:Nx
            row_ux = _row_ux(i, j, Nx)
            row_uy = _row_uy(i, j, Nx)
            ip1f = _ip1_modular(i, Nx, Tx_top)
            jp1f = _jp1_modular(j, Ny, Ty_top)
            Uxbar[ip1f, j, 1] = x[row_ux]
            Uybar[i, jp1f, 1] = x[row_uy]
        end
        if Tx_top === Bounded
            @views Uxbar[1, :, :] .= Uxbar[2, :, :]
        end
        if Ty_top === Bounded
            @views Uybar[:, 1, :] .= Uybar[:, 2, :]
        end

        # Step 9b — NaN-scrub + ssa_vel_max clamp (same safety net as SSA).
        ssa_vel_max = p_ydyn.ssa_vel_max
        nan_seen = false
        @inbounds for k in eachindex(Uxbar)
            v = Uxbar[k]
            if isnan(v)
                Uxbar[k] = 0.0; nan_seen = true
            else
                Uxbar[k] = clamp(v, -ssa_vel_max, +ssa_vel_max)
            end
        end
        @inbounds for k in eachindex(Uybar)
            v = Uybar[k]
            if isnan(v)
                Uybar[k] = 0.0; nan_seen = true
            else
                Uybar[k] = clamp(v, -ssa_vel_max, +ssa_vel_max)
            end
        end
        if nan_seen
            @warn "calc_velocity_diva!: NaN entries from linear solve; replaced with 0 and clamped." iter = iter ssa_vel_max = ssa_vel_max
        end

        # Step 10 — Picard velocity relax.
        if iter > 1
            picard_relax_vel!(y.dyn.ux_bar, y.dyn.uy_bar,
                              sc.diva_picard_ux_bar_nm1,
                              sc.diva_picard_uy_bar_nm1,
                              ssa.picard_relax)
        end

        # Step 11 — zero face velocities at fully-empty margins.
        set_inactive_margins!(y.dyn.ux_bar, y.dyn.uy_bar, y.tpo.f_ice)

        # Step 12 — convergence (L2 relative residual on ux_bar, uy_bar).
        l2_resid = picard_calc_convergence_l2(
            interior(y.dyn.ux_bar), interior(sc.diva_picard_ux_bar_nm1),
            interior(y.dyn.uy_bar), interior(sc.diva_picard_uy_bar_nm1))
        if iter ≤ n_resid_max
            sc.ssa_residuals[iter] = l2_resid
        end
        if l2_resid < ssa.picard_tol && iter > 1
            break
        end
    end

    sc.ssa_iter_now[] = iter_now

    # Post-Picard reconstruction.
    # 1. Basal stress at faces from β_eff and the converged depth-averaged
    #    velocity (Fortran calc_basal_stress at velocity_diva.f90:377).
    calc_basal_stress!(y.dyn.taub_acx, y.dyn.taub_acy,
                       sc.diva_beta_eff_acx, sc.diva_beta_eff_acy,
                       y.dyn.ux_bar, y.dyn.uy_bar)

    # 2. Recover basal velocity via Goldberg (2011) Eq. 34.
    calc_vel_basal_diva!(y.dyn.ux_b, y.dyn.uy_b,
                         y.dyn.ux_bar, y.dyn.uy_bar,
                         y.dyn.taub_acx, y.dyn.taub_acy,
                         sc.diva_F2, y.tpo.f_ice;
                         no_slip = no_slip)

    # 3. Compute F1 cumulative (3D).
    calc_F1_integral_3D!(sc.diva_F1_3D, y.dyn.visc_eff,
                         y.tpo.H_ice, y.tpo.f_ice, zeta_c)

    # 4. Reconstruct 3D ux, uy via Lipscomb (2019) Eq. 29.
    calc_vel_horizontal_3D!(y.dyn.ux, y.dyn.uy,
                            y.dyn.ux_b, y.dyn.uy_b,
                            y.dyn.taub_acx, y.dyn.taub_acy,
                            sc.diva_F1_3D, y.tpo.f_ice)

    # 5. Shearing component ux_i = ux − ux_b, depth-averaged ux_i_bar
    #    is also useful for diagnostics.
    Ux  = interior(y.dyn.ux)
    Uy  = interior(y.dyn.uy)
    Uxi = interior(y.dyn.ux_i)
    Uyi = interior(y.dyn.uy_i)
    Uxb = interior(y.dyn.ux_b)
    Uyb = interior(y.dyn.uy_b)
    @inbounds for k in 1:Nz
        @views Uxi[:, :, k] .= Ux[:, :, k] .- Uxb[:, :, 1]
        @views Uyi[:, :, k] .= Uy[:, :, k] .- Uyb[:, :, 1]
    end

    return y
end
