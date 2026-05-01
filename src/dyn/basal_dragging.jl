# ----------------------------------------------------------------------
# Bed-roughness coefficient chain at aa-cell centres.
#
# Three kernels, all writing 2D Center fields:
#
#   - `calc_cb_ref!` — reference till-friction coefficient (`cb_tgt`):
#     stochastic Gaussian sample over `n_sd` perturbations of `z_bed`
#     (uses `z_bed_sd`), then bedrock-elevation scaling (`scale_zb` ∈
#     {0, 1, 2}) and sediment-cover scaling (`scale_sed` ∈ {0, 1, 2, 3}).
#
#   - `calc_c_bed!` — basal drag coefficient `c_bed = c · N_eff`,
#     where `c = cb_ref` (Pa) or `tan(cb_ref°) · N_eff` (when
#     `is_angle = true`, Bueler & van Pelt 2015 till-strength angle
#     formulation). Optional thermal scaling (`scale_T = 1`) blends
#     toward a frozen-bed reference value as `T'_b` drops below 0.
#
# Both fed from the dyn_step! orchestrator after lateral-stress; both
# operate on Center fields (no staggered indexing).
#
# Port of `yelmo/src/physics/basal_dragging.f90`:
#   - `calc_cb_ref` (line 62)
#   - `calc_c_bed` (line 278)
#   - `calc_lambda_bed_lin` (line 843), `calc_lambda_bed_exp` (line 870)
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.Grids: topology, Bounded, Periodic, AbstractTopology

export calc_cb_ref!, calc_c_bed!, calc_beta!, stagger_beta!

# Linear bedrock-elevation scaling for cb_ref. Returns λ ∈ [0, 1] with
# λ=0 below `z0` (low / no friction) and λ=1 above `z1` (full
# friction). `z_sl` is currently unused — the Fortran reference uses
# present-day sea-level (i.e. `z_rel = z_bed`); the commented-out
# alternative `z_rel = z_sl - z_bed` is kept for parity but inactive.
@inline function _lambda_bed_lin(z_bed::Float64, z_sl::Float64,
                                 z0::Float64, z1::Float64)
    z_rel = z_bed
    lam = (z_rel - z0) / (z1 - z0)
    return clamp(lam, 0.0, 1.0)
end

# Exponential bedrock-elevation scaling. λ = exp((z_bed - z1) / (z1 - z0)),
# clamped to ≤ 1. Equals 1 at `z_bed = z1` and decays exponentially below.
@inline function _lambda_bed_exp(z_bed::Float64, z_sl::Float64,
                                 z0::Float64, z1::Float64)
    z_rel = z_bed
    lam = exp((z_rel - z1) / (z1 - z0))
    return min(lam, 1.0)
end

# n_sd-point Gaussian sampling weights for the stochastic z_bed
# perturbation. Mirrors the `f_sd` / `w_sd` arrays in
# `calc_cb_ref` (line 102–125): linear-spaced abscissae over [-1, 1]
# (i.e. ±1σ) with N(0, 1) weights, normalised to sum to 1. For
# `n_sd = 1` returns single-point quadrature `(0, 1)`.
function _gaussian_z_sd_quadrature(n_sd::Int)
    n_sd ≥ 1 || error("calc_cb_ref!: ytill.n_sd must be ≥ 1; got $n_sd")
    n_sd == 1 && return [0.0], [1.0]
    f_sd = collect(range(-1.0, 1.0; length=n_sd))
    w_sd = (1.0 / sqrt(2π)) .* exp.(-f_sd .^ 2 ./ 2)
    w_sd ./= sum(w_sd)
    return f_sd, w_sd
end

"""
    calc_cb_ref!(cb_ref, z_bed, z_bed_sd, z_sl, H_sed,
                 f_sed, H_sed_min, H_sed_max,
                 cf_ref, cf_min, z0, z1, n_sd,
                 scale_zb, scale_sed) -> cb_ref

Reference till-friction coefficient on aa-cells. Two-stage:

  1. **Elevation scaling** (`scale_zb`): per-cell, Gaussian-sample
     `z_bed` ± `f_sd · z_bed_sd` (n_sd points), evaluate
     `lambda_bed`, take a weighted mean → `cb_ref = cf_ref · λ̄`,
     clamped to ≥ `cf_min`. `scale_zb = 0` skips elevation
     scaling and writes `cb_ref = cf_ref`. `1` → linear `λ`,
     `2` → exponential `λ`.
  2. **Sediment scaling** (`scale_sed`): adjusts the elevation
     result with `H_sed` cover. `0` skips. `1` writes
     `min(cb_ref, sed_lin)` where `sed_lin` is a linear blend
     `cf_min ↔ cf_ref` over `H_sed ∈ [H_sed_min, H_sed_max]`.
     `2`/`3` apply a multiplicative factor `1 - (1 - f_sed) · λ_sed`;
     `2` reapplies the `cf_min` floor, `3` does NOT (Schannwell
     et al. 2023 method).

Port of `basal_dragging.f90:62 calc_cb_ref`.
"""
function calc_cb_ref!(cb_ref,
                      z_bed, z_bed_sd, z_sl, H_sed,
                      f_sed::Real, H_sed_min::Real, H_sed_max::Real,
                      cf_ref::Real, cf_min::Real,
                      z0::Real, z1::Real,
                      n_sd::Int,
                      scale_zb::Int, scale_sed::Int)
    f_sd_arr, w_sd_arr = _gaussian_z_sd_quadrature(n_sd)

    cb_ref_int = interior(cb_ref)
    Zb_int     = interior(z_bed)
    Zsd_int    = interior(z_bed_sd)
    Zsl_int    = interior(z_sl)
    Hs_int     = interior(H_sed)

    Nx, Ny = size(cb_ref_int, 1), size(cb_ref_int, 2)
    cf_ref_f = Float64(cf_ref)
    cf_min_f = Float64(cf_min)
    z0_f, z1_f = Float64(z0), Float64(z1)

    # === 1. Elevation scaling ===
    if scale_zb == 0
        # No elevation scaling — uniform reference value.
        fill!(cb_ref_int, cf_ref_f)
    elseif scale_zb == 1 || scale_zb == 2
        λ_fn = scale_zb == 1 ? _lambda_bed_lin : _lambda_bed_exp
        @inbounds for j in 1:Ny, i in 1:Nx
            zb_ij  = Zb_int[i, j, 1]
            zsd_ij = Zsd_int[i, j, 1]
            zsl_ij = Zsl_int[i, j, 1]
            acc = 0.0
            for q in eachindex(f_sd_arr)
                λ = λ_fn(zb_ij + f_sd_arr[q] * zsd_ij, zsl_ij, z0_f, z1_f)
                cbq = cf_ref_f * λ
                cbq < cf_min_f && (cbq = cf_min_f)
                acc += cbq * w_sd_arr[q]
            end
            cb_ref_int[i, j, 1] = acc
        end
    else
        error("calc_cb_ref!: ytill.scale_zb must be 0, 1, or 2; got $scale_zb")
    end

    # === 2. Sediment scaling ===
    if scale_sed == 0
        # No sediment scaling — leave elevation result alone.
    elseif scale_sed == 1
        @inbounds for j in 1:Ny, i in 1:Nx
            λ = clamp((Hs_int[i, j, 1] - H_sed_min) / (H_sed_max - H_sed_min), 0.0, 1.0)
            cb_ref_now = cf_min_f * λ + cf_ref_f * (1.0 - λ)
            cb_ref_int[i, j, 1] = min(cb_ref_int[i, j, 1], cb_ref_now)
        end
    elseif scale_sed == 2 || scale_sed == 3
        @inbounds for j in 1:Ny, i in 1:Nx
            λ = clamp((Hs_int[i, j, 1] - H_sed_min) / (H_sed_max - H_sed_min), 0.0, 1.0)
            scale = 1.0 - (1.0 - Float64(f_sed)) * λ
            cb_ref_int[i, j, 1] *= scale
            if scale_sed == 2 && cb_ref_int[i, j, 1] < cf_min_f
                cb_ref_int[i, j, 1] = cf_min_f
            end
        end
    else
        error("calc_cb_ref!: ytill.scale_sed must be 0, 1, 2, or 3; got $scale_sed")
    end

    return cb_ref
end

"""
    calc_c_bed!(c_bed, cb_ref, N_eff, T_prime_b,
                is_angle, cf_ref, T_frz, scale_T) -> c_bed

Basal drag coefficient at aa-cells:

    c_bed = c · N_eff

with `c = cb_ref` (Pa) by default, or `c = tan(cb_ref°)` when
`is_angle = true` (Bueler & van Pelt 2015 till-strength angle
formulation; `cb_ref` is then in degrees).

If `scale_T = 1`, a per-cell linear blend toward the frozen-bed
reference value `c_bed_frz = c_frz · N_eff` is applied as the
homologous basal temperature `T'_b` drops below 0:

    λ = clamp((T'_b - T_frz) / (0 - T_frz), 0, 1)
    c_bed = c_bed · λ + c_bed_frz · (1 - λ)

`T_frz` must be < 0; an error is raised otherwise.

`T_prime_b` is the homologous temperature `T - T_pmp` at the base —
0 at melting, negative when frozen, expressed in degC despite the
`K` label in the variable schema (the value semantics is degrees-
relative-to-melting, not Kelvin).

Port of `basal_dragging.f90:278 calc_c_bed`.
"""
function calc_c_bed!(c_bed, cb_ref, N_eff, T_prime_b,
                     is_angle::Bool, cf_ref::Real,
                     T_frz::Real, scale_T::Int)
    c_int    = interior(c_bed)
    cb_int   = interior(cb_ref)
    N_int    = interior(N_eff)
    Tp_int   = interior(T_prime_b)

    Nx, Ny = size(c_int, 1), size(c_int, 2)
    cf_ref_f = Float64(cf_ref)
    T_frz_f  = Float64(T_frz)

    # First pass: c_bed = c · N_eff with the appropriate transform.
    cf_frz = if is_angle
        @inbounds for j in 1:Ny, i in 1:Nx
            c_int[i, j, 1] = tan(cb_int[i, j, 1] * π / 180) * N_int[i, j, 1]
        end
        tan(cf_ref_f * π / 180)
    else
        @inbounds for j in 1:Ny, i in 1:Nx
            c_int[i, j, 1] = cb_int[i, j, 1] * N_int[i, j, 1]
        end
        cf_ref_f
    end

    # Optional thermal scaling toward c_bed_frz as T'_b → T_frz.
    if scale_T == 0
        # No-op.
    elseif scale_T == 1
        T_frz_f >= 0.0 && error(
            "calc_c_bed!: ydyn.T_frz must be < 0 for scale_T = 1; got $T_frz_f")
        @inbounds for j in 1:Ny, i in 1:Nx
            λ = clamp((Tp_int[i, j, 1] - T_frz_f) / (0.0 - T_frz_f), 0.0, 1.0)
            c_bed_frz = cf_frz * N_int[i, j, 1]
            c_int[i, j, 1] = c_int[i, j, 1] * λ + c_bed_frz * (1.0 - λ)
        end
    else
        error("calc_c_bed!: ydyn.scale_T must be 0 or 1; got $scale_T")
    end

    return c_bed
end

# ----------------------------------------------------------------------
# SSA basal-friction coefficient beta on aa-cells.
#
# Three groups of helpers:
#
#   1. Per-cell `calc_beta_aa_*` kernels (private):
#        - power-plastic (Bueler & van Pelt 2015)
#        - regularized Coulomb (Joughin et al. 2019)
#      Both compute `beta` on aa-cells from `c_bed`, basal velocity
#      `(ux_b, uy_b)` (acx/acy faces), and exponent / regularization
#      params. Each averages over Gauss-quadrature nodes per cell using
#      the 4-node 2D Gauss-Legendre rule from `quadrature.jl`. The
#      `simple_stagger` flag bypasses Gauss-quadrature corner-staggering
#      and uses central averages instead — matches the Fortran
#      `simple_stagger=.TRUE.` branch (case 4/5 of `beta_method`),
#      useful for the Schoof slab test.
#
#   2. `scale_beta_gl_*` post-scaling helpers (private):
#        - fraction (`beta_gl_scale=0`): GL cells multiplied by f_gl.
#        - Hgrnd    (`beta_gl_scale=1`): linear blend toward 0 as
#          H_grnd → H_grnd_lim.
#        - zstar    (`beta_gl_scale=2`): Gladstone et al. 2017
#          floatation-fraction scaling.
#        - f_grnd   (`beta_gl_scale=3`): in-line `beta = f_grnd * beta`.
#
#   3. Public `calc_beta!` driver (line-numbered to Fortran source):
#        - dispatches on `beta_method` ∈ [-1, 5];
#        - applies `beta_gl_scale` post-scaling;
#        - zeros beta on purely floating cells;
#        - clamps `beta_min` floor on positive entries.
#
#   4. Public `stagger_beta!` driver:
#        - dispatches on `beta_gl_stag` ∈ [-1, 0, 1, 2, 3, 4];
#        - first applies the standard mean staggering (case 0);
#        - then applies the GL-zone modifier per `beta_gl_stag`;
#        - clamps `beta_min` on positive faces.
#
# Port of `yelmo/src/physics/basal_dragging.f90`:
#   - calc_beta                    (line 356)
#   - stagger_beta                 (line 528)
#   - calc_beta_aa_power_plastic   (line 902)
#   - calc_beta_aa_reg_coulomb     (line 1003)
#   - scale_beta_gl_fraction       (line 1122)
#   - scale_beta_gl_Hgrnd          (line 1174)
#   - scale_beta_gl_zstar          (line 1212)
#   - stagger_beta_aa_mean         (line 1275)
#   - stagger_beta_aa_gl_upstream  (line 1350)
#   - stagger_beta_aa_gl_downstream (line 1415)
#   - stagger_beta_aa_gl_subgrid   (line 1481)
#   - stagger_beta_aa_gl_subgrid_flux (line 1559)
#   - calc_beta_gl_flux_weight     (line 1655)
#
# Yelmo.jl indexing convention (matches `driving_stress.jl` /
# `velocity_sia.jl`): an XFaceField value for "Fortran ux_b(i, j)" lives
# at `interior(ux_b)[i+1, j, 1]`; YFaceField value for "Fortran uy_b(i,
# j)" lives at `interior(uy_b)[i, j+1, 1]`. The leading face slot
# `[1, :, :]` / `[:, 1, :]` is conventionally a replicate of the
# adjacent interior face. The `boundaries` parameter is ignored here —
# Yelmo.jl uses Oceananigans-style halo / clamped indices, matching the
# Fortran `infinite` BC.
# ----------------------------------------------------------------------

const _UB_MIN     = 1e-3       # [m/yr] basal-velocity floor (Fortran ub_min)
const _UB_SQ_MIN  = _UB_MIN^2

# --- Private helper: power-plastic friction law (Fortran line 902) ---
#
# beta_aa = mean over 4 Gauss-quadrature nodes of:
#     c_bed(node) * (uxy(node) / u_0)^q * (1 / uxy(node))
# with uxy(node) = sqrt(ux(node)^2 + uy(node)^2 + ub_sq_min).
#
# `simple_stagger=true` short-circuits the 4-node Gauss-quadrature
# averaging: cbn = c_bed(i, j); uxn = mean of acx neighbours of cell
# (i, j); uyn = mean of acy neighbours. Used by `beta_method=4`
# (Schoof slab test).
#
# Boundary handling: topology-dispatched neighbour helpers
# `_neighbor_im1` / `_ip1_modular` etc. — under Bounded these clamp /
# resolve to k+1 (Fortran `infinite`-BC behaviour); under Periodic
# they wrap modularly. Mirrors Fortran `get_neighbor_indices_bc_codes`
# (`yelmo_tools.f90:162`) which dispatches on BC.
function _calc_beta_aa_power_plastic!(beta_int::AbstractArray,
                                      ux_int::AbstractArray, uy_int::AbstractArray,
                                      c_int::AbstractArray, fi_int::AbstractArray,
                                      q::Float64, u_0::Float64,
                                      simple_stagger::Bool,
                                      Tx_top::Type{<:AbstractTopology},
                                      Ty_top::Type{<:AbstractTopology})
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)

    # Initialize friction to zero everywhere (Fortran line 949).
    fill!(beta_int, 0.0)

    # 2D Gauss-Legendre nodes (4 nodes for n=2 — counter-clockwise from
    # SW corner; matches Fortran `gq2D_init`).
    xr, yr, wt, wt_tot = gq2d_nodes(2)

    @inbounds for j in 1:Ny, i in 1:Nx
        # Topology-aware neighbour indices (Fortran line 955).
        im1 = _neighbor_im1(i, Nx, Tx_top)
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jm1 = _neighbor_jm1(j, Ny, Ty_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)

        # Yelmo face-array indices for Fortran "ux_b(k, j)" / "uy_b(i, k)".
        # `_ip1_modular(k, Nx, Tx)` → `k+1` under Bounded, wrap under Periodic.
        ux_im1 = _ip1_modular(im1, Nx, Tx_top)
        ux_i   = _ip1_modular(i,   Nx, Tx_top)
        uy_jm1 = _jp1_modular(jm1, Ny, Ty_top)
        uy_j   = _jp1_modular(j,   Ny, Ty_top)

        if fi_int[i, j, 1] == 1.0
            # Fully ice-covered point.

            # Convert Fortran "ux_b(i, j)" reads to Yelmo.jl indices.
            # ---- 4-corner staggering: c_bed (aa-grid) ----
            cb_ij = c_int[i, j, 1]
            if simple_stagger
                # Use central c_bed for all nodes.
                cbn = (cb_ij, cb_ij, cb_ij, cb_ij)
                # Velocity components: simple central avg.
                ux_aa = 0.5 * (ux_int[ux_i, j, 1] + ux_int[ux_im1, j, 1])
                uy_aa = 0.5 * (uy_int[i, uy_j, 1] + uy_int[i, uy_jm1, 1])
                uxn = (ux_aa, ux_aa, ux_aa, ux_aa)
                uyn = (uy_aa, uy_aa, uy_aa, uy_aa)
            else
                # 4-corner aa-stagger for c_bed (Fortran line 350-353).
                v_ab_c = (
                    0.25 * (c_int[im1, jm1, 1] + c_int[i, jm1, 1] + c_int[im1, j, 1] + c_int[i, j, 1]),  # SW
                    0.25 * (c_int[i, jm1, 1] + c_int[ip1, jm1, 1] + c_int[i, j, 1] + c_int[ip1, j, 1]),  # SE
                    0.25 * (c_int[i, j, 1] + c_int[ip1, j, 1] + c_int[i, jp1, 1] + c_int[ip1, jp1, 1]),  # NE
                    0.25 * (c_int[im1, j, 1] + c_int[i, j, 1] + c_int[im1, jp1, 1] + c_int[i, jp1, 1]),  # NW
                )
                # acx-stagger for ux_b (Fortran line 406-409). Yelmo
                # ux array slot for Fortran `ux(k, j)` is
                # `_ip1_modular(k, Nx, Tx)`.
                v_ab_ux = (
                    0.5 * (ux_int[ux_im1, jm1, 1] + ux_int[ux_im1, j,   1]),  # SW: ux(im1, jm1) avg ux(im1, j)
                    0.5 * (ux_int[ux_i,   jm1, 1] + ux_int[ux_i,   j,   1]),  # SE: ux(i,   jm1) avg ux(i,   j)
                    0.5 * (ux_int[ux_i,   j,   1] + ux_int[ux_i,   jp1, 1]),  # NE: ux(i,   j  ) avg ux(i,   jp1)
                    0.5 * (ux_int[ux_im1, j,   1] + ux_int[ux_im1, jp1, 1]),  # NW: ux(im1, j  ) avg ux(im1, jp1)
                )
                # acy-stagger for uy_b (Fortran line 434-437).
                v_ab_uy = (
                    0.5 * (uy_int[im1, uy_jm1, 1] + uy_int[i,   uy_jm1, 1]),  # SW
                    0.5 * (uy_int[i,   uy_jm1, 1] + uy_int[ip1, uy_jm1, 1]),  # SE
                    0.5 * (uy_int[i,   uy_j,   1] + uy_int[ip1, uy_j,   1]),  # NE
                    0.5 * (uy_int[im1, uy_j,   1] + uy_int[i,   uy_j,   1]),  # NW
                )

                # Map corner values to the 4 quadrature nodes via the
                # bilinear shape functions.
                cbn = ntuple(p -> gq2d_interp_to_node(v_ab_c,  xr[p], yr[p]), 4)
                uxn = ntuple(p -> gq2d_interp_to_node(v_ab_ux, xr[p], yr[p]), 4)
                uyn = ntuple(p -> gq2d_interp_to_node(v_ab_uy, xr[p], yr[p]), 4)
            end

            # Fortran line 983: uxyn = sqrt(uxn² + uyn² + ub_sq_min).
            # Fortran line 986: betan = c_bed(i, j) * (uxyn / u_0)^q * (1 / uxyn)
            #                       NOTE the c_bed(i, j) in the formula —
            #                       not cbn — for the power-plastic case
            #                       (matches Fortran line 986 verbatim).
            acc = 0.0
            for p in 1:4
                uxy = sqrt(uxn[p]^2 + uyn[p]^2 + _UB_SQ_MIN)
                # Mirrors Fortran line 986 `c_bed(i,j)` (not cbn(p)).
                # `cbn` is computed but only used by the reg_coulomb
                # variant below.
                betan = cb_ij * (uxy / u_0)^q * (1.0 / uxy)
                acc += betan * wt[p]
            end
            beta_int[i, j, 1] = acc / wt_tot
        else
            # Not fully ice-covered: simple non-staggered fallback
            # (Fortran line 989-993).
            uxy_b = _UB_MIN
            beta_int[i, j, 1] = c_int[i, j, 1] * (uxy_b / u_0)^q * (1.0 / uxy_b)
        end
    end
    return beta_int
end

# --- Private helper: regularized-Coulomb friction (Fortran line 1003) ---
#
# Same shape as power-plastic but with the regularized Coulomb form
# (Joughin et al. 2019, GRL Eq. 2):
#     beta_aa = mean over nodes of:
#         cbn(node) * (uxy(node) / (uxy(node) + u_0))^q * (1 / uxy(node))
# Note the use of `cbn(node)` (not `c_bed(i, j)`) — Fortran line 1096.
function _calc_beta_aa_reg_coulomb!(beta_int::AbstractArray,
                                    ux_int::AbstractArray, uy_int::AbstractArray,
                                    c_int::AbstractArray, fi_int::AbstractArray,
                                    q::Float64, u_0::Float64,
                                    simple_stagger::Bool,
                                    Tx_top::Type{<:AbstractTopology},
                                    Ty_top::Type{<:AbstractTopology})
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    fill!(beta_int, 0.0)
    xr, yr, wt, wt_tot = gq2d_nodes(2)

    @inbounds for j in 1:Ny, i in 1:Nx
        # Topology-aware neighbour indices (Fortran line 1066, "infinite" BC
        # under Bounded; periodic wrap under Periodic).
        im1 = _neighbor_im1(i, Nx, Tx_top)
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jm1 = _neighbor_jm1(j, Ny, Ty_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)

        # Yelmo face-array indices for Fortran "ux_b(k, j)" / "uy_b(i, k)".
        ux_im1 = _ip1_modular(im1, Nx, Tx_top)
        ux_i   = _ip1_modular(i,   Nx, Tx_top)
        uy_jm1 = _jp1_modular(jm1, Ny, Ty_top)
        uy_j   = _jp1_modular(j,   Ny, Ty_top)

        cb_ij = c_int[i, j, 1]

        if fi_int[i, j, 1] == 1.0
            if simple_stagger
                cbn = (cb_ij, cb_ij, cb_ij, cb_ij)
                ux_aa = 0.5 * (ux_int[ux_i, j, 1] + ux_int[ux_im1, j, 1])
                uy_aa = 0.5 * (uy_int[i, uy_j, 1] + uy_int[i, uy_jm1, 1])
                uxn = (ux_aa, ux_aa, ux_aa, ux_aa)
                uyn = (uy_aa, uy_aa, uy_aa, uy_aa)
            else
                v_ab_c = (
                    0.25 * (c_int[im1, jm1, 1] + c_int[i, jm1, 1] + c_int[im1, j, 1] + c_int[i, j, 1]),
                    0.25 * (c_int[i, jm1, 1] + c_int[ip1, jm1, 1] + c_int[i, j, 1] + c_int[ip1, j, 1]),
                    0.25 * (c_int[i, j, 1] + c_int[ip1, j, 1] + c_int[i, jp1, 1] + c_int[ip1, jp1, 1]),
                    0.25 * (c_int[im1, j, 1] + c_int[i, j, 1] + c_int[im1, jp1, 1] + c_int[i, jp1, 1]),
                )
                v_ab_ux = (
                    0.5 * (ux_int[ux_im1, jm1, 1] + ux_int[ux_im1, j,   1]),
                    0.5 * (ux_int[ux_i,   jm1, 1] + ux_int[ux_i,   j,   1]),
                    0.5 * (ux_int[ux_i,   j,   1] + ux_int[ux_i,   jp1, 1]),
                    0.5 * (ux_int[ux_im1, j,   1] + ux_int[ux_im1, jp1, 1]),
                )
                v_ab_uy = (
                    0.5 * (uy_int[im1, uy_jm1, 1] + uy_int[i,   uy_jm1, 1]),
                    0.5 * (uy_int[i,   uy_jm1, 1] + uy_int[ip1, uy_jm1, 1]),
                    0.5 * (uy_int[i,   uy_j,   1] + uy_int[ip1, uy_j,   1]),
                    0.5 * (uy_int[im1, uy_j,   1] + uy_int[i,   uy_j,   1]),
                )
                cbn = ntuple(p -> gq2d_interp_to_node(v_ab_c,  xr[p], yr[p]), 4)
                uxn = ntuple(p -> gq2d_interp_to_node(v_ab_ux, xr[p], yr[p]), 4)
                uyn = ntuple(p -> gq2d_interp_to_node(v_ab_uy, xr[p], yr[p]), 4)
            end

            # Fortran line 1096: betan = cbn * (uxyn / (uxyn+u_0))^q * (1/uxyn)
            acc = 0.0
            for p in 1:4
                uxy = sqrt(uxn[p]^2 + uyn[p]^2 + _UB_SQ_MIN)
                betan = cbn[p] * (uxy / (uxy + u_0))^q * (1.0 / uxy)
                acc += betan * wt[p]
            end
            beta_int[i, j, 1] = acc / wt_tot
        else
            # Fortran line 1099-1103.
            uxy_b = _UB_MIN
            beta_int[i, j, 1] = cb_ij * (uxy_b / (uxy_b + u_0))^q * (1.0 / uxy_b)
        end
    end
    return beta_int
end

# --- Private GL post-scaling helpers (Fortran line 1122 onward) ---

# Fortran line 1122 `scale_beta_gl_fraction`: GL cells (any neighbour
# floating) get `beta *= f_gl`. The Fortran loop iterates `i = 1, nx-1`
# (not `nx`); we mirror that.
function _scale_beta_gl_fraction!(beta_int::AbstractArray,
                                  fg_int::AbstractArray,
                                  f_gl::Float64,
                                  Tx_top::Type{<:AbstractTopology},
                                  Ty_top::Type{<:AbstractTopology})
    (0.0 ≤ f_gl ≤ 1.0) || error("scale_beta_gl_fraction: f_gl must be in [0, 1]; got $f_gl")
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:(Nx - 1)
        # Topology-aware neighbour indices: clamp under Bounded
        # (Fortran "infinite" BC), wrap under Periodic.
        im1 = _neighbor_im1(i, Nx, Tx_top)
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jm1 = _neighbor_jm1(j, Ny, Ty_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        if fg_int[i, j, 1] > 0.0 &&
           (fg_int[im1, j, 1] == 0.0 || fg_int[ip1, j, 1] == 0.0 ||
            fg_int[i, jm1, 1] == 0.0 || fg_int[i, jp1, 1] == 0.0)
            beta_int[i, j, 1] *= f_gl
        end
    end
    return beta_int
end

# Fortran line 1174 `scale_beta_gl_Hgrnd`: linear blend toward 0 as
# H_grnd → 0.
function _scale_beta_gl_Hgrnd!(beta_int::AbstractArray,
                               Hg_int::AbstractArray,
                               H_grnd_lim::Float64)
    H_grnd_lim > 0.0 || error("scale_beta_gl_Hgrnd: H_grnd_lim must be > 0; got $H_grnd_lim")
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        f_scale = max(min(Hg_int[i, j, 1], H_grnd_lim) / H_grnd_lim, 0.0)
        beta_int[i, j, 1] *= f_scale
    end
    return beta_int
end

# Fortran line 1212 `scale_beta_gl_zstar`: Gladstone et al. 2017 floatation
# scaling. `norm=true` always; `f_scale = max(0, H_eff - (z_sl - z_bed) ·
# rho_sw / rho_ice) / H_eff` for marine, `f_scale = 1` for land-based.
function _scale_beta_gl_zstar!(beta_int::AbstractArray,
                               H_int::AbstractArray, fi_int::AbstractArray,
                               zb_int::AbstractArray, zsl_int::AbstractArray,
                               rho_ice::Float64, rho_sw::Float64)
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    rho_sw_ice = rho_sw / rho_ice
    @inbounds for j in 1:Ny, i in 1:Nx
        if fi_int[i, j, 1] > 0.0
            # H_eff = H_ice * f_ice (Fortran calc_H_eff w/ set_frac_zero=true).
            # set_frac_zero means: if f_ice < 1 → H_eff = 0.
            H_eff = fi_int[i, j, 1] == 1.0 ? H_int[i, j, 1] : 0.0
            f_scale = if zb_int[i, j, 1] > zsl_int[i, j, 1]
                H_eff
            else
                max(0.0, H_eff - (zsl_int[i, j, 1] - zb_int[i, j, 1]) * rho_sw_ice)
            end
            # norm=true (Fortran line 461 always passes `.TRUE.`).
            if H_eff > 0.0
                f_scale /= H_eff
            end
            beta_int[i, j, 1] *= f_scale
        end
    end
    return beta_int
end

"""
    calc_beta!(beta, c_bed, ux_b, uy_b, H_ice, f_ice, H_grnd, f_grnd,
               z_bed, z_sl;
               beta_method::Int, beta_const::Real,
               beta_q::Real, beta_u0::Real,
               beta_gl_scale::Int, beta_gl_f::Real,
               H_grnd_lim::Real, beta_min::Real,
               rho_ice::Real, rho_sw::Real) -> beta

SSA basal-friction coefficient on aa-cells. Mirrors Fortran
`basal_dragging.f90:356 calc_beta`.

`beta_method` selects the friction law:
  - `-1`: external (no-op; assumes `beta` already populated).
  -  `0`: constant `beta_const`.
  -  `1`: linear law via power-plastic with `q = 1`.
  -  `2`: power-plastic (Bueler & van Pelt 2015) with `q = beta_q`.
  -  `3`: regularized Coulomb (Joughin et al. 2019) with `q = beta_q`.
  -  `4`: power-plastic with `simple_stagger=true` (Schoof slab test).
  -  `5`: regularized Coulomb with `simple_stagger=true`.

`beta_gl_scale` post-scales beta near the grounding line:
  - `0`: fractional `beta *= beta_gl_f` for GL cells; `beta_gl_f=1.0`
    is a no-op.
  - `1`: `beta *= max(min(H_grnd, H_grnd_lim) / H_grnd_lim, 0)`.
  - `2`: zstar scaling (Gladstone et al. 2017).
  - `3`: `beta *= f_grnd` directly.

Then beta is zeroed on purely floating cells (`f_grnd == 0`), and
positive values are clamped to `≥ beta_min`.

Boundary handling: Yelmo.jl uses Oceananigans halo / clamped indices,
matching the Fortran `infinite` BC (Fortran's MISMIP3D / periodic
special cases are NOT implemented here — those would require a
topology-aware kernel, deferred to a later milestone).

`ux_b` is an XFaceField with values for cell `(i, j)` at array index
`[i+1, j, 1]`; `uy_b` is a YFaceField analogous.
"""
function calc_beta!(beta, c_bed, ux_b, uy_b,
                    H_ice, f_ice, H_grnd, f_grnd,
                    z_bed, z_sl;
                    beta_method::Int,
                    beta_const::Real = 0.0,
                    beta_q::Real,
                    beta_u0::Real,
                    beta_gl_scale::Int,
                    beta_gl_f::Real = 1.0,
                    H_grnd_lim::Real,
                    beta_min::Real,
                    rho_ice::Real, rho_sw::Real)
    beta_int = interior(beta)
    c_int    = interior(c_bed)
    ux_int   = interior(ux_b)
    uy_int   = interior(uy_b)
    H_int    = interior(H_ice)
    fi_int   = interior(f_ice)
    Hg_int   = interior(H_grnd)
    fg_int   = interior(f_grnd)
    zb_int   = interior(z_bed)
    zsl_int  = interior(z_sl)

    Tx_top = topology(beta.grid, 1)
    Ty_top = topology(beta.grid, 2)

    # === Step 1 (Fortran line 393-437): beta_method dispatch ===
    if beta_method == -1
        # External — no-op.
    elseif beta_method == 0
        fill!(beta_int, Float64(beta_const))
    elseif beta_method == 1
        # Linear law via power-plastic (q=1).
        _calc_beta_aa_power_plastic!(beta_int, ux_int, uy_int, c_int, fi_int,
                                     1.0, Float64(beta_u0), false,
                                     Tx_top, Ty_top)
    elseif beta_method == 2
        _calc_beta_aa_power_plastic!(beta_int, ux_int, uy_int, c_int, fi_int,
                                     Float64(beta_q), Float64(beta_u0), false,
                                     Tx_top, Ty_top)
    elseif beta_method == 3
        _calc_beta_aa_reg_coulomb!(beta_int, ux_int, uy_int, c_int, fi_int,
                                   Float64(beta_q), Float64(beta_u0), false,
                                   Tx_top, Ty_top)
    elseif beta_method == 4
        _calc_beta_aa_power_plastic!(beta_int, ux_int, uy_int, c_int, fi_int,
                                     Float64(beta_q), Float64(beta_u0), true,
                                     Tx_top, Ty_top)
    elseif beta_method == 5
        _calc_beta_aa_reg_coulomb!(beta_int, ux_int, uy_int, c_int, fi_int,
                                   Float64(beta_q), Float64(beta_u0), true,
                                   Tx_top, Ty_top)
    else
        error("calc_beta!: beta_method = $beta_method not recognised; expected -1..5.")
    end

    # === Step 2 (Fortran line 444-476): beta_gl_scale post-scaling ===
    if beta_gl_scale == 0
        _scale_beta_gl_fraction!(beta_int, fg_int, Float64(beta_gl_f),
                                 Tx_top, Ty_top)
    elseif beta_gl_scale == 1
        _scale_beta_gl_Hgrnd!(beta_int, Hg_int, Float64(H_grnd_lim))
    elseif beta_gl_scale == 2
        _scale_beta_gl_zstar!(beta_int, H_int, fi_int, zb_int, zsl_int,
                              Float64(rho_ice), Float64(rho_sw))
    elseif beta_gl_scale == 3
        Nx, Ny = size(beta_int, 1), size(beta_int, 2)
        @inbounds for j in 1:Ny, i in 1:Nx
            beta_int[i, j, 1] *= fg_int[i, j, 1]
        end
    else
        error("calc_beta!: beta_gl_scale = $beta_gl_scale not recognised; expected 0..3.")
    end

    # === Step 3 (Fortran line 483): zero beta on purely floating cells.
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        if fg_int[i, j, 1] == 0.0
            beta_int[i, j, 1] = 0.0
        end
    end

    # === Step 4 (Fortran line 515): beta_min floor on positive values.
    bm = Float64(beta_min)
    @inbounds for j in 1:Ny, i in 1:Nx
        b = beta_int[i, j, 1]
        if b > 0.0 && b < bm
            beta_int[i, j, 1] = bm
        end
    end

    return beta
end

# --- Private GL flux-weight helper (Fortran line 1655) ---
#
# Compute weighted beta at acx/acy face when GL-aware subgrid_flux is
# active. Mirrors Fortran exactly, including the 100-segment quadrature.
function _calc_beta_gl_flux_weight(beta_a::Float64, beta_b::Float64,
                                   ux_a::Float64, ux_b::Float64,
                                   H_a::Float64, H_b::Float64,
                                   f_grnd_ac::Float64)
    nseg = 100
    q_a = ux_a * H_a
    q_b = ux_b * H_b
    uu_grnd = 0.0
    uu_tot  = 0.0
    @inbounds for i in 1:nseg
        lambda = (i - 1) / (nseg - 1)
        # Fortran line 1702-1703 mixes lambda counter-intuitively (a
        # weight increases with lambda). Mirror verbatim.
        H_now = H_a * lambda + H_b * (1 - lambda)
        q_now = q_a * lambda + q_b * (1 - lambda)
        # u_now is computed in Fortran but only `q_now` is summed
        # (line 1713). We mirror that.
        uu_tot += q_now
        if lambda < f_grnd_ac
            uu_grnd += q_now
        end
    end
    weight = uu_tot > 0.0 ? uu_grnd / uu_tot : 0.0
    return beta_a * weight
end

# --- Private staggering helpers (Fortran line 1275-1653) ---

# Fortran line 1275 `stagger_beta_aa_mean`. Standard mean staggering
# with margin handling (one-sided face value when one neighbour is
# ice-free).
function _stagger_beta_aa_mean!(beta_acx_int::AbstractArray, beta_acy_int::AbstractArray,
                                beta_int::AbstractArray, fi_int::AbstractArray,
                                fg_int::AbstractArray,
                                Tx_top::Type{<:AbstractTopology},
                                Ty_top::Type{<:AbstractTopology})
    # beta_acx_int has interior shape (Nx+1, Ny, 1) under Bounded and
    # (Nx, Ny, 1) under Periodic. The face-east of cell (i, j) lives at
    # `_ip1_modular(i, Nx, Tx_top)`: `i+1` under Bounded, wrapped under
    # Periodic.
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)

        # acx-face (Fortran line 1305-1322).
        if fg_int[i, j, 1] == 0.0 && fg_int[ip1, j, 1] == 0.0
            beta_acx_int[ip1f, j, 1] = 0.0
        else
            if fi_int[i, j, 1] == 1.0 && fi_int[ip1, j, 1] < 1.0
                beta_acx_int[ip1f, j, 1] = beta_int[i, j, 1]
            elseif fi_int[i, j, 1] < 1.0 && fi_int[ip1, j, 1] == 1.0
                beta_acx_int[ip1f, j, 1] = beta_int[ip1, j, 1]
            else
                beta_acx_int[ip1f, j, 1] = 0.5 * (beta_int[i, j, 1] + beta_int[ip1, j, 1])
            end
        end

        # acy-face (Fortran line 1324-1341).
        if fg_int[i, j, 1] == 0.0 && fg_int[i, jp1, 1] == 0.0
            beta_acy_int[i, jp1f, 1] = 0.0
        else
            if fi_int[i, j, 1] == 1.0 && fi_int[i, jp1, 1] < 1.0
                beta_acy_int[i, jp1f, 1] = beta_int[i, j, 1]
            elseif fi_int[i, j, 1] < 1.0 && fi_int[i, jp1, 1] == 1.0
                beta_acy_int[i, jp1f, 1] = beta_int[i, jp1, 1]
            else
                beta_acy_int[i, jp1f, 1] = 0.5 * (beta_int[i, j, 1] + beta_int[i, jp1, 1])
            end
        end
    end
    # Replicate the leading face slot (matches `calc_driving_stress!`).
    # Bounded only — under Periodic the leading slot is the wrapped
    # eastern/northern face that was just populated by the loop.
    if Tx_top === Bounded
        @views beta_acx_int[1, :, :] .= beta_acx_int[2, :, :]
    end
    if Ty_top === Bounded
        @views beta_acy_int[:, 1, :] .= beta_acy_int[:, 2, :]
    end
    return nothing
end

# Fortran line 1350 `stagger_beta_aa_gl_upstream`. Upstream rule at
# the GL: face uses beta from the grounded neighbour.
function _stagger_beta_aa_gl_upstream!(beta_acx_int::AbstractArray,
                                       beta_acy_int::AbstractArray,
                                       beta_int::AbstractArray,
                                       fi_int::AbstractArray,
                                       fg_int::AbstractArray,
                                       Tx_top::Type{<:AbstractTopology},
                                       Ty_top::Type{<:AbstractTopology})
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)
        # acx-face (Fortran line 1385-1394).
        if fi_int[i, j, 1] == 1.0 && fi_int[ip1, j, 1] == 1.0
            if fg_int[i, j, 1] > 0.0 && fg_int[ip1, j, 1] == 0.0
                beta_acx_int[ip1f, j, 1] = beta_int[i, j, 1]
            elseif fg_int[i, j, 1] == 0.0 && fg_int[ip1, j, 1] > 0.0
                beta_acx_int[ip1f, j, 1] = beta_int[ip1, j, 1]
            end
        end
        # acy-face (Fortran line 1397-1406).
        if fi_int[i, j, 1] == 1.0 && fi_int[i, jp1, 1] == 1.0
            if fg_int[i, j, 1] > 0.0 && fg_int[i, jp1, 1] == 0.0
                beta_acy_int[i, jp1f, 1] = beta_int[i, j, 1]
            elseif fg_int[i, j, 1] == 0.0 && fg_int[i, jp1, 1] > 0.0
                beta_acy_int[i, jp1f, 1] = beta_int[i, jp1, 1]
            end
        end
    end
    return nothing
end

# Fortran line 1415 `stagger_beta_aa_gl_downstream`.
function _stagger_beta_aa_gl_downstream!(beta_acx_int::AbstractArray,
                                         beta_acy_int::AbstractArray,
                                         beta_int::AbstractArray,
                                         fi_int::AbstractArray,
                                         fg_int::AbstractArray,
                                         Tx_top::Type{<:AbstractTopology},
                                         Ty_top::Type{<:AbstractTopology})
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)
        if fi_int[i, j, 1] == 1.0 && fi_int[ip1, j, 1] == 1.0
            if fg_int[i, j, 1] > 0.0 && fg_int[ip1, j, 1] == 0.0
                beta_acx_int[ip1f, j, 1] = beta_int[ip1, j, 1]
            elseif fg_int[i, j, 1] == 0.0 && fg_int[ip1, j, 1] > 0.0
                beta_acx_int[ip1f, j, 1] = beta_int[i, j, 1]
            end
        end
        if fi_int[i, j, 1] == 1.0 && fi_int[i, jp1, 1] == 1.0
            if fg_int[i, j, 1] > 0.0 && fg_int[i, jp1, 1] == 0.0
                beta_acy_int[i, jp1f, 1] = beta_int[i, jp1, 1]
            elseif fg_int[i, j, 1] == 0.0 && fg_int[i, jp1, 1] > 0.0
                beta_acy_int[i, jp1f, 1] = beta_int[i, j, 1]
            end
        end
    end
    return nothing
end

# Fortran line 1481 `stagger_beta_aa_gl_subgrid`. Weight by f_grnd_ac².
function _stagger_beta_aa_gl_subgrid!(beta_acx_int::AbstractArray,
                                      beta_acy_int::AbstractArray,
                                      beta_int::AbstractArray,
                                      fi_int::AbstractArray,
                                      fg_int::AbstractArray,
                                      fg_acx_int::AbstractArray,
                                      fg_acy_int::AbstractArray,
                                      Tx_top::Type{<:AbstractTopology},
                                      Ty_top::Type{<:AbstractTopology})
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)
        if fi_int[i, j, 1] == 1.0 && fi_int[ip1, j, 1] == 1.0
            wt = fg_acx_int[ip1f, j, 1]^2
            if fg_int[i, j, 1] > 0.0 && fg_int[ip1, j, 1] == 0.0
                beta_acx_int[ip1f, j, 1] = wt * beta_int[i, j, 1] +
                                           (1 - wt) * beta_int[ip1, j, 1]
            elseif fg_int[i, j, 1] == 0.0 && fg_int[ip1, j, 1] > 0.0
                beta_acx_int[ip1f, j, 1] = (1 - wt) * beta_int[i, j, 1] +
                                           wt * beta_int[ip1, j, 1]
            end
        end
        if fi_int[i, j, 1] == 1.0 && fi_int[i, jp1, 1] == 1.0
            wt = fg_acy_int[i, jp1f, 1]^2
            if fg_int[i, j, 1] > 0.0 && fg_int[i, jp1, 1] == 0.0
                beta_acy_int[i, jp1f, 1] = wt * beta_int[i, j, 1] +
                                           (1 - wt) * beta_int[i, jp1, 1]
            elseif fg_int[i, j, 1] == 0.0 && fg_int[i, jp1, 1] > 0.0
                beta_acy_int[i, jp1f, 1] = (1 - wt) * beta_int[i, j, 1] +
                                           wt * beta_int[i, jp1, 1]
            end
        end
    end
    return nothing
end

# Fortran line 1559 `stagger_beta_aa_gl_subgrid_flux`. Weight via the
# 100-segment flux helper.
function _stagger_beta_aa_gl_subgrid_flux!(beta_acx_int::AbstractArray,
                                           beta_acy_int::AbstractArray,
                                           beta_int::AbstractArray,
                                           H_int::AbstractArray,
                                           fi_int::AbstractArray,
                                           ux_int::AbstractArray,
                                           uy_int::AbstractArray,
                                           fg_int::AbstractArray,
                                           fg_acx_int::AbstractArray,
                                           fg_acy_int::AbstractArray,
                                           Tx_top::Type{<:AbstractTopology},
                                           Ty_top::Type{<:AbstractTopology})
    Nx, Ny = size(beta_int, 1), size(beta_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = _neighbor_im1(i, Nx, Tx_top)
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jm1 = _neighbor_jm1(j, Ny, Ty_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)

        # Yelmo face-array indices for Fortran "ux(k, j)" / "uy(i, k)".
        ux_im1 = _ip1_modular(im1, Nx, Tx_top)
        ux_i   = _ip1_modular(i,   Nx, Tx_top)
        ux_ip1 = _ip1_modular(ip1, Nx, Tx_top)
        uy_jm1 = _jp1_modular(jm1, Ny, Ty_top)
        uy_j   = _jp1_modular(j,   Ny, Ty_top)
        uy_jp1 = _jp1_modular(jp1, Ny, Ty_top)

        # acx (Fortran line 1597-1620).
        if fi_int[i, j, 1] == 1.0 && fi_int[ip1, j, 1] == 1.0
            if fg_int[i, j, 1] > 0.0 && fg_int[ip1, j, 1] == 0.0
                # Floating to the right (Fortran line 1600-1607).
                ux_aa_a = 0.5 * (ux_int[ux_im1, j, 1] + ux_int[ux_i, j, 1])
                ux_aa_b = 0.5 * (ux_int[ux_ip1, j, 1] + ux_int[ux_i, j, 1])
                beta_acx_int[ip1f, j, 1] = _calc_beta_gl_flux_weight(
                    beta_int[i, j, 1], beta_int[ip1, j, 1],
                    ux_aa_a, ux_aa_b,
                    H_int[i, j, 1], H_int[ip1, j, 1],
                    fg_acx_int[ip1f, j, 1])
            elseif fg_int[i, j, 1] == 0.0 && fg_int[ip1, j, 1] > 0.0
                # Floating to the left (Fortran line 1609-1617).
                ux_aa_a = 0.5 * (ux_int[ux_ip1, j, 1] + ux_int[ux_i, j, 1])
                ux_aa_b = 0.5 * (ux_int[ux_im1, j, 1] + ux_int[ux_i, j, 1])
                beta_acx_int[ip1f, j, 1] = _calc_beta_gl_flux_weight(
                    beta_int[ip1, j, 1], beta_int[i, j, 1],
                    ux_aa_a, ux_aa_b,
                    H_int[ip1, j, 1], H_int[i, j, 1],
                    fg_acx_int[ip1f, j, 1])
            end
        end

        # acy (Fortran line 1623-1646). PR-A.1 polish: applies the
        # Fortran-side typo corrections that the user has fixed upstream
        # in `basal_dragging.f90`:
        #   - Line 1635: `fg(ip1, j)` → `fg(i, jp1)` (gating condition).
        #   - Line 1642: `fg_acx(i, j)` → `fg_acy(i, j)` (downstream call).
        # Previously the Julia port mirrored the typos verbatim per the
        # "no autonomous shortcuts" constraint; with the Fortran source
        # corrected, follow it.
        if fi_int[i, j, 1] == 1.0 && fi_int[i, jp1, 1] == 1.0
            if fg_int[i, j, 1] > 0.0 && fg_int[i, jp1, 1] == 0.0
                uy_aa_a = 0.5 * (uy_int[i, uy_jm1, 1] + uy_int[i, uy_j, 1])
                uy_aa_b = 0.5 * (uy_int[i, uy_jp1, 1] + uy_int[i, uy_j, 1])
                beta_acy_int[i, jp1f, 1] = _calc_beta_gl_flux_weight(
                    beta_int[i, j, 1], beta_int[i, jp1, 1],
                    uy_aa_a, uy_aa_b,
                    H_int[i, j, 1], H_int[i, jp1, 1],
                    fg_acy_int[i, jp1f, 1])
            elseif fg_int[i, j, 1] == 0.0 && fg_int[i, jp1, 1] > 0.0
                # Follows fixed upstream Fortran (basal_dragging.f90:1635)
                # — gating condition is `fg(i, jp1)`, not `fg(ip1, j)`.
                uy_aa_a = 0.5 * (uy_int[i, uy_jp1, 1] + uy_int[i, uy_j, 1])
                uy_aa_b = 0.5 * (uy_int[i, uy_jm1, 1] + uy_int[i, uy_j, 1])
                beta_acy_int[i, jp1f, 1] = _calc_beta_gl_flux_weight(
                    beta_int[i, jp1, 1], beta_int[i, j, 1],
                    uy_aa_a, uy_aa_b,
                    H_int[i, jp1, 1], H_int[i, j, 1],
                    # Follows fixed upstream Fortran (basal_dragging.f90:1642)
                    # — downstream call uses `fg_acy`, not `fg_acx`.
                    fg_acy_int[i, jp1f, 1])
            end
        end
    end
    return nothing
end

"""
    stagger_beta!(beta_acx, beta_acy, beta,
                  H_ice, f_ice, ux, uy,
                  f_grnd, f_grnd_acx, f_grnd_acy;
                  beta_gl_stag::Int, beta_min::Real) -> (beta_acx, beta_acy)

Stagger `beta` from aa-cells to acx/acy faces with GL-aware
modifications. Mirrors Fortran `basal_dragging.f90:528 stagger_beta`.

`beta_gl_stag` selects the staggering rule:
  - `-1`: external (no-op; assumes `beta_acx`, `beta_acy` already populated).
  -  `0`: simple mean staggering only (`stagger_beta_aa_mean`).
  -  `1`: mean staggering + GL upstream rule.
  -  `2`: mean staggering + GL downstream rule.
  -  `3`: mean staggering + GL subgrid (f_grnd_ac² weighted blend).
  -  `4`: mean staggering + GL subgrid_flux (100-segment integration).

After GL modifications, positive face values are clamped to `≥ beta_min`.

Boundary handling: Yelmo.jl uses Oceananigans halo / clamped indices,
matching the Fortran `infinite` BC. The Fortran `periodic` boundary
fix-up (lines 606-619) is NOT mirrored — periodic SSA support requires
topology-aware staggering and is deferred.

`beta_acx` is an XFaceField with face-east of cell `(i, j)` at array
index `[i+1, j, 1]`; `beta_acy` is a YFaceField analogous.
"""
function stagger_beta!(beta_acx, beta_acy, beta,
                       H_ice, f_ice, ux, uy,
                       f_grnd, f_grnd_acx, f_grnd_acy;
                       beta_gl_stag::Int,
                       beta_min::Real)
    bx_int = interior(beta_acx)
    by_int = interior(beta_acy)
    b_int  = interior(beta)
    H_int  = interior(H_ice)
    fi_int = interior(f_ice)
    ux_int = interior(ux)
    uy_int = interior(uy)
    fg_int = interior(f_grnd)
    fgx_int = interior(f_grnd_acx)
    fgy_int = interior(f_grnd_acy)

    Tx_top = topology(beta_acx.grid, 1)
    Ty_top = topology(beta_acy.grid, 2)

    if beta_gl_stag == -1
        # External — no-op (Fortran line 552-554).
        # Skip the standard mean staggering AND the GL block entirely.
    else
        # Fortran line 560 always calls mean staggering first, then
        # (line 563-600) optionally overlays the GL modifier.
        _stagger_beta_aa_mean!(bx_int, by_int, b_int, fi_int, fg_int,
                               Tx_top, Ty_top)

        if beta_gl_stag == 0
            # Already done.
        elseif beta_gl_stag == 1
            _stagger_beta_aa_gl_upstream!(bx_int, by_int, b_int, fi_int, fg_int,
                                          Tx_top, Ty_top)
        elseif beta_gl_stag == 2
            _stagger_beta_aa_gl_downstream!(bx_int, by_int, b_int, fi_int, fg_int,
                                            Tx_top, Ty_top)
        elseif beta_gl_stag == 3
            _stagger_beta_aa_gl_subgrid!(bx_int, by_int, b_int, fi_int, fg_int,
                                         fgx_int, fgy_int, Tx_top, Ty_top)
        elseif beta_gl_stag == 4
            _stagger_beta_aa_gl_subgrid_flux!(bx_int, by_int, b_int,
                                              H_int, fi_int, ux_int, uy_int,
                                              fg_int, fgx_int, fgy_int,
                                              Tx_top, Ty_top)
        else
            error("stagger_beta!: beta_gl_stag = $beta_gl_stag not recognised; expected -1..4.")
        end
    end

    # Fortran line 637-638: beta_min floor on positive face values.
    bm = Float64(beta_min)
    Nx, Ny = size(b_int, 1), size(b_int, 2)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)
        if bx_int[ip1f, j, 1] > 0.0 && bx_int[ip1f, j, 1] < bm
            bx_int[ip1f, j, 1] = bm
        end
        if by_int[i, jp1f, 1] > 0.0 && by_int[i, jp1f, 1] < bm
            by_int[i, jp1f, 1] = bm
        end
    end

    return beta_acx, beta_acy
end
