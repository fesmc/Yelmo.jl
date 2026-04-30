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

export calc_cb_ref!, calc_c_bed!

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
