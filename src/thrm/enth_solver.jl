# ----------------------------------------------------------------------
# Enthalpy column solver — `method = "enth"`. Port-only milestone (PR5).
#
# Direct ports of:
#
#   - `calc_enth_column`           ← `physics/ice_enthalpy.f90:536-727`
#   - `calc_enth_column_internal`  ← `physics/ice_enthalpy.f90:729-888`
#   - `calc_enth_diffusivity`      ← `physics/ice_enthalpy.f90:892-931`
#   - `convert_from_enthalpy_column` ← `physics/thermodynamics.f90:1637`
#
# Note on Fortran fidelity: Fortran `calc_enth_column` ends with a
# `write(*,*) "...routine needs to be updated"` + `stop` — the basal
# mass balance call is commented out in the source. This Julia port
# mirrors the Fortran code path but uses the same `_calc_bmb_grounded`
# closure used by the temp solver to write `bmb_grnd` (so the per-cell
# net-flux mass balance still happens). PR5 is explicitly **port-only,
# not benchmarked** per the agreed plan — Mirror's enth path showed
# issues so a Julia-native validation is the right next step.
#
# Key difference from the temp solver: the state is enthalpy `enth`
# (J/kg), not temperature. Diffusivity per layer is either the cold
# `kappa_cold = kt / (rho_ice * cp)` (cold ice) or the temperate
# `kappa_temp = cr * kappa_cold` (temperate ice with water). At the
# CTS-adjacent ac-node, the harmonic-mean kappa_a is replaced by the
# layer-below kappa to enforce zero diffusive flux into the temperate
# layer (Fortran `if (k == k_cts+1) kappa_a = kappa(k-1)`).
#
# Reference enthalpy `enth_ref = T_ref_ice * 2009 J/kg/K` matches
# Fortran's hard-coded `273.15 * 2009.0`.
# ----------------------------------------------------------------------

const _ENTH_REF = _T_REF_ICE * 2009.0

"""
    calc_enth_diffusivity!(kappa_buf, enth_col, T_pmp_col, cp_col, kt_col,
                           cr, rho_ice) -> kappa_buf

Per-cell diffusivity `kappa[k] = kt[k] / (rho_ice * cp[k])` if cold,
or `cr * kappa_cold` if temperate (`enth[k] >= T_pmp[k] * cp[k]`).
"""
function calc_enth_diffusivity!(kappa_buf::AbstractVector{Float64},
                                enth_col::AbstractVector{Float64},
                                T_pmp_col::AbstractVector{Float64},
                                cp_col::AbstractVector{Float64},
                                kt_col::AbstractVector{Float64},
                                cr::Float64, rho_ice::Float64)
    # `kappa_buf` is a plain Vector — its length is correct.
    nz = length(kappa_buf)
    @inbounds for k in 1:nz
        kappa_cold = kt_col[k] / (rho_ice * cp_col[k])
        if enth_col[k] >= T_pmp_col[k] * cp_col[k]
            kappa_buf[k] = cr * kappa_cold
        else
            kappa_buf[k] = kappa_cold
        end
    end
    return kappa_buf
end

"""
    convert_from_enthalpy_column!(enth_col, T_col, omega_col, T_pmp_col,
                                  cp_col, L_ice) -> nothing

Per-column reverse conversion: split `enth` into temperature `T_col`
and water content `omega_col`. Mirrors Fortran
`convert_from_enthalpy_column` (`physics/thermodynamics.f90:1637`).
The basal layer (k = 1) keeps omega for consistency. The surface
layer (k = nz) clamps `enth = enth_pmp` if temperate (resetting
omega to zero).
"""
function convert_from_enthalpy_column!(enth_col::AbstractVector{Float64},
                                       T_col::AbstractVector{Float64},
                                       omega_col::AbstractVector{Float64},
                                       T_pmp_col::AbstractVector{Float64},
                                       cp_col::AbstractVector{Float64},
                                       L_ice::Float64;
                                       nz::Int=-1)
    # If nz not supplied, derive from `T_col` (assumed plain Vector).
    # Callers wrapping OffsetArray views MUST pass `nz` explicitly.
    if nz < 0
        nz = length(T_col)
    end
    @inbounds begin
        for k in 1:(nz - 1)
            enth_pmp_k = T_pmp_col[k] * cp_col[k]
            if enth_col[k] > enth_pmp_k
                T_col[k]     = T_pmp_col[k]
                omega_col[k] = (enth_col[k] - enth_pmp_k) / L_ice
            else
                T_col[k]     = enth_col[k] / cp_col[k]
                omega_col[k] = 0.0
            end
        end
        # Surface layer.
        enth_pmp_top = T_pmp_col[nz] * cp_col[nz]
        if enth_col[nz] >= enth_pmp_top
            enth_col[nz]   = enth_pmp_top
            T_col[nz]      = enth_col[nz] / cp_col[nz]
            omega_col[nz]  = 0.0
        else
            T_col[nz]      = enth_col[nz] / cp_col[nz]
            omega_col[nz]  = 0.0
        end
    end
    return nothing
end

"""
    _calc_enth_column_internal!(enth, kappa, uz, advecxy, Q_strn,
                                val_base, val_srf, thickness,
                                zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                                enth_ref, dt, k_cts, is_basal_flux,
                                subd, diag, supd, rhs, solution,
                                cp_tri, dp_tri) -> enth

Implicit 1D enthalpy solver. Differs from the temp variant in:

  - Base BC has 3 cases: Neumann flux (frozen base), zero-flux into
    a temperate-layer-above-base (`k_cts >= 2`), or Dirichlet at
    PMP enthalpy.
  - Surface BC is always Dirichlet at `min(T_srf, T0) * cp[nz]`.
  - Diffusivity at the CTS interface is forced to the layer-below
    kappa (`if k == k_cts+1: kappa_a = kappa[k-1]`).
"""
function _calc_enth_column_internal!(enth::AbstractVector{Float64},
                                     kappa::AbstractVector{Float64},
                                     uz::AbstractVector{Float64},
                                     advecxy::AbstractVector{Float64},
                                     Q_strn::AbstractVector{Float64},
                                     val_base::Float64, val_srf::Float64,
                                     thickness::Float64,
                                     zeta_aa::Vector{Float64},
                                     zeta_ac::Vector{Float64},
                                     dzeta_a::Vector{Float64},
                                     dzeta_b::Vector{Float64},
                                     enth_ref::Float64, dt::Float64,
                                     k_cts::Int, is_basal_flux::Bool,
                                     subd::Vector{Float64},
                                     diag::Vector{Float64},
                                     supd::Vector{Float64},
                                     rhs::Vector{Float64},
                                     solution::Vector{Float64},
                                     cp_tri::Vector{Float64},
                                     dp_tri::Vector{Float64})
    nz_aa = length(zeta_aa)

    @inbounds begin
        # -- Base BC --
        if is_basal_flux
            dz       = thickness * (zeta_aa[2] - zeta_aa[1])
            subd[1]  = 0.0
            diag[1]  = 1.0
            supd[1]  = -1.0
            rhs[1]   = val_base * dz
        elseif k_cts >= 2
            # Layer above base also temperate → zero flux.
            subd[1]  = 0.0
            diag[1]  = 1.0
            supd[1]  = -1.0
            rhs[1]   = 0.0
        else
            subd[1]  = 0.0
            diag[1]  = 1.0
            supd[1]  = 0.0
            rhs[1]   = val_base - enth_ref
        end

        H2 = thickness * thickness
        for k in 2:(nz_aa - 1)
            dz1     = zeta_ac[k - 1] - zeta_aa[k - 1]
            dz2     = zeta_aa[k]     - zeta_ac[k - 1]
            kappa_a = _calc_wtd_harmonic_mean(kappa[k - 1], kappa[k], dz1, dz2)

            dz1     = zeta_ac[k]     - zeta_aa[k]
            dz2     = zeta_aa[k + 1] - zeta_ac[k]
            kappa_b = _calc_wtd_harmonic_mean(kappa[k], kappa[k + 1], dz1, dz2)

            # CTS-adjacent override (Fortran line 826).
            if k == k_cts + 1
                kappa_a = kappa[k - 1]
            end

            fac_a = -kappa_a * dzeta_a[k] * dt / H2
            fac_b = -kappa_b * dzeta_b[k] * dt / H2
            uz_aa = 0.5 * (uz[k] + uz[k + 1])

            dzeta = zeta_aa[k + 1] - zeta_aa[k - 1]
            dz    = thickness * dzeta

            subd[k] = fac_a - uz_aa * dt / dz
            supd[k] = fac_b + uz_aa * dt / dz
            diag[k] = 1.0 - fac_a - fac_b
            rhs[k]  = (enth[k] - enth_ref) - dt * advecxy[k] + dt * Q_strn[k]
        end

        # -- Surface BC (Dirichlet only) --
        subd[nz_aa] = 0.0
        diag[nz_aa] = 1.0
        supd[nz_aa] = 0.0
        rhs[nz_aa]  = val_srf - enth_ref

        solve_tridiag!(solution, subd, diag, supd, rhs, cp_tri, dp_tri)

        for k in 1:nz_aa
            enth[k] = solution[k] + enth_ref
        end
    end
    return enth
end

"""
    calc_enth_column!(enth_col, T_col, omega_col, T_pmp_col, cp_col, kt_col,
                      advecxy_col, uz_col, Q_strn_col,
                      Q_b, Q_rock, T_srf, T_shlf,
                      H_ice, H_w, f_grnd, bmb_grnd_in,
                      zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                      cr, omega_max, T0, rho_ice, rho_w, L_ice, sec_year, dt,
                      kappa_buf, Q_strn_kg_buf, subd, diag, supd, rhs,
                      solution, cp_tri, dp_tri)
        -> (Q_ice_b, bmb_grnd, H_cts)

Per-column enthalpy solver. Fortran `calc_enth_column`. Returns the
basal heat flux, basal mass balance, and CTS height for the caller.
"""
function calc_enth_column!(enth_col::AbstractVector{Float64},
                           T_col::AbstractVector{Float64},
                           omega_col::AbstractVector{Float64},
                           T_pmp_col::AbstractVector{Float64},
                           cp_col::AbstractVector{Float64},
                           kt_col::AbstractVector{Float64},
                           advecxy_col::AbstractVector{Float64},
                           uz_col::AbstractVector{Float64},
                           Q_strn_col::AbstractVector{Float64},
                           Q_b::Float64, Q_rock::Float64,
                           T_srf::Float64, T_shlf::Float64,
                           H_ice::Float64, H_w::Float64,
                           f_grnd::Float64, bmb_grnd_in::Float64,
                           zeta_aa::Vector{Float64},
                           zeta_ac::Vector{Float64},
                           dzeta_a::Vector{Float64},
                           dzeta_b::Vector{Float64},
                           cr::Float64, omega_max::Float64, T0::Float64,
                           rho_ice::Float64, rho_w::Float64,
                           L_ice::Float64, sec_year::Float64, dt::Float64,
                           kappa_buf::Vector{Float64},
                           Q_strn_kg_buf::Vector{Float64},
                           subd::Vector{Float64},
                           diag::Vector{Float64},
                           supd::Vector{Float64},
                           rhs::Vector{Float64},
                           solution::Vector{Float64},
                           cp_tri::Vector{Float64},
                           dp_tri::Vector{Float64})
    nz_aa = length(zeta_aa)

    @inbounds begin
        Q_rock_now = Q_rock * 1e-3 * sec_year   # mW → J a⁻¹ m⁻²
        Q_b_now    = Q_b    * 1e-3 * sec_year

        # CTS index from current state.
        k_cts = 1
        for k in 1:nz_aa
            if enth_col[k] >= T_pmp_col[k] * cp_col[k]
                k_cts = k
            else
                break
            end
        end

        calc_enth_diffusivity!(kappa_buf, enth_col, T_pmp_col,
                               cp_col, kt_col, cr, rho_ice)

        # Q_strn in J/kg/a (note Fortran lines 617: divides by rho_ice
        # only — the cp factor is NOT applied for enthalpy form).
        for k in 1:nz_aa
            Q_strn_kg_buf[k] = Q_strn_col[k] / rho_ice
        end

        # Surface BC.
        val_srf = min(T_srf, T0) * cp_col[nz_aa]

        # Basal BC.
        is_basal_flux = false
        local val_base::Float64
        if f_grnd < 1.0
            val_base = (f_grnd * T_pmp_col[1] + (1.0 - f_grnd) * T_shlf) * cp_col[1]
        else
            H_w_predicted = H_w - (bmb_grnd_in * (rho_w / rho_ice)) * dt
            enth_pmp_base = T_pmp_col[1] * cp_col[1]
            if H_w_predicted > 0.0
                val_base = enth_pmp_base
            elseif enth_col[1] < enth_pmp_base || H_w_predicted < 0.0
                val_base       = (Q_b_now + Q_rock_now) / kt_col[1] * cp_col[1]
                is_basal_flux  = true
            else
                val_base = enth_pmp_base
            end
        end

        _calc_enth_column_internal!(enth_col, kappa_buf, uz_col,
                                     advecxy_col, Q_strn_kg_buf,
                                     val_base, val_srf, H_ice,
                                     zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                                     _ENTH_REF, dt, k_cts,
                                     is_basal_flux,
                                     subd, diag, supd, rhs, solution,
                                     cp_tri, dp_tri)

        # Stability boost: if layer above base is temperate, copy
        # enth(2) down to enth(1). Fortran line 674.
        if enth_col[2] >= T_pmp_col[2] * cp_col[2]
            enth_col[1] = enth_col[2]
        end

        # Recover (T, omega) from enth. Pass `nz_aa` explicitly because
        # `T_col` here is an OffsetArray view (halo-inclusive length).
        convert_from_enthalpy_column!(enth_col, T_col, omega_col,
                                      T_pmp_col, cp_col, L_ice;
                                      nz=nz_aa)

        # Internal-melt cleanup: clip omega above omega_max.
        melt_internal = 0.0
        for k in (nz_aa - 1):-1:2
            omega_excess = omega_col[k] - omega_max
            if omega_excess > 0.0
                dz = H_ice * (zeta_ac[k] - zeta_ac[k - 1])
                melt_internal += (omega_excess * dz) / dt
                omega_col[k] = omega_max
            end
        end
        if omega_col[1] > omega_max
            omega_col[1] = omega_max
        end

        # Re-derive enth to stay consistent with clipped omega.
        for k in 1:nz_aa
            enth_col[k] = convert_to_enthalpy(T_col[k], omega_col[k],
                                              T_pmp_col[k], cp_col[k],
                                              L_ice)
        end

        # Q_ice_b: Fortran uses T-gradient × kt (line 707), not enth.
        local Q_ice_b_now::Float64
        if H_ice > 0.0
            dz_base     = H_ice * (zeta_aa[2] - zeta_aa[1])
            Q_ice_b_now = kt_col[1] * (T_col[2] - T_col[1]) / dz_base
        else
            Q_ice_b_now = 0.0
        end
        Q_ice_b = Q_ice_b_now * 1e3 / sec_year

        # bmb_grnd: Fortran's enth path comments out the
        # `calc_bmb_grounded` call and stops. We mirror the temp solver
        # here so the field stays meaningful — flagged in the file
        # header.
        local bmb_grnd::Float64
        if f_grnd > 0.0
            bmb_grnd = _calc_bmb_grounded(T_col[1] - T_pmp_col[1],
                                          Q_ice_b_now, Q_b_now,
                                          Q_rock_now, rho_ice, L_ice)
            bmb_grnd -= melt_internal
            bmb_grnd >  _BMB_GRND_LIM && (bmb_grnd =  _BMB_GRND_LIM)
            bmb_grnd < -_BMB_GRND_LIM && (bmb_grnd = -_BMB_GRND_LIM)
        else
            bmb_grnd = 0.0
        end

        H_cts = _calc_cts_height_column(enth_col, T_pmp_col, cp_col,
                                        H_ice, zeta_aa)
    end
    return Q_ice_b, bmb_grnd, H_cts
end

"""
    calc_enth_3D!(enth_field, T_ice_field, omega_field,
                  bmb_grnd_field, Q_ice_b_field, H_cts_field,
                  T_pmp_field, cp_field, kt_field, advecxy_field,
                  uz_field, Q_strn_field, Q_b_field, Q_rock_field,
                  T_srf_field, T_shlf_field, H_ice_field, f_ice_field,
                  H_w_field, f_grnd_field, H_grnd_field,
                  zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                  cr, omega_max, T0, rho_ice, rho_sw, rho_w, L_ice,
                  sec_year, dt)

3D wrapper around `calc_enth_column!`. Same gating + marginal-cell
fallback as `calc_temp_3D!`.
"""
function calc_enth_3D!(enth_field, T_ice_field, omega_field,
                       bmb_grnd_field, Q_ice_b_field, H_cts_field,
                       T_pmp_field, cp_field, kt_field, advecxy_field,
                       uz_field, Q_strn_field,
                       Q_b_field, Q_rock_field,
                       T_srf_field, T_shlf_field,
                       H_ice_field, f_ice_field, H_w_field,
                       f_grnd_field, H_grnd_field,
                       zeta_aa::AbstractVector{<:Real},
                       zeta_ac::AbstractVector{<:Real},
                       dzeta_a::AbstractVector{<:Real},
                       dzeta_b::AbstractVector{<:Real},
                       cr::Real, omega_max::Real, T0::Real,
                       rho_ice::Real, rho_sw::Real, rho_w::Real,
                       L_ice::Real, sec_year::Real, dt::Real)
    enth = enth_field.data
    T    = T_ice_field.data
    om   = omega_field.data
    bmg  = bmb_grnd_field.data
    Qib  = Q_ice_b_field.data
    Hcts = H_cts_field.data
    Tp   = T_pmp_field.data
    cp   = cp_field.data
    kt   = kt_field.data
    adv  = advecxy_field.data
    uz   = uz_field.data
    Qs   = Q_strn_field.data
    Qb   = Q_b_field.data
    Qr   = Q_rock_field.data
    Ts   = T_srf_field.data
    Tsh  = T_shlf_field.data
    H    = H_ice_field.data
    Fi   = f_ice_field.data
    Hw   = H_w_field.data
    Fg   = f_grnd_field.data
    Hg   = H_grnd_field.data
    Nx   = T_ice_field.grid.Nx
    Ny   = T_ice_field.grid.Ny
    Nz   = T_ice_field.grid.Nz

    zeta_aa_v = collect(Float64, zeta_aa)
    zeta_ac_v = collect(Float64, zeta_ac)
    dzeta_a_v = collect(Float64, dzeta_a)
    dzeta_b_v = collect(Float64, dzeta_b)

    kappa_buf     = Vector{Float64}(undef, Nz)
    Q_strn_kg_buf = Vector{Float64}(undef, Nz)
    subd          = Vector{Float64}(undef, Nz)
    diag          = Vector{Float64}(undef, Nz)
    supd          = Vector{Float64}(undef, Nz)
    rhs           = Vector{Float64}(undef, Nz)
    solution      = Vector{Float64}(undef, Nz)
    cp_tri        = Vector{Float64}(undef, Nz)
    dp_tri        = Vector{Float64}(undef, Nz)

    return _calc_enth_3D_kernel!(enth, T, om, bmg, Qib, Hcts,
                                 Tp, cp, kt, adv, uz, Qs, Qb, Qr,
                                 Ts, Tsh, H, Fi, Hw, Fg, Hg,
                                 zeta_aa_v, zeta_ac_v,
                                 dzeta_a_v, dzeta_b_v,
                                 Float64(cr), Float64(omega_max),
                                 Float64(T0),
                                 Float64(rho_ice), Float64(rho_sw),
                                 Float64(rho_w),
                                 Float64(L_ice), Float64(sec_year),
                                 Float64(dt),
                                 kappa_buf, Q_strn_kg_buf,
                                 subd, diag, supd, rhs, solution,
                                 cp_tri, dp_tri,
                                 Nx, Ny, Nz)
end

function _calc_enth_3D_kernel!(enth, T, om, bmg, Qib, Hcts,
                               Tp, cp, kt, adv, uz, Qs, Qb, Qr,
                               Ts, Tsh, H, Fi, Hw, Fg, Hg,
                               zeta_aa::Vector{Float64},
                               zeta_ac::Vector{Float64},
                               dzeta_a::Vector{Float64},
                               dzeta_b::Vector{Float64},
                               cr::Float64, omega_max::Float64,
                               T0::Float64,
                               rho_ice::Float64, rho_sw::Float64,
                               rho_w::Float64,
                               L_ice::Float64, sec_year::Float64,
                               dt::Float64,
                               kappa_buf::Vector{Float64},
                               Q_strn_kg_buf::Vector{Float64},
                               subd::Vector{Float64},
                               diag::Vector{Float64},
                               supd::Vector{Float64},
                               rhs::Vector{Float64},
                               solution::Vector{Float64},
                               cp_tri::Vector{Float64},
                               dp_tri::Vector{Float64},
                               Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 2:(Ny - 1), i in 2:(Nx - 1)
        f_ice_ij  = Fi[i, j, 1]
        f_grnd_ij = Fg[i, j, 1]
        H_ice_raw = H[i, j, 1]
        H_ice_now = f_ice_ij > 0.0 ? H_ice_raw / f_ice_ij : H_ice_raw

        T_shlf_ij = if f_grnd_ij < 1.0
            _calc_T_base_shlf_approx(H_ice_now, Tp[i, j, 1],
                                     Hg[i, j, 1], T0, rho_ice, rho_sw)
        else
            Tp[i, j, 1]
        end

        T_col       = view(T,    i, j, :)
        omega_col   = view(om,   i, j, :)
        T_pmp_col   = view(Tp,   i, j, :)
        cp_col      = view(cp,   i, j, :)
        kt_col      = view(kt,   i, j, :)
        enth_col    = view(enth, i, j, :)
        advecxy_col = view(adv,  i, j, :)
        Q_strn_col  = view(Qs,   i, j, :)
        uz_col      = view(uz,   i, j, :)

        if f_ice_ij == 1.0 && H_ice_now > _H_ICE_THIN
            Q_ice_b_v, bmb_v, H_cts_v = calc_enth_column!(
                enth_col, T_col, omega_col,
                T_pmp_col, cp_col, kt_col,
                advecxy_col, uz_col, Q_strn_col,
                Qb[i, j, 1], Qr[i, j, 1],
                Ts[i, j, 1], T_shlf_ij,
                H_ice_now, Hw[i, j, 1], f_grnd_ij,
                bmg[i, j, 1],
                zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                cr, omega_max, T0,
                rho_ice, rho_w, L_ice, sec_year, dt,
                kappa_buf, Q_strn_kg_buf,
                subd, diag, supd, rhs, solution,
                cp_tri, dp_tri,
            )
            Qib[i, j, 1]  = Q_ice_b_v
            bmg[i, j, 1]  = bmb_v
            Hcts[i, j, 1] = H_cts_v
        else
            T_base = f_grnd_ij < 1.0 ? T_shlf_ij : T_pmp_col[1]
            T_top  = min(Ts[i, j, 1], T0)
            for k in 1:Nz
                if k == 1
                    T_col[k] = T_base
                elseif k == Nz
                    T_col[k] = T_top
                else
                    T_col[k] = T_base + zeta_aa[k] * (T_top - T_base)
                end
                omega_col[k] = 0.0
                enth_col[k]  = convert_to_enthalpy(T_col[k], 0.0,
                                                   T_pmp_col[k],
                                                   cp_col[k], L_ice)
            end
            bmg[i, j, 1]  = 0.0
            Qib[i, j, 1]  = 0.0
            Hcts[i, j, 1] = 0.0
        end
    end
    return nothing
end
