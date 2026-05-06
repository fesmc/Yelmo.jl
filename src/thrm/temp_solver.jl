# ----------------------------------------------------------------------
# Per-column wrapper around `_calc_temp_column_internal!` for the
# `method = "temp"` (cold-ice implicit) thermodynamics path, plus the
# 3D wrapper that loops over (i, j) calling the single-column kernel.
#
# Direct port of Fortran `calc_temp_column`
# (`physics/ice_enthalpy.f90:24-248`). Per-column responsibilities:
#
#   1. Convert flux units (mW m^-2 → J a^-1 m^-2).
#   2. Diffusivity `kappa = kt / (rho_ice * cp)` and strain heat in
#      K/a (`Q_strn / (rho_ice * cp)`).
#   3. Surface BC: Dirichlet at `min(T_srf, T0)`.
#   4. Basal BC: floating → Dirichlet weighted PMP/T_shlf;
#      grounded + predicted H_w > 0 → Dirichlet PMP;
#      grounded + cold base → Neumann flux from Q_b + Q_rock;
#      grounded + temperate → Dirichlet PMP.
#   5. Implicit solve via `_calc_temp_column_internal!`.
#   6. Internal-melt cleanup: any T > T_pmp is reset to T_pmp and the
#      released energy folded into `melt_internal` for `bmb_grnd`.
#   7. Floor at T_min_lim = 200 K, cap basal at T_pmp.
#   8. Set `omega = 0` and update `enth = (1 - omega) cp T + omega
#      (cp T_pmp + L)` for the column.
#   9. Compute `Q_ice_b`, `bmb_grnd` (grounded only), and `H_cts`.
#
# The column holds only the interior layers (ζ_aa[1] > 0,
# ζ_aa[nz_aa] < 1). Boundary temperatures at ζ = 0 and ζ = 1 live in
# the 2D `T_ice_b` / `T_ice_s` fields and are passed as `T_ice_b_val`
# / `T_pmp_b_val` kwargs. The BC selection and post-solve diagnostics
# (Q_ice_b, bmb_grnd) use the boundary values; the column solver
# stencil uses the derived `kappa_basal` / `kappa_surf` diffusivities
# at the phantom boundary cells.
#
# All scratch (length-Nz_aa workspace) is owned by the caller — the
# 3D wrapper allocates one set per call and the column kernel re-uses
# it across (i, j).
# ----------------------------------------------------------------------

const _T_MIN_LIM    = 200.0
const _BMB_GRND_LIM = 10.0       # m/yr safety cap on bmb_grnd
const _H_ICE_THIN   = 10.0       # m — Fortran calc_ytherm_enthalpy_3D threshold

"""
    calc_temp_column!(enth_col, T_col, omega_col, T_pmp_col, cp_col, kt_col,
                      advecxy_col, uz_col, Q_strn_col, Q_b, Q_rock,
                      T_srf, T_shlf, H_ice, H_w, f_grnd, bmb_grnd_in,
                      zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                      omega_max, T0, rho_ice, rho_w, L_ice, sec_year, dt,
                      kappa_buf, Q_strn_K_buf, subd, diag, supd, rhs,
                      solution, cp_tri, dp_tri;
                      T_ice_b_val, T_pmp_b_val)
        -> (Q_ice_b, bmb_grnd, H_cts, T_ice_b_new, T_ice_s_new)

Per-column implicit temperature solver. Updates `enth_col`, `T_col`,
`omega_col` in place; returns `(Q_ice_b, bmb_grnd, H_cts, T_ice_b_new,
T_ice_s_new)`.

`T_ice_b_val` / `T_pmp_b_val` are the ice and pressure-melting temperatures
at the base (ζ = 0, the 2D `_b` boundary fields). `T_ice_b_new` on return is
the resolved basal temperature (the Dirichlet BC value, or the
gradient-extrapolated value for the Neumann case). `T_ice_s_new` is the
surface Dirichlet value `min(T_srf, T0)`.
"""
function calc_temp_column!(enth_col::AbstractVector{Float64},
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
                           omega_max::Float64, T0::Float64,
                           rho_ice::Float64, rho_w::Float64,
                           L_ice::Float64, sec_year::Float64, dt::Float64,
                           kappa_buf::Vector{Float64},
                           Q_strn_K_buf::Vector{Float64},
                           subd::Vector{Float64},
                           diag::Vector{Float64},
                           supd::Vector{Float64},
                           rhs::Vector{Float64},
                           solution::Vector{Float64},
                           cp_tri::Vector{Float64},
                           dp_tri::Vector{Float64};
                           T_ice_b_val::Float64,
                           T_pmp_b_val::Float64)
    nz_aa = length(zeta_aa)

    @inbounds begin
        Q_rock_now = Q_rock * 1e-3 * sec_year   # mW/m² → J a⁻¹ m⁻²
        Q_b_now    = Q_b    * 1e-3 * sec_year

        for k in 1:nz_aa
            kappa_buf[k]    = kt_col[k] / (rho_ice * cp_col[k])
            Q_strn_K_buf[k] = Q_strn_col[k] / (rho_ice * cp_col[k])
        end

        # Surface BC (always Dirichlet in the cold-ice solver).
        val_srf      = min(T_srf, T0)
        is_surf_flux = false

        # Boundary diffusivities from the boundary temperatures.
        # kappa_basal uses T_ice_b_val; kappa_surf uses val_srf (the current
        # Dirichlet surface BC, more consistent than the previous-step T_ice_s).
        kt_b        = calc_thermal_conductivity(T_ice_b_val, sec_year)
        cp_b        = calc_specific_heat_capacity(T_ice_b_val)
        kappa_basal = kt_b / (rho_ice * cp_b)
        kt_s        = calc_thermal_conductivity(val_srf, sec_year)
        cp_s        = calc_specific_heat_capacity(val_srf)
        kappa_surf  = kt_s / (rho_ice * cp_s)
        kt_base     = kt_b

        # Basal BC selection using the boundary values.
        local val_base::Float64
        local T_ice_b_new::Float64
        local is_basal_flux::Bool = false
        if f_grnd < 1.0
            val_base    = f_grnd * T_pmp_b_val + (1.0 - f_grnd) * T_shlf
            T_ice_b_new = val_base
        else
            H_w_predicted = H_w - (bmb_grnd_in * (rho_w / rho_ice)) * dt
            if H_w_predicted > 0.0
                val_base    = T_pmp_b_val
                T_ice_b_new = val_base
            elseif T_ice_b_val < T_pmp_b_val || H_w_predicted < 0.0
                # Neumann flux BC. Use kt_col[1] (first interior cell kt) so
                # that Q_ice_b computed from the cell-to-cell gradient cancels
                # to Q_b — mirrors Fortran where boundary == first cell.
                val_base      = -(Q_b_now + Q_rock_now) / kt_col[1]
                is_basal_flux = true
                T_ice_b_new   = NaN  # resolved post-solve via extrapolation
            else
                val_base    = T_pmp_b_val
                T_ice_b_new = val_base
            end
        end

        _calc_temp_column_internal!(T_col, kappa_buf, uz_col,
                                    advecxy_col, Q_strn_K_buf,
                                    val_base, val_srf, H_ice,
                                    zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                                    _T_REF_ICE, dt,
                                    is_basal_flux, is_surf_flux,
                                    subd, diag, supd, rhs, solution,
                                    cp_tri, dp_tri;
                                    kappa_basal = kappa_basal,
                                    kappa_surf  = kappa_surf)

        # Internal-melt cleanup, descending from surface-1 to base+1.
        melt_internal = 0.0
        for k in (nz_aa - 1):-1:2
            T_excess = T_col[k] - T_pmp_col[k]
            if T_excess > 0.0
                melt_internal += T_excess * H_ice *
                                  (zeta_ac[k] - zeta_ac[k - 1]) *
                                  cp_col[k] / (L_ice * dt)
                T_col[k] = T_pmp_col[k]
            end
        end
        if T_col[1] > T_pmp_col[1]
            T_col[1] = T_pmp_col[1]
        end

        # Floor + omega + enth update.
        for k in 1:nz_aa
            if T_col[k] < _T_MIN_LIM
                T_col[k] = _T_MIN_LIM
            end
            omega_col[k] = 0.0
            enth_col[k]  = convert_to_enthalpy(T_col[k], 0.0,
                                               T_pmp_col[k], cp_col[k],
                                               L_ice)
        end

        # Q_ice_b: heat flux from ice to bed (J a⁻¹ m⁻²).
        local Q_ice_b_now::Float64
        if H_ice > 0.0
            if is_basal_flux
                # Neumann: approximate from first two interior cells;
                # extrapolate T_ice_b to ζ=0 using the prescribed gradient.
                dz_12 = H_ice * (zeta_aa[2] - zeta_aa[1])
                Q_ice_b_now = -kt_col[1] * (T_col[2] - T_col[1]) / dz_12
                T_ice_b_new = T_col[1] - val_base * H_ice * zeta_aa[1]
            else
                # Dirichlet: gradient from boundary (T_ice_b_new = val_base)
                # to first interior centre.
                dz_base     = H_ice * zeta_aa[1]
                Q_ice_b_now = -kt_base * (T_col[1] - T_ice_b_new) / dz_base
            end
        else
            Q_ice_b_now = 0.0
            if is_basal_flux
                T_ice_b_new = T_pmp_b_val  # thin-ice fallback
            end
        end
        Q_ice_b = Q_ice_b_now * 1e3 / sec_year   # → mW/m²

        # bmb_grnd: net heat-flux melt + internal-melt fold-in,
        # grounded only.
        local bmb_grnd::Float64
        if f_grnd > 0.0
            dT_b = T_ice_b_new - T_pmp_b_val
            bmb_grnd = _calc_bmb_grounded(dT_b, Q_ice_b_now, Q_b_now,
                                          Q_rock_now, rho_ice, L_ice)
            bmb_grnd -= melt_internal
            bmb_grnd >  _BMB_GRND_LIM && (bmb_grnd =  _BMB_GRND_LIM)
            bmb_grnd < -_BMB_GRND_LIM && (bmb_grnd = -_BMB_GRND_LIM)
        else
            bmb_grnd = 0.0
        end

        H_cts = _calc_cts_height_column(enth_col, T_pmp_col, cp_col,
                                        H_ice, zeta_aa)

        T_ice_s_new = val_srf
    end

    return Q_ice_b, bmb_grnd, H_cts, T_ice_b_new, T_ice_s_new
end

"""
    calc_temp_3D!(enth_field, T_ice_field, omega_field,
                  bmb_grnd_field, Q_ice_b_field, H_cts_field,
                  T_pmp_field, cp_field, kt_field, advecxy_field,
                  uz_field, Q_strn_field, Q_b_field, Q_rock_field,
                  T_srf_field, T_shlf_field, H_ice_field, f_ice_field,
                  H_w_field, f_grnd_field, H_grnd_field,
                  zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                  omega_max, T0, rho_ice, rho_sw, rho_w, L_ice,
                  sec_year, dt;
                  T_ice_b_field, T_pmp_b_field, T_ice_s_field)

3D wrapper around `calc_temp_column!`. Mirrors Fortran
`calc_ytherm_enthalpy_3D` (`yelmo_thermodynamics.f90:268`):

  - Loop interior `(i, j)` (skip boundary cells `i = 1, Nx`,
    `j = 1, Ny`).
  - For `f_ice = 1` and effective thickness `H_ice/f_ice > 10 m`:
    run the implicit column solver.
  - Otherwise: prescribe a linear T profile between basal (PMP for
    grounded, shelf-approx for floating) and surface. Zero
    `bmb_grnd`, `Q_ice_b`, `H_cts`.
  - For floating cells (`f_grnd < 1`), `T_shlf` is computed
    locally via `_calc_T_base_shlf_approx`, overriding
    `bnd.T_shlf` (Fortran-faithful).

Boundary temperatures at ζ=0 and ζ=1 live in `T_ice_b_field` /
`T_pmp_b_field` / `T_ice_s_field`; per-column scalars are forwarded
to `calc_temp_column!` and the solved values are written back.
"""
function calc_temp_3D!(enth_field, T_ice_field, omega_field,
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
                       omega_max::Real, T0::Real,
                       rho_ice::Real, rho_sw::Real, rho_w::Real,
                       L_ice::Real, sec_year::Real, dt::Real;
                       T_ice_b_field,
                       T_pmp_b_field,
                       T_ice_s_field)
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

    # Per-column scratch (re-used across (i, j)).
    kappa_buf    = Vector{Float64}(undef, Nz)
    Q_strn_K_buf = Vector{Float64}(undef, Nz)
    subd         = Vector{Float64}(undef, Nz)
    diag         = Vector{Float64}(undef, Nz)
    supd         = Vector{Float64}(undef, Nz)
    rhs          = Vector{Float64}(undef, Nz)
    solution     = Vector{Float64}(undef, Nz)
    cp_tri       = Vector{Float64}(undef, Nz)
    dp_tri       = Vector{Float64}(undef, Nz)

    Tib = T_ice_b_field.data
    Tmb = T_pmp_b_field.data
    Tis = T_ice_s_field.data

    return _calc_temp_3D_kernel!(enth, T, om, bmg, Qib, Hcts,
                                 Tp, cp, kt, adv, uz, Qs, Qb, Qr,
                                 Ts, Tsh, H, Fi, Hw, Fg, Hg,
                                 zeta_aa_v, zeta_ac_v,
                                 dzeta_a_v, dzeta_b_v,
                                 Float64(omega_max), Float64(T0),
                                 Float64(rho_ice), Float64(rho_sw),
                                 Float64(rho_w),
                                 Float64(L_ice), Float64(sec_year),
                                 Float64(dt),
                                 kappa_buf, Q_strn_K_buf,
                                 subd, diag, supd, rhs, solution,
                                 cp_tri, dp_tri,
                                 Tib, Tmb, Tis,
                                 Nx, Ny, Nz)
end

function _calc_temp_3D_kernel!(enth, T, om, bmg, Qib, Hcts,
                               Tp, cp, kt, adv, uz, Qs, Qb, Qr,
                               Ts, Tsh, H, Fi, Hw, Fg, Hg,
                               zeta_aa::Vector{Float64},
                               zeta_ac::Vector{Float64},
                               dzeta_a::Vector{Float64},
                               dzeta_b::Vector{Float64},
                               omega_max::Float64, T0::Float64,
                               rho_ice::Float64, rho_sw::Float64,
                               rho_w::Float64,
                               L_ice::Float64, sec_year::Float64,
                               dt::Float64,
                               kappa_buf::Vector{Float64},
                               Q_strn_K_buf::Vector{Float64},
                               subd::Vector{Float64},
                               diag::Vector{Float64},
                               supd::Vector{Float64},
                               rhs::Vector{Float64},
                               solution::Vector{Float64},
                               cp_tri::Vector{Float64},
                               dp_tri::Vector{Float64},
                               Tib::Array{Float64,3},
                               Tmb::Array{Float64,3},
                               Tis::Array{Float64,3},
                               Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 2:(Ny - 1), i in 2:(Nx - 1)
        f_ice_ij  = Fi[i, j, 1]
        f_grnd_ij = Fg[i, j, 1]
        H_ice_raw = H[i, j, 1]

        # Effective thickness (Fortran calc_ytherm_enthalpy_3D:344-348).
        H_ice_now = f_ice_ij > 0.0 ? H_ice_raw / f_ice_ij : H_ice_raw

        T_pmp_b_ij = Tmb[i, j, 1]

        # Marine-shelf basal temperature (Fortran lines 353-363).
        T_shlf_ij = if f_grnd_ij < 1.0
            _calc_T_base_shlf_approx(H_ice_now, T_pmp_b_ij,
                                     Hg[i, j, 1], T0, rho_ice, rho_sw)
        else
            T_pmp_b_ij
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
            Q_ice_b_v, bmb_v, H_cts_v, T_ice_b_v, T_ice_s_v = calc_temp_column!(
                enth_col, T_col, omega_col,
                T_pmp_col, cp_col, kt_col,
                advecxy_col, uz_col, Q_strn_col,
                Qb[i, j, 1], Qr[i, j, 1],
                Ts[i, j, 1], T_shlf_ij,
                H_ice_now, Hw[i, j, 1], f_grnd_ij,
                bmg[i, j, 1],
                zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                omega_max, T0,
                rho_ice, rho_w, L_ice, sec_year, dt,
                kappa_buf, Q_strn_K_buf,
                subd, diag, supd, rhs, solution,
                cp_tri, dp_tri;
                T_ice_b_val = Tib[i, j, 1],
                T_pmp_b_val = Tmb[i, j, 1],
            )
            Qib[i, j, 1]  = Q_ice_b_v
            bmg[i, j, 1]  = bmb_v
            Hcts[i, j, 1] = H_cts_v
            Tib[i, j, 1]  = T_ice_b_v
            Tis[i, j, 1]  = T_ice_s_v
        else
            # Marginal cell: prescribe linear T profile between
            # basal (PMP grounded / T_shlf floating) and surface.
            T_base = f_grnd_ij < 1.0 ? T_shlf_ij : T_pmp_b_ij
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
            Tib[i, j, 1]  = T_base
            Tis[i, j, 1]  = T_top
        end
    end
    return nothing
end
