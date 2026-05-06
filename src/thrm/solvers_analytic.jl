# ----------------------------------------------------------------------
# Analytic temperature solvers — `linear`, `robin`, `robin-cold`.
#
# Direct ports of Fortran `physics/thermodynamics.f90`:
#
#   - `define_temp_linear_3D`              (line 1092)
#   - `define_temp_linear_column`          (line 1146)
#   - `define_temp_robin_3D`               (line 1172)
#   - `define_temp_robin_column`           (line 1244)
#   - `error_function`                     (line 1487)
#
# All three methods are closed-form (no advection-diffusion solve),
# operate column-by-column, and write `T_ice` + `omega` (always 0 for
# these methods); `enth` is then computed downstream by
# `convert_to_enthalpy_3D!` in `therm_step!`.
#
# Wrapper-+-kernel template per
# `~/.claude/.../memory/wrapper_kernel_template.md`. Wrappers lift
# `Field.data`, materialise `zeta_aa` once as `Vector{Float64}`, and
# dispatch into a parametric kernel that loops over (i, j) calling a
# single-column inner routine. The inner routine takes raw column
# slices + scalar parameters, so the hot path is alloc-free.
#
# Numerical note on `_error_function`: the Fortran erf is a
# Chebyshev-style power-series for |x| < 3.5 plus an asymptotic series
# for |x| ≥ 3.5. Faithful port — does NOT defer to `SpecialFunctions.erf`
# because Yelmo.jl does not currently depend on SpecialFunctions
# directly.
# ----------------------------------------------------------------------

# Constant from Robin's stub-base ocean temperature; matches Fortran's
# `T_ocn = 271.15 K` parameter inside `define_temp_robin_column`.
const _ROBIN_T_OCN     = 271.15
const _ROBIN_H_ICE_MIN = 0.1

"""
    _error_function(x) -> erf(x)

Custom erf implementation, ported faithfully from Fortran
`thermodynamics.f90:1487-1533`. Uses a power series for `|x| < 3.5`
and an asymptotic series otherwise. Sign-corrected for `x < 0` in the
asymptotic branch.
"""
function _error_function(x::Float64)
    eps_tol = 1.0e-15
    x2      = x * x
    if abs(x) < 3.5
        er = 1.0
        r  = 1.0
        err = 0.0
        @inbounds for k in 1:50
            r  = r * x2 / (Float64(k) + 0.5)
            er = er + r
            if abs(r) < abs(er) * eps_tol
                c0  = 2.0 / sqrt(pi) * x * exp(-x2)
                err = c0 * er
                break
            end
        end
        return err
    else
        er = 1.0
        r  = 1.0
        err = 0.0
        @inbounds for k in 1:12
            r  = -r * (Float64(k) - 0.5) / x2
            er = er + r
            c0 = exp(-x2) / (abs(x) * sqrt(pi))
            err = 1.0 - c0 * er
            x < 0.0 && (err = -err)
        end
        return err
    end
end

# ---------------------------------------------------------------------------
# `linear` method — linear vertical T profile, frozen base (T_pmp(z=0) - 10).
# ---------------------------------------------------------------------------

"""
    define_temp_linear_column!(T_col, T_srf, T_base, T0, zeta_aa) -> T_col

Per-column linear temperature profile from `T_base` at the bed
(`zeta=0`) to `min(T_srf, T0)` at the surface (`zeta=1`).
"""
function define_temp_linear_column!(T_col::AbstractVector{Float64},
                                    T_srf::Float64, T_base::Float64,
                                    T0::Float64,
                                    zeta_aa::Vector{Float64})
    T_srf_now = min(T_srf, T0)
    Nz = length(zeta_aa)
    @inbounds for k in 1:Nz
        T_col[k] = T_base + zeta_aa[k] * (T_srf_now - T_base)
    end
    return T_col
end

"""
    define_temp_linear_3D!(T_ice_field, omega_field, H_ice_field, T_srf_field,
                           zeta_aa, T0, T_pmp_beta, rho_ice, g) -> T_ice_field

Fill `T_ice` with a linear profile per column (frozen-base linear
between `T_pmp(zeta=0) - 10 K` and `min(T_srf, T0)`); set `omega` to
zero. `enth` is updated separately by `convert_to_enthalpy_3D!` in
`therm_step!` — the Fortran routine does this inline using an
uninitialised local `T_pmp` scalar, which is masked by `omega = 0`;
splitting the conversion out makes the dependency explicit and avoids
porting the latent bug.
"""
function define_temp_linear_3D!(T_ice_field, omega_field,
                                H_ice_field, T_srf_field,
                                zeta_aa::AbstractVector{<:Real},
                                T0::Real, T_pmp_beta::Real,
                                rho_ice::Real, g::Real)
    T_d  = T_ice_field.data
    om_d = omega_field.data
    H_d  = H_ice_field.data
    Ts_d = T_srf_field.data
    Nx   = T_ice_field.grid.Nx
    Ny   = T_ice_field.grid.Ny
    Nz   = T_ice_field.grid.Nz
    zeta = collect(Float64, zeta_aa)
    return _temp_linear_3D_kernel!(T_d, om_d, H_d, Ts_d, zeta,
                                   Float64(T0), Float64(T_pmp_beta),
                                   Float64(rho_ice), Float64(g),
                                   Nx, Ny, Nz)
end

function _temp_linear_3D_kernel!(T, om, H, Ts, zeta::Vector{Float64},
                                 T0::Float64, T_pmp_beta::Float64,
                                 rho_ice::Float64, g::Float64,
                                 Nx::Int, Ny::Int, Nz::Int)
    pref = T_pmp_beta * rho_ice * g
    @inbounds for j in 1:Ny, i in 1:Nx
        H_ij = H[i, j, 1]
        if H_ij > 0.0
            # Frozen-base linear profile. T_pmp at zeta=0 minus 10 K.
            T_pmp_base = T0 - pref * (H_ij * (1.0 - zeta[1]))
            T_base     = T_pmp_base - 10.0
            T_col      = view(T, i, j, :)
            define_temp_linear_column!(T_col, Ts[i, j, 1], T_base, T0, zeta)
        else
            for k in 1:Nz
                T[i, j, k] = T0
            end
        end
        for k in 1:Nz
            om[i, j, k] = 0.0
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# `robin` / `robin-cold` methods — closed-form analytic Robin solution.
# ---------------------------------------------------------------------------

"""
    define_temp_robin_column!(T_col, zeta_aa, T_pmp_col, kt_col, cp_col,
                              rho_ice, H_ice, T_srf, mb_net, Q_rock,
                              is_float, sec_year) -> T_col

Per-column Robin solution with analytic fallback branches. Matches
Fortran `define_temp_robin_column` (line 1244) exactly.

Branches (in order of test):

  1. Grounded + `H_ice > 0.1` + `mb_net > 0` → Robin analytic via
     `erf(z/ll) - erf(H_ice/ll)` with thermal length-scale `ll =
     sqrt(2 κ H / mb)`.
  2. Grounded + `H_ice > 0.1` (mb ≤ 0)      → linear with temperate
     base at `T_pmp[1]`.
  3. Floating                                 → linear from
     `T_ocn = 271.15` at base to `T_srf` at surface.
  4. `H_ice ≤ 0.1`                            → uniform `T_pmp`.

Final cap: `T_col[k] = min(T_col[k], T_pmp[k])` everywhere.
"""
function define_temp_robin_column!(T_col::AbstractVector{Float64},
                                   zeta_aa::Vector{Float64},
                                   T_pmp_col::AbstractVector{Float64},
                                   kt_col::AbstractVector{Float64},
                                   cp_col::AbstractVector{Float64},
                                   rho_ice::Float64,
                                   H_ice::Float64,
                                   T_srf::Float64,
                                   mb_net::Float64,
                                   Q_rock::Float64,
                                   is_float::Bool,
                                   sec_year::Float64)
    Nz         = length(zeta_aa)
    Q_rock_now = Q_rock * 1e-3 * sec_year      # [mW m-2] -> [J a-1 m-2]
    dTdz_b     = -Q_rock_now / kt_col[1]

    sqrt_pi = sqrt(pi)

    if !is_float && H_ice > _ROBIN_H_ICE_MIN && mb_net > 0.0
        mb_now = mb_net
        @inbounds for k in 1:Nz
            z      = zeta_aa[k] * H_ice
            kappa  = kt_col[k] / (cp_col[k] * rho_ice)
            ll     = sqrt(2.0 * kappa * H_ice / mb_now)
            T_col[k] = (sqrt_pi / 2.0) * ll * dTdz_b *
                       (_error_function(z / ll) -
                        _error_function(H_ice / ll)) + T_srf
        end
    elseif !is_float && H_ice > _ROBIN_H_ICE_MIN
        T_col[Nz] = T_srf
        T_col[1]  = T_pmp_col[1]
        @inbounds for k in 2:(Nz - 1)
            T_col[k] = T_col[1] + zeta_aa[k] * (T_col[Nz] - T_col[1])
        end
    elseif is_float
        T_col[Nz] = T_srf
        T_col[1]  = _ROBIN_T_OCN
        @inbounds for k in 2:(Nz - 1)
            T_col[k] = T_col[1] + zeta_aa[k] * (T_col[Nz] - T_col[1])
        end
    else
        @inbounds for k in 1:Nz
            T_col[k] = T_pmp_col[k]
        end
    end

    @inbounds for k in 1:Nz
        if T_col[k] > T_pmp_col[k]
            T_col[k] = T_pmp_col[k]
        end
    end
    return T_col
end

"""
    define_temp_robin_3D!(T_ice_field, omega_field, T_pmp_field,
                          cp_field, kt_field,
                          Q_rock_field, T_srf_field, H_ice_field,
                          smb_field, bmb_grnd_field, f_grnd_field,
                          zeta_aa, rho_ice, sec_year; cold)
        -> T_ice_field

Fill `T_ice` and `omega` per the Robin analytic solution. With
`cold = true` the column is averaged with a cold linear profile
(`T_pmp(zeta=0) - 10 K` to `T_srf`) — the `robin-cold` variant.
`omega` is set to zero everywhere; the caller is responsible for
calling `convert_to_enthalpy_3D!` afterwards. `mb_net = smb +
bmb_grnd` matches Fortran's `bnd%smb + thrm%bmb_grnd`.
"""
function define_temp_robin_3D!(T_ice_field, omega_field,
                               T_pmp_field, cp_field, kt_field,
                               Q_rock_field, T_srf_field, H_ice_field,
                               smb_field, bmb_grnd_field, f_grnd_field,
                               zeta_aa::AbstractVector{<:Real},
                               rho_ice::Real, sec_year::Real;
                               cold::Bool)
    T_d   = T_ice_field.data
    om_d  = omega_field.data
    Tp_d  = T_pmp_field.data
    cp_d  = cp_field.data
    kt_d  = kt_field.data
    Qr_d  = Q_rock_field.data
    Ts_d  = T_srf_field.data
    H_d   = H_ice_field.data
    smb_d = smb_field.data
    bmb_d = bmb_grnd_field.data
    Fg_d  = f_grnd_field.data
    Nx    = T_ice_field.grid.Nx
    Ny    = T_ice_field.grid.Ny
    Nz    = T_ice_field.grid.Nz
    zeta  = collect(Float64, zeta_aa)
    return _temp_robin_3D_kernel!(T_d, om_d, Tp_d, cp_d, kt_d,
                                  Qr_d, Ts_d, H_d, smb_d, bmb_d, Fg_d,
                                  zeta, Float64(rho_ice),
                                  Float64(sec_year), cold,
                                  Nx, Ny, Nz)
end

function _temp_robin_3D_kernel!(T, om, Tp, cp, kt,
                                Qr, Ts, H, smb, bmb, Fg,
                                zeta::Vector{Float64},
                                rho_ice::Float64, sec_year::Float64,
                                cold::Bool,
                                Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        H_ij     = H[i, j, 1]
        T_srf_ij = Ts[i, j, 1]
        is_float = Fg[i, j, 1] == 0.0
        mb_net   = smb[i, j, 1] + bmb[i, j, 1]
        Q_rock_ij = Qr[i, j, 1]

        T_col   = view(T,  i, j, :)
        Tp_col  = view(Tp, i, j, :)
        cp_col  = view(cp, i, j, :)
        kt_col  = view(kt, i, j, :)

        define_temp_robin_column!(T_col, zeta, Tp_col, kt_col, cp_col,
                                  rho_ice, H_ij, T_srf_ij, mb_net,
                                  Q_rock_ij, is_float, sec_year)

        if cold
            # Blend with a cold linear profile: T_base = T_pmp(z=0) - 10,
            # T_top = T_srf, intermediate layers linear in zeta. Matches
            # Fortran robin_3D's local `T1(:)` averaging block.
            T_top  = T_srf_ij
            T_base = Tp_col[1] - 10.0
            for k in 1:Nz
                if k == 1
                    T_lin_k = T_base
                elseif k == Nz
                    T_lin_k = T_top
                else
                    T_lin_k = T_base + zeta[k] * (T_top - T_base)
                end
                T_col[k] = 0.5 * (T_col[k] + T_lin_k)
                # Re-cap at T_pmp post-blend (Fortran does NOT re-cap
                # after the 0.5*blend, but the blend can lift T above
                # T_pmp where the Robin branch returned exactly T_pmp
                # and the linear branch lifted slightly above. Keep
                # faithful to Fortran: do not re-cap.)
            end
        end

        for k in 1:Nz
            om[i, j, k] = 0.0
        end
    end
    return nothing
end
