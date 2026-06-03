# ----------------------------------------------------------------------
# Glen-law rate-factor `ATT` parameterisations and water-content
# scaling. Ports of three `elemental` Fortran routines:
#
#   - `calc_rate_factor`         (deformation.f90:389) — Greve & Blatter
#                                 (2009) Arrhenius-law constants
#   - `calc_rate_factor_eismint` (deformation.f90:427) — EISMINT2 /
#                                 Payne et al. (2000) constants
#   - `scale_rate_factor_water`  (deformation.f90:461) — Lliboutry &
#                                 Duval (1985) water-content scaling
#
# All three take a temperature-coupled ice state. They are wired into
# `mat_step!` when `rf_method = 1` (temperature-coupled Arrhenius);
# `mat_step!` reads `T_ice`, `T_pmp`, and `omega` from the therm port,
# branches on `ymat.rf_use_eismint2` between the EISMINT2 and Greve &
# Blatter constants, and applies `scale_rate_factor_water!` when
# `ymat.rf_with_water = true`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using LoopVectorization: @turbo

export calc_rate_factor!, calc_rate_factor_eismint!,
       scale_rate_factor_water!


# Greve & Blatter (2009) Arrhenius constants — used by `calc_rate_factor`.
const _RF_GB_T_PRIME_LIM = 263.15           # [K] split between low/high-T
const _RF_GB_A0_1        = 1.25671e-5       # [1/yr / Pa^3]  T' ≤ T_lim
const _RF_GB_A0_2        = 6.0422976e10     # [1/yr / Pa^3]  T' >  T_lim
const _RF_GB_Q_1         =  60.0e3          # [J/mol]
const _RF_GB_Q_2         = 139.0e3          # [J/mol]

# EISMINT2 / Payne et al. (2000) Arrhenius constants — used by
# `calc_rate_factor_eismint` (no T' floor / cap clamps).
const _RF_E2_T_PRIME_LIM = 263.15           # [K]
const _RF_E2_A0_1        = 1.139205e-5      # [1/yr / Pa^3]
const _RF_E2_A0_2        = 5.459348198e10   # [1/yr / Pa^3]
const _RF_E2_Q_1         =  60.0e3          # [J/mol]
const _RF_E2_Q_2         = 139.0e3          # [J/mol]

const _RF_R = 8.314                         # [J/mol/K] gas constant

# T_prime clamps applied by `calc_rate_factor` only (the GB form).
# `calc_rate_factor_eismint` skips them — Fortran lines 451-455.
const _RF_GB_T_PRIME_FLOOR = 220.0          # [K] avoid exp underflow

# Water-content scaling factor — Greve & Blatter (2016) Eq. 14.
const _RF_WATER_FACTOR = 181.25             # ATT *= (1 + 181.25 · omega)


# Per-cell Greve & Blatter Arrhenius rate factor.
# Replicates Fortran clamps `T_prime = clamp(T_prime, 220, T_pmp)`.
@inline function _rate_factor_gb(T_ice::Real, T_pmp::Real,
                                 enh::Real, T0::Real)
    Tp = Float64(T_ice) - Float64(T_pmp) + Float64(T0)
    Tp = Tp < _RF_GB_T_PRIME_FLOOR ? _RF_GB_T_PRIME_FLOOR : Tp
    Tp = Tp > Float64(T_pmp)       ? Float64(T_pmp)       : Tp
    if Tp <= _RF_GB_T_PRIME_LIM
        return Float64(enh) * _RF_GB_A0_1 * exp(-_RF_GB_Q_1 / (_RF_R * Tp))
    else
        return Float64(enh) * _RF_GB_A0_2 * exp(-_RF_GB_Q_2 / (_RF_R * Tp))
    end
end

# Per-cell EISMINT2 / Payne (2000) Arrhenius rate factor. No T_prime
# clamps — Fortran lines 451-455 only branch on T_prime_lim.
@inline function _rate_factor_eismint(T_ice::Real, T_pmp::Real,
                                      enh::Real, T0::Real)
    Tp = Float64(T_ice) - Float64(T_pmp) + Float64(T0)
    if Tp <= _RF_E2_T_PRIME_LIM
        return Float64(enh) * _RF_E2_A0_1 * exp(-_RF_E2_Q_1 / (_RF_R * Tp))
    else
        return Float64(enh) * _RF_E2_A0_2 * exp(-_RF_E2_Q_2 / (_RF_R * Tp))
    end
end


"""
    calc_rate_factor!(ATT, T_ice, T_pmp, enh, T0) -> ATT

Greve & Blatter (2009) Arrhenius rate factor (`yelmo/src/physics/
deformation.f90:389 calc_rate_factor`). Per-cell formula:

    T_prime = clamp(T_ice - T_pmp + T0, 220 K, T_pmp)
    ATT     = enh · A0 · exp(-Q / (R · T_prime))

with `(A0, Q)` switching at `T_prime = 263.15 K`. All inputs are 3D
`CenterField`s (Center vertical staggering); `T0` is a scalar from
`y.bnd.c.T0` / `YelmoConstants.T0`.

Valid only for Glen exponent `n = 3` (the Arrhenius pre-factors are
calibrated for `n = 3`; mixing with other `n_glen` violates the
calibration assumptions).
"""
function calc_rate_factor!(ATT, T_ice, T_pmp, enh, T0::Real)
    A    = interior(ATT)
    Ti   = interior(T_ice)
    Tpmp = interior(T_pmp)
    En   = interior(enh)

    Nx, Ny, Nz = size(A)
    @assert size(Ti)   == size(A) "calc_rate_factor!: T_ice size mismatch"
    @assert size(Tpmp) == size(A) "calc_rate_factor!: T_pmp size mismatch"
    @assert size(En)   == size(A) "calc_rate_factor!: enh size mismatch"

    # Inlined + branchless: the original looped over a non-SIMDable
    # `@inline` helper with an internal `if Tp <= T_LIM` branch on the
    # Arrhenius constant pair. Here both constants are selected via
    # `ifelse` so the loop body is straight-line, and `@turbo` lifts the
    # single `exp` call to SLEEFPirates' SIMD intrinsics.
    T0_f  = Float64(T0)
    Tflr  = _RF_GB_T_PRIME_FLOOR
    Tlim  = _RF_GB_T_PRIME_LIM
    A0_1  = _RF_GB_A0_1
    A0_2  = _RF_GB_A0_2
    Q_1   = _RF_GB_Q_1
    Q_2   = _RF_GB_Q_2
    R     = _RF_R
    @turbo for k in 1:Nz, j in 1:Ny, i in 1:Nx
        t_pmp   = Tpmp[i, j, k]
        t_prime = Ti[i, j, k] - t_pmp + T0_f
        # Clamp T_prime to [T_FLOOR, T_pmp] (Fortran lines 408-409).
        t_prime = ifelse(t_prime < Tflr,  Tflr,  t_prime)
        t_prime = ifelse(t_prime > t_pmp, t_pmp, t_prime)
        low     = t_prime <= Tlim
        A0      = ifelse(low, A0_1, A0_2)
        Q       = ifelse(low, Q_1,  Q_2)
        A[i, j, k] = En[i, j, k] * A0 * exp(-Q / (R * t_prime))
    end
    return ATT
end


"""
    calc_rate_factor_eismint!(ATT, T_ice, T_pmp, enh, T0) -> ATT

EISMINT2 / Payne et al. (2000) Arrhenius rate factor (`yelmo/src/
physics/deformation.f90:427 calc_rate_factor_eismint`). Same form as
`calc_rate_factor!` but with EISMINT2 calibration constants and **no**
T_prime clamps (Fortran lines 451-455 only switch on `T_prime_lim`).

Selected by `mat.par.rf_use_eismint2 = true` in the `rf_method = 1`
branch.
"""
function calc_rate_factor_eismint!(ATT, T_ice, T_pmp, enh, T0::Real)
    A    = interior(ATT)
    Ti   = interior(T_ice)
    Tpmp = interior(T_pmp)
    En   = interior(enh)

    Nx, Ny, Nz = size(A)
    @assert size(Ti)   == size(A) "calc_rate_factor_eismint!: T_ice size mismatch"
    @assert size(Tpmp) == size(A) "calc_rate_factor_eismint!: T_pmp size mismatch"
    @assert size(En)   == size(A) "calc_rate_factor_eismint!: enh size mismatch"

    # Same shape as `calc_rate_factor!` but without the T_prime clamps
    # (Fortran lines 451-455 only branch on T_PRIME_LIM).
    T0_f  = Float64(T0)
    Tlim  = _RF_E2_T_PRIME_LIM
    A0_1  = _RF_E2_A0_1
    A0_2  = _RF_E2_A0_2
    Q_1   = _RF_E2_Q_1
    Q_2   = _RF_E2_Q_2
    R     = _RF_R
    @turbo for k in 1:Nz, j in 1:Ny, i in 1:Nx
        t_prime = Ti[i, j, k] - Tpmp[i, j, k] + T0_f
        low     = t_prime <= Tlim
        A0      = ifelse(low, A0_1, A0_2)
        Q       = ifelse(low, Q_1,  Q_2)
        A[i, j, k] = En[i, j, k] * A0 * exp(-Q / (R * t_prime))
    end
    return ATT
end


"""
    scale_rate_factor_water!(ATT, omega) -> ATT

In-place water-content scaling of the rate factor (`yelmo/src/
physics/deformation.f90:461 scale_rate_factor_water`). Per cell:

    ATT *= (1 + 181.25 · omega)    if omega > 0

Cells with `omega <= 0` are left unchanged. Used by `mat_step!` when
`mat.par.rf_with_water = true` after computing the temperature-only
rate factor.

Parameterisation: Greve & Blatter (2016) Eq. 14, following Lliboutry
& Duval (1985).
"""
function scale_rate_factor_water!(ATT, omega)
    A = interior(ATT)
    Ω = interior(omega)

    @assert size(Ω) == size(A) "scale_rate_factor_water!: omega size mismatch"

    @inbounds for I in eachindex(A)
        if Ω[I] > 0.0
            A[I] = A[I] * (1.0 + _RF_WATER_FACTOR * Ω[I])
        end
    end
    return ATT
end
