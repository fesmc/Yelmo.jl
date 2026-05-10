# ----------------------------------------------------------------------
# Thermodynamic property kernels.
#
# Direct ports of the elemental property functions in Fortran
# `physics/thermodynamics.f90:930-1050`:
#
#   - `calc_specific_heat_capacity` — Greve & Blatter (2009) Eq. 4.39
#     (Ritz 1987). Linear in T.
#   - `calc_thermal_conductivity` — Greve & Blatter (2009) Eq. 4.37
#     (Ritz 1987). Exponential in T, with the `*sec_year` factor baked
#     in so the returned conductivity is in J m^-1 K^-1 a^-1 (the unit
#     used throughout the Fortran column solvers, despite the
#     `W m^-1 K^-1` label in `yelmo-variables-ytherm.md`).
#   - `calc_T_pmp` — Greve & Blatter (2009) Eq. 4.13. Pressure-corrected
#     melting point. `zeta` is the fractional height in the column;
#     overlying ice thickness is `H_ice * (1 - zeta)`.
#   - `calc_f_pmp!` — Greve (2005), Hindmarsh & Le Meur (2001). Smooth
#     decay function for the fraction of a basal cell at the pressure
#     melting point. Floating cells are temperate by default.
#
# Wrapper-+-parametric-kernel template (per
# `~/.claude/.../memory/wrapper_kernel_template.md`):
#
#   - The `calc_X` scalar functions are inline-able elementals.
#   - The `calc_*_3D!` wrappers lift `Field.data` into `OffsetArray`
#     handles and dispatch into a parametric kernel taking
#     concrete-typed `Float64` scalars and explicit `Nx, Ny, Nz` ints.
#   - For `calc_T_pmp_3D!` the per-layer `zeta_aa[k]` is read from a
#     materialised `Vector{Float64}` (the Oceananigans `znodes(grid)`
#     range collected once in the wrapper), so the inner loop sees a
#     concretely-typed vector.
# ----------------------------------------------------------------------

"""
    calc_specific_heat_capacity(T_ice) -> cp

Specific heat capacity of ice [J kg^-1 K^-1] at temperature `T_ice`
[K]. Greve & Blatter (2009) Eq. 4.39; Ritz (1987).
"""
@inline calc_specific_heat_capacity(T_ice::Float64) = 146.3 + 7.253 * T_ice

"""
    calc_thermal_conductivity(T_ice, sec_year) -> kt

Thermal conductivity of ice at temperature `T_ice` [K], multiplied by
`sec_year` to convert from W m^-1 K^-1 to J m^-1 K^-1 a^-1 (the unit
expected by the column solvers). Greve & Blatter (2009) Eq. 4.37;
Ritz (1987).
"""
@inline calc_thermal_conductivity(T_ice::Float64, sec_year::Float64) =
    9.828 * exp(-0.0057 * T_ice) * sec_year

"""
    calc_T_pmp(H_ice, zeta, T0, beta, rho_ice, g) -> T_pmp

Pressure-corrected melting point [K] at fractional column height
`zeta ∈ [0, 1]` in ice of total thickness `H_ice` [m]. The overlying
ice thickness is `H_ice*(1 - zeta)`. Greve & Blatter (2009) Eq. 4.13.
"""
@inline calc_T_pmp(H_ice::Float64, zeta::Float64,
                   T0::Float64, beta::Float64,
                   rho_ice::Float64, g::Float64) =
    T0 - (beta * rho_ice * g) * (H_ice * (1.0 - zeta))


# ---------------------------------------------------------------------------
# 3D wrappers + parametric kernels (Field-aware).
# ---------------------------------------------------------------------------

"""
    calc_cp_3D!(cp_field, T_field) -> cp_field

Fill `cp_field` with the specific heat capacity at every cell of the
3D temperature field `T_field`. Both fields must share the same
Oceananigans grid.
"""
function calc_cp_3D!(cp_field, T_field)
    cp_d = cp_field.data
    T_d  = T_field.data
    Nx   = T_field.grid.Nx
    Ny   = T_field.grid.Ny
    Nz   = T_field.grid.Nz
    return _calc_cp_kernel!(cp_d, T_d, Nx, Ny, Nz)
end

function _calc_cp_kernel!(cp, T, Nx::Int, Ny::Int, Nz::Int)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        cp[i, j, k] = calc_specific_heat_capacity(T[i, j, k])
    end
    return nothing
end

"""
    calc_kt_3D!(kt_field, T_field, sec_year) -> kt_field

Fill `kt_field` with the thermal conductivity (multiplied by
`sec_year`, returning J m^-1 K^-1 a^-1) at every cell of the 3D
temperature field `T_field`.
"""
function calc_kt_3D!(kt_field, T_field, sec_year::Real)
    kt_d = kt_field.data
    T_d  = T_field.data
    Nx   = T_field.grid.Nx
    Ny   = T_field.grid.Ny
    Nz   = T_field.grid.Nz
    return _calc_kt_kernel!(kt_d, T_d, Float64(sec_year), Nx, Ny, Nz)
end

function _calc_kt_kernel!(kt, T, sec_year::Float64, Nx::Int, Ny::Int, Nz::Int)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        kt[i, j, k] = calc_thermal_conductivity(T[i, j, k], sec_year)
    end
    return nothing
end

"""
    calc_T_pmp_3D!(T_pmp_field, H_ice_field, zeta_aa, T0, beta, rho_ice, g)
        -> T_pmp_field

Fill `T_pmp_field` with the pressure-corrected melting point at every
3D cell. `zeta_aa` is the cell-centre vertical coordinate; values are
collected to a concrete `Vector{Float64}` once in the wrapper so the
inner loop sees a concretely-typed vector.
"""
function calc_T_pmp_3D!(T_pmp_field, H_ice_field,
                        zeta_aa::AbstractVector{<:Real},
                        T0::Real, beta::Real,
                        rho_ice::Real, g::Real)
    Tp_d  = T_pmp_field.data
    H_d   = H_ice_field.data
    Nx    = T_pmp_field.grid.Nx
    Ny    = T_pmp_field.grid.Ny
    Nz    = T_pmp_field.grid.Nz
    zeta  = collect(Float64, zeta_aa)
    return _calc_T_pmp_kernel!(Tp_d, H_d, zeta,
                               Float64(T0), Float64(beta),
                               Float64(rho_ice), Float64(g),
                               Nx, Ny, Nz)
end

function _calc_T_pmp_kernel!(Tp, H, zeta::Vector{Float64},
                             T0::Float64, beta::Float64,
                             rho_ice::Float64, g::Float64,
                             Nx::Int, Ny::Int, Nz::Int)
    pref = beta * rho_ice * g
    @inbounds for k in 1:Nz
        zk = zeta[k]
        for j in 1:Ny, i in 1:Nx
            Tp[i, j, k] = T0 - pref * (H[i, j, 1] * (1.0 - zk))
        end
    end
    return nothing
end

"""
    calc_T_pmp_boundaries_2D!(T_pmp_b, T_pmp_s, H_ice, T0, beta, rho_ice, g)
        -> nothing

Fill the 2D basal (`T_pmp_b`, ζ=0) and surface (`T_pmp_s`, ζ=1)
pressure-corrected melting-point fields. At the bed the overlying ice
column is the full `H_ice`; at the surface it is zero, so
`T_pmp_s ≡ T0` everywhere. Companion of [`calc_T_pmp_3D!`](@ref) for
the boundary fields registered in `BOUNDARY_FIELD_REGISTRY_ICE`.
"""
function calc_T_pmp_boundaries_2D!(T_pmp_b_field, T_pmp_s_field, H_ice_field,
                                   T0::Real, beta::Real,
                                   rho_ice::Real, g::Real)
    Tpb_d = T_pmp_b_field.data
    Tps_d = T_pmp_s_field.data
    H_d   = H_ice_field.data
    Nx    = T_pmp_b_field.grid.Nx
    Ny    = T_pmp_b_field.grid.Ny
    pref  = Float64(beta) * Float64(rho_ice) * Float64(g)
    T0_f  = Float64(T0)
    @inbounds for j in 1:Ny, i in 1:Nx
        Tpb_d[i, j, 1] = T0_f - pref * H_d[i, j, 1]   # ζ = 0 (basal)
        Tps_d[i, j, 1] = T0_f                          # ζ = 1 (surface)
    end
    return nothing
end

"""
    calc_f_pmp!(f_pmp_field, T_ice_b_field, T_pmp_b_field, f_grnd_field; gamma)
        -> f_pmp_field

Compute the fraction of each basal cell at the pressure melting point
(`f_pmp ∈ [0, 1]`):

  - Floating cells (`f_grnd == 0`) are temperate by default → `f_pmp = 1`.
  - With `gamma == 0` the result is binary (`T_ice_b ≥ T_pmp_b`).
  - With `gamma > 0` a smooth decay `exp(min(T_ice_b - T_pmp_b, 0) / gamma)`
    is applied, clamped to `[0, 1]` outside `[1e-2, 1 - 1e-2]`, with the
    inner `dT` floored at `-20 K` to avoid underflow at very cold cells.

Greve (2005); Hindmarsh & Le Meur (2001). Direct port of Fortran
`thermodynamics.f90:991-1050`. The basal temperature and
pressure-melting temperature live in the dedicated 2D `T_ice_b` /
`T_pmp_b` boundary fields rather than in `T_ice[:, :, 1]` / `T_pmp[:, :, 1]`
(which are the first interior layer at ζ_aa[1], not the boundary at
ζ = 0).
"""
function calc_f_pmp!(f_pmp_field, T_ice_b_field, T_pmp_b_field, f_grnd_field;
                     gamma::Real)
    fp_d  = f_pmp_field.data
    Tib_d = T_ice_b_field.data
    Tpb_d = T_pmp_b_field.data
    Fg_d  = f_grnd_field.data
    Nx    = f_pmp_field.grid.Nx
    Ny    = f_pmp_field.grid.Ny
    return _calc_f_pmp_kernel!(fp_d, Tib_d, Tpb_d, Fg_d, Float64(gamma), Nx, Ny)
end

function _calc_f_pmp_kernel!(fp, Tib, Tpb, Fg, gamma::Float64, Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        if Fg[i, j, 1] == 0.0
            fp[i, j, 1] = 1.0
        elseif gamma == 0.0
            fp[i, j, 1] = (Tib[i, j, 1] >= Tpb[i, j, 1]) ? 1.0 : 0.0
        else
            dT = min(Tib[i, j, 1] - Tpb[i, j, 1], 0.0)
            dT = max(dT, -20.0)
            v  = exp(dT / gamma)
            v  = (v < 1e-2)         ? 0.0 : v
            v  = (v > 1.0 - 1e-2)   ? 1.0 : v
            fp[i, j, 1] = v
        end
    end
    return nothing
end
