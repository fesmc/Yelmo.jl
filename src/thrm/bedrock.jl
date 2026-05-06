# ----------------------------------------------------------------------
# Bedrock thermodynamics — `rock_method ∈ {fixed, equil, active}`.
#
# Direct ports:
#
#   - `define_temp_bedrock_3D`     ← `physics/thermodynamics.f90:1336`
#       Linear-equilibrium bedrock profile from `T_bed` (ice base) at
#       the top to a deep slope `dT/dz = -Q_geo / kt_rock`. Used for
#       `rock_method = "equil"`.
#   - `define_temp_bedrock_column` ← `physics/thermodynamics.f90:1381`
#       Per-column linear interpolation: descend from surface to base
#       with constant slope.
#   - `calc_Q_bedrock_column`      ← `physics/thermodynamics.f90:1457`
#       Diagnose `Q_rock` (mW m^-2) from the upwind T_rock gradient at
#       the bed surface (top of bedrock column).
#   - `calc_temp_bedrock_column`   ← `physics/ice_enthalpy.f90:250-366`
#       Implicit-solve variant. Used for `rock_method = "active"`.
#       Reuses `_calc_temp_column_internal!` with no advection,
#       no strain heat, basal Neumann from Q_geo, surface Dirichlet
#       at T_ice basal temperature.
#
# Bedrock conventions:
#
#   - `zeta_aa` for bedrock starts at `0` (deep) and ends at `1` (bed
#     surface). The "surface" of the bedrock column (`k = nz_aa`)
#     equals the BASE of the ice column.
#   - `cp_rock`, `kt_rock` are scalar (constant per Yelmo
#     convention) — the param struct stores them as `cp_rock`,
#     `kt_rock`.
#
# `rock_method = "fixed"` is a no-op handled inline in `therm_step!`.
# `rock_method = "active"` is implemented via `define_temp_bedrock_active_3D!`,
# which mirrors Fortran's `calc_ytherm_enthalpy_bedrock_3D`.
# ----------------------------------------------------------------------

"""
    define_temp_bedrock_column!(T_rock_col, kt_rock, H_rock, T_bed, Q_geo,
                                zeta_aa, sec_year) -> T_rock_col

Per-column equilibrium bedrock profile. Walks from the bedrock
surface (k = nz_aa, anchored at `T_rock = T_bed`) downward with slope
`-Q_geo / kt_rock`.
"""
function define_temp_bedrock_column!(T_rock_col::AbstractVector{Float64},
                                     kt_rock::Float64, H_rock::Float64,
                                     T_bed::Float64, Q_geo::Float64,
                                     zeta_aa::Vector{Float64},
                                     sec_year::Float64)
    nz_aa     = length(zeta_aa)
    Q_geo_now = Q_geo * 1e-3 * sec_year     # mW/m² → J m⁻² a⁻¹
    dTdz      = -Q_geo_now / kt_rock        # K/m (negative; cooling upward)

    @inbounds begin
        T_rock_col[nz_aa] = T_bed
        for k in (nz_aa - 1):-1:1
            dz = H_rock * (zeta_aa[k + 1] - zeta_aa[k])
            T_rock_col[k] = T_rock_col[k + 1] - dTdz * dz
        end
    end
    return T_rock_col
end

"""
    calc_Q_bedrock_column(T_rock_col, kt_rock, H_rock, zeta_aa, sec_year)
        -> Q_rock_mW

Diagnose Q_rock at the bedrock surface (mW m⁻²) from the 1st-order
upwind temperature gradient between `T_rock[nz_aa]` and `T_rock[nz_aa-1]`.
"""
@inline function calc_Q_bedrock_column(T_rock_col::AbstractVector{Float64},
                                       kt_rock::Float64, H_rock::Float64,
                                       zeta_aa::Vector{Float64},
                                       sec_year::Float64)
    nz_aa    = length(zeta_aa)
    @inbounds dz = H_rock * (zeta_aa[nz_aa] - zeta_aa[nz_aa - 1])
    @inbounds Q_rock_now = -kt_rock *
                            (T_rock_col[nz_aa] - T_rock_col[nz_aa - 1]) / dz
    return Q_rock_now * 1e3 / sec_year   # → mW/m²
end

"""
    define_temp_bedrock_3D!(enth_rock_field, T_rock_field, Q_rock_field,
                            T_ice_basal_field, Q_geo_field,
                            cp_rock, kt_rock, H_rock,
                            zeta_aa, sec_year) -> nothing

Fill `T_rock`, `Q_rock`, `enth_rock` for the `rock_method = "equil"`
mode: linear profile at every (i, j) anchored at `T_bed = T_ice[i,j,1]`
(ice base, the top of the bedrock column) with deep slope from
`Q_geo`. `enth_rock = cp_rock * T_rock` (Fortran calls
`convert_to_enthalpy(enth, T, omega=0, T_pmp=0, cp, L=0)`).
"""
function define_temp_bedrock_3D!(enth_rock_field, T_rock_field,
                                 Q_rock_field, T_ice_basal_field,
                                 Q_geo_field,
                                 cp_rock::Real, kt_rock::Real,
                                 H_rock::Real,
                                 zeta_aa_rock::AbstractVector{<:Real},
                                 sec_year::Real)
    er_d  = enth_rock_field.data
    Tr_d  = T_rock_field.data
    Qr_d  = Q_rock_field.data
    Tib_d = T_ice_basal_field.data
    Qg_d  = Q_geo_field.data
    Nx    = T_rock_field.grid.Nx
    Ny    = T_rock_field.grid.Ny
    Nz    = T_rock_field.grid.Nz
    zeta  = collect(Float64, zeta_aa_rock)
    return _define_temp_bedrock_3D_kernel!(er_d, Tr_d, Qr_d, Tib_d, Qg_d,
                                           Float64(cp_rock),
                                           Float64(kt_rock),
                                           Float64(H_rock),
                                           zeta, Float64(sec_year),
                                           Nx, Ny, Nz)
end

function _define_temp_bedrock_3D_kernel!(er, Tr, Qr, Tib, Qg,
                                         cp_rock::Float64, kt_rock::Float64,
                                         H_rock::Float64,
                                         zeta_aa::Vector{Float64},
                                         sec_year::Float64,
                                         Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        T_rock_col = view(Tr, i, j, :)
        define_temp_bedrock_column!(T_rock_col, kt_rock, H_rock,
                                    Tib[i, j, 1], Qg[i, j, 1],
                                    zeta_aa, sec_year)
        Qr[i, j, 1] = calc_Q_bedrock_column(T_rock_col, kt_rock,
                                            H_rock, zeta_aa, sec_year)
        for k in 1:Nz
            er[i, j, k] = cp_rock * T_rock_col[k]
        end
    end
    return nothing
end

"""
    define_temp_bedrock_active_3D!(enth_rock_field, T_rock_field,
                                   Q_rock_field, T_ice_basal_field,
                                   Q_geo_field,
                                   cp_rock, kt_rock, rho_rock, H_rock,
                                   zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                                   sec_year, dt) -> nothing

Implicit-solve variant for `rock_method = "active"`. Mirrors Fortran
`calc_temp_bedrock_column` (`physics/ice_enthalpy.f90:250-366`) per
column. No advection, no strain heat — surface (top, bedrock-ice
interface) is Dirichlet at the ice basal temperature, base (deep) is
Neumann from Q_geo.
"""
function define_temp_bedrock_active_3D!(enth_rock_field, T_rock_field,
                                        Q_rock_field, T_ice_basal_field,
                                        Q_geo_field,
                                        cp_rock::Real, kt_rock::Real,
                                        rho_rock::Real, H_rock::Real,
                                        zeta_aa_rock::AbstractVector{<:Real},
                                        zeta_ac_rock::AbstractVector{<:Real},
                                        dzeta_a_rock::AbstractVector{<:Real},
                                        dzeta_b_rock::AbstractVector{<:Real},
                                        sec_year::Real, dt::Real)
    er_d  = enth_rock_field.data
    Tr_d  = T_rock_field.data
    Qr_d  = Q_rock_field.data
    Tib_d = T_ice_basal_field.data
    Qg_d  = Q_geo_field.data
    Nx    = T_rock_field.grid.Nx
    Ny    = T_rock_field.grid.Ny
    Nz    = T_rock_field.grid.Nz

    zeta_aa = collect(Float64, zeta_aa_rock)
    zeta_ac = collect(Float64, zeta_ac_rock)
    dzeta_a = collect(Float64, dzeta_a_rock)
    dzeta_b = collect(Float64, dzeta_b_rock)

    # Per-column scratch.
    kappa_buf    = fill(Float64(kt_rock) / (Float64(rho_rock) * Float64(cp_rock)), Nz)
    advecxy_zero = zeros(Float64, Nz)
    Q_strn_zero  = zeros(Float64, Nz)
    uz_zero      = zeros(Float64, Nz + 1)
    subd         = Vector{Float64}(undef, Nz)
    diag         = Vector{Float64}(undef, Nz)
    supd         = Vector{Float64}(undef, Nz)
    rhs          = Vector{Float64}(undef, Nz)
    solution     = Vector{Float64}(undef, Nz)
    cp_tri       = Vector{Float64}(undef, Nz)
    dp_tri       = Vector{Float64}(undef, Nz)

    return _define_temp_bedrock_active_3D_kernel!(er_d, Tr_d, Qr_d, Tib_d, Qg_d,
                                                  Float64(cp_rock),
                                                  Float64(kt_rock),
                                                  Float64(H_rock),
                                                  zeta_aa, zeta_ac,
                                                  dzeta_a, dzeta_b,
                                                  Float64(sec_year),
                                                  Float64(dt),
                                                  kappa_buf,
                                                  advecxy_zero, Q_strn_zero,
                                                  uz_zero,
                                                  subd, diag, supd,
                                                  rhs, solution,
                                                  cp_tri, dp_tri,
                                                  Nx, Ny, Nz)
end

function _define_temp_bedrock_active_3D_kernel!(er, Tr, Qr, Tib, Qg,
                                                cp_rock::Float64,
                                                kt_rock::Float64,
                                                H_rock::Float64,
                                                zeta_aa::Vector{Float64},
                                                zeta_ac::Vector{Float64},
                                                dzeta_a::Vector{Float64},
                                                dzeta_b::Vector{Float64},
                                                sec_year::Float64,
                                                dt::Float64,
                                                kappa_buf::Vector{Float64},
                                                advecxy_zero::Vector{Float64},
                                                Q_strn_zero::Vector{Float64},
                                                uz_zero::Vector{Float64},
                                                subd::Vector{Float64},
                                                diag::Vector{Float64},
                                                supd::Vector{Float64},
                                                rhs::Vector{Float64},
                                                solution::Vector{Float64},
                                                cp_tri::Vector{Float64},
                                                dp_tri::Vector{Float64},
                                                Nx::Int, Ny::Int, Nz::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        T_col = view(Tr, i, j, :)

        # Surface Dirichlet at ice basal temperature.
        val_srf      = Tib[i, j, 1]
        is_surf_flux = false

        # Base Neumann from Q_geo.
        Q_geo_now    = Qg[i, j, 1] * 1e-3 * sec_year   # → J a⁻¹ m⁻²
        val_base     = -Q_geo_now / kt_rock
        is_basal_flux = true

        _calc_temp_column_internal!(T_col, kappa_buf, uz_zero,
                                    advecxy_zero, Q_strn_zero,
                                    val_base, val_srf, H_rock,
                                    zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                                    _T_REF_ICE, dt,
                                    is_basal_flux, is_surf_flux,
                                    subd, diag, supd, rhs, solution,
                                    cp_tri, dp_tri)

        Qr[i, j, 1] = calc_Q_bedrock_column(T_col, kt_rock, H_rock,
                                            zeta_aa, sec_year)
        for k in 1:Nz
            er[i, j, k] = cp_rock * T_col[k]
        end
    end
    return nothing
end
