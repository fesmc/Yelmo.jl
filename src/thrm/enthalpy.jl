# ----------------------------------------------------------------------
# Enthalpy / temperature / water-content conversion.
#
# Direct ports of `convert_to_enthalpy` and (later, in PR5)
# `convert_from_enthalpy_column` from Fortran
# `physics/thermodynamics.f90:1619+`. The forward conversion is the only
# piece needed for the analytic-solver milestone (PR2) — it closes the
# loop after the temperature solvers (linear / robin / robin-cold) have
# written `T_ice` and `omega` so the `enth` field stays consistent.
#
# Wrapper-+-parametric-kernel template per
# `~/.claude/.../memory/wrapper_kernel_template.md`.
# ----------------------------------------------------------------------

"""
    convert_to_enthalpy(temp, omega, T_pmp, cp, L) -> enth

Per-cell forward enthalpy conversion (Fortran
`convert_to_enthalpy`):

    enth = (1 - omega) * cp * temp + omega * (cp * T_pmp + L)

Units: `temp`, `T_pmp` in K; `cp` in J kg^-1 K^-1; `L` in J kg^-1;
`omega` dimensionless. Returns enthalpy in J kg^-1 (multiply by
density to get J m^-3 if needed — the Fortran/Yelmo convention is
to label this "J m^-3" though the formula itself does not include a
density factor).
"""
@inline convert_to_enthalpy(temp::Float64, omega::Float64,
                            T_pmp::Float64, cp::Float64, L::Float64) =
    (1.0 - omega) * (cp * temp) + omega * (cp * T_pmp + L)

"""
    convert_to_enthalpy_3D!(enth_field, T_ice_field, omega_field,
                            T_pmp_field, cp_field, L) -> enth_field

Fill `enth_field` from the per-cell enthalpy formula. All four of
`T_ice_field`, `omega_field`, `T_pmp_field`, `cp_field` must share the
3D grid of `enth_field`. `L` is the latent heat of fusion (scalar,
J kg^-1).
"""
function convert_to_enthalpy_3D!(enth_field, T_ice_field, omega_field,
                                 T_pmp_field, cp_field, L::Real)
    enth_d  = enth_field.data
    T_d     = T_ice_field.data
    om_d    = omega_field.data
    Tp_d    = T_pmp_field.data
    cp_d    = cp_field.data
    Nx      = T_ice_field.grid.Nx
    Ny      = T_ice_field.grid.Ny
    Nz      = T_ice_field.grid.Nz
    return _convert_to_enthalpy_kernel!(enth_d, T_d, om_d, Tp_d, cp_d,
                                        Float64(L), Nx, Ny, Nz)
end

function _convert_to_enthalpy_kernel!(enth, T, om, Tp, cp,
                                      L::Float64,
                                      Nx::Int, Ny::Int, Nz::Int)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        enth[i, j, k] = convert_to_enthalpy(T[i, j, k], om[i, j, k],
                                            Tp[i, j, k], cp[i, j, k], L)
    end
    return nothing
end
