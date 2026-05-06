# ----------------------------------------------------------------------
# Implicit 1D vertical advection-diffusion solver for a single column.
#
# Direct port of Fortran `calc_temp_column_internal`
# (`physics/ice_enthalpy.f90:368-534`). Produces a tridiagonal system
# `M · solution = rhs` with backward-Euler diffusion, centered-difference
# vertical advection, and explicit horizontal advection / strain
# heating in the RHS, then solves it via the Thomas algorithm
# (`solve_tridiag!`).
#
# Boundary conditions:
#
#   - Bottom (k = 1):  Dirichlet (`is_basal_flux = false`,
#                       `solution[1] = val_base - T_ref`) OR
#                      Neumann via backward-Euler flux
#                       (`is_basal_flux = true`,
#                        coefficients `subd=0, diag=-1, supd=+1`,
#                        rhs=`val_base * dz`).
#   - Top (k = nz):    Dirichlet OR Neumann mirror of the bottom.
#
# `T_ref` is a numerical conditioning offset (273.15 K for ice;
# 273.15 K also for bedrock per Fortran). The kernel returns the
# solved column already shifted back to absolute K
# (`temp = solution + T_ref`).
#
# The kernel takes caller-supplied scratch vectors `subd`, `diag`,
# `supd`, `rhs`, `solution`, `cp_buf`, `dp_buf` so the 3D outer loop
# can reuse one set of length-Nz buffers across all (i, j) cells.
# Mirrors the wrapper-+-kernel template — the 3D outer loop allocates
# scratch once per call and passes it down.
#
# Fortran flag: there's a hard-coded `if (.TRUE.)` branch toggle inside
# `calc_temp_column_internal` for centered-difference vs.
# variable-layer-thickness vertical advection. The default-true branch
# is the centered scheme used in Yelmo; we port the centered branch
# only. The variable-layer alternative is left out (it's unused in
# practice and the comment notes "this can get unstable in some rare
# cases").
# ----------------------------------------------------------------------

const _T_REF_ICE  = 273.15

"""
    _calc_temp_column_internal!(temp, kappa, uz, advecxy, Q_strn,
                                val_base, val_srf, thickness,
                                zeta_aa, zeta_ac, dzeta_a, dzeta_b,
                                T_ref, dt,
                                is_basal_flux, is_surf_flux,
                                subd, diag, supd, rhs, solution,
                                cp_buf, dp_buf) -> temp

Solve the implicit 1D thermo equation in the column. `temp` is updated
in place (input current state, output new state).
"""
function _calc_temp_column_internal!(temp::AbstractVector{Float64},
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
                                     T_ref::Float64, dt::Float64,
                                     is_basal_flux::Bool,
                                     is_surf_flux::Bool,
                                     subd::Vector{Float64},
                                     diag::Vector{Float64},
                                     supd::Vector{Float64},
                                     rhs::Vector{Float64},
                                     solution::Vector{Float64},
                                     cp_buf::Vector{Float64},
                                     dp_buf::Vector{Float64};
                                     kappa_basal::Float64,
                                     kappa_surf::Float64)
    nz_aa = length(zeta_aa)

    @inbounds begin
        H2 = thickness * thickness

        # -- Base BC --
        if !is_basal_flux
            # Dirichlet: `val_base` lives at z=0, `temp[1]` at zeta_aa[1]
            # (first interior centre). Absorb the T[0]=val_base contribution
            # into the rhs; kappa_basal is the diffusivity at the z=0 face.
            kappa_a = kappa_basal
            dz1     = zeta_ac[2]   - zeta_aa[1]
            dz2     = zeta_aa[2]   - zeta_ac[2]
            kappa_n = _calc_wtd_harmonic_mean(kappa[1], kappa[2], dz1, dz2)

            fac_a = -kappa_a * dzeta_a[1] * dt / H2
            fac_b = -kappa_n * dzeta_b[1] * dt / H2

            uz_aa     = 0.5 * (uz[1] + uz[2])
            dzeta_adv = zeta_aa[2] - 0.0
            dz_adv    = thickness * dzeta_adv

            absorb = fac_a - uz_aa * dt / dz_adv

            subd[1] = 0.0
            supd[1] = fac_b + uz_aa * dt / dz_adv
            diag[1] = 1.0 - fac_a - fac_b
            rhs[1]  = (temp[1] - T_ref) - dt * advecxy[1] + dt * Q_strn[1] -
                      absorb * (val_base - T_ref)
        else
            # Neumann: backward-Euler flux at the base.
            dz       = thickness * (zeta_aa[2] - zeta_aa[1])
            subd[1]  = 0.0
            diag[1]  = -1.0
            supd[1]  = 1.0
            rhs[1]   = val_base * dz
        end

        # -- Interior layers --
        for k in 2:(nz_aa - 1)
            # Harmonic-mean kappa onto ac-nodes k and k+1.
            dz1     = zeta_ac[k]   - zeta_aa[k - 1]
            dz2     = zeta_aa[k]   - zeta_ac[k]
            kappa_a = _calc_wtd_harmonic_mean(kappa[k - 1], kappa[k], dz1, dz2)

            dz1     = zeta_ac[k + 1] - zeta_aa[k]
            dz2     = zeta_aa[k + 1] - zeta_ac[k + 1]
            kappa_b = _calc_wtd_harmonic_mean(kappa[k], kappa[k + 1], dz1, dz2)

            fac_a = -kappa_a * dzeta_a[k] * dt / H2
            fac_b = -kappa_b * dzeta_b[k] * dt / H2

            uz_aa = 0.5 * (uz[k] + uz[k + 1])

            # Centered-difference vertical advection (Fortran's
            # default `if (.TRUE.)` branch).
            dzeta = zeta_aa[k + 1] - zeta_aa[k - 1]
            dz    = thickness * dzeta

            subd[k] = fac_a - uz_aa * dt / dz
            supd[k] = fac_b + uz_aa * dt / dz
            diag[k] = 1.0 - fac_a - fac_b
            rhs[k]  = (temp[k] - T_ref) - dt * advecxy[k] + dt * Q_strn[k]
        end

        # -- Surface BC --
        if !is_surf_flux
            # Dirichlet: `val_srf` lives at z=1, `temp[nz_aa]` at the last
            # interior centre. Symmetric to the basal case; kappa_surf is
            # the diffusivity at the z=1 face.
            dz1     = zeta_ac[nz_aa]   - zeta_aa[nz_aa - 1]
            dz2     = zeta_aa[nz_aa]   - zeta_ac[nz_aa]
            kappa_a = _calc_wtd_harmonic_mean(kappa[nz_aa - 1], kappa[nz_aa],
                                              dz1, dz2)
            kappa_n = kappa_surf

            fac_a = -kappa_a * dzeta_a[nz_aa] * dt / H2
            fac_b = -kappa_n * dzeta_b[nz_aa] * dt / H2

            uz_aa     = 0.5 * (uz[nz_aa] + uz[nz_aa + 1])
            dzeta_adv = 1.0 - zeta_aa[nz_aa - 1]
            dz_adv    = thickness * dzeta_adv

            absorb = fac_b + uz_aa * dt / dz_adv

            subd[nz_aa] = fac_a - uz_aa * dt / dz_adv
            supd[nz_aa] = 0.0
            diag[nz_aa] = 1.0 - fac_a - fac_b
            rhs[nz_aa]  = (temp[nz_aa] - T_ref) - dt * advecxy[nz_aa] +
                          dt * Q_strn[nz_aa] - absorb * (val_srf - T_ref)
        else
            # Neumann: backward-Euler flux at the surface.
            dz             = thickness * (zeta_aa[nz_aa] - zeta_aa[nz_aa - 1])
            subd[nz_aa]    = -1.0
            diag[nz_aa]    =  1.0
            supd[nz_aa]    =  0.0
            rhs[nz_aa]     = val_srf * dz
        end

        # -- Tridiagonal solve --
        solve_tridiag!(solution, subd, diag, supd, rhs, cp_buf, dp_buf)

        # -- Shift back to absolute K --
        for k in 1:nz_aa
            temp[k] = solution[k] + T_ref
        end
    end
    return temp
end
