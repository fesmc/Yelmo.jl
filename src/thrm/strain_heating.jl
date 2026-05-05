# ----------------------------------------------------------------------
# Strain heating sources for the implicit-diffusion / analytic
# temperature solvers.
#
# Direct ports of:
#
#   - `calc_strain_heating`     (`physics/thermodynamics.f90:465`)
#       General 3D form `Q_strn = 4 * visc * de^2`, where `de` is the
#       effective strain rate (from `dyn.strn.de`) and `visc` is the
#       3D Glen-law viscosity (from `mat.visc`). Greve & Blatter
#       (2009) Eqs. 4.7 + 5.65; Cuffey & Patterson (2010) Eq. 9.30.
#
#   - `calc_strain_heating_sia` (`physics/thermodynamics.f90:516`)
#       SIA approximation `Q_strn = -rho*g*depth*(duxdz*dzsdx +
#       duydz*dzsdy)`, evaluated on aa-nodes by averaging the
#       acx/acy-staggered velocities from the four surrounding
#       ac-stagger cells. Skips i=1 and j=1 (Fortran loops i=2..nx,
#       j=2..ny) — the boundary cells inherit whatever Q_strn value
#       they had last step.
#
# Both routines blend a fresh `Q_strn` value with the previous-step
# value via Forward-Euler weights `(beta1, beta2) = (1.0, 0.0)` —
# Yelmo's only currently-supported timestepping for thrm. The Fortran
# `dt_beta` array is never overridden away from this default, so
# beta1 / beta2 are baked in here as constants and the Field-aware
# wrappers don't carry them.
# ----------------------------------------------------------------------

"""
    calc_strain_heating!(Q_strn_field, de_field, visc_field) -> Q_strn_field

General 3D strain heating

    Q_strn = 4 * visc * de^2

written into `Q_strn_field`. Negatives clamped to zero. Forward-Euler
weighting (beta1=1, beta2=0) — fully overwritten each call.
"""
function calc_strain_heating!(Q_strn_field, de_field, visc_field)
    Q_d  = Q_strn_field.data
    de_d = de_field.data
    v_d  = visc_field.data
    Nx   = Q_strn_field.grid.Nx
    Ny   = Q_strn_field.grid.Ny
    Nz   = Q_strn_field.grid.Nz
    return _calc_strain_heating_kernel!(Q_d, de_d, v_d, Nx, Ny, Nz)
end

function _calc_strain_heating_kernel!(Q, de, visc,
                                      Nx::Int, Ny::Int, Nz::Int)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        q = 4.0 * visc[i, j, k] * de[i, j, k]^2
        Q[i, j, k] = q < 0.0 ? 0.0 : q
    end
    return nothing
end

"""
    calc_strain_heating_sia!(Q_strn_field, ux_field, uy_field,
                             dzsdx_field, dzsdy_field, H_ice_field,
                             zeta_aa, zeta_ac, rho_ice, g) -> Q_strn_field

SIA-approximation strain heating

    Q_strn(i, j, k) = -rho*g*depth*(duxdz*dzsdx_aa + duydz*dzsdy_aa)

evaluated at aa-nodes. The horizontal velocities are 3D
acx/acy-staggered fields (`ux_field`, `uy_field`). Surface slopes are
2D acx/acy fields (`dzsdx_field`, `dzsdy_field`) staggered on the
opposite face from velocity in the corresponding direction. The
boundary cells `i = 1` and `j = 1` are skipped (Fortran:
`do j = 2, ny; do i = 2, nx`); their Q_strn keeps the previous-step
value via the (1, 0) weighting that overwrites only updated cells.

Internal indexing convention for aa-stagger:

  - Fortran `ux(i-1, j, k)` averages with `ux(i, j, k)` (and the
    `k+1`/`k-1` neighbours) to get the aa-node velocity at the
    "upper" / "lower" half-layer interfaces. Yelmo's XFaceField
    stores ac-x velocities at `ux_int[ip1, j, 1]` for the eastern
    face of cell (i, j), so Fortran `ux(i-1, j, k)` and `ux(i, j, k)`
    both get a `+1` shift on the x-axis.
  - Same pattern in y for `uy`.
"""
function calc_strain_heating_sia!(Q_strn_field, ux_field, uy_field,
                                  dzsdx_field, dzsdy_field,
                                  H_ice_field,
                                  zeta_aa::AbstractVector{<:Real},
                                  zeta_ac::AbstractVector{<:Real},
                                  rho_ice::Real, g::Real)
    Q_d   = Q_strn_field.data
    ux_d  = ux_field.data
    uy_d  = uy_field.data
    dx_d  = dzsdx_field.data
    dy_d  = dzsdy_field.data
    H_d   = H_ice_field.data
    Nx    = Q_strn_field.grid.Nx
    Ny    = Q_strn_field.grid.Ny
    Nz    = Q_strn_field.grid.Nz
    zaa   = collect(Float64, zeta_aa)
    zac   = collect(Float64, zeta_ac)
    return _calc_strain_heating_sia_kernel!(Q_d, ux_d, uy_d, dx_d, dy_d, H_d,
                                            zaa, zac,
                                            Float64(rho_ice), Float64(g),
                                            Nx, Ny, Nz)
end

function _calc_strain_heating_sia_kernel!(Q, ux, uy, dzsdx, dzsdy, H,
                                          zeta_aa::Vector{Float64},
                                          zeta_ac::Vector{Float64},
                                          rho_ice::Float64, g::Float64,
                                          Nx::Int, Ny::Int, Nz::Int)
    rho_g = rho_ice * g
    @inbounds for j in 2:Ny, i in 2:Nx
        H_ij = H[i, j, 1]
        if H_ij > 0.0
            for k in 2:(Nz - 1)
                # Yelmo XFaceField: Fortran `ux(k, j)` is the eastern
                # face of cell (k, j) → Yelmo index `ux[k+1, j, 1]`.
                # Fortran `ux(i-1, j, k)` → Yelmo `ux[i, j, k]`,
                # Fortran `ux(i,   j, k)` → Yelmo `ux[i+1, j, k]`.
                ux_aa_up  = 0.25 * (ux[i, j, k]     + ux[i+1, j, k]   +
                                    ux[i, j, k+1]   + ux[i+1, j, k+1])
                ux_aa_dwn = 0.25 * (ux[i, j, k]     + ux[i+1, j, k]   +
                                    ux[i, j, k-1]   + ux[i+1, j, k-1])

                uy_aa_up  = 0.25 * (uy[i, j, k]     + uy[i, j+1, k]   +
                                    uy[i, j, k+1]   + uy[i, j+1, k+1])
                uy_aa_dwn = 0.25 * (uy[i, j, k]     + uy[i, j+1, k]   +
                                    uy[i, j, k-1]   + uy[i, j+1, k-1])

                dz       = H_ij * (zeta_ac[k+1] - zeta_ac[k])
                duxdz    = (ux_aa_up - ux_aa_dwn) / dz
                duydz    = (uy_aa_up - uy_aa_dwn) / dz

                # Surface-slope acx/acy → aa stagger (avg of two
                # adjacent face values; same `+1` Yelmo shift as
                # velocity).
                dzsdx_aa = 0.5 * (dzsdx[i, j, 1] + dzsdx[i+1, j, 1])
                dzsdy_aa = 0.5 * (dzsdy[i, j, 1] + dzsdy[i, j+1, 1])

                depth = H_ij * (1.0 - zeta_aa[k])
                q     = (-rho_g * depth) * (duxdz * dzsdx_aa + duydz * dzsdy_aa)
                Q[i, j, k] = q < 0.0 ? 0.0 : q
            end
        else
            for k in 1:Nz
                Q[i, j, k] = 0.0
            end
        end
    end
    return nothing
end
