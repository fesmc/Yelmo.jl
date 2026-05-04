# ----------------------------------------------------------------------
# 3D Glen-law viscosity (`calc_viscosity_glen!`) and depth-integrated
# viscosity (`calc_visc_int!`) for the `mat` component.
#
# Ports of `yelmo/src/physics/deformation.f90`:
#   - `calc_viscosity_glen` (line 174) — pure Glen-law per layer
#   - `calc_visc_int`       (line 321) — column trapezoidal integral
#
# Both produce diagnostic fields stored in `mat`. They are NOT used by
# the dyn solver chain — dyn has its own SSA-specific viscosity
# pipeline in `src/dyn/viscosity.jl` (`calc_visc_eff_3D_aa!`,
# `calc_visc_eff_int!`) that operates on velocity-derived strain
# rates. The mat versions consume the dyn-supplied symmetrised
# strain-rate tensor (`y.dyn.strn.de`) plus the mat rate factor
# (`y.mat.ATT`) to compute material-property viscosities.
#
# Boundary endpoint convention (deviation from Fortran):
#
# In Fortran, `visc(:,:,1:nz_aa)` is allocated on the full aa-grid
# where `zeta_aa(1) = 0` (bed) and `zeta_aa(nz_aa) = 1` (surface), so
# `integrate_trapezoid1D_pt` covers the full `[0, 1]` interval using
# the per-layer values directly.
#
# In Yelmo.jl under the Option C convention, 3D fields use Center
# vertical staggering — interior layer midpoints only, length `Nz`,
# excluding the bed and surface endpoints. Mat does NOT carry
# explicit `visc_b` / `visc_s` 2D boundary fields (unlike dyn's
# `visc_eff_b` / `visc_eff_s` which exist for SSA matrix accuracy),
# so the depth integral falls back to constant extrapolation: the
# bed value is taken as `visc[k=1]` and the surface as `visc[k=Nz]`.
# This is exact for uniform vertical viscosity and an O(Δz) error in
# general — acceptable for diagnostic-only output. Per-field
# benchmark tolerances cover the resulting deviation from the
# Fortran reference.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

# `vert_int_trapz_boundary!` is the same trapezoidal-with-explicit-
# boundary helper used by dyn's `calc_visc_eff_int!`. Lives in the
# top-level `YelmoIntegration` module (`src/integration.jl`) so both
# `mat` and `dyn` consume it as a peer.
using ..YelmoIntegration: vert_int_trapz_boundary!

export calc_viscosity_glen!, calc_visc_int!, depth_average!


"""
    calc_viscosity_glen!(visc, de, ATT, f_ice;
                         n_glen, visc_min, eps_0) -> visc

Glen-law viscosity per layer (Greve & Blatter 2009, Eq. 4.22):

    visc = 0.5 · ATT^(-1/n) · (sqrt(de² + eps_0²))^((1-n)/n)

clamped to `visc_min` from below. Sets `visc[i, j, k] = 0` on cells
with `f_ice != 1` (matches Fortran's ice-mask gate at line 222).

Port of `yelmo/src/physics/deformation.f90:174 calc_viscosity_glen`.

Inputs (all Center-staggered):
  - `visc`  (3D CenterField, output, Pa·yr)
  - `de`    (3D CenterField, second invariant of strain rate, 1/yr)
  - `ATT`   (3D CenterField, rate factor, Pa^-n · yr^-1)
  - `f_ice` (2D CenterField, fractional ice cover ∈ {0, partial, 1})

Keyword inputs (scalars):
  - `n_glen`   — Glen exponent (`y.p.ymat.n_glen`)
  - `visc_min` — viscosity floor in Pa·yr (`y.p.ymat.visc_min`)
  - `eps_0`    — strain-rate regularization in 1/yr (read from
                 `y.p.ydyn.eps_0` at the call site, matching Fortran)
"""
function calc_viscosity_glen!(visc, de, ATT, f_ice;
                              n_glen::Real, visc_min::Real, eps_0::Real)
    V  = interior(visc)
    De = interior(de)
    A  = interior(ATT)
    fi = interior(f_ice)

    Nx, Ny, Nz = size(V)
    @assert size(De) == size(V) "calc_viscosity_glen!: de size mismatch"
    @assert size(A)  == size(V) "calc_viscosity_glen!: ATT size mismatch"
    @assert size(fi, 1) == Nx && size(fi, 2) == Ny "calc_viscosity_glen!: f_ice size mismatch"

    exp1   = -1.0 / Float64(n_glen)
    exp2   = (1.0 - Float64(n_glen)) / Float64(n_glen)
    eps0sq = Float64(eps_0)^2
    vmin   = Float64(visc_min)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] == 1.0
            de_reg = sqrt(De[i, j, k]^2 + eps0sq)
            v = 0.5 * A[i, j, k]^exp1 * de_reg^exp2
            V[i, j, k] = v < vmin ? vmin : v
        else
            V[i, j, k] = 0.0
        end
    end
    return visc
end


"""
    calc_visc_int!(visc_int, visc, H_ice, f_ice, zeta_aa) -> visc_int

Depth-integrated viscosity in Pa·yr·m:

    visc_int(i, j) = ∫₀¹ visc(i, j, ζ) dζ · H_ice(i, j)    if f_ice == 1
                   = 0                                       otherwise

Port of `yelmo/src/physics/deformation.f90:321 calc_visc_int`.

Boundary endpoint values for the integration come from constant
extrapolation off the nearest interior layer (`visc[k=1]` for the
bed, `visc[k=Nz]` for the surface) — see file-header comment for the
rationale.

Boundary halo treatment from the Fortran routine
(`periodic` / `periodic-x` / `infinite` blocks at lines 362-383) is
not replicated here. Oceananigans handles topology via field
staggering, and downstream callers refresh halos via
`fill_halo_regions!` as needed.

Inputs:
  - `visc_int` (2D CenterField, output, Pa·yr·m)
  - `visc`     (3D CenterField, Pa·yr)
  - `H_ice`    (2D CenterField, m)
  - `f_ice`    (2D CenterField)
  - `zeta_aa`  (Vector, length Nz, dimensionless interior-Center ζ)
"""
function calc_visc_int!(visc_int, visc, H_ice, f_ice,
                        zeta_aa::AbstractVector{<:Real})
    Vi = interior(visc_int)
    V  = interior(visc)
    H  = interior(H_ice)
    fi = interior(f_ice)

    Nx = size(Vi, 1)
    Ny = size(Vi, 2)
    Nz = size(V, 3)
    Nz == length(zeta_aa) || error(
        "calc_visc_int!: visc has Nz=$(Nz) but zeta_aa has length $(length(zeta_aa))")

    # Constant-extrapolation bed/surface views — see file-header comment.
    Vb = @view V[:, :, 1:1]
    Vs = @view V[:, :, Nz:Nz]
    vert_int_trapz_boundary!(Vi, V, Vb, Vs, zeta_aa)

    @inbounds for j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] == 1.0
            Vi[i, j, 1] = Vi[i, j, 1] * H[i, j, 1]
        else
            Vi[i, j, 1] = 0.0
        end
    end
    return visc_int
end


"""
    depth_average!(out2D, var3D, zeta_aa) -> out2D

Pure depth-average `∫₀¹ var(ζ) dζ` of a 3D Center-staggered field
into a 2D Center-staggered field. Uses the same constant-extrapolation
boundary convention as `calc_visc_int!` (the bed value is taken as
`var3D[k=1]` and the surface as `var3D[k=Nz]`).

Used by `mat_step!` to populate `mat.enh_bar`, `mat.ATT_bar`, and
`mat.visc_bar` from their respective 3D fields. Fortran analog:
`calc_vertical_integrated_2D` from `yelmo/src/yelmo_tools.f90`.
"""
function depth_average!(out2D, var3D, zeta_aa::AbstractVector{<:Real})
    Vi = interior(out2D)
    V  = interior(var3D)
    Nz = size(V, 3)
    Nz == length(zeta_aa) || error(
        "depth_average!: var3D has Nz=$(Nz) but zeta_aa has length $(length(zeta_aa))")

    Vb = @view V[:, :, 1:1]
    Vs = @view V[:, :, Nz:Nz]
    vert_int_trapz_boundary!(Vi, V, Vb, Vs, zeta_aa)
    return out2D
end
