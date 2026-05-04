# ----------------------------------------------------------------------
# Shared depth-integration utilities for Yelmo.jl. Lives at the top
# level (not inside any sub-module) so that both `dyn` and `mat` (and
# any future consumer) can depend on it without an inter-module
# layering smell. Mirrors the `YelmoSolvers` pattern in `dyn/solvers.jl`.
#
#   - `vert_int_trapz_boundary!` — trapezoidal ∫₀¹ var(ζ) dζ over a 3D
#     field that is `Center()`-staggered in the vertical, with explicit
#     2D bed (zeta = 0) and surface (zeta = 1) boundary values. Used by
#     both the SIA wrapper (depth-average velocity) and the SSA viscosity
#     pipeline (`calc_visc_eff_int!`), and by the upcoming `mat` module.
#
# The helper is unit-agnostic: it returns the pure depth integral over
# zeta (no thickness multiplication, no floor). Callers that need
# Pa·yr·m units (depth-integrated viscosity for the SSA matrix) multiply
# by `H_ice` themselves, and apply any `visc_min` floor downstream.
#
# The Center-staggered layers do NOT include the boundary endpoints
# under the Option C convention, so the integration scheme MUST consume
# the bed and surface boundary values explicitly to cover the full
# [0, 1] interval. Otherwise the `[0, zeta_c[1]]` and `[zeta_c[Nz], 1]`
# end-strips are silently dropped (e.g. for Nz = 4 with uniform spacing,
# the integrator covers only `(Nz - 1) / Nz = 0.75` of the column).
# ----------------------------------------------------------------------

module YelmoIntegration

export vert_int_trapz_boundary!

# Local copy of the underflow tolerance — same value as
# `YelmoModelDyn`'s `TOL_UNDERFLOW` (`src/dyn/diagnostics.jl`). Mirrors
# the existing local-const pattern in `velocity_uz.jl` /
# `velocity_ssa.jl`.
const TOL_UNDERFLOW = 1e-15

"""
    vert_int_trapz_boundary!(out2D, var3D, var_bed, var_surf, zeta_c)
        -> out2D

Trapezoidal depth-integral of `var3D(zeta)` over `zeta ∈ [0, 1]`,
written to `out2D[:, :, 1]`. The 3D input lives on Oceananigans
`Center()` z-stagger (interior layer midpoints, length `Nz`) and the
boundary endpoints `var_bed[:, :, 1]` (zeta = 0) / `var_surf[:, :, 1]`
(zeta = 1) are supplied as 2D fields by the caller.

Segments (Nz interior centres → Nz + 1 trapezoidal segments):

  - bed:        `[0, zeta_c[1]]`        → `½ (var_bed + var3D[k=1]) · zeta_c[1]`
  - k = 2..Nz:  `[zeta_c[k-1], zeta_c[k]]` → `½ (var3D[k-1] + var3D[k]) · Δzeta_k`
  - surface:    `[zeta_c[Nz], 1]`       → `½ (var3D[k=Nz] + var_surf) · (1 - zeta_c[Nz])`

`TOL_UNDERFLOW` clip is applied to each segment midpoint, mirroring
Fortran's `integrate_trapezoid1D_pt`.

The helper iterates over the leading 2D dims of `out2D` / `var3D`. The
caller is responsible for any face-staggered offset (e.g. for an
`XFaceField`, pass shifted views of `interior(...)` so that the loop
indexing lines up with the face-cell convention). The helper itself
writes only `out2D[:, :, 1]`.

This is a pure depth-integral over zeta (dimensionless): the output
units equal the input units. Callers that need a depth-integrated
quantity in physical units multiply by `H_ice` themselves (e.g. SSA's
`calc_visc_eff_int!`).
"""
@inline function vert_int_trapz_boundary!(
        out2D::AbstractArray, var3D::AbstractArray,
        var_bed::AbstractArray, var_surf::AbstractArray,
        zeta_c::AbstractVector{<:Real})
    Nx = size(var3D, 1)
    Ny = size(var3D, 2)
    Nz = size(var3D, 3)
    @assert length(zeta_c) == Nz
    @assert size(out2D, 1) == Nx && size(out2D, 2) == Ny
    @assert size(var_bed,  1) == Nx && size(var_bed,  2) == Ny
    @assert size(var_surf, 1) == Nx && size(var_surf, 2) == Ny

    @inbounds for j in 1:Ny, i in 1:Nx
        # Bed segment: zeta in [0, zeta_c[1]], midpoint =
        # 0.5 * (var_bed + var3D[k=1]).
        v_mid = 0.5 * (var_bed[i, j, 1] + var3D[i, j, 1])
        abs(v_mid) < TOL_UNDERFLOW && (v_mid = 0.0)
        acc = v_mid * (Float64(zeta_c[1]) - 0.0)

        # Interior segments k = 2 ... Nz between consecutive Centers.
        for k in 2:Nz
            v_mid = 0.5 * (var3D[i, j, k-1] + var3D[i, j, k])
            abs(v_mid) < TOL_UNDERFLOW && (v_mid = 0.0)
            acc += v_mid * (Float64(zeta_c[k]) - Float64(zeta_c[k-1]))
        end

        # Surface segment: zeta in [zeta_c[Nz], 1], midpoint =
        # 0.5 * (var3D[k=Nz] + var_surf).
        v_mid = 0.5 * (var3D[i, j, Nz] + var_surf[i, j, 1])
        abs(v_mid) < TOL_UNDERFLOW && (v_mid = 0.0)
        acc += v_mid * (1.0 - Float64(zeta_c[Nz]))

        out2D[i, j, 1] = acc
    end
    return out2D
end

end # module YelmoIntegration
