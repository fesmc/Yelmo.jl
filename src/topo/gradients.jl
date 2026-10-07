# ----------------------------------------------------------------------
# Margin-aware horizontal gradients on staggered ac-faces.
#
#   - `calc_gradient_acx!` — `∂var/∂x` on an XFaceField, with an
#     optional `zero_outside` mode (clip ice-free aa-cells to 0).
#   - `calc_gradient_acy!` — `∂var/∂y` on a YFaceField, same modes.
#
# Both are written in the per-cell `(i, j, k, grid, args...)` shape so
# they can be lifted into `KernelFunctionOperation` at GPU-portability
# time without changing the body. For now they're driven from a plain
# CPU loop over interior face indices, with halos refreshed once per
# call. Boundary handling comes from the input fields' grid topology
# + BCs (Neumann clamp by default; periodic wrap on Periodic axes).
#
# Port of `yelmo/src/yelmo_tools.f90 calc_gradient_acx` and
# `calc_gradient_acy` (yelmo dev removed the `margin2nd` option).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, Field
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_gradient_acx!, calc_gradient_acy!

# Per-cell gradient kernel using the Yelmo CenterField storage
# convention for ac-staggered diagnostics: `dvardx[i, j]` is the
# *east* face of aa-cell (i, j), i.e. the face between aa-cells
# (i, j) and (i+1, j). The gradient there is
# `(var[i+1] - var[i]) / dx`. This matches the Fortran reference's
# face-indexing convention used by `calc_gradient_acx` in
# `yelmo_tools.f90` and the `dzsdx`/`dHidx`/`dzbdx` schema entries
# in `yelmo-variables-ytopo.md`, which are loaded as plain Center
# fields (no `_acx` suffix → no XFace allocation).
#
# `zero_outside` clips partially-covered (`f_ice < 1`) aa-cells to 0
# before differencing — used for `dHidx`/`dHidy` so the ice thickness
# drops cleanly to 0 at the margin.
@inline function _gradient_acx_kernel(i::Int, j::Int, k::Int,
                                       var, f_ice,
                                       dx::Float64,
                                       zero_outside::Bool)
    V0 = var[i,   j, k]
    V1 = var[i+1, j, k]
    f0 = f_ice[i,   j, 1]
    f1 = f_ice[i+1, j, 1]

    if zero_outside
        f0 < 1.0 && (V0 = 0.0)
        f1 < 1.0 && (V1 = 0.0)
    end

    grad = (V1 - V0) / dx

    return grad
end

@inline function _gradient_acy_kernel(i::Int, j::Int, k::Int,
                                       var, f_ice,
                                       dy::Float64,
                                       zero_outside::Bool)
    V0 = var[i, j,   k]
    V1 = var[i, j+1, k]
    f0 = f_ice[i, j,   1]
    f1 = f_ice[i, j+1, 1]

    if zero_outside
        f0 < 1.0 && (V0 = 0.0)
        f1 < 1.0 && (V1 = 0.0)
    end

    grad = (V1 - V0) / dy

    return grad
end

"""
    calc_gradient_acx!(dvardx, var, f_ice, dx;
                       grad_lim=Inf,
                       zero_outside=false,
                       slope_bg=0.0) -> dvardx

Compute the per-cell `∂var/∂x` on the acx-staggered diagnostic
`dvardx` (a Center field per the Yelmo schema, with the convention
that interior index `i` is the *east* face of aa-cell `i`).
`var` is Center-located; `f_ice` is the binary/fractional ice mask.

Modes:

  - `zero_outside=true`: aa-cells with `f_ice < 1` are treated as
    `var = 0` before differencing. Used for `dHidx`/`dHidy` so
    margin gradients reflect the actual ice/ocean step.
  - `slope_bg`: uniform background slope (m/m) not contained in
    `var`, added to every gradient before the `grad_lim` clamp (so
    the limit bounds the total slope). For periodic domains whose
    geometry is tilted (`ytopo.slope_bg_x/y`): `var` holds only the
    periodic part.
  - `grad_lim`: clamp `|dvardx|` to `≤ grad_lim` (matches Fortran's
    final `minmax` post-processing). Default `Inf` means no clamp.

Halo handling: `var` and `f_ice` halos are filled via
`fill_halo_regions!`. Boundary behaviour at domain edges is then
driven by each field's BC.

Port of `yelmo_tools.f90 calc_gradient_acx`.
"""
function calc_gradient_acx!(dvardx, var, f_ice, dx::Real;
                            grad_lim::Real = Inf,
                            zero_outside::Bool = false,
                            slope_bg::Real = 0.0)
    fill_halo_regions!(var)
    fill_halo_regions!(f_ice)

    Dx = interior(dvardx)
    dx_f = Float64(dx)
    cap  = Float64(grad_lim)
    sbg  = Float64(slope_bg)

    @inbounds for k in axes(Dx, 3), j in axes(Dx, 2), i in axes(Dx, 1)
        g = _gradient_acx_kernel(i, j, k, var, f_ice,
                                  dx_f, zero_outside) + sbg
        if isfinite(cap)
            g = clamp(g, -cap, cap)
        end
        Dx[i, j, k] = g
    end
    return dvardx
end

"""
    calc_gradient_acy!(dvardy, var, f_ice, dy;
                       grad_lim=Inf,
                       zero_outside=false,
                       slope_bg=0.0) -> dvardy

`∂var/∂y` on the acy-staggered diagnostic `dvardy` (Center field per
the Yelmo schema, interior index `j` ↔ *north* face of aa-cell `j`).
Same options as [`calc_gradient_acx!`](@ref). Port of
`yelmo_tools.f90 calc_gradient_acy`.
"""
function calc_gradient_acy!(dvardy, var, f_ice, dy::Real;
                            grad_lim::Real = Inf,
                            zero_outside::Bool = false,
                            slope_bg::Real = 0.0)
    fill_halo_regions!(var)
    fill_halo_regions!(f_ice)

    Dy = interior(dvardy)
    dy_f = Float64(dy)
    cap  = Float64(grad_lim)
    sbg  = Float64(slope_bg)

    @inbounds for k in axes(Dy, 3), j in axes(Dy, 2), i in axes(Dy, 1)
        g = _gradient_acy_kernel(i, j, k, var, f_ice,
                                  dy_f, zero_outside) + sbg
        if isfinite(cap)
            g = clamp(g, -cap, cap)
        end
        Dy[i, j, k] = g
    end
    return dvardy
end
