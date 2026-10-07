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
using Oceananigans.Grids: topology, Bounded, Periodic

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
                                       zero_outside::Bool,
                                       Nx::Int,
                                       ::Type{Tx},
                                       periodic_offset::Float64) where {Tx}
    V0 = var[i,   j, k]
    V1 = var[i+1, j, k]
    f0 = f_ice[i,   j, 1]
    f1 = f_ice[i+1, j, 1]

    # Periodic-wrap offset: at the wrap face (i = Nx under Periodic-x),
    # the halo read `var[Nx+1] == var[1]` is the raw periodic image.
    # For benchmark fields that are *additively non-periodic* with a
    # known wrap step (e.g. HOM-C `z_srf = -x · tan α`), shift the halo
    # value by the configured offset so the FD recovers the true slope.
    # Under Bounded, the offset is meaningless (no wrap face) and is
    # silently ignored — the FD then sees the Neumann-clamped halo.
    if Tx === Periodic && i == Nx && periodic_offset != 0.0
        V1 += periodic_offset
    end

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
                                       zero_outside::Bool,
                                       Ny::Int,
                                       ::Type{Ty},
                                       periodic_offset::Float64) where {Ty}
    V0 = var[i, j,   k]
    V1 = var[i, j+1, k]
    f0 = f_ice[i, j,   1]
    f1 = f_ice[i, j+1, 1]

    # Periodic-wrap offset on the y-axis: see `_gradient_acx_kernel`
    # docstring above. Under Bounded-y, the offset is silently ignored.
    if Ty === Periodic && j == Ny && periodic_offset != 0.0
        V1 += periodic_offset
    end

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
                       periodic_offset=0.0) -> dvardx

Compute the per-cell `∂var/∂x` on the acx-staggered diagnostic
`dvardx` (a Center field per the Yelmo schema, with the convention
that interior index `i` is the *east* face of aa-cell `i`).
`var` is Center-located; `f_ice` is the binary/fractional ice mask.

Modes:

  - `zero_outside=true`: aa-cells with `f_ice < 1` are treated as
    `var = 0` before differencing. Used for `dHidx`/`dHidy` so
    margin gradients reflect the actual ice/ocean step.
  - `grad_lim`: clamp `|dvardx|` to `≤ grad_lim` (matches Fortran's
    final `minmax` post-processing). Default `Inf` means no clamp.
  - `periodic_offset`: signed `Δvar` (in the units of `var`) added
    to the wrap-face halo read on a Periodic-x grid. Used for
    benchmark fields that are *additively non-periodic* with a known
    wrap step (e.g. HOM-C `z_srf = -x · tan α`, with
    `periodic_offset = -tan(α) · Lx_m`). Under Bounded-x the offset
    is silently ignored — the FD is already correct via the Neumann
    clamp. Default `0.0` preserves the legacy behaviour.

Halo handling: `var` and `f_ice` halos are filled via
`fill_halo_regions!`. Boundary behaviour at domain edges is then
driven by each field's BC.

Port of `yelmo_tools.f90 calc_gradient_acx`.
"""
function calc_gradient_acx!(dvardx, var, f_ice, dx::Real;
                            grad_lim::Real = Inf,
                            zero_outside::Bool = false,
                            periodic_offset::Real = 0.0)
    fill_halo_regions!(var)
    fill_halo_regions!(f_ice)

    Dx = interior(dvardx)
    dx_f = Float64(dx)
    cap  = Float64(grad_lim)
    off  = Float64(periodic_offset)
    Tx   = topology(var.grid, 1)
    Nx   = size(Dx, 1)

    @inbounds for k in axes(Dx, 3), j in axes(Dx, 2), i in axes(Dx, 1)
        g = _gradient_acx_kernel(i, j, k, var, f_ice,
                                  dx_f, zero_outside,
                                  Nx, Tx, off)
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
                       periodic_offset=0.0) -> dvardy

`∂var/∂y` on the acy-staggered diagnostic `dvardy` (Center field per
the Yelmo schema, interior index `j` ↔ *north* face of aa-cell `j`).
Same options as [`calc_gradient_acx!`](@ref); `periodic_offset` is
applied to the y-axis wrap face under Periodic-y. Port of
`yelmo_tools.f90 calc_gradient_acy`.
"""
function calc_gradient_acy!(dvardy, var, f_ice, dy::Real;
                            grad_lim::Real = Inf,
                            zero_outside::Bool = false,
                            periodic_offset::Real = 0.0)
    fill_halo_regions!(var)
    fill_halo_regions!(f_ice)

    Dy = interior(dvardy)
    dy_f = Float64(dy)
    cap  = Float64(grad_lim)
    off  = Float64(periodic_offset)
    Ty   = topology(var.grid, 2)
    Ny   = size(Dy, 2)

    @inbounds for k in axes(Dy, 3), j in axes(Dy, 2), i in axes(Dy, 1)
        g = _gradient_acy_kernel(i, j, k, var, f_ice,
                                  dy_f, zero_outside,
                                  Ny, Ty, off)
        if isfinite(cap)
            g = clamp(g, -cap, cap)
        end
        Dy[i, j, k] = g
    end
    return dvardy
end
