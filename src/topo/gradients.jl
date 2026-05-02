# ----------------------------------------------------------------------
# Margin-aware horizontal gradients on staggered ac-faces.
#
#   - `calc_gradient_acx!` — `∂var/∂x` on an XFaceField, with
#     optional `zero_outside` (clip ice-free aa-cells to 0) and
#     `margin2nd` (Saito et al. 2007 second-order one-sided
#     difference at ice/ocean margins) modes.
#   - `calc_gradient_acy!` — `∂var/∂y` on a YFaceField, same modes.
#
# Both are written in the per-cell `(i, j, k, grid, args...)` shape so
# they can be lifted into `KernelFunctionOperation` at GPU-portability
# time without changing the body. For now they're driven from a plain
# CPU loop over interior face indices, with halos refreshed once per
# call. Boundary handling comes from the input fields' grid topology
# + BCs (Neumann clamp by default; periodic wrap on Periodic axes).
#
# Port of `yelmo/src/yelmo_tools.f90:723 calc_gradient_acx` and
# `:829 calc_gradient_acy`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, Field
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: topology, Bounded, Periodic

using ..YelmoCore: fill_corner_halos!

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
# drops cleanly to 0 at the margin. `margin2nd` switches to a
# 2nd-order upwind difference at ice/ocean margins (Saito et al. 2007),
# which reduces margin-thickness bias in the SIA driving stress.
@inline function _gradient_acx_kernel(i::Int, j::Int, k::Int,
                                       var, f_ice,
                                       dx::Float64,
                                       margin2nd::Bool,
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

    if margin2nd
        # Ice on left (centre cell), ice-free on right: 2nd-order
        # upstream stencil reaching into (i-1, j) on the ice side.
        if f0 == 1.0 && f1 < 1.0
            f_far = f_ice[i-1, j, 1]
            if f_far == 1.0
                Va = var[i+1, j, k]; zero_outside && f1 < 1.0 && (Va = 0.0)
                Vb = var[i,   j, k]
                Vc = var[i-1, j, k]
                grad = (Vc - 4.0*Vb + 3.0*Va) / dx
            else
                grad = 0.0
            end
        elseif f0 < 1.0 && f1 == 1.0
            # Ice on right, ice-free on left: 2nd-order stencil
            # reaching into (i+2, j).
            f_far = f_ice[i+2, j, 1]
            if f_far == 1.0
                Va = var[i,   j, k]; zero_outside && f0 < 1.0 && (Va = 0.0)
                Vb = var[i+1, j, k]
                Vc = var[i+2, j, k]
                grad = -(Vc - 4.0*Vb + 3.0*Va) / dx
            else
                grad = 0.0
            end
        end
    end

    return grad
end

@inline function _gradient_acy_kernel(i::Int, j::Int, k::Int,
                                       var, f_ice,
                                       dy::Float64,
                                       margin2nd::Bool,
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

    if margin2nd
        if f0 == 1.0 && f1 < 1.0
            f_far = f_ice[i, j-1, 1]
            if f_far == 1.0
                Va = var[i, j+1, k]; zero_outside && f1 < 1.0 && (Va = 0.0)
                Vb = var[i, j,   k]
                Vc = var[i, j-1, k]
                grad = (Vc - 4.0*Vb + 3.0*Va) / dy
            else
                grad = 0.0
            end
        elseif f0 < 1.0 && f1 == 1.0
            f_far = f_ice[i, j+2, 1]
            if f_far == 1.0
                Va = var[i, j,   k]; zero_outside && f0 < 1.0 && (Va = 0.0)
                Vb = var[i, j+1, k]
                Vc = var[i, j+2, k]
                grad = -(Vc - 4.0*Vb + 3.0*Va) / dy
            else
                grad = 0.0
            end
        end
    end

    return grad
end

"""
    calc_gradient_acx!(dvardx, var, f_ice, dx;
                       grad_lim=Inf, margin2nd=false,
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
  - `margin2nd=true`: at ice/ice-free margin faces, use a 2nd-order
    one-sided upwind difference (Saito et al. 2007) when the
    upstream cell is fully covered. Falls back to 0 otherwise.
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
`fill_halo_regions!`; corner halos are filled when `margin2nd=true`
since the 2nd-order stencil reaches into diagonals. Boundary
behaviour at domain edges is then driven by each field's BC.

Port of `yelmo_tools.f90:723 calc_gradient_acx`.
"""
function calc_gradient_acx!(dvardx, var, f_ice, dx::Real;
                            grad_lim::Real = Inf,
                            margin2nd::Bool = false,
                            zero_outside::Bool = false,
                            periodic_offset::Real = 0.0)
    if margin2nd && periodic_offset != 0
        error("calc_gradient_acx!: 2nd-order margin extrapolation " *
              "(`margin2nd=true`) with non-zero `periodic_offset` is " *
              "not yet supported. The 2nd-order stencil reaches one " *
              "cell further across the periodic wrap and needs a " *
              "wrap-aware reach-2 implementation; this will land when " *
              "a benchmark exercises both modes simultaneously.")
    end

    fill_halo_regions!(var)
    fill_halo_regions!(f_ice)
    if margin2nd
        # The 2nd-order kernel reads (i-2, j) / (i+1, j). For the deeper
        # halo cell (i-2 at i=1 → halo[-1]) we don't strictly need
        # corner halos, but periodicity on the y-axis makes diagonal
        # access reasonable to anticipate. Cheap and harmless.
        fill_corner_halos!(var)
        fill_corner_halos!(f_ice)
    end

    Dx = interior(dvardx)
    dx_f = Float64(dx)
    cap  = Float64(grad_lim)
    off  = Float64(periodic_offset)
    Tx   = topology(var.grid, 1)
    Nx   = size(Dx, 1)

    @inbounds for k in axes(Dx, 3), j in axes(Dx, 2), i in axes(Dx, 1)
        g = _gradient_acx_kernel(i, j, k, var, f_ice,
                                  dx_f, margin2nd, zero_outside,
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
                       grad_lim=Inf, margin2nd=false,
                       zero_outside=false,
                       periodic_offset=0.0) -> dvardy

`∂var/∂y` on the acy-staggered diagnostic `dvardy` (Center field per
the Yelmo schema, interior index `j` ↔ *north* face of aa-cell `j`).
Same options as [`calc_gradient_acx!`](@ref); `periodic_offset` is
applied to the y-axis wrap face under Periodic-y. Port of
`yelmo_tools.f90:829 calc_gradient_acy`.
"""
function calc_gradient_acy!(dvardy, var, f_ice, dy::Real;
                            grad_lim::Real = Inf,
                            margin2nd::Bool = false,
                            zero_outside::Bool = false,
                            periodic_offset::Real = 0.0)
    if margin2nd && periodic_offset != 0
        error("calc_gradient_acy!: 2nd-order margin extrapolation " *
              "(`margin2nd=true`) with non-zero `periodic_offset` is " *
              "not yet supported. The 2nd-order stencil reaches one " *
              "cell further across the periodic wrap and needs a " *
              "wrap-aware reach-2 implementation; this will land when " *
              "a benchmark exercises both modes simultaneously.")
    end

    fill_halo_regions!(var)
    fill_halo_regions!(f_ice)
    if margin2nd
        fill_corner_halos!(var)
        fill_corner_halos!(f_ice)
    end

    Dy = interior(dvardy)
    dy_f = Float64(dy)
    cap  = Float64(grad_lim)
    off  = Float64(periodic_offset)
    Ty   = topology(var.grid, 2)
    Ny   = size(Dy, 2)

    @inbounds for k in axes(Dy, 3), j in axes(Dy, 2), i in axes(Dy, 1)
        g = _gradient_acy_kernel(i, j, k, var, f_ice,
                                  dy_f, margin2nd, zero_outside,
                                  Ny, Ty, off)
        if isfinite(cap)
            g = clamp(g, -cap, cap)
        end
        Dy[i, j, k] = g
    end
    return dvardy
end
