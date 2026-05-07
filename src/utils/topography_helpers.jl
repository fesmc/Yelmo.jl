# ----------------------------------------------------------------------
# Plain-array topography helpers used when loading reference / present-
# day datasets (e.g. `data_load!` in `src/data/YelmoModelData.jl`).
#
#   - `remove_englacial_lakes!`  — port of
#     `physics/topography.f90:1306 remove_englacial_lakes`. Diagnoses
#     where ice should be grounded but a sub-ice gap exists, and
#     adjusts `H_ice` so the basal surface sits on `z_bed`.
#   - `smooth_gauss_2D!`         — port of
#     `yelmo_tools.f90:1584 smooth_gauss_2D`. Masked Gaussian smoother:
#     smooths only `mask_apply` cells using only `mask_use` neighbours.
#     Replicate (clamped-edge) boundary conditions, kernel half-width
#     `3 * ceil(f_sigma)`, σ = `f_sigma * dx`.
#   - `adjust_topography_gradients!` — port of
#     `yelmo_tools.f90:1750 adjust_topography_gradients`. Iteratively
#     identifies cells where |∂z_bed/∂x| or |∂z_bed/∂y| exceeds
#     `grad_lim`, then Gaussian-smooths `z_bed` and `H_ice` over the
#     offending cells. Up to `iter_max` passes; exits early when no
#     cell is over the limit.
#
# These all operate on plain `AbstractMatrix{Float64}` rather than
# Oceananigans `Field`s — load-time helpers only.
# ----------------------------------------------------------------------

export remove_englacial_lakes!, smooth_gauss_2D!, adjust_topography_gradients!

"""
    remove_englacial_lakes!(H_ice, z_bed, z_srf, z_sl, rho_ice, rho_sw) -> H_ice

Adjust `H_ice` so the ice base sits on `z_bed` wherever a present-day
reference dataset exhibits an englacial / sub-ice lake artefact. Two
cases are handled:

  1. Cell is grounded (`H_grnd > 0`) but `z_srf - H_ice > z_bed` — a
     gap exists. Set `H_ice = z_srf - z_bed` to fill it.
  2. `z_srf - H_ice < z_bed` — ice is erroneously thick. Clip to
     `H_ice = z_srf - z_bed`.

Mirrors `physics/topography.f90:1306 remove_englacial_lakes`. Used at
present-day reference load (`data_load!`) and when initialising the
model state from observations.
"""
function remove_englacial_lakes!(H_ice::AbstractMatrix{<:AbstractFloat},
                                 z_bed::AbstractMatrix{<:AbstractFloat},
                                 z_srf::AbstractMatrix{<:AbstractFloat},
                                 z_sl::AbstractMatrix{<:AbstractFloat},
                                 rho_ice::Real, rho_sw::Real)
    nx, ny = size(H_ice)
    size(z_bed) == (nx, ny) || error("remove_englacial_lakes!: size(z_bed) != size(H_ice)")
    size(z_srf) == (nx, ny) || error("remove_englacial_lakes!: size(z_srf) != size(H_ice)")
    size(z_sl)  == (nx, ny) || error("remove_englacial_lakes!: size(z_sl) != size(H_ice)")

    rho_sw_ice = Float64(rho_sw) / Float64(rho_ice)

    @inbounds for j in 1:ny, i in 1:nx
        H_grnd_now = H_ice[i, j] - rho_sw_ice * max(z_sl[i, j] - z_bed[i, j], 0.0)
        gap        = z_srf[i, j] - H_ice[i, j] - z_bed[i, j]

        if H_grnd_now > 0.0 && gap > 0.0
            H_ice[i, j] = z_srf[i, j] - z_bed[i, j]
        elseif gap < 0.0
            H_ice[i, j] = z_srf[i, j] - z_bed[i, j]
        end
    end
    return H_ice
end

"""
    smooth_gauss_2D!(var, dx, f_sigma; mask_apply=nothing, mask_use=nothing) -> var

In-place 2D Gaussian smoothing with optional `mask_apply` (which cells
to smooth) and `mask_use` (which cells contribute to the kernel sum).
σ = `f_sigma * dx`; kernel half-width n2 = `3 * ceil(f_sigma)`;
replicate (clamped-edge) boundary conditions.

When `mask_apply` is `nothing`, every cell is smoothed. When `mask_use`
is `nothing`, it defaults to `mask_apply` (or all-true if neither is
supplied), matching the Fortran `smooth_gauss_2D` semantics.

Mirrors `yelmo_tools.f90:1584 smooth_gauss_2D`. Differs from
`gaussian_filter!` in `scrip_map.jl` only in the masked-neighbourhood
support — for unmasked smoothing, prefer `gaussian_filter!` for its
separable convolution.
"""
function smooth_gauss_2D!(var::AbstractMatrix{<:AbstractFloat},
                          dx::Real, f_sigma::Real;
                          mask_apply::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          mask_use::Union{Nothing,AbstractMatrix{Bool}}   = nothing)
    f_sigma >= 1.0 || error("smooth_gauss_2D!: f_sigma must be >= 1, got $f_sigma.")
    dx       > 0   || error("smooth_gauss_2D!: dx must be > 0, got $dx.")

    nx, ny = size(var)
    n2     = 3 * ceil(Int, f_sigma)
    sigma  = Float64(dx) * Float64(f_sigma)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    apply_all = mask_apply === nothing
    use_all   = mask_use   === nothing
    if !apply_all
        size(mask_apply) == (nx, ny) || error("smooth_gauss_2D!: size(mask_apply) != size(var)")
    end
    if !use_all
        size(mask_use) == (nx, ny) || error("smooth_gauss_2D!: size(mask_use) != size(var)")
    end

    # Build kernel weights `w[di+n2+1, dj+n2+1] = exp(-(di² + dj²) dx² / 2σ²)`.
    n      = 2 * n2 + 1
    kernel = Array{Float64}(undef, n, n)
    @inbounds for dj in -n2:n2, di in -n2:n2
        kernel[di + n2 + 1, dj + n2 + 1] = exp(-(di * di + dj * dj) * (Float64(dx) * Float64(dx)) * inv2s2)
    end

    # Snapshot input for read-only access during the convolution pass.
    var_old = copy(var)

    @inbounds for j in 1:ny, i in 1:nx
        smooth_here = apply_all || mask_apply[i, j]
        smooth_here || continue

        wsum = 0.0
        vsum = 0.0
        for dj in -n2:n2, di in -n2:n2
            ii = clamp(i + di, 1, nx)
            jj = clamp(j + dj, 1, ny)
            usable = use_all || mask_use[ii, jj]
            usable || continue
            w = kernel[di + n2 + 1, dj + n2 + 1]
            wsum += w
            vsum += w * Float64(var_old[ii, jj])
        end

        if wsum > 0.0
            var[i, j] = vsum / wsum
        end
    end

    return var
end

# Plain-array gradient on the east face of cell (i, j):
#   dvardx[i, j] = (var[i+1, j] - var[i, j]) / dx
# with `var[nx+1] = var[nx]` (Neumann clamp). Used internally by
# `adjust_topography_gradients!`.
@inline function _grad_x_clamp(var, i, j, nx, dx)
    ip1 = min(i + 1, nx)
    return (var[ip1, j] - var[i, j]) / dx
end
@inline function _grad_y_clamp(var, i, j, ny, dy)
    jp1 = min(j + 1, ny)
    return (var[i, jp1] - var[i, j]) / dy
end

"""
    adjust_topography_gradients!(z_bed, H_ice, grad_lim, dx, boundaries;
                                  iter_max=50, f_sigma=2.0) -> (z_bed, H_ice)

Iteratively smooth `z_bed` (and `H_ice` co-mutated to avoid spurious
patterns) at cells where `|∂z_bed/∂x|` or `|∂z_bed/∂y|` exceed
`grad_lim`. Each pass identifies offending cells (and the +1 face
neighbour), then calls `smooth_gauss_2D!` with `f_sigma`. Exits early
when no cell remains over the limit, or after `iter_max` passes.

`boundaries` is accepted for parity with the Fortran caller signature
but currently ignored — the gradient stencil clamps at edges (Neumann)
in all directions. The inner loop runs over `3:nx-3 × 3:ny-3` exactly
as the Fortran reference, so the choice of edge-handling rarely
matters in practice.

Mirrors `yelmo_tools.f90:1750 adjust_topography_gradients`.
"""
function adjust_topography_gradients!(z_bed::AbstractMatrix{<:AbstractFloat},
                                      H_ice::AbstractMatrix{<:AbstractFloat},
                                      grad_lim::Real, dx::Real,
                                      boundaries::AbstractString;
                                      iter_max::Int = 50,
                                      f_sigma::Real = 2.0)
    nx, ny = size(z_bed)
    size(H_ice) == (nx, ny) || error("adjust_topography_gradients!: size(H_ice) != size(z_bed)")
    grad_lim >= 0 || error("adjust_topography_gradients!: grad_lim must be >= 0.")
    dx        > 0 || error("adjust_topography_gradients!: dx must be > 0.")

    dy         = dx
    mask_apply = falses(nx, ny)

    for q in 1:iter_max
        fill!(mask_apply, false)

        @inbounds for j in 3:(ny - 3), i in 3:(nx - 3)
            gx = _grad_x_clamp(z_bed, i, j, nx, dx)
            gy = _grad_y_clamp(z_bed, i, j, ny, dy)
            if abs(gx) >= grad_lim
                mask_apply[i, j]                    = true
                mask_apply[min(i + 1, nx), j]       = true
            end
            if abs(gy) >= grad_lim
                mask_apply[i, j]                    = true
                mask_apply[i, min(j + 1, ny)]       = true
            end
        end

        n_apply = count(mask_apply)
        @info "adjust_topography_gradients!: pass $q  applied=$n_apply"
        n_apply == 0 && break

        smooth_gauss_2D!(z_bed, dx, f_sigma; mask_apply = mask_apply)
        smooth_gauss_2D!(H_ice, dx, f_sigma; mask_apply = mask_apply)
    end

    return (z_bed, H_ice)
end
