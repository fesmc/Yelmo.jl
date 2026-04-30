# ----------------------------------------------------------------------
# Level-set function (LSF) for flux-form calving.
#
# Convention: `lsf < 0` ⇒ ice domain, `lsf > 0` ⇒ ocean. The zero
# level set is the calving front.
#
# Public surface:
#
#   - `lsf_init!`        — initialise φ to ±1 from H_ice / z_bed / z_sl.
#   - `lsf_update!`      — advect φ at w = u_bar + cr using
#                          `advect_tracer!`, then saturate to [-1, 1].
#                          Velocity is extrapolated outside the ice
#                          along its natural axis so the upwind
#                          advection near the front sees a sensible w.
#   - `lsf_redistance!`  — Sussman/Osher Hamilton-Jacobi redistancing
#                          to restore |∇φ| ≈ 1 without moving the
#                          zero level set. Replaces the Fortran
#                          neighbour-based reset and the periodic
#                          ±1 re-flag.
#
# The Fortran reference is `yelmo/src/physics/calving/lsf_module.f90`
# and the embedded reset block in
# `yelmo_topography.f90:calc_ytopo_calving_lsf` (lines 786–817). The
# Yelmo.jl port intentionally drops the post-advection neighbour reset
# in favour of redistancing.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, location

export lsf_init!, lsf_update!, lsf_redistance!,
       extrapolate_ocn_acx!, extrapolate_ocn_acy!

# Clamped read for lsf — Neumann (zero-gradient) at domain edges. Using
# Dirichlet-zero as in `_aa_or_zero` would inject spurious zero level
# sets at the boundary; for redistancing of a near-saturated field, a
# zero-gradient extension is the right boundary condition.
@inline function _clamp_read(F::AbstractMatrix, i::Int, j::Int,
                             nx::Int, ny::Int)
    ii = clamp(i, 1, nx)
    jj = clamp(j, 1, ny)
    return F[ii, jj]
end

# Resolve a Field or a bare 3-D array view down to a 2-D matrix. Lets
# callers hand us either an Oceananigans Field or `interior(field)` /
# `copy(interior(field))`; the kernels below only need 2-D access.
_view2d(field) = view(interior(field), :, :, 1)
_view2d(arr::AbstractArray{<:Any, 3}) = view(arr, :, :, 1)
_view2d(arr::AbstractMatrix) = arr

"""
    lsf_init!(lsf, H_ice, z_bed, z_sl) -> lsf

Initialise `lsf = -1` where ice is present (`H_ice > 0`) or where the
bed sits above sea level (a permanent land cell that the level set
must never invade). Elsewhere `lsf = +1` (open ocean).

Port of `LSFinit` in `lsf_module.f90`.
"""
function lsf_init!(lsf, H_ice, z_bed, z_sl)
    L  = interior(lsf)
    H  = interior(H_ice)
    Zb = interior(z_bed)
    Zs = interior(z_sl)
    @inbounds for j in axes(L, 2), i in axes(L, 1)
        if H[i, j, 1] > 0.0 || Zb[i, j, 1] > Zs[i, j, 1]
            L[i, j, 1] = -1.0
        else
            L[i, j, 1] = 1.0
        end
    end
    return lsf
end

# Single-pass forward+backward sweep along x. Cells with `is_ocean` get
# the value of their nearest filled x-neighbour. O(nx·ny) total work
# instead of the iterated `do while` used in Fortran's
# `extrapolate_ocn_acx`. A row with no filled cells is left untouched.
"""
    extrapolate_ocn_acx!(w_acx; reference=w_acx) -> w_acx

Fill in-place the ocean cells of an XFaceField `w_acx` along the
x-axis by nearest-filled-neighbour sweep. A face is treated as
"ocean" if the corresponding face of `reference` is exactly zero;
this matches Fortran's convention of using the raw ice velocity
field as the ice/ocean mask for the velocity extrapolation.

Used inside `lsf_update!` to propagate the calving-front velocity
`w = u_bar + cr` outward into ocean cells where it would otherwise
be zero, so upwind advection of `lsf` near the front sees a
non-zero front velocity.
"""
function extrapolate_ocn_acx!(w_acx; reference = w_acx)
    W = _view2d(w_acx)
    R = _view2d(reference)
    nx, ny = size(W)
    filled = R .!= 0.0    # local mask; never mutate the reference field

    @inbounds for j in 1:ny
        # Forward sweep: rightward fill.
        for i in 2:nx
            if !filled[i, j] && filled[i - 1, j]
                W[i, j] = W[i - 1, j]
                filled[i, j] = true
            end
        end
        # Backward sweep: leftward fill (catches any unfilled left tail).
        for i in (nx - 1):-1:1
            if !filled[i, j] && filled[i + 1, j]
                W[i, j] = W[i + 1, j]
                filled[i, j] = true
            end
        end
    end
    return w_acx
end

"""
    extrapolate_ocn_acy!(w_acy; reference=w_acy) -> w_acy

Same as [`extrapolate_ocn_acx!`](@ref) but along the y-axis for a
YFaceField.
"""
function extrapolate_ocn_acy!(w_acy; reference = w_acy)
    W = _view2d(w_acy)
    R = _view2d(reference)
    nx, ny = size(W)
    filled = R .!= 0.0

    @inbounds for i in 1:nx
        for j in 2:ny
            if !filled[i, j] && filled[i, j - 1]
                W[i, j] = W[i, j - 1]
                filled[i, j] = true
            end
        end
        for j in (ny - 1):-1:1
            if !filled[i, j] && filled[i, j + 1]
                W[i, j] = W[i, j + 1]
                filled[i, j] = true
            end
        end
    end
    return w_acy
end

"""
    lsf_update!(lsf, u_bar, v_bar, cr_acx, cr_acy, dt;
                cfl_safety=0.1) -> lsf

Advance `lsf` by `dt` years. Builds the calving-front velocity
`w = u_bar + cr`, extrapolates it into the ocean along its natural
axis, advects `lsf` at `w` via `advect_tracer!`, and saturates the
result to `[-1, 1]`. The two scratch face-fields for `w` are
allocated internally (cheap; one pair per call).

Port of `LSFupdate` in `lsf_module.f90`. Differences from Fortran:

  - Uses `advect_tracer!` (the same upwind kernel used for `H_ice`)
    instead of the separate `calc_advec2D` path.
  - The neighbour-based "if all four neighbours share sign, snap to
    ±1" reset (Fortran lines 808–818) is removed; redistancing
    handles drift instead.
"""
function lsf_update!(lsf,
                     u_bar, v_bar,
                     cr_acx, cr_acy,
                     dt::Real;
                     cfl_safety::Real = 0.1)
    grid = lsf.grid
    w_acx = XFaceField(grid)
    w_acy = YFaceField(grid)

    Wx = interior(w_acx)
    Wy = interior(w_acy)
    Ux = interior(u_bar)
    Uy = interior(v_bar)
    Cx = interior(cr_acx)
    Cy = interior(cr_acy)

    @inbounds for k in eachindex(Wx)
        Wx[k] = Ux[k] + Cx[k]
    end
    @inbounds for k in eachindex(Wy)
        Wy[k] = Uy[k] + Cy[k]
    end

    # Extrapolate w outward into the ocean using the raw ice velocity
    # as the ice/ocean mask (zero ⇒ ocean).
    extrapolate_ocn_acx!(w_acx; reference = u_bar)
    extrapolate_ocn_acy!(w_acy; reference = v_bar)

    advect_tracer!(lsf, w_acx, w_acy, dt; cfl_safety = cfl_safety)

    # Saturate to [-1, 1] to keep the field bounded under upwind
    # diffusion. Redistancing is what restores the slope; this is just
    # a guardrail.
    L = interior(lsf)
    @inbounds for k in eachindex(L)
        if L[k] >  1.0
            L[k] =  1.0
        elseif L[k] < -1.0
            L[k] = -1.0
        end
    end
    return lsf
end

"""
    lsf_redistance!(lsf, dx, dy; n_iter=5) -> lsf

Restore `|∇lsf| ≈ 1` near the zero level set without moving it. Uses
the Sussman/Osher Hamilton-Jacobi redistancing PDE

    ∂φ/∂τ + sgn(φ₀) (|∇φ| − 1) = 0

discretised with the Godunov upwind scheme for `|∇φ|` and a smoothed
sign function `sgn(φ₀) = φ₀ / √(φ₀² + ε²)` with `ε = max(dx, dy)`.
`φ₀` is the lsf at the start of redistancing, held fixed for the
duration of the iteration (this is what prevents the zero level set
from drifting).

Pseudo-timestep is `dτ = 0.5 · min(dx, dy)`, satisfying the CFL
limit for the explicit Godunov scheme.

Boundary condition: zero-gradient (Neumann), so out-of-domain reads
copy the edge value rather than introducing a spurious zero level
set at the boundary.

Replaces the Fortran `dt_lsf` periodic re-flag (`lsf > 0 → 1`,
`lsf ≤ 0 → -1`) and the post-advection neighbour-snap. `n_iter`
defaults to 5; near-saturated input fields converge well within
that.
"""
function lsf_redistance!(lsf, dx::Real, dy::Real; n_iter::Int = 5)
    n_iter >= 0 || error("lsf_redistance!: n_iter must be ≥ 0 (got $n_iter).")
    L = view(interior(lsf), :, :, 1)
    nx, ny = size(L)

    phi0 = copy(L)
    eps  = max(dx, dy)
    dtau = 0.5 * min(dx, dy)
    new_phi = similar(L)

    for _ in 1:n_iter
        @inbounds for j in 1:ny, i in 1:nx
            phi = L[i, j]
            a = (phi - _clamp_read(L, i - 1, j, nx, ny)) / dx
            b = (_clamp_read(L, i + 1, j, nx, ny) - phi) / dx
            c = (phi - _clamp_read(L, i, j - 1, nx, ny)) / dy
            d = (_clamp_read(L, i, j + 1, nx, ny) - phi) / dy

            s0  = phi0[i, j]
            sgn = s0 / sqrt(s0 * s0 + eps * eps)

            if sgn > 0
                gx2 = max(max(a, 0.0)^2, min(b, 0.0)^2)
                gy2 = max(max(c, 0.0)^2, min(d, 0.0)^2)
            else
                gx2 = max(min(a, 0.0)^2, max(b, 0.0)^2)
                gy2 = max(min(c, 0.0)^2, max(d, 0.0)^2)
            end
            grad = sqrt(gx2 + gy2)

            new_phi[i, j] = phi - dtau * sgn * (grad - 1.0)
        end
        L .= new_phi
    end
    return lsf
end
