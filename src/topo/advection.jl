# ----------------------------------------------------------------------
# Generic 2D tracer advection on a regular RectilinearGrid.
#
# `advect_tracer!` advances a CenterField tracer `c` by `dt_outer`
# using Forward Euler in time and
# `Oceananigans.Advection.UpwindBiased(order=1)` in space (the
# `expl-upwind` solver). Internally CFL-aware: each call may take many
# sub-steps with `dt_sub ≤ cfl_safety * min(dx/|u|_max, dy/|v|_max)`
# until the requested outer `dt` is reached.
#
# Used in `topo_step!` for both ice thickness `H_ice` (advected at the
# depth-averaged ice velocity `(ux_bar, uy_bar)`) and the level-set
# function `lsf` (advected at `u_bar + cr` — see `lsf_update!`).
#
# `impl-lis` (implicit) lands as a separate later milestone; the
# function signature accepts `scheme` so an implicit solver can plug
# in without changing call sites.
# ----------------------------------------------------------------------

import Oceananigans
using Oceananigans.Advection: div_Uc
using Oceananigans.BoundaryConditions: fill_halo_regions!

export advect_tracer!

const _DEFAULT_UPWIND = UpwindBiased(order=1)

"""
    advect_tracer!(c, ux, uy, dt; scheme, cfl_safety, fill_velocity_halos)
        -> c

Advance a 2D cell-centred tracer `c` (CenterField) by `dt` years using
the given advection `scheme` (default first-order upwind). Velocities
`ux` (XFaceField) and `uy` (YFaceField) are held fixed during the
call; they should already be in their final values for this outer
step.

Used by `topo_step!` for ice thickness `H_ice` and by `calving_step!`
for the level-set function `lsf`.

Internally sub-steps with a CFL-safe `dt_sub` so the caller can pass
any outer `dt` without thinking about CFL. `cfl_safety` (default 0.1,
matching Yelmo Fortran's `ytopo.cfl_max`) is the Courant-number
fraction; lower values trade speed for stability headroom.

If `fill_velocity_halos = true` (default) the velocity halos are
refreshed once at the start of the call. Pass `false` if the caller
already filled them and the velocities won't change between calls.
"""
function advect_tracer!(c, ux, uy, dt::Real;
                        scheme = _DEFAULT_UPWIND,
                        cfl_safety::Real = 0.1,
                        fill_velocity_halos::Bool = true)
    grid = c.grid
    U = (u=ux, v=uy, w=Oceananigans.Fields.ZeroField())

    if fill_velocity_halos
        fill_halo_regions!(ux)
        fill_halo_regions!(uy)
    end

    tend = similar(interior(c))
    elapsed = 0.0
    while elapsed < dt
        cfl_dt = _cfl_dt(grid, ux, uy, cfl_safety)
        dt_sub = min(dt - elapsed, cfl_dt)

        fill_halo_regions!(c)
        @inbounds for k in axes(tend, 3), j in axes(tend, 2), i in axes(tend, 1)
            tend[i, j, k] = -div_Uc(i, j, k, grid, scheme, U, c)
        end
        interior(c) .+= dt_sub .* tend

        elapsed += dt_sub
    end

    fill_halo_regions!(c)
    return c
end

# Largest dt for which `cfl_safety * (|u|·dt/dx + |v|·dt/dy) ≤ 1` over
# the whole grid. Returns `Inf` if velocity is identically zero (then
# only the outer `dt` constrains the loop).
function _cfl_dt(grid, ux, uy, cfl_safety::Real)
    dx = _dx(grid)
    dy = _dy(grid)

    umax = maximum(abs, interior(ux))
    vmax = maximum(abs, interior(uy))

    inv_dt = umax / dx + vmax / dy
    inv_dt > 0 || return Inf
    return cfl_safety / inv_dt
end

# Cell sizes from a regularly-spaced RectilinearGrid. We don't yet
# support stretched grids in the advection kernel; flag explicitly.
function _dx(grid::RectilinearGrid)
    Δx = grid.Δxᶜᵃᵃ
    Δx isa Number || error("advect_tracer! requires a uniform x-spacing for now (got $(typeof(Δx))).")
    return abs(Δx)
end

function _dy(grid::RectilinearGrid)
    Δy = grid.Δyᵃᶜᵃ
    Δy isa Number || error("advect_tracer! requires a uniform y-spacing for now (got $(typeof(Δy))).")
    return abs(Δy)
end
