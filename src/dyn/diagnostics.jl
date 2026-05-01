# ----------------------------------------------------------------------
# Post-solver dyn diagnostics.
#
# This file collects the kernels that run after the velocity solver
# (currently a no-op for `solver = "fixed"`):
#
#   - `_clip_underflow!`       — drop sub-`TOL_UNDERFLOW` floats to zero.
#   - `calc_ice_flux!`         — `qq_acx/y = H_face · dx · u_bar` on
#                                 ac-staggered faces.
#   - `calc_magnitude_from_staggered!` — `√(u² + v²)` at aa-cell
#                                 centres using the symmetric face
#                                 average of an X/Y-Face pair, masked
#                                 to `f_ice == 1`. Operates on 2D and
#                                 3D fields uniformly (loops over `k`).
#   - `calc_vel_ratio!`        — `f_vbvs = min(1, u_b / u_s)` with the
#                                 zero-surface-velocity edge case.
#
# Surface / basal velocity slicing and `duxydt` time-difference are
# inline in `dyn_step!` since they're one-liners that are only ever
# done once per step.
#
# Port of the diagnostic block at `yelmo_dynamics.f90:212–294`,
# `velocity_general.f90:1727 calc_ice_flux`, `:1775 calc_vel_ratio`,
# and `yelmo_tools.f90:248 calc_magnitude_from_staggered`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.Grids: topology, Bounded, Periodic
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_ice_flux!, calc_magnitude_from_staggered!, calc_vel_ratio!

# Yelmo Fortran constants — `yelmo_defs.f90:44`.
const TOL           = 1e-5
const TOL_UNDERFLOW = 1e-15

# Drop sub-`TOL_UNDERFLOW` magnitudes to zero. Fortran uses
# `where (abs(x) .lt. TOL_UNDERFLOW) x = 0.0` for `ux/uy/ux_bar/uy_bar`
# right after the solver to keep tiny denormals out of downstream
# arithmetic. Operates on the field's interior view.
@inline function _clip_underflow!(field)
    int = interior(field)
    @. int = ifelse(abs(int) < TOL_UNDERFLOW, 0.0, int)
    return field
end

"""
    calc_ice_flux!(qq_acx, qq_acy, ux_bar, uy_bar, H_ice, dx, dy)
        -> (qq_acx, qq_acy)

Compute the ice flux on staggered ac-faces:

    qq_acx[i+1, j] = ½(H_ice[i, j] + H_ice[i+1, j]) · dx · ux_bar[i+1, j]
    qq_acy[i, j+1] = ½(H_ice[i, j] + H_ice[i, j+1]) · dy · uy_bar[i, j+1]

Units: m³ / yr (H · dx · u). Fortran loops `i = 1..nx-1` /
`j = 1..ny-1` and leaves the rightmost / northernmost face row at the
initial zero value; we mirror that by zeroing `qq_acx`/`qq_acy` first
and only writing on the `i = 1..Nx-1` / `j = 1..Ny-1` interior loop.
The leftmost/southernmost extra face slot is replicated for parity
with the YelmoMirror loader convention.

Port of `velocity_general.f90:1727 calc_ice_flux`.
"""
function calc_ice_flux!(qq_acx, qq_acy, ux_bar, uy_bar, H_ice,
                        dx::Real, dy::Real)
    fill_halo_regions!(H_ice)

    Qx = interior(qq_acx)
    Qy = interior(qq_acy)
    fill!(Qx, 0.0)
    fill!(Qy, 0.0)

    Nx = size(interior(H_ice), 1)
    Ny = size(interior(H_ice), 2)
    dx_f = Float64(dx)
    dy_f = Float64(dy)

    Tx_top = topology(qq_acx.grid, 1)
    Ty_top = topology(qq_acy.grid, 2)

    # Loop ranges 1..Nx-1 / 1..Ny-1 mirror the Fortran convention of
    # leaving the rightmost / northernmost face row at zero. The
    # `_ip1_modular` / `_jp1_modular` wraps don't trigger inside that
    # range (since i < Nx), so the index expression is identical to the
    # original `i+1` under both Bounded and Periodic.
    @inbounds for j in 1:Ny, i in 1:Nx-1
        ip1f = _ip1_modular(i, Nx, Tx_top)
        H_face = 0.5 * (H_ice[i, j, 1] + H_ice[i+1, j, 1]) * dx_f
        Qx[ip1f, j, 1] = H_face * ux_bar[ip1f, j, 1]
    end
    if Tx_top === Bounded
        @views Qx[1, :, :] .= Qx[2, :, :]
    end

    @inbounds for j in 1:Ny-1, i in 1:Nx
        jp1f = _jp1_modular(j, Ny, Ty_top)
        H_face = 0.5 * (H_ice[i, j, 1] + H_ice[i, j+1, 1]) * dy_f
        Qy[i, jp1f, 1] = H_face * uy_bar[i, jp1f, 1]
    end
    if Ty_top === Bounded
        @views Qy[:, 1, :] .= Qy[:, 2, :]
    end

    return qq_acx, qq_acy
end

"""
    calc_magnitude_from_staggered!(umag, u, v, f_ice) -> umag

Centred-cell magnitude of an ac-staggered vector field. At each
aa-cell `(i, j, k)`:

    u_centre = ½(u[i,   j, k] + u[i+1, j,   k])
    v_centre = ½(v[i,   j, k] + v[i,   j+1, k])
    umag     = √(u_centre² + v_centre²)

Cells with `f_ice[i, j, 1] != 1` are zeroed (matches the Fortran
"only fully-covered cells get a meaningful magnitude" convention).
Underflow clipping (`TOL_UNDERFLOW`) is applied to both face-averaged
components and to the final magnitude.

Operates on 2D fields (interior shape `(Nx, Ny, 1)`) and on 3D fields
(`(Nx, Ny, Nz)`) uniformly — the inner loop iterates over the umag
interior's third axis. `u`/`v` are the staggered XFace/YFace pair on
the same horizontal grid as `umag`.

Port of `yelmo_tools.f90:248 calc_magnitude_from_staggered`.
"""
function calc_magnitude_from_staggered!(umag, u, v, f_ice)
    fill_halo_regions!(u)
    fill_halo_regions!(v)

    M = interior(umag)
    Nx = size(M, 1)
    Ny = size(M, 2)
    Nz = size(M, 3)

    fill!(M, 0.0)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if f_ice[i, j, 1] == 1.0
            unow = 0.5 * (u[i,   j,   k] + u[i+1, j,   k])
            vnow = 0.5 * (v[i,   j,   k] + v[i,   j+1, k])
            abs(unow) < TOL_UNDERFLOW && (unow = 0.0)
            abs(vnow) < TOL_UNDERFLOW && (vnow = 0.0)
            mag = sqrt(unow * unow + vnow * vnow)
            M[i, j, k] = abs(mag) < TOL_UNDERFLOW ? 0.0 : mag
        end
    end
    return umag
end

"""
    calc_vel_ratio!(f_vbvs, uxy_b, uxy_s) -> f_vbvs

Per-cell basal-to-surface velocity ratio:

    f_vbvs = min(1, uxy_b / uxy_s)   if uxy_s > 0
             1                       otherwise

Fortran's `calc_vel_ratio` is `elemental`; Julia version is the
broadcast over the three Center fields' interior views.

Port of `velocity_general.f90:1775 calc_vel_ratio`.
"""
function calc_vel_ratio!(f_vbvs, uxy_b, uxy_s)
    Fb = interior(f_vbvs)
    Ub = interior(uxy_b)
    Us = interior(uxy_s)
    @. Fb = ifelse(Us > 0.0, min(1.0, Ub / Us), 1.0)
    return f_vbvs
end
