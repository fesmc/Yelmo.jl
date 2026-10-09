# ----------------------------------------------------------------------
# Post-solver dyn diagnostics.
#
# This file collects the kernels that run after the velocity solver
# (currently a no-op for `solver = "fixed"`):
#
#   - `_clip_underflow!`       — drop sub-`TOL_UNDERFLOW` floats to zero.
#   - `calc_ice_flux!`         — `qq_acx/y = H_up · dy · u_bar` on
#                                 ac-staggered faces (upwind thickness).
#   - `calc_grounding_line_flux!` — `qq_gl_acx/y`, the face flux across
#                                 the grounding line.
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
using Oceananigans.Grids: topology, Bounded, Periodic, AbstractTopology
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_ice_flux!, calc_grounding_line_flux!, calc_magnitude_from_staggered!, calc_vel_ratio!

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

Ice flux through each cell face [m³/yr] with the upwind ice thickness,
as in the advection solvers:

    qq_acx[i+1, j] = H_up · dy · ux_bar[i+1, j],  H_up = H_ice[i] if ux_bar ≥ 0 else H_ice[i+1]

(likewise `qq_acy` with `dx`). The last face of a Bounded direction (the
domain edge) stays zero; in a Periodic direction it wraps. The leading
face slot of a Bounded direction is replicated for parity with the
YelmoMirror loader convention. The model passes the actual thickness
`tpo.H_ice`.

Port of `velocity_general.f90:calc_ice_flux` (yelmo dev).
"""
function calc_ice_flux!(qq_acx, qq_acy, ux_bar, uy_bar, H_ice,
                        dx::Real, dy::Real)
    Qx, Qy = interior(qq_acx), interior(qq_acy)
    Ux, Uy = interior(ux_bar), interior(uy_bar)
    H      = interior(H_ice)
    Nx, Ny = size(H, 1), size(H, 2)
    Tx = topology(qq_acx.grid, 1)
    Ty = topology(qq_acy.grid, 2)
    _calc_ice_flux_kernel!(Qx, Qy, Ux, Uy, H, Float64(dx), Float64(dy), Tx, Ty, Nx, Ny)
    return qq_acx, qq_acy
end

function _calc_ice_flux_kernel!(Qx, Qy, Ux, Uy, H, dx::Float64, dy::Float64,
                                ::Type{Tx}, ::Type{Ty}, Nx::Int, Ny::Int
                               ) where {Tx<:AbstractTopology, Ty<:AbstractTopology}
    fill!(Qx, 0.0)
    fill!(Qy, 0.0)

    # The last face is an interior face only in a periodic direction.
    i2 = Tx === Periodic ? Nx : Nx - 1
    j2 = Ty === Periodic ? Ny : Ny - 1

    @inbounds for j in 1:Ny, i in 1:i2
        ip1  = _neighbor_ip1(i, Nx, Tx)
        ip1f = _ip1_modular(i, Nx, Tx)
        u    = Ux[ip1f, j, 1]
        Qx[ip1f, j, 1] = (u >= 0.0 ? H[i, j, 1] : H[ip1, j, 1]) * dy * u
    end
    @inbounds for j in 1:j2, i in 1:Nx
        jp1  = _neighbor_jp1(j, Ny, Ty)
        jp1f = _jp1_modular(j, Ny, Ty)
        v    = Uy[i, jp1f, 1]
        Qy[i, jp1f, 1] = (v >= 0.0 ? H[i, j, 1] : H[i, jp1, 1]) * dx * v
    end

    if Tx === Bounded
        @views Qx[1, :, :] .= Qx[2, :, :]
    end
    if Ty === Bounded
        @views Qy[:, 1, :] .= Qy[:, 2, :]
    end
    return nothing
end

"""
    calc_grounding_line_flux!(qq_gl_acx, qq_gl_acy, qq_acx, qq_acy, f_grnd, f_ice)
        -> (qq_gl_acx, qq_gl_acy)

Ice flux across the grounding line [m³/yr]: the face flux `qq_acx/acy`
(see [`calc_ice_flux!`](@ref)) through faces between a (partially)
grounded cell (`f_grnd > 0`) and a floating ice cell (`f_grnd == 0`,
`f_ice > 0`), zero elsewhere. The sign gives the direction (+x/+y).
Neighbours follow the grid topology (clamped when Bounded).

Port of `velocity_general.f90:calc_grounding_line_flux` (yelmo dev).
"""
function calc_grounding_line_flux!(qq_gl_acx, qq_gl_acy, qq_acx, qq_acy, f_grnd, f_ice)
    Gx, Gy = interior(qq_gl_acx), interior(qq_gl_acy)
    Qx, Qy = interior(qq_acx), interior(qq_acy)
    Fg, Fi = interior(f_grnd), interior(f_ice)
    Nx, Ny = size(Fg, 1), size(Fg, 2)
    Tx = topology(qq_gl_acx.grid, 1)
    Ty = topology(qq_gl_acy.grid, 2)
    _calc_grounding_line_flux_kernel!(Gx, Gy, Qx, Qy, Fg, Fi, Tx, Ty, Nx, Ny)
    return qq_gl_acx, qq_gl_acy
end

# Face between a (partially) grounded cell and a floating ice cell.
@inline _is_gl_face(fg0, fg1, fi0, fi1) =
    (fg0 > 0.0 && fg1 == 0.0 && fi1 > 0.0) || (fg1 > 0.0 && fg0 == 0.0 && fi0 > 0.0)

function _calc_grounding_line_flux_kernel!(Gx, Gy, Qx, Qy, Fg, Fi,
                                           ::Type{Tx}, ::Type{Ty}, Nx::Int, Ny::Int
                                          ) where {Tx<:AbstractTopology, Ty<:AbstractTopology}
    fill!(Gx, 0.0)
    fill!(Gy, 0.0)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip1  = _neighbor_ip1(i, Nx, Tx)
        jp1  = _neighbor_jp1(j, Ny, Ty)
        ip1f = _ip1_modular(i, Nx, Tx)
        jp1f = _jp1_modular(j, Ny, Ty)
        if _is_gl_face(Fg[i, j, 1], Fg[ip1, j, 1], Fi[i, j, 1], Fi[ip1, j, 1])
            Gx[ip1f, j, 1] = Qx[ip1f, j, 1]
        end
        if _is_gl_face(Fg[i, j, 1], Fg[i, jp1, 1], Fi[i, j, 1], Fi[i, jp1, 1])
            Gy[i, jp1f, 1] = Qy[i, jp1f, 1]
        end
    end
    if Tx === Bounded
        @views Gx[1, :, :] .= Gx[2, :, :]
    end
    if Ty === Bounded
        @views Gy[:, 1, :] .= Gy[:, 2, :]
    end
    return nothing
end

# uz_srf_err = uz_star at the surface + smb on fully covered cells, else 0.
function _calc_uz_srf_err!(uz_srf_err, uz_star, smb, f_ice)
    E, Us, S, F = interior(uz_srf_err), interior(uz_star), interior(smb), interior(f_ice)
    nz = size(Us, 3)
    @inbounds for j in axes(E, 2), i in axes(E, 1)
        E[i, j, 1] = F[i, j, 1] == 1.0 ? Us[i, j, nz] + S[i, j, 1] : 0.0
    end
    return uz_srf_err
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
    # Wrapper: fill halos on the staggered face fields, lift Field
    # views to plain SubArrays, look up topology, dispatch into the
    # parametric kernel below. Same wrapper-+-parametric-kernel template
    # as the dyn 3D series (#45 / #47 / #48 / #49 / #50) and the DIVA
    # viscosity refactor (#51).
    fill_halo_regions!(u)
    fill_halo_regions!(v)

    M  = interior(umag)
    Ux = interior(u)
    Uy = interior(v)
    Fi = interior(f_ice)
    Nx = size(M, 1)
    Ny = size(M, 2)
    Nz = size(M, 3)

    Tx_top = topology(umag.grid, 1)
    Ty_top = topology(umag.grid, 2)

    _calc_magnitude_from_staggered_kernel!(M, Ux, Uy, Fi,
                                           Tx_top, Ty_top, Nx, Ny, Nz)
    return umag
end

# Compute kernel — parametric topology, plain arrays, no Field accesses.
# Under Bounded the i+1 / j+1 face slots are in-bounds (face fields
# have Nx+1 / Ny+1 columns), under Periodic they wrap modularly via
# `_ip1_modular` / `_jp1_modular`.
function _calc_magnitude_from_staggered_kernel!(
        M, Ux, Uy, Fi,
        ::Type{Tx}, ::Type{Ty}, Nx::Int, Ny::Int, Nz::Int,
    ) where {Tx<:AbstractTopology, Ty<:AbstractTopology}

    fill!(M, 0.0)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if Fi[i, j, 1] == 1.0
            ip1 = _ip1_modular(i, Nx, Tx)
            jp1 = _jp1_modular(j, Ny, Ty)
            unow = 0.5 * (Ux[i,   j,   k] + Ux[ip1, j,   k])
            vnow = 0.5 * (Uy[i,   j,   k] + Uy[i,   jp1, k])
            abs(unow) < TOL_UNDERFLOW && (unow = 0.0)
            abs(vnow) < TOL_UNDERFLOW && (vnow = 0.0)
            mag = sqrt(unow * unow + vnow * vnow)
            M[i, j, k] = abs(mag) < TOL_UNDERFLOW ? 0.0 : mag
        end
    end
    return nothing
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
