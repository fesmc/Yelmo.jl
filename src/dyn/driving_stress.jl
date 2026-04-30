# ----------------------------------------------------------------------
# Driving stress on staggered ac-faces.
#
#   - `calc_driving_stress!` — `taud = ρ_ice · g · H_mid · ∂z_s/∂{x,y}`
#     on the X/Y-Face fields `taud_acx` / `taud_acy`, with the Fortran
#     margin-aware `H_mid` averaging (use the upstream cell's `H_ice`
#     when one side is partially-covered, the symmetric average
#     otherwise) and a hard `taud_lim` clamp.
#
#   - `calc_driving_stress_gl!` — optional grounding-line refinement
#     over `taud_acx` / `taud_acy`, dispatched on
#     `y.p.ydyn.taud_gl_method`. Methods −1, 1, 2, 3 mirror the
#     Fortran reference; method 0 is a no-op (gated upstream in
#     `dyn_step!`). Method −1 with `beta_gl_stag=2` is the only
#     branch that actually mutates anything; with `beta_gl_stag=1`
#     the Fortran body is wrapped in `if (.FALSE.)` (dead code) and
#     we mirror that as a no-op. The orchestrator passes
#     `beta_gl_stag=1` unconditionally.
#
# Boundary handling: H_ice halos satisfy Dirichlet `H = 0`; f_ice halos
# Neumann-replicate. This differs from the Fortran `infinite` BC code
# (which extends `H_ice` across the right/north edge) by one face row
# at each edge — the lockstep test compares interior-only.
#
# Port of `yelmo/src/physics/velocity_general.f90`:
#   - `calc_driving_stress` (line 985)
#   - `calc_driving_stress_gl` (line 1130)
#   - `integrate_gl_driving_stress_linear` (line 1600), used by case 3.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_driving_stress!, calc_driving_stress_gl!

# Margin-aware face-averaged ice thickness used by the SIA driving
# stress: when one side of the face is fully-covered (`f == 1`) and
# the other is partially-covered (`f < 1`), use the fully-covered
# cell's thickness paired with `0`; otherwise use the plain average.
# Mirrors the Fortran branch in `calc_driving_stress`.
@inline function _H_mid_margin(H0, H1, f0, f1)
    if f0 == 1.0 && f1 < 1.0
        return 0.5 * H0
    elseif f0 < 1.0 && f1 == 1.0
        return 0.5 * H1
    else
        return 0.5 * (H0 + H1)
    end
end

"""
    calc_driving_stress!(taud_acx, taud_acy,
                         H_ice_dyn, f_ice_dyn, dzsdx, dzsdy,
                         dx, taud_lim, rho_ice, g) -> (taud_acx, taud_acy)

Compute the driving stress on the ac-staggered faces:

    taud_acx[i+1, j] = ρ_ice · g · H_mid_x(i, j) · dzsdx[i, j]
    taud_acy[i, j+1] = ρ_ice · g · H_mid_y(i, j) · dzsdy[i, j]

where `H_mid_x` is the margin-aware face-average of `H_ice_dyn` between
aa-cells `(i, j)` and `(i+1, j)` (analogous in y). Each component is
clamped to `|taud| ≤ taud_lim`. The leftmost / southernmost extra face
slot of the X/Y-Face fields (interior index 1) is replicated from the
adjacent interior face to match the YelmoMirror NetCDF convention.

`H_ice_dyn` and `f_ice_dyn` are the dynamic-ice fields produced by
`extend_floating_slab!` / `calc_dynamic_ice_fields!`; `dzsdx` /
`dzsdy` are CenterField gradients staggered to ac-faces (per the
Yelmo schema). `dx` is in metres; `taud_lim` is in Pa.

Port of `velocity_general.f90:985 calc_driving_stress`. The
`if (.FALSE.) ... end if` "special cases" block in the Fortran
reference is dead code and is omitted.
"""
function calc_driving_stress!(taud_acx, taud_acy,
                              H_ice_dyn, f_ice_dyn,
                              dzsdx, dzsdy,
                              dx::Real, taud_lim::Real,
                              rho_ice::Real, g::Real)
    # H_ice halo (Dirichlet 0) and f_ice halo (Neumann replicate)
    # supply the ip1 / jp1 reads at the eastern / northern domain edge.
    # Reads use field-indexed access (which goes through halos via
    # the underlying OffsetArray); writes use the interior view.
    fill_halo_regions!(H_ice_dyn)
    fill_halo_regions!(f_ice_dyn)

    rhog = Float64(rho_ice) * Float64(g)
    lim  = Float64(taud_lim)

    Tx = interior(taud_acx)
    Ty = interior(taud_acy)
    Nx = size(interior(H_ice_dyn), 1)
    Ny = size(interior(H_ice_dyn), 2)

    # x-direction: Fortran taud_acx[i, j] = east face of cell i.
    # Julia XFaceField stores it at interior(taud_acx)[i+1, j, 1].
    @inbounds for j in 1:Ny, i in 1:Nx
        H_mid = _H_mid_margin(H_ice_dyn[i, j, 1], H_ice_dyn[i+1, j, 1],
                              f_ice_dyn[i, j, 1], f_ice_dyn[i+1, j, 1])
        taud  = rhog * H_mid * dzsdx[i, j, 1]
        Tx[i+1, j, 1] = clamp(taud, -lim, lim)
    end
    # Replicate first-face slot per the YelmoMirror loader convention.
    @views Tx[1, :, :] .= Tx[2, :, :]

    # y-direction: same pattern.
    @inbounds for j in 1:Ny, i in 1:Nx
        H_mid = _H_mid_margin(H_ice_dyn[i, j, 1], H_ice_dyn[i, j+1, 1],
                              f_ice_dyn[i, j, 1], f_ice_dyn[i, j+1, 1])
        taud  = rhog * H_mid * dzsdy[i, j, 1]
        Ty[i, j+1, 1] = clamp(taud, -lim, lim)
    end
    @views Ty[:, 1, :] .= Ty[:, 2, :]

    return taud_acx, taud_acy
end

# Subgrid driving stress at a grounded↔floating face by piecewise-
# linear integration of (H · dzs/dx) along the face span (Gladstone
# et al. 2010, Eq. 27). `H_a`/`zb_a`/`zsl_a` are at the grounded cell;
# `H_b`/`zb_b`/`zsl_b` at the floating cell. `dx` is the cell spacing.
# Returns `taud` (Pa).
@inline function _integrate_gl_driving_stress_linear(H_a::Float64, H_b::Float64,
                                                     zb_a::Float64, zb_b::Float64,
                                                     zsl_a::Float64, zsl_b::Float64,
                                                     dx::Float64,
                                                     rho_ice::Float64, rho_sw::Float64,
                                                     g::Float64)
    ntot = 100
    rho_sw_ice = rho_sw / rho_ice
    rho_ice_sw = rho_ice / rho_sw
    dl = 1.0 / (ntot - 1.0)
    dx_ab = dx * dl

    taud = 0.0
    @inbounds for n in 1:ntot
        Ha  = H_a   + (H_b - H_a)   * dl * (n - 1)
        Hb  = H_a   + (H_b - H_a)   * dl *  n
        Ba  = zb_a  + (zb_b - zb_a) * dl * (n - 1)
        Bb  = zb_a  + (zb_b - zb_a) * dl *  n
        sla = zsl_a + (zsl_b - zsl_a) * dl * (n - 1)
        slb = zsl_a + (zsl_b - zsl_a) * dl *  n

        Sa = Ha < rho_sw_ice * (sla - Ba) ? (1.0 - rho_ice_sw) * Ha : Ba + Ha
        Sb = Hb < rho_sw_ice * (slb - Bb) ? (1.0 - rho_ice_sw) * Hb : Bb + Hb

        H_mid = 0.5 * (Ha + Hb)
        dzsdx = (Sb - Sa) / dx_ab
        taud += H_mid * dzsdx * dl
    end
    return rho_ice * g * taud
end

"""
    calc_driving_stress_gl!(taud_acx, taud_acy,
                            H_ice_dyn, z_srf, z_bed, z_sl, H_grnd,
                            f_grnd, f_grnd_acx, f_grnd_acy,
                            dx, rho_ice, rho_sw, g,
                            method, beta_gl_stag) -> (taud_acx, taud_acy)

Refine the driving stress at the grounding line. Dispatch on `method`:

  - `-1`: pick upstream / downstream face value depending on
    `beta_gl_stag`. With `beta_gl_stag=1` (the value passed by
    `dyn_step!`) the Fortran body is `if (.FALSE.) ... end if` —
    no-op, mirrored here. With `beta_gl_stag=2` the upstream/
    downstream choice flips: stress at the floating-side face is set
    to its grounded neighbour, stress at the grounded-side face is set
    to its floating neighbour. Other `beta_gl_stag` values error.
  - `1`: weighted average of grounded and "virtual floating" surface
    slope using `f_grnd_acx`/`f_grnd_acy` (only at sub-grid GL faces),
    clamped to `|dzsdx| ≤ slope_max = 0.05`.
  - `2`: one-sided differences (Feldmann et al. 2014). Uses the
    grounded surface slope when `H_grnd_mid > 0`, ice-thickness slope
    otherwise. Clamped to `|dzsdx| ≤ slope_max`.
  - `3`: piecewise-linear integration along the GL face (Gladstone
    et al. 2010, Eq. 27) via `_integrate_gl_driving_stress_linear`.
  - any other value: no-op.

Indexing convention matches `calc_driving_stress!`: face `(i, j)` in
Fortran corresponds to `interior(taud_acx)[i+1, j, 1]` in Julia.

Port of `velocity_general.f90:1130 calc_driving_stress_gl`.
"""
function calc_driving_stress_gl!(taud_acx, taud_acy,
                                 H_ice_dyn, z_srf, z_bed, z_sl, H_grnd,
                                 f_grnd, f_grnd_acx, f_grnd_acy,
                                 dx::Real, rho_ice::Real, rho_sw::Real, g::Real,
                                 method::Int, beta_gl_stag::Int)
    # Halos for any field that the kernels read at i+1 / j+1.
    fill_halo_regions!(H_ice_dyn)
    fill_halo_regions!(z_srf)
    fill_halo_regions!(z_bed)
    fill_halo_regions!(z_sl)
    fill_halo_regions!(H_grnd)
    fill_halo_regions!(f_grnd)
    fill_halo_regions!(f_grnd_acx)
    fill_halo_regions!(f_grnd_acy)

    rhog       = Float64(rho_ice) * Float64(g)
    dx_f       = Float64(dx)
    slope_max  = 0.05

    Tx = interior(taud_acx)
    Ty = interior(taud_acy)
    Nx = size(interior(H_ice_dyn), 1)
    Ny = size(interior(H_ice_dyn), 2)

    if method == -1
        if beta_gl_stag == 1
            # Fortran wraps this branch in `if (.FALSE.)` — dead code.
            # Intentionally no-op.
        elseif beta_gl_stag == 2
            # Downstream beta at GL ⇒ float-side face takes grounded
            # neighbour's stress, grounded-side face takes floating's.
            @inbounds for j in 1:Ny, i in 2:Nx-1
                if f_grnd[i, j, 1] == 0.0 && f_grnd[i+1, j, 1] > 0.0
                    Tx[i+1, j, 1] = Tx[(i-1)+1, j, 1]    # taud_acx[i] = taud_acx[i-1]
                elseif f_grnd[i, j, 1] > 0.0 && f_grnd[i+1, j, 1] == 0.0
                    Tx[i+1, j, 1] = Tx[(i+1)+1, j, 1]    # taud_acx[i] = taud_acx[i+1]
                end
            end
            @inbounds for j in 2:Ny-1, i in 1:Nx
                if f_grnd[i, j, 1] == 0.0 && f_grnd[i, j+1, 1] > 0.0
                    Ty[i, j+1, 1] = Ty[i, (j-1)+1, 1]
                elseif f_grnd[i, j, 1] > 0.0 && f_grnd[i, j+1, 1] == 0.0
                    Ty[i, j+1, 1] = Ty[i, (j+1)+1, 1]
                end
            end
        else
            error("calc_driving_stress_gl!: method=-1 requires beta_gl_stag in (1, 2); got $beta_gl_stag.")
        end

    elseif method == 1
        @inbounds for j in 1:Ny, i in 1:Nx-1
            fgx_now = f_grnd_acx[i, j, 1]
            if 0.0 < fgx_now < 1.0
                H_gl    = 0.5 * (H_ice_dyn[i, j, 1] + H_ice_dyn[i+1, j, 1])
                dzsdx_1 = (z_srf[i+1, j, 1] - z_srf[i, j, 1]) / dx_f
                dzsdx   = fgx_now * dzsdx_1   # `dzsdx_2 = 0` per Fortran
                dzsdx   = clamp(dzsdx, -slope_max, slope_max)
                Tx[i+1, j, 1] = rhog * H_gl * dzsdx
            end
        end
        @inbounds for j in 1:Ny-1, i in 1:Nx
            fgy_now = f_grnd_acy[i, j, 1]
            if 0.0 < fgy_now < 1.0
                H_gl    = 0.5 * (H_ice_dyn[i, j, 1] + H_ice_dyn[i, j+1, 1])
                dzsdy_1 = (z_srf[i, j+1, 1] - z_srf[i, j, 1]) / dx_f
                dzsdy   = fgy_now * dzsdy_1
                dzsdy   = clamp(dzsdy, -slope_max, slope_max)
                Ty[i, j+1, 1] = rhog * H_gl * dzsdy
            end
        end

    elseif method == 2
        @inbounds for j in 1:Ny, i in 1:Nx-1
            fgx_now = f_grnd_acx[i, j, 1]
            if 0.0 < fgx_now < 1.0
                H_grnd_mid = 0.5 * (H_grnd[i, j, 1] + H_grnd[i+1, j, 1])
                dzsdx = if H_grnd_mid > 0.0
                    (z_srf[i+1, j, 1] - z_srf[i, j, 1]) / dx_f
                else
                    (H_ice_dyn[i+1, j, 1] - H_ice_dyn[i, j, 1]) / dx_f
                end
                dzsdx = clamp(dzsdx, -slope_max, slope_max)
                H_gl  = 0.5 * (H_ice_dyn[i, j, 1] + H_ice_dyn[i+1, j, 1])
                Tx[i+1, j, 1] = rhog * H_gl * dzsdx
            end
        end
        @inbounds for j in 1:Ny-1, i in 1:Nx
            fgy_now = f_grnd_acy[i, j, 1]
            if 0.0 < fgy_now < 1.0
                H_grnd_mid = 0.5 * (H_grnd[i, j, 1] + H_grnd[i, j+1, 1])
                dzsdy = if H_grnd_mid > 0.0
                    (z_srf[i, j+1, 1] - z_srf[i, j, 1]) / dx_f
                else
                    (H_ice_dyn[i, j+1, 1] - H_ice_dyn[i, j, 1]) / dx_f
                end
                dzsdy = clamp(dzsdy, -slope_max, slope_max)
                H_gl  = 0.5 * (H_ice_dyn[i, j, 1] + H_ice_dyn[i, j+1, 1])
                Ty[i, j+1, 1] = rhog * H_gl * dzsdy
            end
        end

    elseif method == 3
        rho_ice_f = Float64(rho_ice)
        rho_sw_f  = Float64(rho_sw)
        g_f       = Float64(g)
        @inbounds for j in 1:Ny, i in 1:Nx
            # x-direction: Fortran does max(1, i-1) / min(nx, i+1) for
            # neighbour clamps; we use cells (i, i+1) directly. The
            # `i+1` read at i = Nx pulls through halo — for `infinite`-
            # like Neumann BCs that's a replicate; the case-3 kernel
            # never trips at the boundary in realistic runs.
            ip1 = min(Nx, i + 1)
            jp1 = min(Ny, j + 1)

            if H_grnd[i, j, 1] > 0.0 && H_grnd[ip1, j, 1] <= 0.0
                Tx[i+1, j, 1] = _integrate_gl_driving_stress_linear(
                    H_ice_dyn[i, j, 1], H_ice_dyn[ip1, j, 1],
                    z_bed[i, j, 1],     z_bed[ip1, j, 1],
                    z_sl[i, j, 1],      z_sl[ip1, j, 1],
                    dx_f, rho_ice_f, rho_sw_f, g_f)
            elseif H_grnd[i, j, 1] <= 0.0 && H_grnd[ip1, j, 1] > 0.0
                t = _integrate_gl_driving_stress_linear(
                    H_ice_dyn[ip1, j, 1], H_ice_dyn[i, j, 1],
                    z_bed[ip1, j, 1],     z_bed[i, j, 1],
                    z_sl[ip1, j, 1],      z_sl[i, j, 1],
                    dx_f, rho_ice_f, rho_sw_f, g_f)
                Tx[i+1, j, 1] = -t
            end

            if H_grnd[i, j, 1] > 0.0 && H_grnd[i, jp1, 1] <= 0.0
                Ty[i, j+1, 1] = _integrate_gl_driving_stress_linear(
                    H_ice_dyn[i, j, 1], H_ice_dyn[i, jp1, 1],
                    z_bed[i, j, 1],     z_bed[i, jp1, 1],
                    z_sl[i, j, 1],      z_sl[i, jp1, 1],
                    dx_f, rho_ice_f, rho_sw_f, g_f)
            elseif H_grnd[i, j, 1] <= 0.0 && H_grnd[i, jp1, 1] > 0.0
                t = _integrate_gl_driving_stress_linear(
                    H_ice_dyn[i, jp1, 1], H_ice_dyn[i, j, 1],
                    z_bed[i, jp1, 1],     z_bed[i, j, 1],
                    z_sl[i, jp1, 1],      z_sl[i, j, 1],
                    dx_f, rho_ice_f, rho_sw_f, g_f)
                Ty[i, j+1, 1] = -t
            end
        end

    else
        # Other values (including 0): no-op. Fortran's `case DEFAULT`
        # is also a no-op. Method 0 is gated upstream in `dyn_step!`.
    end

    return taud_acx, taud_acy
end
