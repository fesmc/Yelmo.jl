# ----------------------------------------------------------------------
# Shallow-Ice Approximation (SIA) velocity kernels.
#
#   - `calc_shear_stress_3D!` — vertical shear stresses on
#     (acx, acy)-faces × `zeta_aa` (layer centres):
#         tau_xz(i, j, k) = -(1 - zeta_aa[k]) * taud_acx(i, j)
#         tau_yz(i, j, k) = -(1 - zeta_aa[k]) * taud_acy(i, j)
#
#   - `calc_uxy_sia_3D!` — depth-recurrence for the SIA horizontal
#     velocity components:
#         u(k) = u(k-1) + 0.5 * fact_ac * (tau(k) + tau(k-1))
#     where `fact_ac` is built from a 4-cell ATT/H average plus an
#     effective-stress factor `tau_eff^((n_glen-1)/2)`. See the
#     Fortran reference for the full averaging recipe.
#
#   - `calc_velocity_sia!` — Option-C SIA wrapper that fills halos,
#     calls the two kernels, computes the surface boundary value
#     (ux_i_s / uy_i_s) and the depth-averaged ux_i_bar / uy_i_bar.
#
# Vertical-stagger convention (Option C): the 3D SIA fields
# (`ux_i`, `uy_i`, `tau_xz`, `tau_yz`, `ATT`, …) live at Oceananigans
# `Center()` z-positions — interior layer midpoints, length Nz_aa
# cells. Bed (zeta = 0) and surface (zeta = 1) endpoints are NOT in
# `zeta_aa`; their boundary values are handled explicitly by the
# kernel (bed segment) and the wrapper (surface segment).
#
# Staggering:
#
#   - `tau_xz` lives on x-faces and is allocated as an `XFaceField` on
#     the 3D ice grid `y.gt`. The Yelmo.jl convention (matching
#     `driving_stress.jl`) is that an x-face value for cell `(i, j)`
#     lives at array index `[i+1, j, :]`. Y-face analogous: y-face
#     value for cell `(i, j)` lives at `[i, j+1, :]`.
#
#   - These two fields are SIA-only solver scratch buffers — they are
#     recomputed every `dyn_step!`, are not part of the model state,
#     and therefore are NOT in the `dyn` schema. They are exposed as
#     `y.dyn.scratch.sia_tau_xz` / `y.dyn.scratch.sia_tau_yz`.
#
# Boundary handling:
#
#   - `calc_shear_stress_3D!` does not read i±1 / j±1 neighbours of
#     any input — each cell uses `taud_acx[i+1, j, 1]` and
#     `taud_acy[i, j+1, 1]` directly. No `fill_halo_regions!` is
#     required.
#
#   - `calc_uxy_sia_3D!` reads i±1 / j±1 neighbours of `tau_xz`,
#     `tau_yz`, `ATT`, `H_ice`. Callers must `fill_halo_regions!`
#     those four fields *before* invoking this kernel. The kernel
#     does not call `fill_halo_regions!` itself; the SIA wrapper
#     (Commit 2) will.
#
#   - The local `fact_ab::Matrix{Float64}` buffer in kernel 2 is a
#     plain `Array` with no halo, so `fact_ab[i, jm1]` / `fact_ab[im1, j]`
#     reads use a `max(·, 1)` clamp. This is correct for the
#     `(Bounded, Bounded, Bounded)` topology only (matching the
#     Fortran `BND_INFINITE` semantics). Kernel 2 asserts on entry
#     that the input grid has Bounded x and y; periodic-y support is
#     deferred to milestone 3d.
#
# Port of `yelmo/src/physics/velocity_sia.f90`:
#   - `calc_shear_stress_3D` (line 63)
#   - `calc_uxy_sia_3D` (line 110)
# `calc_velocity_sia` (the wrapper combining the two + the depth
# integration) is deliberately NOT ported in this commit; that, plus
# the `solver == "sia"` dispatch in `dyn_step!`, lands in Commit 2 of
# milestone 3c.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.Grids: topology, Bounded
using Oceananigans.BoundaryConditions: fill_halo_regions!

export calc_shear_stress_3D!, calc_uxy_sia_3D!, calc_velocity_sia!

"""
    calc_shear_stress_3D!(tau_xz, tau_yz,
                          taud_acx, taud_acy,
                          f_ice,
                          zeta_aa::AbstractVector{<:Real})
        -> (tau_xz, tau_yz)

Compute the per-layer vertical shear stresses on x- and y-faces:

    tau_xz(i, j, k) = -(1 - zeta_aa[k]) * taud_acx(i, j)
    tau_yz(i, j, k) = -(1 - zeta_aa[k]) * taud_acy(i, j)

Indexing convention: x-face values for cell `(i, j)` live at array
index `[i+1, j, :]`; y-face values at `[i, j+1, :]`. The leftmost
slot `[1, :, k]` of `tau_xz` and the southernmost slot `[:, 1, k]`
of `tau_yz` are replicated from the adjacent interior face per layer
to match the YelmoMirror loader convention used elsewhere in the dyn
module (mirroring `calc_driving_stress!`).

`f_ice` is currently unused in the kernel body but kept for Fortran
signature parity (`velocity_sia.f90:75`); future masking work that
zeroes shear stress outside grounded ice may consume it.

`zeta_aa` is the cell-centre vertical coordinate vector (length
`Nz_aa`), obtained from `znodes(y.gt, Center())`.

This kernel does NOT read i±1 / j±1 neighbours of any input, so no
`fill_halo_regions!` is required before calling it.

Port of `velocity_sia.f90:63 calc_shear_stress_3D`.
"""
function calc_shear_stress_3D!(tau_xz, tau_yz,
                               taud_acx, taud_acy,
                               f_ice,
                               zeta_aa::AbstractVector{<:Real})
    Txz = interior(tau_xz)
    Tyz = interior(tau_yz)
    Tx  = interior(taud_acx)
    Ty  = interior(taud_acy)

    fill!(Txz, 0.0)
    fill!(Tyz, 0.0)

    Nx, Ny = size(interior(f_ice), 1), size(interior(f_ice), 2)
    Nz = length(zeta_aa)

    @inbounds for k in 1:Nz
        one_m_zk = 1.0 - Float64(zeta_aa[k])
        for j in 1:Ny, i in 1:Nx
            # X-face: Fortran tau_xz(i, j, k) ↔ Tx index (i+1, j, 1).
            Txz[i+1, j, k] = -one_m_zk * Tx[i+1, j, 1]
            # Y-face: Fortran tau_yz(i, j, k) ↔ Ty index (i, j+1, 1).
            Tyz[i, j+1, k] = -one_m_zk * Ty[i, j+1, 1]
        end
    end

    # Replicate the leading face slot per-layer (matches the
    # YelmoMirror loader convention and `calc_driving_stress!`).
    @views Txz[1, :, :] .= Txz[2, :, :]
    @views Tyz[:, 1, :] .= Tyz[:, 2, :]

    return tau_xz, tau_yz
end

"""
    calc_uxy_sia_3D!(ux, uy,
                     tau_xz, tau_yz,
                     taud_acx, taud_acy,
                     H_ice, f_ice,
                     ATT, n_glen::Real,
                     zeta_aa::AbstractVector{<:Real})
        -> (ux, uy)

Compute the SIA horizontal velocity components by depth recurrence:

    ux(i, j, k) = ux(i, j, k-1) + 0.5 * fact_ac_x * (tau_xz(k) + tau_xz(k-1))
    uy(i, j, k) = uy(i, j, k-1) + 0.5 * fact_ac_y * (tau_yz(k) + tau_yz(k-1))

For Option C in Yelmo.jl, `zeta_aa` carries the Oceananigans `Center`
positions (interior layer midpoints; length `Nz_aa`); the bed
(zeta = 0) and surface (zeta = 1) endpoints are NOT in `zeta_aa`. To
preserve "Yelmo intent" (the SIA shear contribution from the bed up to
the topmost Center) the kernel integrates an extra k=1 bed segment
from zeta = 0 to zeta_aa[1] using the SIA no-slip BC `ux = uy = 0` at
the bed and the closed-form shear stress at zeta = 0:

    tau_xz_bed = -taud_acx
    tau_yz_bed = -taud_acy

For the bed segment, ATT and H at the bed are approximated by their
nearest-Center values (4-cell average of ATT[:,:,1] and H_ice).

For each interior recurrence step the kernel first builds an ab-node
factor

    fact_ab(i, j) = 2 * ATT_n * (dzeta * H_n) * tau_eff_sq_n^p1
                    with p1 = (n_glen - 1) / 2

via a 4-cell aa-cell average of `ATT` and `H_ice` and a 4-corner
average of the (acx, acy) shear stresses to the ab corner. Then the
ab-node factor is staggered onto the (acx, acy) faces via
`fact_ac = 0.5 * (fact_ab[i, j] + fact_ab[i, jm1])` for x-faces and
`fact_ac = 0.5 * (fact_ab[i, j] + fact_ab[im1, j])` for y-faces.

The surface segment (zeta_aa[Nz] → 1) is computed by the SIA wrapper
`calc_velocity_sia!`, NOT by this kernel.

Indexing convention: x-face values for cell `(i, j)` live at array
index `[i+1, j, :]` (so reads of "Fortran `ux(i, j, k)`" become
`ux[i+1, j, k]`); y-face values at `[i, j+1, :]`.

Halo handling: the kernel reads `tau_xz[i+1, j+1, ...]`,
`tau_yz[i+1, j+1, ...]`, `ATT[i+1, j+1, k]`, `H_ice[i+1, j+1, 1]`,
`taud_acx[i+1, j+1, 1]`, and `taud_acy[i+1, j+1, 1]` at the
(i, j) = (Nx, Ny) corner, so callers must `fill_halo_regions!` those
six fields *before* invocation. The local `fact_ab` matrix is plain
(no halo), and its `[i, jm1]` / `[im1, j]` reads use a `max(·, 1)`
clamp — correct for the `(Bounded, Bounded, Bounded)` topology only.
The kernel asserts on entry that x and y are Bounded; periodic-y
support is deferred to milestone 3d.

`f_ice` is unused in the kernel body but kept for Fortran signature
parity (`velocity_sia.f90:122`).

`n_glen` is the Glen-flow exponent (Float64, default 3.0).
`zeta_aa` is the cell-centre vertical coordinate vector at
Oceananigans `Center()` nodes (length Nz_aa, midpoints of zeta_ac).

Port of `velocity_sia.f90:110 calc_uxy_sia_3D`, extended for Option C
with an explicit bed segment from zeta = 0 to zeta_aa[1].
"""
function calc_uxy_sia_3D!(ux, uy,
                          tau_xz, tau_yz,
                          taud_acx, taud_acy,
                          H_ice, f_ice,
                          ATT, n_glen::Real,
                          zeta_aa::AbstractVector{<:Real})
    grid = ux.grid
    Tx_top = topology(grid, 1)
    Ty_top = topology(grid, 2)
    if !(Tx_top === Bounded && Ty_top === Bounded)
        error("calc_uxy_sia_3D!: requires (Bounded, Bounded, *) horizontal topology; " *
              "got ($(Tx_top), $(Ty_top), …). Periodic-y support is deferred to milestone 3d.")
    end

    Ux  = interior(ux)
    Uy  = interior(uy)

    Nx = size(interior(H_ice), 1)
    Ny = size(interior(H_ice), 2)
    Nz = length(zeta_aa)

    # Sanity asserts — ATT must share the 3D ice grid stagger.
    @assert size(interior(ATT), 1) == Nx
    @assert size(interior(ATT), 2) == Ny
    @assert size(interior(ATT), 3) == Nz

    # Initialise output velocities to zero (matches the Fortran
    # `ux = 0; uy = 0` lines 168-169). The bed boundary condition
    # ux_bed = uy_bed = 0 is implicit in this initialisation: the
    # bed segment integrates from zeta = 0 (where u = 0) up to
    # zeta_aa[1].
    fill!(Ux, 0.0)
    fill!(Uy, 0.0)

    p1 = (Float64(n_glen) - 1.0) / 2.0
    fact_ab = Matrix{Float64}(undef, Nx, Ny)

    # ---- Bed segment: integrate from zeta = 0 to zeta_aa[1] ----
    #
    # SIA closed-form shear stress at zeta = 0:
    #   tau_xz_bed(i, j) = -taud_acx(i, j)
    #   tau_yz_bed(i, j) = -taud_acy(i, j)
    # Bed ATT / H approximated by their k = 1 Center values (4-cell
    # average matches the Fortran averaging recipe for layer k = 1).
    let k = 1
        dzeta_bed = Float64(zeta_aa[1])

        for j in 1:Ny, i in 1:Nx
            # Shear stress 4-corner average at zeta = 0 (closed form):
            # tau_xz_bed at face (i+1, j) and (i+1, j+1).
            txz_up_bed = 0.5 * (-taud_acx[i+1, j,   1] + (-taud_acx[i+1, j+1, 1]))
            txz_up_k1  = 0.5 * (tau_xz[i+1, j,   k] + tau_xz[i+1, j+1, k])
            txz_n      = 0.5 * (txz_up_bed + txz_up_k1)

            tyz_up_bed = 0.5 * (-taud_acy[i,   j+1, 1] + (-taud_acy[i+1, j+1, 1]))
            tyz_up_k1  = 0.5 * (tau_yz[i,   j+1, k] + tau_yz[i+1, j+1, k])
            tyz_n      = 0.5 * (tyz_up_bed + tyz_up_k1)

            tau_eff_sq_n = txz_n^2 + tyz_n^2

            # ATT_bed approximated by ATT[k=1] (4-cell average); ATT_up
            # is also from k=1, so the average is just ATT_k1_avg.
            ATT_k1 = 0.25 * (ATT[i, j, k]   + ATT[i+1, j, k]   +
                             ATT[i, j+1, k] + ATT[i+1, j+1, k])
            ATT_n  = ATT_k1   # bed ≈ k = 1

            H_n = 0.25 * (H_ice[i, j, 1]   + H_ice[i+1, j, 1] +
                          H_ice[i, j+1, 1] + H_ice[i+1, j+1, 1])

            fact_ab[i, j] = if p1 != 0.0
                2.0 * ATT_n * (dzeta_bed * H_n) * tau_eff_sq_n^p1
            else
                2.0 * ATT_n * (dzeta_bed * H_n)
            end
        end

        for j in 1:Ny, i in 1:Nx
            jm1 = max(j - 1, 1)
            im1 = max(i - 1, 1)

            # x-face: bed segment contribution. ux_bed = 0, so
            # ux[k=1] = 0 + 0.5 * fact_ac_x * (tau_xz[k=1] + tau_xz_bed).
            fact_ac_x = 0.5 * (fact_ab[i, j] + fact_ab[i, jm1])
            Ux[i+1, j, k] = 0.0 +
                            fact_ac_x * 0.5 *
                            (tau_xz[i+1, j, k] + (-taud_acx[i+1, j, 1]))

            fact_ac_y = 0.5 * (fact_ab[i, j] + fact_ab[im1, j])
            Uy[i, j+1, k] = 0.0 +
                            fact_ac_y * 0.5 *
                            (tau_yz[i, j+1, k] + (-taud_acy[i, j+1, 1]))
        end
    end

    # ---- Interior recurrence k = 2 ... Nz ----
    @inbounds for k in 2:Nz
        dzeta = Float64(zeta_aa[k]) - Float64(zeta_aa[k-1])

        # First inner loop: compute `fact_ab[i, j]` (ab-node SIA
        # factor) for layer k via 4-cell averages of ATT/H plus the
        # effective shear-stress factor.
        for j in 1:Ny, i in 1:Nx
            # tau_xz: Fortran tau_xz(i,   j,   k) ↔ tau_xz[i+1, j,   k]
            #         Fortran tau_xz(i,   j+1, k) ↔ tau_xz[i+1, j+1, k]
            txz_up = 0.5 * (tau_xz[i+1, j, k]   + tau_xz[i+1, j+1, k])
            txz_dn = 0.5 * (tau_xz[i+1, j, k-1] + tau_xz[i+1, j+1, k-1])
            txz_n  = 0.5 * (txz_up + txz_dn)

            # tau_yz: Fortran tau_yz(i,   j, k) ↔ tau_yz[i,   j+1, k]
            #         Fortran tau_yz(i+1, j, k) ↔ tau_yz[i+1, j+1, k]
            tyz_up = 0.5 * (tau_yz[i, j+1, k]   + tau_yz[i+1, j+1, k])
            tyz_dn = 0.5 * (tau_yz[i, j+1, k-1] + tau_yz[i+1, j+1, k-1])
            tyz_n  = 0.5 * (tyz_up + tyz_dn)

            tau_eff_sq_n = txz_n^2 + tyz_n^2

            # ATT: CenterField, Fortran ATT(i, j, k) ↔ ATT[i, j, k].
            ATT_up = 0.25 * (ATT[i, j, k]   + ATT[i+1, j, k]   +
                             ATT[i, j+1, k] + ATT[i+1, j+1, k])
            ATT_dn = 0.25 * (ATT[i, j, k-1]   + ATT[i+1, j, k-1] +
                             ATT[i, j+1, k-1] + ATT[i+1, j+1, k-1])
            ATT_n  = 0.5 * (ATT_up + ATT_dn)

            # H_ice: CenterField, 2D — read with k = 1.
            H_n = 0.25 * (H_ice[i, j, 1]   + H_ice[i+1, j, 1] +
                          H_ice[i, j+1, 1] + H_ice[i+1, j+1, 1])

            fact_ab[i, j] = if p1 != 0.0
                2.0 * ATT_n * (dzeta * H_n) * tau_eff_sq_n^p1
            else
                2.0 * ATT_n * (dzeta * H_n)
            end
        end

        # Second inner loop: stagger fact_ab to acx / acy faces, then
        # apply the depth recurrence.
        for j in 1:Ny, i in 1:Nx
            jm1 = max(j - 1, 1)
            im1 = max(i - 1, 1)

            # x-face: Fortran ux(i, j, k) ↔ Ux[i+1, j, k].
            fact_ac_x = 0.5 * (fact_ab[i, j] + fact_ab[i, jm1])
            Ux[i+1, j, k] = Ux[i+1, j, k-1] +
                            fact_ac_x * 0.5 * (tau_xz[i+1, j, k] + tau_xz[i+1, j, k-1])

            # y-face: Fortran uy(i, j, k) ↔ Uy[i, j+1, k].
            fact_ac_y = 0.5 * (fact_ab[i, j] + fact_ab[im1, j])
            Uy[i, j+1, k] = Uy[i, j+1, k-1] +
                            fact_ac_y * 0.5 * (tau_yz[i, j+1, k] + tau_yz[i, j+1, k-1])
        end
    end

    return ux, uy
end

# ---------------------------------------------------------------------
# Trapezoidal depth-integration with explicit bed / surface boundaries.
# ---------------------------------------------------------------------
#
# Mirrors Fortran's `integrate_trapezoid1D_pt` (with the
# `TOL_UNDERFLOW` clip on each segment midpoint) but with an extra
# bed segment from zeta = 0 to zeta_c[1] and surface segment from
# zeta_c[Nz] to 1. Used by the Option C SIA wrapper to depth-average
# the 3D velocity field — the Center-staggered layers do NOT include
# the boundary endpoints, so the integration scheme must consume the
# bed and surface boundary values explicitly.
#
# The caller is responsible for face-staggered offset: pass slices of
# `interior(...)` already shifted (e.g. for an XFaceField, work with
# `var3D[i+1, j, 1:Nz]` etc.). The helper itself iterates over the
# leading dims of `out2D` / `var3D` and writes `out2D[:, :, 1]`.
@inline function _vert_int_trapz_with_boundary!(
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

"""
    calc_velocity_sia!(ux_i, uy_i, ux_i_bar, uy_i_bar,
                       ux_i_s, uy_i_s,
                       tau_xz_buf, tau_yz_buf,
                       H_ice, f_ice, taud_acx, taud_acy, ATT,
                       zeta_c::AbstractVector{<:Real},
                       n_glen::Real)
        -> (ux_i, uy_i, ux_i_bar, uy_i_bar, ux_i_s, uy_i_s)

Option C SIA velocity wrapper. Fills 2D / 3D halos, computes the SIA
shear stresses, the 3D Center-staggered SIA velocity field, the 2D
surface boundary value (`ux_i_s`, `uy_i_s`), and the depth-averaged
2D field (`ux_i_bar`, `uy_i_bar`).

Vertical convention (Option C):

  - `ux_i`, `uy_i`, `ATT`, `tau_xz_buf`, `tau_yz_buf` are 3D fields
    with Oceananigans `Center()` z-stagger (interior layer midpoints,
    length `Nz_aa`). Bed (zeta = 0) and surface (zeta = 1) endpoints
    are NOT in `zeta_c`; their boundary values are handled
    explicitly:
      * bed: u = 0 (no-slip), tau_xz_bed = -taud_acx (closed form),
        ATT_bed ≈ ATT[k=1].
      * surface: tau_xz_surf = 0 (closed form), ATT_surf ≈ ATT[k=Nz];
        u_surf is computed in the wrapper from the topmost Center
        value plus the surface segment.

  - `ux_i_bar`, `uy_i_bar`, `ux_i_s`, `uy_i_s` are 2D fields. The
    depth-average `ux_i_bar` integrates `ux_i` over zeta ∈ [0, 1]
    using a trapezoidal scheme with explicit bed (u = 0) and surface
    (`ux_i_s`) endpoints — see `_vert_int_trapz_with_boundary!`.

Boundary ATT assumed equal to nearest-Center ATT. Exact for uniform
ATT (e.g. BUELER tests) and approximate for temperature-dependent ATT;
revisit when therm wires temperature-dependent ATT (milestone 3g).

Halo handling: the wrapper fills halos on `H_ice`, `f_ice`, `ATT`,
`taud_acx`, `taud_acy` (so the 4-cell averages in the kernels see the
Bounded-Neumann replica), and on `tau_xz_buf` / `tau_yz_buf` after
they are populated by `calc_shear_stress_3D!`. The output 3D / 2D
fields are NOT halo-filled; the caller (e.g. `dyn_step!`) handles any
downstream halo refresh.

Mirrors `velocity_sia.f90:17 calc_velocity_sia` and the SIA branch of
`yelmo_dynamics.f90:343-444`, with the Option C extension for the bed
and surface boundary segments.
"""
function calc_velocity_sia!(ux_i, uy_i, ux_i_bar, uy_i_bar,
                            ux_i_s, uy_i_s,
                            tau_xz_buf, tau_yz_buf,
                            H_ice, f_ice, taud_acx, taud_acy, ATT,
                            zeta_c::AbstractVector{<:Real},
                            n_glen::Real)

    # 1. Fill halos on the 2D / 3D inputs that the kernels read with
    # i+1 / j+1 stencils.
    fill_halo_regions!(H_ice)
    fill_halo_regions!(f_ice)
    fill_halo_regions!(ATT)
    fill_halo_regions!(taud_acx)
    fill_halo_regions!(taud_acy)

    # 2. Per-layer SIA shear stress at Oceananigans Center positions.
    # Pass `zeta_c` as the kernel's `zeta_aa` argument: the kernel
    # computes tau_xz[k] = -(1 - zeta_c[k]) * taud_acx, which is the
    # closed-form tau_xz at the Center positions. Bed (zeta = 0) and
    # surface (zeta = 1) tau values are handled separately (kernel 2's
    # bed segment and the wrapper's surface segment, respectively).
    calc_shear_stress_3D!(tau_xz_buf, tau_yz_buf,
                          taud_acx, taud_acy, f_ice, zeta_c)
    fill_halo_regions!(tau_xz_buf)
    fill_halo_regions!(tau_yz_buf)

    # 3. 3D SIA velocity at the Centers (kernel handles the bed
    # segment internally — see `calc_uxy_sia_3D!` docstring).
    calc_uxy_sia_3D!(ux_i, uy_i,
                     tau_xz_buf, tau_yz_buf,
                     taud_acx, taud_acy,
                     H_ice, f_ice, ATT, n_glen, zeta_c)

    # 4. Surface segment: integrate from zeta_c[Nz] to 1 starting at
    # `ux_i[k=Nz]`, with tau_xz_surf = 0 (closed form) and
    # ATT_surf ≈ ATT[k=Nz] (Option C limitation; documented in the
    # docstring).
    Ux_i  = interior(ux_i)
    Uy_i  = interior(uy_i)
    Ux_is = interior(ux_i_s)
    Uy_is = interior(uy_i_s)

    Nx = size(interior(H_ice), 1)
    Ny = size(interior(H_ice), 2)
    Nz = length(zeta_c)

    p1         = (Float64(n_glen) - 1.0) / 2.0
    dzeta_surf = 1.0 - Float64(zeta_c[Nz])
    fact_ab_surf = Matrix{Float64}(undef, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        # 4-corner shear-stress average at zeta = zeta_c[Nz]
        # (Center) and zeta = 1 (surface, where tau = 0 closed form).
        txz_up_surf = 0.0
        txz_up_Nz   = 0.5 * (tau_xz_buf[i+1, j,   Nz] +
                             tau_xz_buf[i+1, j+1, Nz])
        txz_n       = 0.5 * (txz_up_Nz + txz_up_surf)

        tyz_up_surf = 0.0
        tyz_up_Nz   = 0.5 * (tau_yz_buf[i,   j+1, Nz] +
                             tau_yz_buf[i+1, j+1, Nz])
        tyz_n       = 0.5 * (tyz_up_Nz + tyz_up_surf)

        tau_eff_sq_n = txz_n^2 + tyz_n^2

        ATT_Nz = 0.25 * (ATT[i, j, Nz]   + ATT[i+1, j, Nz]   +
                         ATT[i, j+1, Nz] + ATT[i+1, j+1, Nz])
        ATT_n  = ATT_Nz   # surface ≈ k = Nz under Option C

        H_n = 0.25 * (H_ice[i, j, 1]   + H_ice[i+1, j, 1] +
                      H_ice[i, j+1, 1] + H_ice[i+1, j+1, 1])

        fact_ab_surf[i, j] = if p1 != 0.0
            2.0 * ATT_n * (dzeta_surf * H_n) * tau_eff_sq_n^p1
        else
            2.0 * ATT_n * (dzeta_surf * H_n)
        end
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        jm1 = max(j - 1, 1)
        im1 = max(i - 1, 1)

        fact_ac_x = 0.5 * (fact_ab_surf[i, j] + fact_ab_surf[i, jm1])
        # tau_xz_surf = 0, so the segment increment uses (0 + tau_xz[Nz]).
        Ux_is[i+1, j, 1] = Ux_i[i+1, j, Nz] +
                           fact_ac_x * 0.5 *
                           (0.0 + tau_xz_buf[i+1, j, Nz])

        fact_ac_y = 0.5 * (fact_ab_surf[i, j] + fact_ab_surf[im1, j])
        Uy_is[i, j+1, 1] = Uy_i[i, j+1, Nz] +
                           fact_ac_y * 0.5 *
                           (0.0 + tau_yz_buf[i, j+1, Nz])
    end

    # Replicate the leading face slot to match the YelmoMirror loader
    # convention used elsewhere in dyn (mirrors `calc_driving_stress!`).
    @views Ux_is[1, :, :] .= Ux_is[2, :, :]
    @views Uy_is[:, 1, :] .= Uy_is[:, 2, :]

    # 5. Depth-averages via trapezoidal integration with explicit
    # bed (= 0) and surface (= ux_i_s) boundary values. `Ux_i` is
    # XFace-staggered: pass slices [2:Nx+1, :, :] to the helper, which
    # internally treats them as Nx × Ny × Nz aa-style arrays.
    Ux_bar = interior(ux_i_bar)
    Uy_bar = interior(uy_i_bar)

    # Allocate a small zero-buffer for the bed boundary value (u = 0).
    # Sized for the face-shifted slices.
    zero_xface = zeros(Nx, Ny, 1)
    zero_yface = zeros(Nx, Ny, 1)

    # X-face: write to Ux_bar[2:Nx+1, :, 1] reading
    # Ux_i[2:Nx+1, :, 1:Nz] / Ux_is[2:Nx+1, :, 1].
    let
        out_view  = @view Ux_bar[2:Nx+1, :, :]
        var_view  = @view Ux_i[2:Nx+1, :, :]
        surf_view = @view Ux_is[2:Nx+1, :, :]
        _vert_int_trapz_with_boundary!(out_view, var_view,
                                       zero_xface, surf_view, zeta_c)
    end
    # Y-face: write to Uy_bar[:, 2:Ny+1, 1] reading
    # Uy_i[:, 2:Ny+1, 1:Nz] / Uy_is[:, 2:Ny+1, 1].
    let
        out_view  = @view Uy_bar[:, 2:Ny+1, :]
        var_view  = @view Uy_i[:, 2:Ny+1, :]
        surf_view = @view Uy_is[:, 2:Ny+1, :]
        _vert_int_trapz_with_boundary!(out_view, var_view,
                                       zero_yface, surf_view, zeta_c)
    end

    # Replicate the leading face slot for the depth-average (matches
    # the leading-slot replication on the XFace / YFace fields above).
    @views Ux_bar[1, :, :] .= Ux_bar[2, :, :]
    @views Uy_bar[:, 1, :] .= Uy_bar[:, 2, :]

    return ux_i, uy_i, ux_i_bar, uy_i_bar, ux_i_s, uy_i_s
end
