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

export calc_shear_stress_3D!, calc_uxy_sia_3D!

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

starting from `ux(:,:,1) = uy(:,:,1) = 0` and looping `k = 2:Nz_aa`.
For each layer the kernel first builds an ab-node factor

    fact_ab(i, j) = 2 * ATT_n * (dzeta * H_n) * tau_eff_sq_n^p1
                    with p1 = (n_glen - 1) / 2

via a 4-cell aa-cell average of `ATT` and `H_ice` and a 4-corner
average of the (acx, acy) shear stresses to the ab corner. Then the
ab-node factor is staggered onto the (acx, acy) faces via
`fact_ac = 0.5 * (fact_ab[i, j] + fact_ab[i, jm1])` for x-faces and
`fact_ac = 0.5 * (fact_ab[i, j] + fact_ab[im1, j])` for y-faces.

Indexing convention: x-face values for cell `(i, j)` live at array
index `[i+1, j, :]` (so reads of "Fortran `ux(i, j, k)`" become
`ux[i+1, j, k]`); y-face values at `[i, j+1, :]`.

Halo handling: the kernel reads `tau_xz[i+1, j+1, ...]`,
`tau_yz[i+1, j+1, ...]`, `ATT[i+1, j+1, k]`, and `H_ice[i+1, j+1, 1]`
at the (i, j) = (Nx, Ny) corner, so callers must
`fill_halo_regions!` those four fields *before* invocation. The
local `fact_ab` matrix is plain (no halo), and its `[i, jm1]` /
`[im1, j]` reads use a `max(·, 1)` clamp — correct for the
`(Bounded, Bounded, Bounded)` topology only. The kernel asserts on
entry that x and y are Bounded; periodic-y support is deferred to
milestone 3d.

`taud_acx`, `taud_acy`, and `f_ice` are unused in the kernel body
but kept for Fortran signature parity (`velocity_sia.f90:119-122`).

`n_glen` is the Glen-flow exponent (Float64, default 3.0).
`zeta_aa` is the cell-centre vertical coordinate vector.

Port of `velocity_sia.f90:110 calc_uxy_sia_3D`.
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
    # `ux = 0; uy = 0` lines 168-169).
    fill!(Ux, 0.0)
    fill!(Uy, 0.0)

    p1 = (Float64(n_glen) - 1.0) / 2.0
    fact_ab = Matrix{Float64}(undef, Nx, Ny)

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
