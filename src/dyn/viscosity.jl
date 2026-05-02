# ----------------------------------------------------------------------
# SSA effective viscosity helpers.
#
# Three driver kernels and one staggering helper:
#
#   - `calc_visc_eff_3D_aa!`   — direct aa-cell evaluation of the
#     Glen-flow effective viscosity (`velocity_ssa.f90:530`). Reads
#     `(ux, uy, ATT, H_ice, f_ice, n_glen, eps_0)`. Strain rates are
#     centered finite differences at aa-cells; ATT is read at aa.
#
#   - `calc_visc_eff_3D_nodes!` — Gauss-quadrature evaluation
#     (`velocity_ssa.f90:337`). The strain-rate derivatives are
#     pre-computed on the (acx, acy) faces via the upstream-aware
#     finite-difference formula (`calc_strain_rate_horizontal_2D` in
#     `deformation.f90:1667`), then bilinearly interpolated to the 4
#     Gauss nodes per aa-cell (the same `gq2d_nodes(2)` rule used by
#     `calc_beta_aa_power_plastic`).
#
#   - `calc_visc_eff_int!` — depth-integrate `visc_eff` to a 2D field
#     `visc_eff_int = (∫₀¹ visc dζ) · H_ice` via the trapezoidal rule
#     with explicit 2D bed and surface boundary visc inputs
#     (`velocity_ssa.f90:634`, with the Option C end-strip closure).
#     Includes the Fortran `visc_min = 1e5` floor.
#
#   - `stagger_visc_aa_ab!` — average a 2D `visc_eff_int` aa-field to
#     the 4-corner ab-grid (`solver_ssa_ac.f90:1160`). Only ice-covered
#     neighbours (`f_ice == 1`) contribute; partial-ice neighbours are
#     skipped.
#
# All four mirror the Fortran sources. The driver kernels use Yelmo.jl
# face-staggered indexing conventions throughout (Fortran `ux(i, j)` →
# `interior(ux)[i+1, j, 1]`, Fortran `uy(i, j)` → `interior(uy)[i, j+1, 1]`,
# centered ATT/visc → `interior(ATT)[i, j, k]`).
#
# The corner-staggered output of `stagger_visc_aa_ab!` is a
# `Field((Face(), Face(), Center()), g)` with interior shape
# `(Nx+1, Ny+1, 1)`; the face-east-and-north corner of cell `(i, j)`
# (Fortran `visc_ab(i, j)`) lives at array index `[i+1, j+1, 1]`.
#
# Boundary handling: Yelmo.jl uses Oceananigans halo / clamped indices
# (matching Fortran `infinite` BC). Periodic-y support arrives in
# commit 3.
#
# Port of:
#   - `velocity_ssa.f90:337-528 calc_visc_eff_3D_nodes`
#   - `velocity_ssa.f90:530-632 calc_visc_eff_3D_aa`
#   - `velocity_ssa.f90:634-679 calc_visc_eff_int`
#   - `velocity_ssa.f90:679`-:  `calc_basal_stress` is a 1-line kernel
#     `taub_acx = beta_acx * ux_b` — defer to PR-A.2 / PR-B (no scratch
#     needed, but bundling with the SSA solve avoids a small standalone
#     port now).
#   - `solver_ssa_ac.f90:1160 stagger_visc_aa_ab`
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior
using Oceananigans.Grids: topology, Bounded, Periodic, AbstractTopology

export calc_visc_eff_3D_aa!, calc_visc_eff_3D_nodes!, calc_visc_eff_int!,
       stagger_visc_aa_ab!

const _VISC_MIN = 1e5      # [Pa·yr] safety floor (Fortran line 386 / 651)

# --- calc_strain_rate_horizontal_2D port (deformation.f90:1667) ---
#
# Compute aa-cell horizontal strain-rate derivatives `(dudx, dudy,
# dvdx, dvdy)` from acx/acy face velocities. Uses centered second-order
# finite differences in the interior, with one-sided second-order
# corrections at ice margins (`f_ice(i,j) == 1` adjacent to
# `f_ice(neigh) < 1`).
#
# The cross-term margin corrections (dudy / dvdx) are intentionally NOT
# applied — Fortran line 1760-1761: "do not treat cross terms as
# symmetry breaks down. Better to keep it clean."
function _calc_strain_rate_horizontal_2D!(dudx::AbstractArray, dudy::AbstractArray,
                                          dvdx::AbstractArray, dvdy::AbstractArray,
                                          ux_int::AbstractArray, uy_int::AbstractArray,
                                          fi_int::AbstractArray,
                                          dx::Float64, dy::Float64,
                                          Tx::Type{<:AbstractTopology},
                                          Ty::Type{<:AbstractTopology})
    Nx, Ny = size(dudx, 1), size(dudx, 2)

    fill!(dudx, 0.0); fill!(dudy, 0.0)
    fill!(dvdx, 0.0); fill!(dvdy, 0.0)

    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = _neighbor_im1(i, Nx, Tx)
        ip1 = _neighbor_ip1(i, Nx, Tx)
        jm1 = _neighbor_jm1(j, Ny, Ty)
        jp1 = _neighbor_jp1(j, Ny, Ty)

        # Yelmo ux idx for Fortran ux(k, j) at array slot
        # `_ip1_modular(k, Nx, Tx)` (i+1 under Bounded; mod1(i+1, Nx)
        # under Periodic). Same for uy on the y-axis with `_jp1_modular`.
        ux_i   = _ip1_modular(i,   Nx, Tx)
        ux_im1 = _ip1_modular(im1, Nx, Tx)
        ux_ip1 = _ip1_modular(ip1, Nx, Tx)
        uy_j   = _jp1_modular(j,   Ny, Ty)
        uy_jm1 = _jp1_modular(jm1, Ny, Ty)
        uy_jp1 = _jp1_modular(jp1, Ny, Ty)

        # Centered, second-order.
        dudx[i, j] = (ux_int[ux_ip1, j, 1] - ux_int[ux_im1, j, 1]) / (2 * dx)
        dudy[i, j] = (ux_int[ux_i,   jp1, 1] - ux_int[ux_i, jm1, 1]) / (2 * dy)
        dvdx[i, j] = (uy_int[ip1, uy_j, 1]   - uy_int[im1, uy_j, 1]) / (2 * dx)
        dvdy[i, j] = (uy_int[i,   uy_jp1, 1] - uy_int[i,   uy_jm1, 1]) / (2 * dy)

        # Margin one-sided corrections (Fortran line 1716 onward; the
        # `if (.TRUE.) then` block is always active). Margin-guard
        # semantics under Periodic (im1>1 / ip1<Nx tests for one-cell-
        # from-boundary detection) are not reworked here — HOM-C is
        # full-ice and doesn't exercise this branch. Revisit when
        # calving lands.
        # dudx (Fortran line 1721-1739).
        if fi_int[i, j, 1] == 1.0 && fi_int[ip1, j, 1] < 1.0
            if fi_int[im1, j, 1] == 1.0 && im1 > 1
                im2 = im1 - 1
                ux_im2 = _ip1_modular(im2, Nx, Tx)
                dudx[i, j] = (1 * ux_int[ux_im2, j, 1] - 4 * ux_int[ux_im1, j, 1] +
                              3 * ux_int[ux_i,   j, 1]) / (2 * dx)
            else
                dudx[i, j] = (ux_int[ux_i, j, 1] - ux_int[ux_im1, j, 1]) / dx
            end
        elseif fi_int[i, j, 1] < 1.0 && fi_int[ip1, j, 1] == 1.0
            if ip1 < Nx
                ip2 = ip1 + 1
                if fi_int[ip2, j, 1] == 1.0
                    ux_ip2 = _ip1_modular(ip2, Nx, Tx)
                    dudx[i, j] = -(1 * ux_int[ux_ip2, j, 1] - 4 * ux_int[ux_ip1, j, 1] +
                                   3 * ux_int[ux_i,   j, 1]) / (2 * dx)
                else
                    dudx[i, j] = (ux_int[ux_ip1, j, 1] - ux_int[ux_i, j, 1]) / dx
                end
            else
                dudx[i, j] = (ux_int[ux_ip1, j, 1] - ux_int[ux_i, j, 1]) / dx
            end
        end

        # dvdy (Fortran line 1742-1758).
        if fi_int[i, j, 1] == 1.0 && fi_int[i, jp1, 1] < 1.0
            if fi_int[i, jm1, 1] == 1.0 && jm1 > 1
                jm2 = jm1 - 1
                uy_jm2 = _jp1_modular(jm2, Ny, Ty)
                dvdy[i, j] = (1 * uy_int[i, uy_jm2, 1] - 4 * uy_int[i, uy_jm1, 1] +
                              3 * uy_int[i, uy_j,   1]) / (2 * dy)
            else
                dvdy[i, j] = (uy_int[i, uy_j, 1] - uy_int[i, uy_jm1, 1]) / dy
            end
        elseif fi_int[i, j, 1] < 1.0 && fi_int[i, jp1, 1] == 1.0
            if jp1 < Ny
                jp2 = jp1 + 1
                if fi_int[i, jp2, 1] == 1.0
                    uy_jp2 = _jp1_modular(jp2, Ny, Ty)
                    dvdy[i, j] = -(1 * uy_int[i, uy_jp2, 1] - 4 * uy_int[i, uy_jp1, 1] +
                                   3 * uy_int[i, uy_j,   1]) / (2 * dy)
                else
                    dvdy[i, j] = (uy_int[i, uy_jp1, 1] - uy_int[i, uy_j, 1]) / dy
                end
            end
        end
    end
    return nothing
end

"""
    calc_visc_eff_3D_aa!(visc_eff, ux, uy, ATT, H_ice, f_ice,
                         zeta_aa, dx, dy, n_glen, eps_0) -> visc_eff

Glen-flow effective viscosity at aa-cells (Lipscomb 2019 Eq. 2):

    visc = 0.5 · eps_eff_sq^p1 · ATT^p2

with `p1 = (1 - n_glen) / (2 n_glen)`, `p2 = -1 / n_glen`, and the
total effective squared strain rate from L19 Eq. 21:

    eps_eff_sq = dudx² + dvdy² + dudx·dvdy + 0.25·(dudy + dvdx)² + eps_0²

Strain rates are centered finite differences at aa-cells (no margin
correction in this `_aa` variant — Fortran lines 593-603 use plain
`(ux(i,j) - ux(im1,j))/dx` etc.); ATT is read centred at aa-cells.

Cells with `f_ice(i, j) < 1.0` are zeroed (Fortran line 619-624). The
output is initialised to `visc_min = 1e5` then overwritten on iced
cells (Fortran line 583).

Port of `velocity_ssa.f90:530 calc_visc_eff_3D_aa`. The Fortran
`zeta_aa` and `boundaries` arguments are kept in the signature for
parity but are unused in this variant.
"""
function calc_visc_eff_3D_aa!(visc_eff, ux, uy, ATT, H_ice, f_ice,
                              zeta_aa::AbstractVector{<:Real},
                              dx::Real, dy::Real,
                              n_glen::Real, eps_0::Real)
    V       = interior(visc_eff)
    ux_int  = interior(ux)
    uy_int  = interior(uy)
    ATT_int = interior(ATT)
    fi_int  = interior(f_ice)

    Nx, Ny = size(V, 1), size(V, 2)
    Nz     = size(V, 3)
    Nz == length(zeta_aa) || error(
        "calc_visc_eff_3D_aa!: visc_eff has Nz=$(Nz) but zeta_aa has length $(length(zeta_aa))")

    # Topology-dispatched neighbour helpers — under Periodic the i±1 /
    # j±1 indices wrap modularly; under Bounded they clamp. Mirrors the
    # Fortran `get_neighbor_indices_bc_codes` (`yelmo_tools.f90`) which
    # dispatches on BC. The Yelmo face-array slot `[i+1, j, 1]` for
    # Fortran `ux(i, j)` similarly wraps under Periodic via
    # `_ip1_modular`.
    Tx_top = topology(visc_eff.grid, 1)
    Ty_top = topology(visc_eff.grid, 2)

    p1 = (1.0 - Float64(n_glen)) / (2.0 * Float64(n_glen))
    p2 = -1.0 / Float64(n_glen)
    eps_0_sq = Float64(eps_0)^2

    fill!(V, _VISC_MIN)

    @inbounds for j in 1:Ny, i in 1:Nx
        if fi_int[i, j, 1] == 1.0
            im1 = _neighbor_im1(i, Nx, Tx_top)
            ip1 = _neighbor_ip1(i, Nx, Tx_top)
            jm1 = _neighbor_jm1(j, Ny, Ty_top)
            jp1 = _neighbor_jp1(j, Ny, Ty_top)

            ux_i   = _ip1_modular(i,   Nx, Tx_top)
            ux_im1 = _ip1_modular(im1, Nx, Tx_top)
            uy_j   = _jp1_modular(j,   Ny, Ty_top)
            uy_jm1 = _jp1_modular(jm1, Ny, Ty_top)

            # Fortran lines 594-603. Face-slot indices are `_ip1_modular` /
            # `_jp1_modular`, which under Bounded resolve to k+1 (the Yelmo
            # face-array convention) and under Periodic wrap modularly.
            dudx_aa = (ux_int[ux_i, j, 1] - ux_int[ux_im1, j, 1]) / Float64(dx)
            dvdy_aa = (uy_int[i, uy_j, 1] - uy_int[i, uy_jm1, 1]) / Float64(dy)

            dudy_aa_1 = (ux_int[ux_i,   jp1, 1] - ux_int[ux_i,   jm1, 1]) / (2 * Float64(dy))
            dudy_aa_2 = (ux_int[ux_im1, jp1, 1] - ux_int[ux_im1, jm1, 1]) / (2 * Float64(dy))
            dudy_aa   = 0.5 * (dudy_aa_1 + dudy_aa_2)

            dvdx_aa_1 = (uy_int[ip1, uy_j,   1] - uy_int[im1, uy_j,   1]) / (2 * Float64(dx))
            dvdx_aa_2 = (uy_int[ip1, uy_jm1, 1] - uy_int[im1, uy_jm1, 1]) / (2 * Float64(dx))
            dvdx_aa   = 0.5 * (dvdx_aa_1 + dvdx_aa_2)

            eps_sq_aa = dudx_aa^2 + dvdy_aa^2 + dudx_aa * dvdy_aa +
                        0.25 * (dudy_aa + dvdx_aa)^2 + eps_0_sq

            for k in 1:Nz
                ATT_aa = ATT_int[i, j, k]
                V[i, j, k] = 0.5 * eps_sq_aa^p1 * ATT_aa^p2
            end
        else
            # Ice-free cell: Fortran line 619-624 sets visc to 0
            # explicitly (overrides the visc_min initial value).
            for k in 1:Nz
                V[i, j, k] = 0.0
            end
        end
    end
    return visc_eff
end

"""
    calc_visc_eff_3D_nodes!(visc_eff, ux, uy, ATT, H_ice, f_ice,
                            zeta_aa, dx, dy, n_glen, eps_0) -> visc_eff

Glen-flow effective viscosity via Gauss quadrature on aa-cells. Same
formula as `calc_visc_eff_3D_aa!` but the strain rates and ATT are
bilinearly interpolated to 4 Gauss nodes per cell, then the viscosity
is averaged across the nodes:

    eps_sq_n   = dudx_n² + dvdy_n² + dudx_n·dvdy_n + 0.25·(dudy_n + dvdx_n)² + eps_0²
    visc_n     = 0.5 · eps_sq_n^p1 · ATT_n^p2
    visc(i, j) = sum_p (visc_n(p) · wt(p)) / wt_tot

The 4 corner-staggered values used as input to the bilinear interpolation
follow the Fortran `gq2D_to_nodes_*` recipes:

  - `gq2D_to_nodes_acx(dudx, ...)`: corner = mean of the 2 acx-faces
    east of `(im1, jm1)`–`(im1, j)` etc.
  - `gq2D_to_nodes_acy(dvdy, ...)`: corner = mean of the 2 acy-faces
    north of `(im1, jm1)`–`(i, jm1)` etc.
  - `gq2D_to_nodes_aa(ATT, ...)`: corner = 4-cell mean of ATT.

The strain-rate inputs `dudx`, `dudy`, `dvdx`, `dvdy` are computed
once per call via the upstream-aware `_calc_strain_rate_horizontal_2D!`
(`deformation.f90:1667`) and stored in scratch matrices on the aa-grid.
Note the Fortran convention: `dudx`, `dudy`, `dvdx`, `dvdy` are all on
**aa-nodes** (computed at cell centres). The Fortran call is then
`gq2D_to_nodes_acx(gq2D, dudxn, dudx(:,:,1), …)` — the variable lives
on aa, but the Fortran `_acx` corner-staggering recipe is applied to
it. We reproduce that exact recipe here.

Port of `velocity_ssa.f90:337 calc_visc_eff_3D_nodes` (use_gq3D=false
branch — 2D Gauss quadrature only; the 3D variant is hard-coded off in
Fortran via `logical, parameter :: use_gq3D = .FALSE.`).
"""
function calc_visc_eff_3D_nodes!(visc_eff, ux, uy, ATT, H_ice, f_ice,
                                 zeta_aa::AbstractVector{<:Real},
                                 dx::Real, dy::Real,
                                 n_glen::Real, eps_0::Real)
    V       = interior(visc_eff)
    ux_int  = interior(ux)
    uy_int  = interior(uy)
    ATT_int = interior(ATT)
    fi_int  = interior(f_ice)

    Nx, Ny = size(V, 1), size(V, 2)
    Nz     = size(V, 3)
    Nz == length(zeta_aa) || error(
        "calc_visc_eff_3D_nodes!: visc_eff has Nz=$(Nz) but zeta_aa has length $(length(zeta_aa))")

    # Topology-dispatched neighbour helpers — under Periodic the i±1 /
    # j±1 indices wrap modularly; under Bounded they clamp. Mirrors the
    # Fortran `get_neighbor_indices_bc_codes` (`yelmo_tools.f90`).
    Tx_top = topology(visc_eff.grid, 1)
    Ty_top = topology(visc_eff.grid, 2)

    p1 = (1.0 - Float64(n_glen)) / (2.0 * Float64(n_glen))
    p2 = -1.0 / Float64(n_glen)
    eps_0_sq = Float64(eps_0)^2

    # Fortran lines 421-429: compute strain rates once per call (the
    # `_nodes` variant uses the same dudx/dudy/dvdx/dvdy across all
    # k-layers because horizontal velocity is depth-averaged here).
    dudx = Matrix{Float64}(undef, Nx, Ny)
    dudy = Matrix{Float64}(undef, Nx, Ny)
    dvdx = Matrix{Float64}(undef, Nx, Ny)
    dvdy = Matrix{Float64}(undef, Nx, Ny)
    _calc_strain_rate_horizontal_2D!(dudx, dudy, dvdx, dvdy,
                                     ux_int, uy_int, fi_int,
                                     Float64(dx), Float64(dy),
                                     Tx_top, Ty_top)

    xr, yr, wt, wt_tot = gq2d_nodes(2)
    fill!(V, _VISC_MIN)

    @inbounds for j in 1:Ny, i in 1:Nx
        if fi_int[i, j, 1] == 1.0
            im1 = _neighbor_im1(i, Nx, Tx_top)
            ip1 = _neighbor_ip1(i, Nx, Tx_top)
            jm1 = _neighbor_jm1(j, Ny, Ty_top)
            jp1 = _neighbor_jp1(j, Ny, Ty_top)

            # Corner-staggering: the 4 Gauss nodes interpolate from
            # corner values, computed via the relevant Fortran recipe.
            # `dudx`, `dudy` are inputs to `gq2D_to_nodes_acx` (Fortran
            # line 455-456). `dvdx`, `dvdy` use the `_acy` recipe
            # (Fortran line 458-459).
            #
            # The `_acx` corner staggering (Fortran line 405-409):
            #   v_ab(SW) = 0.5 * (var(im1, jm1) + var(im1, j))
            #   v_ab(SE) = 0.5 * (var(i,   jm1) + var(i,   j))
            #   v_ab(NE) = 0.5 * (var(i,   j  ) + var(i,   jp1))
            #   v_ab(NW) = 0.5 * (var(im1, j  ) + var(im1, jp1))
            v_ab_dudx = (
                0.5 * (dudx[im1, jm1] + dudx[im1, j]),
                0.5 * (dudx[i,   jm1] + dudx[i,   j]),
                0.5 * (dudx[i,   j]   + dudx[i,   jp1]),
                0.5 * (dudx[im1, j]   + dudx[im1, jp1]),
            )
            v_ab_dudy = (
                0.5 * (dudy[im1, jm1] + dudy[im1, j]),
                0.5 * (dudy[i,   jm1] + dudy[i,   j]),
                0.5 * (dudy[i,   j]   + dudy[i,   jp1]),
                0.5 * (dudy[im1, j]   + dudy[im1, jp1]),
            )
            # The `_acy` corner staggering (Fortran line 433-437):
            #   v_ab(SW) = 0.5 * (var(im1, jm1) + var(i, jm1))
            #   v_ab(SE) = 0.5 * (var(i,   jm1) + var(ip1, jm1))
            #   v_ab(NE) = 0.5 * (var(i,   j)   + var(ip1, j))
            #   v_ab(NW) = 0.5 * (var(im1, j)   + var(i,   j))
            v_ab_dvdx = (
                0.5 * (dvdx[im1, jm1] + dvdx[i,   jm1]),
                0.5 * (dvdx[i,   jm1] + dvdx[ip1, jm1]),
                0.5 * (dvdx[i,   j]   + dvdx[ip1, j]),
                0.5 * (dvdx[im1, j]   + dvdx[i,   j]),
            )
            v_ab_dvdy = (
                0.5 * (dvdy[im1, jm1] + dvdy[i,   jm1]),
                0.5 * (dvdy[i,   jm1] + dvdy[ip1, jm1]),
                0.5 * (dvdy[i,   j]   + dvdy[ip1, j]),
                0.5 * (dvdy[im1, j]   + dvdy[i,   j]),
            )

            # Bilinear interpolation to the 4 Gauss nodes.
            dudxn = ntuple(p -> gq2d_interp_to_node(v_ab_dudx, xr[p], yr[p]), 4)
            dudyn = ntuple(p -> gq2d_interp_to_node(v_ab_dudy, xr[p], yr[p]), 4)
            dvdxn = ntuple(p -> gq2d_interp_to_node(v_ab_dvdx, xr[p], yr[p]), 4)
            dvdyn = ntuple(p -> gq2d_interp_to_node(v_ab_dvdy, xr[p], yr[p]), 4)

            # Effective squared strain rate at each node (Fortran line 462).
            eps_sq_n = ntuple(p ->
                dudxn[p]^2 + dvdyn[p]^2 + dudxn[p] * dvdyn[p] +
                0.25 * (dudyn[p] + dvdxn[p])^2 + eps_0_sq, 4)

            # Loop over layers — ATT can vary in z (Fortran line 464-474).
            for k in 1:Nz
                # ATT corner staggering (`gq2D_to_nodes_aa`, Fortran
                # line 350-353):
                #   v_ab(SW) = 0.25 * (var(im1, jm1) + var(i, jm1) +
                #                       var(im1, j) + var(i, j))
                v_ab_ATT = (
                    0.25 * (ATT_int[im1, jm1, k] + ATT_int[i,   jm1, k] +
                            ATT_int[im1, j,   k] + ATT_int[i,   j,   k]),
                    0.25 * (ATT_int[i,   jm1, k] + ATT_int[ip1, jm1, k] +
                            ATT_int[i,   j,   k] + ATT_int[ip1, j,   k]),
                    0.25 * (ATT_int[i,   j,   k] + ATT_int[ip1, j,   k] +
                            ATT_int[i,   jp1, k] + ATT_int[ip1, jp1, k]),
                    0.25 * (ATT_int[im1, j,   k] + ATT_int[i,   j,   k] +
                            ATT_int[im1, jp1, k] + ATT_int[i,   jp1, k]),
                )
                ATTn = ntuple(p -> gq2d_interp_to_node(v_ab_ATT, xr[p], yr[p]), 4)

                # Fortran line 471: viscn = 0.5 * eps_sq_n^p1 * ATTn^p2.
                # Fortran line 472: visc(i,j,k) = sum(viscn * wt) / wt_tot.
                acc = 0.0
                for p in 1:4
                    viscn = 0.5 * eps_sq_n[p]^p1 * ATTn[p]^p2
                    acc += viscn * wt[p]
                end
                V[i, j, k] = acc / wt_tot
            end
        end
        # else: leave at the visc_min initial value — the Fortran
        # `_nodes` variant doesn't override visc with 0 on ice-free
        # cells (unlike `_aa`); see Fortran line 433 vs 619-624.
    end
    return visc_eff
end

"""
    calc_visc_eff_int!(visc_eff_int, visc_eff, visc_eff_b, visc_eff_s,
                       H_ice, f_ice, zeta_aa) -> visc_eff_int

Depth-integrate the 3D effective viscosity to a 2D `visc_eff_int`
field via the trapezoidal rule with explicit bed and surface boundary
values, scaled by ice thickness:

    visc_eff_int(i, j) = (∫₀¹ visc(ζ) dζ) · H_ice(i, j)

For ice-free cells (`f_ice < 1`), `visc_eff_int` is set to 0 (Fortran
line 666). A `visc_min = 1e5` floor is then applied (Fortran line 670)
— so partially-iced cells with `H_ice > 0` end up at `visc_min` rather
than `(∫ visc dζ) · H_ice`.

Vertical convention (Option C): `visc_eff` is a 3D `Center()`-staggered
field whose `Nz` layers do NOT include the bed (zeta = 0) or surface
(zeta = 1) endpoints. The integration thus consumes two 2D boundary
fields — `visc_eff_b` at the bed and `visc_eff_s` at the surface — to
cover the full [0, 1] interval. The shared `vert_int_trapz_boundary!`
helper performs the trapezoidal sum.

Boundary visc handling at SSA call sites:

  - `visc_method = 0` (constant): caller fills the boundary fields with
    `visc_const`.
  - `visc_method = 1, 2` (Glen-flow): caller approximates the boundary
    visc by the nearest-Center value (`visc_eff_b ≈ visc_eff[:, :, 1]`,
    `visc_eff_s ≈ visc_eff[:, :, end]`). Exact for isothermal (uniform
    ATT) cases; approximate for temperature-dependent ATT — matches the
    existing SIA convention. Revisit when therm wires
    temperature-dependent ATT (milestone 3g).

Port of `velocity_ssa.f90:634 calc_visc_eff_int`. The Fortran
`integrate_trapezoid1D_pt(visc_eff(i, j, :), zeta_aa)` is the centre-
only quadrature; the Option C extension here closes the bed and
surface end-strips that the centre-only scheme silently drops (e.g.
0.75 × correct for Nz = 4 with uniform spacing).
"""
function calc_visc_eff_int!(visc_eff_int, visc_eff,
                            visc_eff_b, visc_eff_s,
                            H_ice, f_ice,
                            zeta_aa::AbstractVector{<:Real})
    Vi = interior(visc_eff_int)
    V  = interior(visc_eff)
    Vb = interior(visc_eff_b)
    Vs = interior(visc_eff_s)
    H  = interior(H_ice)
    fi = interior(f_ice)

    Nx, Ny = size(Vi, 1), size(Vi, 2)
    Nz     = size(V, 3)
    Nz == length(zeta_aa) || error(
        "calc_visc_eff_int!: visc_eff has Nz=$(Nz) but zeta_aa has length $(length(zeta_aa))")

    # Pure ∫₀¹ visc dζ via the shared trapezoidal-with-boundary helper.
    # Output is dimensionless ∫ over zeta — units of visc · dζ = Pa·yr.
    # The H multiplication and ice-mask / floor are applied below.
    vert_int_trapz_boundary!(Vi, V, Vb, Vs, zeta_aa)

    @inbounds for j in 1:Ny, i in 1:Nx
        if fi[i, j, 1] == 1.0
            Vi[i, j, 1] = Vi[i, j, 1] * H[i, j, 1]
        else
            # Fortran line 666: visc_eff_int = 0 on ice-free cells.
            Vi[i, j, 1] = 0.0
        end

        # Fortran line 670: visc_min floor.
        if Vi[i, j, 1] < _VISC_MIN
            Vi[i, j, 1] = _VISC_MIN
        end
    end
    return visc_eff_int
end

"""
    stagger_visc_aa_ab!(visc_ab, visc, H_ice, f_ice) -> visc_ab

Average a 2D aa-cell viscosity field `visc` to the 4-corner ab-grid.
Only ice-covered neighbours (`f_ice == 1`) contribute; partial- or
no-ice neighbours are skipped. If no neighbours are ice-covered, the
corner value stays at 0.

Port of `solver_ssa_ac.f90:1160 stagger_visc_aa_ab`.

Indexing: `visc_ab` is a `Field((Face(), Face(), Center()), g)` with
interior shape `(Nx+1, Ny+1, 1)`. The corner east-and-north of cell
`(i, j)` (Fortran `visc_ab(i, j)`) is at array index `[i+1, j+1, 1]`.

`H_ice` is unused in the kernel body but kept for signature parity
(Fortran line 1167) — future work that gates corner contribution on
H_ice may consume it.
"""
function stagger_visc_aa_ab!(visc_ab, visc, H_ice, f_ice)
    Vab = interior(visc_ab)
    V   = interior(visc)
    fi  = interior(f_ice)

    Nx, Ny = size(V, 1), size(V, 2)
    fill!(Vab, 0.0)

    Tx_top = topology(visc_ab.grid, 1)
    Ty_top = topology(visc_ab.grid, 2)

    @inbounds for j in 1:Ny, i in 1:Nx
        ip1 = _neighbor_ip1(i, Nx, Tx_top)
        jp1 = _neighbor_jp1(j, Ny, Ty_top)
        ip1f = _ip1_modular(i, Nx, Tx_top)
        jp1f = _jp1_modular(j, Ny, Ty_top)

        acc = 0.0
        k_count = 0
        if fi[i, j, 1] == 1.0
            acc += V[i, j, 1]; k_count += 1
        end
        if fi[ip1, j, 1] == 1.0
            acc += V[ip1, j, 1]; k_count += 1
        end
        if fi[i, jp1, 1] == 1.0
            acc += V[i, jp1, 1]; k_count += 1
        end
        if fi[ip1, jp1, 1] == 1.0
            acc += V[ip1, jp1, 1]; k_count += 1
        end

        if k_count > 0
            Vab[ip1f, jp1f, 1] = acc / k_count
        end
    end
    return visc_ab
end
