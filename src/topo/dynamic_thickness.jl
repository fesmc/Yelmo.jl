# ----------------------------------------------------------------------
# Ice-thickness fields exclusively for the dynamics solver.
#
# `tpo.H_ice_dyn` / `tpo.f_ice_dyn` are alternative thickness / cover
# fields handed to the SSA / DIVA / L1L2 dynamics step. They differ
# from `H_ice` / `f_ice` only when `ydyn.ssa_lat_bc` selects a non-
# default lateral-BC mode that wants the dynamic margin to look
# different from the prognostic margin (e.g. "slab" or "slab-ext").
#
#   - "slab":     `H_ice_dyn = H_ice` everywhere, with `H_ice_dyn = 1`
#                 over `f_ice < 1` cells. Dynamic ice fraction is the
#                 binary cover of `H_ice_dyn`.
#   - "slab-ext": same as "slab" plus a `n_ext`-cell extension of a
#                 thin slab outward from grounded marine margins (see
#                 `extend_floating_slab!`).
#   - default:    pass-through, `H_ice_dyn ≡ H_ice`, `f_ice_dyn ≡ f_ice`.
#
# Port of the trailing `select case(trim(dyn%par%ssa_lat_bc))` block
# in `yelmo_topography.f90:calc_ytopo_diagnostic` (lines 1022–1058)
# and `extend_floating_slab` in
# `physics/mass_conservation.f90:965`.
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior, CenterField
using Oceananigans.BoundaryConditions: fill_halo_regions!

export extend_floating_slab!, calc_dynamic_ice_fields!

"""
    extend_floating_slab!(H_ice, f_grnd; H_slab=1.0, n_ext=4) -> H_ice

Iteratively grow a thin floating "slab" of thickness `H_slab` (m)
outward from grounded marine margins by `n_ext` cells. Each iteration
visits every floating ice-free cell (`f_grnd == 0` and `H_ice == 0`)
and sets `H_ice = H_slab` if any direct neighbour is either a
grounded ice point or itself a slab cell from a previous iteration.

Used to seed a thin shelf in front of land-terminating glaciers so
the SSA solver has a non-zero ice cover to integrate the lateral BC
against. `H_slab = 1.0 m` is the Fortran default; smaller values
produce a less-perturbative seed.

Halo handling: `H_ice` and `f_grnd` halos are filled via
`fill_halo_regions!` once per iteration so neighbour reads honour
the grid topology + BCs.

Port of `physics/mass_conservation.f90:965 extend_floating_slab`.
"""
function extend_floating_slab!(H_ice, f_grnd;
                                H_slab::Real = 1.0,
                                n_ext::Integer = 4)
    n_ext > 0 || return H_ice

    Hi = interior(H_ice)
    Fg = interior(f_grnd)
    nx, ny = size(Hi, 1), size(Hi, 2)

    # The "is this cell already a slab cell?" mask. Seeded false; a
    # cell joins the slab the first iteration its neighbour test fires
    # and stays in the mask thereafter (so the front advances by one
    # ring per iteration).
    mask_slab = falses(nx, ny)
    H_slab_f = Float64(H_slab)

    for _ in 1:n_ext
        fill_halo_regions!(H_ice)
        fill_halo_regions!(f_grnd)

        # Snapshot of the current state; modifications happen on a
        # working copy and are applied at the end of the sweep.
        H_new = copy(Hi)

        @inbounds for j in 1:ny, i in 1:nx
            (Fg[i, j, 1] == 0.0 && Hi[i, j, 1] == 0.0) || continue

            grnd_W = (f_grnd[i-1, j, 1] > 0.0) && (H_ice[i-1, j, 1] > 0.0)
            grnd_E = (f_grnd[i+1, j, 1] > 0.0) && (H_ice[i+1, j, 1] > 0.0)
            grnd_S = (f_grnd[i,   j-1, 1] > 0.0) && (H_ice[i,   j-1, 1] > 0.0)
            grnd_N = (f_grnd[i,   j+1, 1] > 0.0) && (H_ice[i,   j+1, 1] > 0.0)

            slab_W = (i > 1)  ? mask_slab[i-1, j  ] : false
            slab_E = (i < nx) ? mask_slab[i+1, j  ] : false
            slab_S = (j > 1)  ? mask_slab[i,   j-1] : false
            slab_N = (j < ny) ? mask_slab[i,   j+1] : false

            if grnd_W | grnd_E | grnd_S | grnd_N |
               slab_W | slab_E | slab_S | slab_N
                H_new[i, j, 1] = H_slab_f
            end
        end

        # Apply the sweep + refresh the slab mask in lockstep with
        # the Fortran reference.
        Hi .= H_new
        @inbounds for j in 1:ny, i in 1:nx
            mask_slab[i, j] = Hi[i, j, 1] == H_slab_f
        end
    end

    return H_ice
end

"""
    calc_dynamic_ice_fields!(H_ice_dyn, f_ice_dyn,
                             H_ice, f_ice, f_grnd,
                             z_bed, z_sl, rho_ice, rho_sw, ssa_lat_bc;
                             flt_subgrid=false,
                             H_slab=1.0, n_ext=4) -> H_ice_dyn

Build the dynamics-only thickness `H_ice_dyn` and cover `f_ice_dyn`
fields from the prognostic state, dispatched on the `ssa_lat_bc`
parameter:

  - `"slab"`     — `H_ice_dyn = H_ice`, with `H_ice_dyn = 1.0` over
    cells where `f_ice < 1` (so the SSA solver sees a fully-covered
    margin). `f_ice_dyn` is recomputed binary from `H_ice_dyn`.
  - `"slab-ext"` — same as `"slab"`, plus a `n_ext`-cell
    `extend_floating_slab!` extension of a thin slab outward from
    grounded marine margins.
  - everything else (e.g. `"floating"`, the default) — pass-through:
    `H_ice_dyn = H_ice` and `f_ice_dyn = f_ice`.

Port of the `select case(trim(dyn%par%ssa_lat_bc))` block at
`yelmo_topography.f90:1022-1058`.
"""
function calc_dynamic_ice_fields!(H_ice_dyn, f_ice_dyn,
                                  H_ice, f_ice, f_grnd,
                                  z_bed, z_sl,
                                  rho_ice::Real, rho_sw::Real,
                                  ssa_lat_bc::AbstractString;
                                  flt_subgrid::Bool = false,
                                  H_slab::Real = 1.0,
                                  n_ext::Integer = 4)
    Hd = interior(H_ice_dyn)
    Fd = interior(f_ice_dyn)
    H  = interior(H_ice)
    F  = interior(f_ice)

    if ssa_lat_bc == "slab"
        # H_ice_dyn = H_ice, then bump f_ice<1 cells up to H=1.
        Hd .= H
        @inbounds for j in axes(Hd, 2), i in axes(Hd, 1)
            F[i, j, 1] < 1.0 && (Hd[i, j, 1] = 1.0)
        end
        # f_ice_dyn from H_ice_dyn (binary).
        calc_f_ice!(f_ice_dyn, H_ice_dyn, z_bed, z_sl, rho_ice, rho_sw;
                    flt_subgrid = flt_subgrid)

    elseif ssa_lat_bc == "slab-ext"
        # Start the same as "slab" — bump partial-cover cells up.
        Hd .= H
        @inbounds for j in axes(Hd, 2), i in axes(Hd, 1)
            if H[i, j, 1] > 0.0 && H[i, j, 1] < 1.0
                Hd[i, j, 1] = 1.0
            end
        end
        # …then extend the slab outward.
        extend_floating_slab!(H_ice_dyn, f_grnd;
                              H_slab = H_slab, n_ext = n_ext)
        calc_f_ice!(f_ice_dyn, H_ice_dyn, z_bed, z_sl, rho_ice, rho_sw;
                    flt_subgrid = flt_subgrid)

    else
        # Default: pass-through.
        Hd .= H
        Fd .= F
    end

    return H_ice_dyn
end

# Convenience dispatch over `YelmoModel`. Threads the parameters
# through the same way the Fortran caller (calc_ytopo_diagnostic)
# does — `flt_subgrid` defaults to false here because the Fortran
# call passes `flt_subgrid=.FALSE.` explicitly for the dynamics-only
# fields.
calc_dynamic_ice_fields!(y::YelmoModel) =
    calc_dynamic_ice_fields!(y.tpo.H_ice_dyn, y.tpo.f_ice_dyn,
                             y.tpo.H_ice, y.tpo.f_ice, y.tpo.f_grnd,
                             y.bnd.z_bed, y.bnd.z_sl,
                             y.c.rho_ice, y.c.rho_sw,
                             y.p.ydyn.ssa_lat_bc;
                             flt_subgrid = false)
