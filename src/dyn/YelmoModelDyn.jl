"""
    YelmoModelDyn

Dynamics (`dyn`) component for the pure-Julia `YelmoModel`. Computes
the horizontal velocity solution from the current `tpo`/`mat`/`thrm`/
`bnd` state, plus all derived stress and velocity-magnitude
diagnostics.

Public surface: `dyn_step!(y::YelmoModel, dt)`, dispatched from
`YelmoCore.step!(::YelmoModel, dt)` after `topo_step!` in the fixed
phase order.

Milestone 3a (current): scaffolding + pre-solver kinematics
(driving stress, lateral BC stress) + post-solver underflow / flux
/ magnitude / surface-slice / `duxydt` diagnostics. Solver dispatch
only handles `solver = "fixed"` (no velocity update); SIA / SSA /
hybrid / DIVA arrive in subsequent milestones (3c–3f). Velocity
Jacobian, vertical velocity `uz`, and strain-rate tensor land in
milestone 3h.
"""
module YelmoModelDyn

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.BoundaryConditions: fill_halo_regions!

using ..YelmoCore: AbstractYelmoModel, YelmoModel

import ..YelmoCore: dyn_step!

export dyn_step!

"""
    dyn_step!(y::YelmoModel, dt) -> y

Advance the dynamics state by `dt` years. Phase order matches
Fortran's `calc_ydyn` body (`yelmo_dynamics.f90:48`):

1. Snapshot `(ux_bar, uy_bar)` into `(ux_bar_prev, uy_bar_prev)`,
   used by higher-order topo timestepping and by the `duxydt`
   diagnostic.
2. Driving stress `taud_acx`/`taud_acy` from surface slope and ice
   thickness (milestone 3a, deferred).
3. Optional grounding-line driving-stress modifier when
   `taud_gl_method != 0` (milestone 3a, deferred).
4. Lateral boundary stress `taul_int_acx`/`taul_int_acy` at the ice
   front (milestone 3a, deferred).
5. Effective pressure `N_eff` and bed-roughness chain `cb_tgt →
   cb_ref → c_bed` (milestone 3b, deferred).
6. Solver dispatch on `y.p.ydyn.solver`. Currently only `"fixed"`
   (no-op) is implemented; other solvers error with a milestone
   pointer.
7. Underflow clip on `ux`/`uy`/`ux_bar`/`uy_bar` (milestone 3a,
   deferred).
8. Velocity Jacobian + vertical velocity `uz` + strain-rate tensor
   (milestone 3h, deferred).
9. Diagnostics: ice flux, stress and velocity magnitudes, surface /
   basal velocity slices, basal-to-surface ratio `f_vbvs`, time
   derivative `duxydt` (milestone 3a, deferred).

`dyn_step!` does NOT advance `y.time` — that is owned by `topo_step!`
which runs first.
"""
function dyn_step!(y::YelmoModel, dt::Float64)
    # 1. Snapshot prev depth-averaged velocity (used by topo PC and
    #    by the `duxydt` diagnostic in step 9).
    interior(y.dyn.ux_bar_prev) .= interior(y.dyn.ux_bar)
    interior(y.dyn.uy_bar_prev) .= interior(y.dyn.uy_bar)

    # 6. Solver dispatch. 3a only handles "fixed".
    solver = y.p.ydyn.solver
    if solver == "fixed"
        # No velocity update.
    else
        error("dyn_step!: solver=\"$solver\" not yet ported. " *
              "Milestone 3a only supports \"fixed\". " *
              "SIA / SSA / hybrid / DIVA land in milestones 3c–3f.")
    end

    return y
end

end # module YelmoModelDyn
