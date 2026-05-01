# Dynamics API

The pure-Julia dynamics component (`y.dyn`). Lives in
`Yelmo.YelmoModelDyn` and is re-exported at the package level. See
the [dynamics page](../physics/dynamics.md) for the narrative
description of the SIA solver, basal-friction chain, and lateral
boundary stress.

The dynamics step is invoked from `step!(y::YelmoModel, dt)` after
[`topo_step!`](@ref). Currently supports `solver = "fixed"` (no
velocity update) and `solver = "sia"` (Option C SIA wrapper); SSA /
hybrid / DIVA solvers land in subsequent milestones.

## Per-step orchestrator

```@docs
dyn_step!
```

## Driving stress

```@docs
calc_driving_stress!
calc_driving_stress_gl!
```

## Lateral boundary stress

```@docs
calc_lateral_bc_stress_2D!
```

## Effective basal pressure

```@docs
calc_ydyn_neff!
```

The kernel dispatches on `y.p.yneff.method` ∈ `{-1, 0, 1, 2, 3, 4, 5}`
— see the docstring for the per-method formulas. Subgrid sampling
(`yneff.nxi > 0`) is not yet ported.

## Basal-roughness chain

```@docs
calc_cb_ref!
calc_c_bed!
```

## SIA velocity solver

The Option C SIA wrapper `Yelmo.YelmoModelDyn.calc_velocity_sia!`
fills halos, computes the SIA shear stresses via
[`calc_shear_stress_3D!`](@ref), runs the depth-recurrence
[`calc_uxy_sia_3D!`](@ref), and adds the surface segment integral to
produce both the 3D velocity and the 2D depth-average and surface
boundary values. See the [dynamics page](../physics/dynamics.md) for
the algorithm and the Option C convention.

The wrapper is unexported at the package level (use the qualified
form `Yelmo.YelmoModelDyn.calc_velocity_sia!` if you need to call it
directly); the two underlying kernels are exported:

```@docs
calc_shear_stress_3D!
calc_uxy_sia_3D!
```

## Diagnostics

```@docs
calc_ice_flux!
calc_magnitude_from_staggered!
calc_vel_ratio!
```
