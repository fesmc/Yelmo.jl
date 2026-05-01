# Topography API

The pure-Julia topography component (`y.tpo`). Lives in
`Yelmo.YelmoModelTopo` and is re-exported at the package level. See
the [topography step page](../physics/topography.md) for the full
narrative pipeline; this page is the API reference.

## Per-step orchestrator and diagnostics refresh

```@docs
topo_step!
update_diagnostics!
```

## Advection

```@docs
advect_tracer!
```

See the [advection page](../physics/advection.md) for the math.

## Mass-balance helpers

```@docs
apply_tendency!
mbal_tendency!
resid_tendency!
```

See the [mass balance page](../physics/mass-balance.md) for the
narrative.

## Source terms

```@docs
calc_bmb_total!
calc_fmb_total!
calc_mb_discharge!
```

## Ice-area fraction

```@docs
calc_f_ice!
```

## Grounding diagnostics

```@docs
calc_H_grnd!
determine_grounded_fractions!
calc_grounded_fractions!
calc_f_grnd_subgrid_linear!
calc_f_grnd_subgrid_area!
calc_f_grnd_pinning_points!
```

The `calc_grounded_fractions!` umbrella dispatches between the linear
(`gl_sep == 1`), area (`gl_sep == 2`), and CISM-quad
(`gl_sep == 3`) subgrid schemes — see the
[grounded fraction page](../physics/grounded-fraction.md) for the
algorithm derivation and analytical-limit fixes used by the CISM
variant.

## Surface and gradient diagnostics

```@docs
calc_z_srf!
calc_gradient_acx!
calc_gradient_acy!
```

## Distance-to-feature fields

```@docs
calc_distance_to_grounding_line!
calc_distance_to_ice_margin!
```

## Bed and front masks

```@docs
calc_grounding_line_zone!
gen_mask_bed!
calc_ice_front!
```

## Dynamic-thickness helpers

The dynamics solver reads `tpo.H_ice_dyn` and `tpo.f_ice_dyn`, which
extend a thin floating slab under the ice front so the SIA pressure
gradient does not collapse to zero one cell upstream of the calving
front.

```@docs
extend_floating_slab!
calc_dynamic_ice_fields!
```

## Relaxation

```@docs
set_tau_relax!
calc_G_relaxation!
```

See the [relaxation page](../physics/relaxation.md) for the modes.

## Calving

```@docs
calving_step!
calc_calving_equil_ac!
calc_calving_threshold_ac!
calc_calving_vonmises_m16_ac!
merge_calving_rates!
```

## Level-set front

```@docs
lsf_init!
lsf_update!
lsf_redistance!
extrapolate_ocn_acx!
extrapolate_ocn_acy!
```

See the [calving page](../physics/calving.md) for the LSF flux
method, redistancing PDE, and the front-velocity merge rule.
