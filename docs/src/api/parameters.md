# Parameters API

`Yelmo.jl` has two parallel parameter trees, one per backend:

| Backend | Type | Module | Constructor / loader |
|---|---|---|---|
| [`YelmoMirror`](mirror.md) | [`YelmoMirrorParameters`](@ref) | `Yelmo.YelmoMirrorPar` | `YelmoMirrorParameters("name"; ...)`, `YelmoMirrorPar.read_nml(filename)` |
| [`YelmoModel`](core.md)    | [`YelmoParameters`](@ref) | `Yelmo.YelmoPar` | `YelmoParameters("name"; ...)`, `YelmoPar.read_nml(filename)` |

The two are structurally identical at the time of writing; they live
in separate modules so the Julia model can grow parameters that the
Fortran-backed mirror does not need (and vice versa) without breaking
either side. The shared functions [`write_nml`](@ref) and
[`compare`](@ref) dispatch on the type so user code can stay
backend-agnostic.

`read_nml` is *not* a generic function — its signature is
`read_nml(::AbstractString)` and the return type differs between the
two modules, so it cannot be a single generic. Call as
`Yelmo.YelmoMirrorPar.read_nml(...)` or `Yelmo.YelmoPar.read_nml(...)`
to disambiguate.

## YelmoModel parameters

```@docs
YelmoParameters
```

The constructor takes one keyword argument per namelist group; each
keyword is a `<group>_params(...)` factory call (or any value of the
matching `Y<group>Params` struct type). Omitted keywords default to
that group's all-defaults factory.

The exported group factories — all `Y<group>Params(; kwargs...)`
shortcuts with the same name as the namelist group:

- `yelmo_params`, `ytopo_params`, `ycalv_params`, `ydyn_params`,
  `ytill_params`, `yneff_params`, `ymat_params`, `ytherm_params`,
  `yelmo_masks_params`, `yelmo_init_topo_params`,
  `yelmo_data_params`.

Defaults match the Yelmo Fortran namelist defaults; pass any subset
of keyword arguments to override.

### Example

```julia
p = YelmoParameters("demo";
    ytopo = ytopo_params(use_bmb = false, topo_rel = 0),
    ydyn  = ydyn_params(solver = "sia"),
)
```

## YelmoMirror parameters

```@docs
YelmoMirrorParameters
```

The Mirror parameter tree mirrors the Fortran namelist exactly
(including a `phys` group with the per-experiment physical
constants). Its group factories follow the same naming as the
`YelmoModel` set, plus `phys_params` and `earth_params` for the
physical-constants group, but they are **not** exported at the package
level (to avoid clobbering the primary `YelmoPar` factories). Call them
namespaced — `YelmoMirrorPar.ydyn_params(...)`,
`YelmoMirrorPar.phys_params(...)`, etc.

## Namelist round-trip

```@docs
write_nml
read_nml
compare
```

## Quick reference: which group does what?

| Group | Drives |
|---|---|
| `yelmo`           | Top-level: domain, grid path, restart, time-step controls, predictor / corrector tuning. |
| `ytopo`           | Topography step: solver, mass-balance gates, grounding-line treatment, residual cleanup thresholds. |
| `ycalv`           | Calving: master switch, front-velocity laws, redistancing cadence, threshold thicknesses. |
| `ydyn`            | Dynamics: solver (`fixed`/`sia`; `ssa`/`hybrid`/`diva` deferred), driving-stress limits, basal sliding parameters. |
| `ytill`           | Till hydrology / friction. |
| `yneff`           | Effective-pressure scheme. |
| `ymat`            | Material: rheology, anisotropy, enhancement factors. |
| `ytherm`          | Thermodynamics: solver, vertical advection method. |
| `yelmo_masks`     | Domain masks (which cells participate in dynamics, which act as boundary). |
| `yelmo_init_topo` | Initial-state construction. |
| `yelmo_data`      | Reference data layer for nudging / spinup. |

For details on every individual field, see the corresponding struct's
default values in `src/YelmoPar.jl` (the `Base.@kwdef struct
Y<group>Params` blocks) or the upstream Yelmo Fortran documentation.
