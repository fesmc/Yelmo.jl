# Core model API

The core layer defines the abstract supertype, the pure-Julia
`YelmoModel` concrete type, the field-allocation machinery, and the
NetCDF restart loader. Everything documented here lives in
`Yelmo.YelmoCore` and is re-exported at the package level.

## Abstract type and concrete model

```@docs
AbstractYelmoModel
YelmoModel
```

## Stepping interface

`YelmoCore` declares the generic `step!` and the per-component
`<comp>_step!` forwards (`topo_step!`, `dyn_step!`, `mat_step!`,
`therm_step!`); each component module adds the actual method body.
For the `YelmoModel` backend, `step!(y, dt)` runs the components in
fixed phase order (`tpo` → `dyn` → `mat` → `therm`); `mat_step!` and
`therm_step!` remain as forward stubs at the current milestone. For
the `YelmoMirror` backend, `step!` delegates to the Fortran
`yelmo_step` via `ccall`.

```@docs
load_state!
```

`step!(y::AbstractYelmoModel, dt)` (`init_state!` likewise) is the
shared stepping interface. Each backend provides its own method
body — see [`topo_step!`](topography.md#per-step-orchestrator-and-diagnostics-refresh)
and [`dyn_step!`](dynamics.md#per-step-orchestrator) for the per-
component bodies inside `YelmoModel.step!`, and
[Yelmo Mirror — Stepping](mirror.md#stepping) for the C-API call
site.

## Field allocation and grid construction

The constructor uses helper functions to allocate one Oceananigans
`Field` per variable on the appropriate grid. The dispatch keys off
the variable name — names matching the patterns in
`XFACE_VARIABLES` go to `XFaceField`, `YFACE_VARIABLES` to
`YFaceField`, `ZFACE_VARIABLES` to `ZFaceField`, and the rest to
`CenterField`.

The pattern lists are exported as
`XFACE_VARIABLES`, `YFACE_VARIABLES`, `ZFACE_VARIABLES`, and
`VERTICAL_DIMS` from `YelmoCore`; the constructor calls
`make_field(varname, grid)` to do the allocation, and
`yelmo_define_grids(g)` to build the three Oceananigans grids from a
named-tuple of coordinate arrays.

## Loading from a NetCDF restart

```@docs
load_grids_from_restart
load_fields_from_restart
```

`load_field_from_dataset_2D(ds, varname, grid)` and
`load_field_from_dataset_3D(ds, varname, grid)` are the per-field
shape-aware loaders used internally by [`load_state!`](@ref); they
accept either an open `NCDataset` or a filename.

## Mask constants

Two families of integer enums encode per-cell categorical state.
They are stored in `Float64` fields (so they share the same
Oceananigans `Field` machinery as continuous-valued fields), but the
constants themselves are `Int`.

**Ice-evolution mask** (`bnd.mask_ice`) — selects how each cell's
`H_ice` is updated by the post-advection mask pass in `topo_step!`:

| Constant | Value | Meaning |
|---|---|---|
| `MASK_ICE_NONE`    | `0` | Force `H_ice = 0` (no ice). |
| `MASK_ICE_FIXED`   | `1` | Hold `H_ice` at its current value. |
| `MASK_ICE_DYNAMIC` | `2` | Evolve `H_ice` freely (default). |

**Bed-state mask** (`tpo.mask_bed`) — diagnostic categorisation of
each cell's bed state, written by `gen_mask_bed!` in the topography
step's diagnostic phase:

| Constant | Value | Meaning |
|---|---|---|
| `MASK_BED_OCEAN`   | `0` | Ice-free ocean. |
| `MASK_BED_LAND`    | `1` | Ice-free land. |
| `MASK_BED_FROZEN`  | `2` | Fully ice-covered, grounded, frozen base. |
| `MASK_BED_STREAM`  | `3` | Fully ice-covered, grounded, temperate base. |
| `MASK_BED_GRLINE`  | `4` | Grounding-line cell. |
| `MASK_BED_FLOAT`   | `5` | Fully ice-covered, floating. |
| `MASK_BED_ISLAND`  | `6` | Reserved (Fortran does not currently emit). |
| `MASK_BED_PARTIAL` | `7` | Partially ice-covered. |

## State comparison

```@docs
compare_state
StateComparison
```
