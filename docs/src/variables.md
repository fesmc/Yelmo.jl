# Variable inventory

Yelmo models carry six component groups (`bnd`, `dta`, `dyn`, `mat`,
`thrm`, `tpo`). The exact set of variables in each group, their
units, and their grid staggering live in **markdown tables** under
`src/variables/{model,mirror}/`. The constructor
([`YelmoModel`](@ref) / [`YelmoMirror`](@ref)) parses these tables
via [`parse_variable_table`](@ref) and pre-allocates one
Oceananigans `Field` per row on the appropriate grid (the
constructor's `make_field` helper matches each name against
the patterns `XFACE_VARIABLES`, `YFACE_VARIABLES`, `ZFACE_VARIABLES`
to pick the right `Field` location).

This page is a curated reference — it inlines the most
user-facing groups (`bnd` and `tpo`) for searchability, and links to
the source markdown files for the larger 3D-state groups (`dyn`,
`mat`, `thrm`) and the data layer (`dta`).

## Two parallel schemas

The two backends use slightly different schemas, both auto-loaded by
their respective constructor:

| Backend | Schema directory | Notes |
|---|---|---|
| [`YelmoModel`](@ref)             | `src/variables/model/`  | Uses `_acx`/`_acy` suffix on staggered fields so the regex-based allocator stages them on the right Oceananigans `Field` location. |
| [`YelmoMirror`](api/mirror.md)   | `src/variables/mirror/` | Mirrors the Fortran NetCDF schema (`_x`/`_y` suffix on staggered fields). |

Variables that appear in only one schema show up in `n_skipped`
during a [`compare_state`](@ref) call rather than as comparison
failures.

## Column meanings

Each row in a variable table follows the same five-column form:

| Column | Meaning |
|---|---|
| `id`         | Stable integer identifier within the group (used by the C interface for the Mirror). |
| `variable`   | Field name. Becomes `y.<group>.<name>` in Julia and the NetCDF variable name on output. |
| `dimensions` | Comma-separated list of axes, e.g. `xc, yc` (2D), `xc, yc, zeta` (3D ice), `xc, yc, zeta_rock` (3D rock). The presence of a vertical axis selects the 3D grid (ice or rock) in the allocator. |
| `units`      | Display units. Not used by kernels; carried through to NetCDF metadata. |
| `long_name`  | Human-readable description for output metadata. |

## Boundary group (`bnd`)

Boundary forcing — read-only inputs to the dynamics. Populated from
the restart, modified by user-supplied climate forcing, and read by
the topography step (e.g. `bnd.smb_ref`, `bnd.z_bed`, `bnd.bmb_shlf`).

| id | variable | dimensions | units | long_name |
|---|---|---|---|---|
|  1 | `z_bed`        | xc, yc | m     | Bedrock elevation |
|  2 | `z_bed_sd`     | xc, yc | m     | Std deviation of bedrock elevation |
|  3 | `z_sl`         | xc, yc | m     | Sea level elevation |
|  4 | `H_sed`        | xc, yc | m     | Sediment thickness |
|  5 | `smb_ref`      | xc, yc | m/yr  | Surface mass balance |
|  6 | `T_srf`        | xc, yc | K     | Surface temperature |
|  7 | `bmb_shlf`     | xc, yc | m/yr  | Basal mass balance for ice shelf |
|  8 | `fmb_shlf`     | xc, yc | m/yr  | Frontal mass balance for ice shelf |
|  9 | `T_shlf`       | xc, yc | K     | Ice shelf temperature |
| 10 | `Q_geo`        | xc, yc | mW/m² | Geothermal heat flow at depth |
| 11 | `enh_srf`      | xc, yc | —     | Enhancement factor at the surface |
| 12 | `basins`       | xc, yc | —     | Basin identification numbers |
| 13 | `basin_mask`   | xc, yc | —     | Mask for basins |
| 14 | `regions`      | xc, yc | —     | Region identification numbers |
| 15 | `region_mask`  | xc, yc | —     | Mask for regions |
| 16 | `ice_allowed`  | xc, yc | —     | Cells where ice thickness can be > 0 |
| 17 | `calv_mask`    | xc, yc | —     | Cells where calving is not allowed |
| 18 | `H_ice_ref`    | xc, yc | m     | Reference ice thickness for relaxation |
| 19 | `z_bed_ref`    | xc, yc | m     | Reference bedrock elevation for relaxation |
| 20 | `domain_mask`  | xc, yc | —     | Domain mask |
| 21 | `mask_ice`     | xc, yc | —     | Per-cell ice-evolution mask: 0=no ice, 1=fixed, 2=dynamic |
| 22 | `tau_relax`    | xc, yc | yr    | User-supplied relaxation timescale (used when `ytopo.topo_rel == -1`) |

## Topography group (`tpo`)

The state evolved by `topo_step!`. Holds the prognostic ice thickness
`H_ice`, all the diagnostic and bookkeeping fields (`H_grnd`,
`f_grnd`, `f_ice`, `z_srf`, `z_base`, …), the seven mass-balance
contributions (`smb`, `bmb`, `fmb`, `dmb`, `cmb`, `mb_relax`,
`mb_resid`), and the calving sub-state (`lsf`, `cr_acx`/`acy`,
`cmb_flt_acx`/`acy`, `cmb_grnd_acx`/`acy`).

A subset for orientation:

| id | variable | dimensions | units | long_name |
|---|---|---|---|---|
|  1 | `H_ice`        | xc, yc | m     | Ice thickness |
|  2 | `dHidt`        | xc, yc | m/yr  | Total ice-thickness rate |
|  3 | `dHidt_dyn`    | xc, yc | m/yr  | Dynamic-only rate (post-advection, pre-mass-balance) |
|  4 | `mb_net`       | xc, yc | m/yr  | Sum of the seven realised mass-balance tendencies |
|  8 | `smb`          | xc, yc | m/yr  | Realised surface mass balance |
|  9 | `bmb`          | xc, yc | m/yr  | Realised basal mass balance |
| 10 | `fmb`          | xc, yc | m/yr  | Realised frontal mass balance |
| 11 | `dmb`          | xc, yc | m/yr  | Realised discharge mass balance |
| 12 | `cmb`          | xc, yc | m/yr  | Realised calving mass balance |
| 18 | `z_srf`        | xc, yc | m     | Surface elevation |
| 23 | `z_base`       | xc, yc | m     | Ice-base elevation |
| 31 | `H_grnd`       | xc, yc | m     | Flotation diagnostic (positive ⇒ grounded) |
| 35 | `f_grnd`       | xc, yc | —     | Subgrid grounded fraction (CISM scheme) |
| 36 | `f_grnd_acx`   | xc, yc | —     | Subgrid grounded fraction at acx nodes |
| 37 | `f_grnd_acy`   | xc, yc | —     | Subgrid grounded fraction at acy nodes |
| 39 | `f_ice`        | xc, yc | —     | Ice-covered fraction (binary stub at v1) |
| 69 | `lsf`          | xc, yc | —     | Level-set function (φ < 0 ice, φ > 0 ocean) |
| 76 | `cr_acx`       | xc, yc | m/yr  | Merged calving-front velocity (acx nodes) |
| 77 | `cr_acy`       | xc, yc | m/yr  | Merged calving-front velocity (acy nodes) |

The full `tpo` schema (77 fields, including predictor / corrector
substructure entries) is in
[`src/variables/model/yelmo-variables-ytopo.md`](https://github.com/fesmc/Yelmo.jl/blob/main/src/variables/model/yelmo-variables-ytopo.md).

## Other groups — links to source

| Group | Schema file (model) |
|---|---|
| `dta`  | [`yelmo-variables-ydata.md`](https://github.com/fesmc/Yelmo.jl/blob/main/src/variables/model/yelmo-variables-ydata.md) — observational reference fields (`pd_*` for present-day). |
| `dyn`  | [`yelmo-variables-ydyn.md`](https://github.com/fesmc/Yelmo.jl/blob/main/src/variables/model/yelmo-variables-ydyn.md) — velocities, drag fields, viscous diagnostics. Mostly 3D ice. |
| `mat`  | [`yelmo-variables-ymat.md`](https://github.com/fesmc/Yelmo.jl/blob/main/src/variables/model/yelmo-variables-ymat.md) — strain-rate and stress invariants, anisotropy, enhancement. Mix of 2D and 3D. |
| `thrm` | [`yelmo-variables-ytherm.md`](https://github.com/fesmc/Yelmo.jl/blob/main/src/variables/model/yelmo-variables-ytherm.md) — temperature on 3D ice + rock grids, basal melt diagnostics. |
| `tpo`  | [`yelmo-variables-ytopo.md`](https://github.com/fesmc/Yelmo.jl/blob/main/src/variables/model/yelmo-variables-ytopo.md) (full table). |
| `bnd`  | [`yelmo-variables-ybound.md`](https://github.com/fesmc/Yelmo.jl/blob/main/src/variables/model/yelmo-variables-ybound.md) (mirrored above). |

The mirror schema lives at the analogous paths under
`src/variables/mirror/`. The differences are minor (mostly variable-
name suffixes and a handful of mirror-only diagnostics that come back
across the C boundary).

## Adding a new field

To add a field to either schema:

1. Add a new row to the relevant table file. Pick a fresh `id` (the
   next integer; avoid reusing IDs even after deletions).
2. Pick `dimensions` based on what kind of field this is:
   - `xc, yc` — 2D scalar (cell-centred, x-face, or y-face — see
     name conventions).
   - `xc, yc, zeta` / `xc, yc, zeta_ac` — 3D ice (centre / face).
   - `xc, yc, zeta_rock` / `xc, yc, zeta_rock_ac` — 3D rock.
3. If the name doesn't match any of the `XFACE_VARIABLES` /
   `YFACE_VARIABLES` / `ZFACE_VARIABLES` patterns (exported from
   `YelmoCore`), the field will land on a `CenterField` (cell-
   centred). For x-face fields, suffix `_acx` (or use a known prefix
   like `ux`); for y-face, `_acy`.
4. Construct the model — the constructor will allocate the new field
   with sensible defaults. If the restart file carries a variable of
   the same name, [`load_state!`](@ref) populates it; otherwise pass
   `strict=false` to the constructor and the field stays at its
   default (zeros).

No code changes outside the markdown table are needed for the
allocator side. Kernels that read the new field of course need to
be written or extended.
