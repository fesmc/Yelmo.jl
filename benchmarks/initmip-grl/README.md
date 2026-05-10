# `initmip-grl`

Steady-state Greenland Ice Sheet simulation under present-day boundary
conditions. The run exercises the full model physics (thermodynamics,
DIVA dynamics, calving) on a real domain forced by observational data,
making it the primary test of Yelmo.jl under realistic conditions.

The setup follows the Fortran reference program
`yelmo/tests/yelmo_initmip.f90` and namelist
`yelmo/par/yelmo_initmip.nml`.

## Domain and grid

| Property | Value |
|----------|-------|
| Domain | Greenland (GRL-16KM) |
| Grid | 106 × 181 cells, 16 km resolution |
| Projection | Polar stereographic (EPSG:3413) |

## Forcing and data sources

| Field | Source | File |
|-------|--------|------|
| Grid (xc, yc) | — | `GRL-16KM_REGIONS.nc` |
| Initial topography (H_ice, z_bed) | Morlighem et al. 2017 (M17) | `GRL-16KM_TOPO-M17-v5.nc` |
| Drainage basins (bnd.basins / basin_mask) | NASA / NEGIS | `GRL-16KM_BASINS-nasa.nc` |
| Region mask (bnd.regions) | — | `GRL-16KM_REGIONS.nc` |
| Surface mass balance (smb_ref) | MARv3.11 / ERA 1961–1990 mean | `GRL-16KM_MARv3.11-ERA_annmean_1961-1990.nc` |
| Surface temperature (T_srf) | MARv3.11 / ERA 1961–1990 mean | `GRL-16KM_MARv3.11-ERA_annmean_1961-1990.nc` |
| Geothermal heat flux (Q_geo) | Shapiro & Ritzwoller 2004 (S04) | `GRL-16KM_GHF-S04.nc` |
| Basal melt (bmb_shlf) | Constant −0.5 m/yr | — |

Basins and regions are loaded by `init_masks!` (Yelmo backend) or by
the Fortran-side `ybound_load_masks` (Mirror backend).
The reference surface velocity file `GRL-16KM_VEL-J18.nc` is present
in `data/` for future use but is not currently loaded.

Unit conversions applied at load time (mirrors Fortran `yelmo_data.f90`):

- `T_srf`: stored in °C in MAR, converted to K (`+ 273.15`).
- `smb`: stored in mm w.e./yr in MAR, converted to m i.e./yr
  (`× 1e-3 · ρ_w / ρ_ice`).
- `Q_geo`: stored in mW/m² in S04 — Yelmo's bedrock solver also expects
  mW/m², so no conversion is applied.

Ice-free cells receive an additional −2 m/yr SMB penalty to suppress
spurious ice growth outside the present-day margin (mirrors Fortran
`yelmo_initmip.f90:183`).

## Physics settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dynamics solver | DIVA | `ydyn.solver = "diva"` |
| Thermodynamics | `method = "temp"` | Full temperature solver |
| Flow law | Glen, `rf_method = 1` | Standard temperature-dependent rate factor |
| Topography solver | `impl-lis` | Implicit advection via LIS |
| Calving | `vm-l19` | Eigencalving (von Mises) |
| Effective pressure | Till pressure (`yneff.method = 3`) | |
| Vertical layers | 10 ice + 5 bedrock | `nz_aa = 10`, `nzr_aa = 5` |
| Timestep method | Adaptive PC (`dt_method = 2`) | `pc_method = "AB-SAM"` |
| Initialisation | `robin-cold` | Cold-based Robin solution |

## References

- Morlighem, M. et al. (2017). BedMachine v3: Complete bed topography
  and ocean bathymetry mapping of Greenland from multi-beam echo
  sounding combined with mass conservation. *Geophysical Research
  Letters*, 44(21), 11051–11061.
- Shapiro, N. M., & Ritzwoller, M. H. (2004). Inferring surface heat
  flux distributions guided by a global seismic model. *Earth and
  Planetary Science Letters*, 223(1–2), 213–224.
- Fettweis, X. et al. (2017). Reconstructions of the 1900–2015 Greenland
  ice sheet surface mass balance using the regional climate MAR model.
  *The Cryosphere*, 11(2), 1015–1033.
- Fortran reference: `yelmo/tests/yelmo_initmip.f90`,
  `yelmo/par/yelmo_initmip.nml`

## How to run

```bash
cd benchmarks/initmip-grl
julia --project=. run.jl                            # yelmo backend (default)
INITMIP_BACKEND=mirror julia --project=. run.jl     # YelmoMirror (Fortran-yelmo C-API)
julia --project=. summary.jl                        # produces summary.json
```

The script must be run from this directory because namelist data paths
are relative to it (`data/GRL-16KM/`). The `cd(@__DIR__)` at the top
of `run.jl` handles this automatically when Julia is invoked from the
repo root via `julia --project=benchmarks/initmip-grl benchmarks/initmip-grl/run.jl`.

### Backend selection

The same `run.jl`, `yelmo_initmip_grl.nml`, and forcing data drive
both backends — only the build path differs:

| | yelmo (default) | mirror |
|---|---|---|
| Construction | `YelmoModel(b, t; p)` | `YelmoMirror(p, t; rundir, overwrite)` |
| Topography load | `init_topo_load!` (Yelmo.jl) | Fortran `yelmo_init` reads it from the nml |
| Mask load | `init_masks!` (Yelmo.jl) | Fortran `ybound_load_masks` reads it from the nml |
| Forcing fields | direct field write to `y.bnd.*` | direct field write to `y.bnd.*`, pushed to Fortran on `init_state!` / `step!` via `yelmo_sync!` |
| Time stepping | Yelmo.jl adaptive PC (`dt_method = 2`) | Fortran's own time-stepping |
| Regions API | yes (`region_domain.nc`) | no — currently `YelmoModel`-only |
| Per-section timer | yes (`y.timer`, prints at end) | no |
| Snapshots / restart | yes (Nz_file = Nz + 2) | yes (Nz_file = Nz; `init_output` / `write_output!` branch on `uses_split_boundary_storage(y)`) |
| `PCsub` / `PCrej` / `SSAit` columns | accurate | always 0 — Fortran's per-step counters aren't surfaced through the C-API. The `yelmo:: timelog:` lines in `run.log` carry per-outer-step `min_dt` / `max_dt` / `n_dtmin` instead |
| `V_sle` | regions API (`V_ice_above_flotation × 1e-3 / 394.7`) | computed inline via `H_af = max(0, H_ice + min(0, z_bed − z_sl)·ρ_sw/ρ_ice)` |
| Per-step `PCsub/PCrej/SSAit` | yes | shown as 0 (counters not surfaced) |

## Outputs

- `output/region_domain.nc` — whole-domain time-series (volume, area,
  SLE, velocities, SMB, BMB, …) via the regions API. Written every
  `DT_OUTER_YR` (1 yr).
- `output/snapshots.nc` — 2D snapshots of all tpo/dyn/mat/thrm/bnd/dta
  fields every `SNAPSHOT_DT_YR` (10 yr).
- `output/restart_final.nc` — full model state at `T_END_YR`; can be
  loaded with `YelmoModel("restart_final.nc", T_END_YR)`.
- `plots/` — figures generated by `summary.jl` (gitignored).
- `summary.json` — committed high-level statistics + metadata.

Per-outer-step diagnostics are also printed to stdout (`PCsub` accepted
adaptive sub-steps, `PCrej` rejected, `SSAit` last Picard count), and
`yelmo.timing = True` triggers a `print_timings` table at the end with
per-section wall-clock totals.

## Notes

- The default run is 10 yr (`T_END_YR = 10.0`, `DT_OUTER_YR = 1.0`) as
  a quick functional smoke test — exercises the full GRL forcing path,
  DIVA + temp solver, calving, and adaptive PC, finishing in a few
  minutes on a single core. For a proper equilibrium spin-up, increase
  to 5000–20 000 yr and consider using the `equil_method = "opt"`
  friction optimisation from the Fortran reference (not yet ported to
  Yelmo.jl).
- With the GRL-16KM topography and present-day forcing, the adaptive
  PC controller (`pc_method = "AB-SAM"`, default `pc_eps`) typically
  takes ~10 sub-steps per 1 yr outer step (effective dt ≈ 0.1 yr).
  Walltime scales linearly: ≈ 1.3 min per 10 yr after first-call
  Julia compilation.
- The Fortran reference runs two brief equilibration passes before the
  main loop (`yelmo_update_equil` with `topo_fixed=false` and
  `topo_fixed=true`). These are not replicated here; the run starts
  directly from the robin-cold state.
- `bmb_shlf = −0.5 m/yr` (constant). The Fortran default is −1.0 m/yr;
  the value here is intentionally more conservative for a test run.
