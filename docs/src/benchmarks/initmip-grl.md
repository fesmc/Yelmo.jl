# initMIP Greenland (initmip-grl)

**Run script**: `benchmarks/initmip-grl/run.jl`  
**Solvers exercised**: full chain — `tpo` (impl-lis advection + calving),
`dyn` (DIVA), `mat` (Glen + temperature-coupled Arrhenius), `thrm`
(implicit temperature solver, bedrock + ice column)  
**Backends supported**: `YelmoModel` (default) and `YelmoMirror`  
**Validation**: side-by-side comparison against the YelmoMirror
backend on the same nml + data; not a CI fixture test

## Setup

Steady-state Greenland Ice Sheet simulation under present-day boundary
conditions on the GRL-16KM grid. The first benchmark in this suite to
exercise the **full Yelmo physics chain on a real domain** rather than
a closed-form analytical setup or a synthetic-grid mock.

| Property | Value |
|---|---|
| Domain | Greenland (GRL-16KM, 106 × 181 cells, 16 km resolution, EPSG:3413) |
| Topography (`H_ice`, `z_bed`) | Morlighem et al. 2017 (M17) |
| SMB / `T_srf` | MARv3.11 / ERA 1961–1990 mean (mm w.e./yr → m i.e./yr) |
| `Q_geo` | Shapiro & Ritzwoller 2004 |
| Drainage basins, region mask | NASA / NEGIS |
| `bmb_shlf` | Constant −0.5 m/yr |
| Vertical layers | 10 ice + 5 bedrock |
| Solver | DIVA (`ydyn.solver = "diva"`) |
| Topography solver | implicit advection (`ytopo.solver = "impl-lis"` for Mirror; `"impl"` for Yelmo) |
| Calving | von Mises (`vm-l19`) |
| Effective pressure | till pressure (`yneff.method = 3`) |
| Time-stepping | adaptive PC, AB-SAM scheme, `dt_method = 2` |
| Default `T_END_YR` | 10 yr (smoke); raise for spin-up |

The two backends share the **same** `yelmo_initmip_grl.nml`,
`data/GRL-16KM/`, and forcing-application logic. Only the build path
differs: `YelmoModel` uses Yelmo.jl's `init_topo_load!` /
`init_masks!` / direct field writes; `YelmoMirror` lets Fortran
`yelmo_init` load topo and masks and writes climate forcing onto
the Julia mirror, which `init_state!` then syncs to Fortran.

For the full forcing-load + unit-conversion details, see
`benchmarks/initmip-grl/README.md`.

## Running

The script lives in its own benchmark project, separate from the
`test/benchmarks/` CI tree. Activate that project, not `test/`:

```bash
cd benchmarks/initmip-grl

# Pure-Julia YelmoModel backend (default). Outputs land in `output/`.
julia --project=. run.jl

# Fortran-yelmo YelmoMirror backend. Outputs land in `output-mirror/`.
INITMIP_BACKEND=mirror julia --project=. run.jl

# Post-processing summary (Yelmo backend only).
julia --project=. summary.jl
```

The Mirror backend additionally requires a built Fortran-yelmo
shared library (`libyelmo_c_api.so`) and the Fortran auxiliary
namelist directory (`input/yelmo_phys_const.nml`,
`input/yelmo-variables-*.md`). The `input/` directory is created as
a per-developer symlink (or copy) of `yelmo/input/`; it's gitignored.

The default 10-yr smoke takes:

| backend | wall-time | notes |
|---|---|---|
| `YelmoModel` | ≈ 200 s | Adaptive PC ≈ 10 sub-steps per outer 1 yr → 100 sub-steps total |
| `YelmoMirror` | ≈ 30 s | Single-threaded; same physics chain through Fortran |

## What it tests

- Real-domain `init_topo_load!` and `init_masks!` against the M17 +
  NASA-basins NetCDFs.
- Forcing unit conversions (`T_srf` °C → K, `smb` mm w.e./yr →
  m i.e./yr) match Fortran `yelmo_data.f90:285-331`.
- DIVA Picard loop with vertical-shear contribution to viscosity
  (commit history under `dyn:` covers the divergence found by this
  benchmark).
- Adaptive PC time-stepping at `pc_method = "AB-SAM"` lands on the
  outer time exactly (`time = 10.0` to float precision; the previous
  version drifted to `10.08` due to a `dt_min` floor bug, fixed
  during this benchmark's bring-up).
- I/O writer round-trips for both storage conventions:
  `YelmoModel`'s split-boundary file (`Nz_file = Nz + 2` with
  `_b`/`_s` glue) and `YelmoMirror`'s interior-extended (`Nz_file = Nz`).
- Cross-backend physics agreement on the GRL domain: bulk numbers
  (V_ice, V_sle, max_H, mean velocity, bulk viscosity) within ~30%
  of YelmoMirror after the DIVA viscosity fix; further agreement
  in fast outlet streams expected as remaining secondary effects
  land.

## Outputs (per backend)

- `region_domain.nc` — whole-domain regions API time-series
  (Yelmo only).
- `snapshots.nc` — 2D snapshots every `SNAPSHOT_DT_YR` (default 10
  yr) of all `tpo / dyn / mat / thrm / bnd / dta` fields.
- `restart_final.nc` — full state at `T_END_YR`. Reloadable with
  `YelmoModel("restart_final.nc", T_END_YR)`.
- `summary.json` — high-level statistics + run metadata. Yelmo
  backend only (the regions API is `YelmoModel`-specific).
- Stdout: per-outer-step `t / V_ice / V_sle / max_H / PCsub /
  PCrej / SSAit` table (PC/SSA columns are 0 for Mirror — those
  counters live inside Fortran's adaptive driver and aren't
  surfaced through the C-API; Fortran's `yelmo:: timelog:` lines
  in `run.log` carry per-outer-step `min_dt` / `max_dt` /
  `n_dtmin` instead).

## Cross-backend comparison

Running both backends produces parallel `output/` and
`output-mirror/` directories so restart fields are available
side-by-side for visual inspection. The recommended workflow:

```bash
cd benchmarks/initmip-grl
julia --project=. run.jl                            # Yelmo → output/
INITMIP_BACKEND=mirror julia --project=. run.jl     # Mirror → output-mirror/

# Open both in your NetCDF viewer of choice.
open output/restart_final.nc output-mirror/restart_final.nc
```

The `_b` / `_s` 2D boundary fields aren't independent disk
variables in Yelmo's restart — they're glued into the unified `ux`
/ `uy` / `uxy` / `T_ice` / etc slabs (basal at `zeta` index 1,
surface at `zeta` index `Nz_file`). Mirror stores the basal /
surface inline at the first / last `zeta` index of the
length-`Nz_file = Nz` slab.
