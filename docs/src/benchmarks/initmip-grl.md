# initMIP Greenland (initmip-grl)

**Run script**: `benchmarks/initmip-grl/run.jl`  
**Solvers exercised**: full chain — `tpo` (implicit advection + calving),
`dyn` (DIVA), `mat` (Glen + temperature-coupled Arrhenius), `thrm`
(implicit temperature solver, bedrock + ice column)  
**Backends supported**: `YelmoModel` (default) and `YelmoMirror`  
**Validation**: side-by-side comparison against the `YelmoMirror`
backend; not a CI fixture test

## Setup

Steady-state Greenland Ice Sheet simulation under present-day boundary
conditions on the GRL-16KM grid — the suite's primary test of the
**full Yelmo physics chain on a real domain**. It follows the Fortran
reference `yelmo/tests/yelmo_initmip.f90` (`set_grl_pd` case).

| Property | Value |
|---|---|
| Domain | Greenland (GRL-16KM, 106 × 181 cells, 16 km, EPSG:3413) |
| Topography (`H_ice`, `z_bed`) | Morlighem et al. 2017 (M17) |
| SMB / `T_srf` | MARv3.11 / ERA 1961–1990 mean, annual (mm w.e./yr → m i.e./yr; °C → K) |
| `Q_geo` | Shapiro & Ritzwoller 2004 |
| Drainage basins, region mask | NASA |
| `bmb_shlf` | constant −0.5 m/yr |
| Vertical layers | 10 ice + 5 bedrock |
| Solver | DIVA (`ydyn.solver = "diva"`) |
| Calving | von Mises (`vm-l19`) |
| Effective pressure | till pressure (`yneff.method = 3`) |
| Time-stepping | adaptive PC, `dt_method = 2` (HEUN on `YelmoModel`, AB-SAM on Mirror) |
| Default `t_end` | 20 yr (functional check; raise for spin-up) |

## Design: pure-Julia-first

The configuration is a native `YelmoParameters` built in `build_params()`
inside `run.jl` — there is **no namelist input file**. The pure-Julia
`YelmoModel` is the primary path, initialised directly from those
parameters and loading its state from topography data (no restart):

```
init_topo_load!  →  init_masks!  →  apply_forcing!  →  init_state!(robin-cold)
```

Selecting `backend = :mirror` runs the Fortran model: the same
`YelmoParameters` is translated to a `YelmoMirrorParameters` via
`to_mirror`, the namelist is written under `output-mirror/`, and Fortran
is initialised from it. Backend-divergent timestepping options
(`MIRROR_DIVERGENT_YELMO`, e.g. `pc_method`) keep their Fortran-native
values rather than being copied from the Julia config; shared controls
(`dt_method`, `dt_min`, `cfl_*`) are carried over. `pc_method` is
therefore HEUN on `YelmoModel` and AB-SAM on the Mirror by design.

## Running

The script lives in its own benchmark project, separate from the
`test/benchmarks/` CI tree. Activate that project, not `test/`:

```bash
cd benchmarks/initmip-grl

# Pure-Julia YelmoModel backend (default). Outputs land in `output/`.
julia --project=. -e 'include("run.jl"); main()'

# Quick one-step sanity check.
julia --project=. -e 'include("run.jl"); main(t_end=1.0)'

# Fortran-yelmo YelmoMirror backend. Outputs land in `output-mirror/`.
julia --project=. -e 'include("run.jl"); main(backend=:mirror)'
```

Configuration is via keyword arguments to `main` — `t_end`, `dt_outer`,
`backend`, `snapshot_dt`, `outdir` — not environment variables. Edit
`build_params()` to change the physics/forcing configuration.

The Mirror backend additionally requires a built Fortran-yelmo shared
library (`libyelmo_c_api.so`); without it `YelmoMirror` construction
fails at load time. The pure-Julia backend has no such dependency.

## What it tests

- Real-domain `init_topo_load!` and `init_masks!` against the M17 +
  NASA-basins NetCDFs, including `bnd.mask_ice` painting from the region
  mask.
- Forcing unit conversions (`T_srf` °C → K, `smb` mm w.e./yr →
  m i.e./yr) match Fortran `yelmo_data.f90`.
- DIVA Picard loop with vertical-shear contribution to viscosity.
- Adaptive PC time-stepping lands on the outer time exactly.
- I/O writer round-trips for both storage conventions: `YelmoModel`'s
  split-boundary file (`Nz_file = Nz + 2`) and `YelmoMirror`'s
  interior-extended (`Nz_file = Nz`).
- `YelmoParameters → YelmoMirrorParameters` translation via `to_mirror`,
  including the backend-divergent timestepping map.

## Outputs (per backend)

Written under `output/` (or `output-mirror/`):

- `region_domain.nc` — whole-domain regions-API time series
  (`YelmoModel` only).
- `snapshots.nc` — 2D snapshots every `snapshot_dt` (default 10 yr) of
  all `tpo / dyn / mat / thrm / bnd / dta` fields.
- `restart_final.nc` — full state at `t_end`; reload with
  `YelmoModel("restart_final.nc", t_end)`.
- `yelmo_timesteps.nc` — adaptive-PC timestep log.
- Stdout: per-outer-step `t / V_ice / V_sle / max_H` table.

## Cross-backend comparison

Running both backends produces parallel `output/` and `output-mirror/`
directories so restart fields are available side-by-side:

```bash
cd benchmarks/initmip-grl
julia --project=. -e 'include("run.jl"); main()'                 # Yelmo  → output/
julia --project=. -e 'include("run.jl"); main(backend=:mirror)'  # Mirror → output-mirror/
open output/restart_final.nc output-mirror/restart_final.nc
```

For full forcing/physics details see
`benchmarks/initmip-grl/README.md`.
