# InitMIP — Greenland & Antarctica

Steady-state ice-sheet simulations under present-day boundary conditions
on real domains, forced by observational data. These are the primary
tests of Yelmo.jl under realistic conditions: full model physics
(thermodynamics, DIVA dynamics, calving) on Greenland (`initmip-grl/`)
and Antarctica (`initmip-ant/`).

Both follow the Fortran reference program `yelmo/tests/yelmo_initmip.f90`
and namelist `yelmo/par/yelmo_initmip.nml` (the `set_grl_pd` and
`set_ant_pd` cases respectively).

## Design: pure-Julia-first

The configuration is a native `YelmoParameters` value built in
`build_params()` inside each `run.jl` — there is **no namelist input
file**. The pure-Julia `YelmoModel` is the primary path and is
initialised directly from those parameters. Initialisation loads the
state from topography data (no restart file):

```
init_topo_load!  →  init_masks!  →  apply_forcing!  →  init_state!(robin-cold)
```

`init_masks!` also paints `bnd.mask_ice` from the region mask, which is
what confines dynamic ice to the domain of interest.

Selecting `backend = :mirror` runs the Fortran model instead. The same
canonical `YelmoParameters` is translated to a `YelmoMirrorParameters`
via `to_mirror`, the namelist is written under `output-mirror/`, and the
Fortran model is initialised from it. Backend-divergent timestepping
options (`MIRROR_DIVERGENT_YELMO`, e.g. `pc_method`) are **not** carried
over — the Mirror keeps its Fortran-native values, while shared controls
(`dt_method`, `dt_min`, `cfl_*`) are copied through.

> `pc_method` differs by backend on purpose: the Julia default is
> `"HEUN"` (fewer adaptive sub-steps on margin-heavy domains), while the
> Mirror uses Fortran's native `"AB-SAM"`. Yelmo.jl's `"HEUN"` is not the
> same scheme as Fortran's `"HEUN"`, so the two are configured
> independently. Setting a divergent parameter on the Julia side and
> requesting `to_mirror` is an error.

## How to run

```bash
cd benchmarks/initmip-grl                            # or initmip-ant
julia --project=. -e 'include("run.jl"); main()'                  # yelmo backend (default), 20 yr
julia --project=. -e 'include("run.jl"); main(t_end=1.0)'         # quick one-step check
julia --project=. -e 'include("run.jl"); main(backend=:mirror)'   # YelmoMirror (Fortran C-API)
```

Configuration is via keyword arguments to `main` (no environment
variables):

| kwarg | default | meaning |
|-------|---------|---------|
| `t_end` | `20.0` | end time [yr] |
| `dt_outer` | `1.0` | outer-loop timestep [yr] |
| `backend` | `:yelmo` | `:yelmo` (pure Julia) or `:mirror` (Fortran) |
| `snapshot_dt` | `10.0` | 2D snapshot cadence [yr] |
| `outdir` | `output/` (`output-mirror/` for mirror) | output directory |

`main` calls `cd(@__DIR__)` so data paths resolve relative to the
benchmark directory regardless of where Julia is launched. To change the
physics/forcing configuration, edit `build_params()` at the top of
`run.jl`.

### Backend differences

| | `:yelmo` (default) | `:mirror` |
|---|---|---|
| Construction | `YelmoModel(b, t; p)` | `YelmoMirror(to_mirror(p), t; rundir, overwrite)` |
| Topography / mask load | `init_topo_load!` / `init_masks!` (Yelmo.jl) | Fortran `yelmo_init` / `ybound_load_masks` from the generated nml |
| Forcing fields | direct write to `y.bnd.*` | direct write to `y.bnd.*`, synced to Fortran on `init_state!`/`step!` |
| Time stepping | Yelmo.jl adaptive PC (`dt_method = 2`, HEUN) | Fortran's own (AB-SAM) |
| Regions API (`region_domain.nc`) | yes | no — `YelmoModel`-only; `V_sle` computed inline |
| Per-section timer (`y.timer`) | yes | no |
| Namelist | none (native params) | generated to `output-mirror/` at runtime |

> The Mirror backend requires the Fortran-yelmo C-API shared library
> (`libyelmo_c_api.so`) to be built; without it `YelmoMirror`
> construction fails at load time. The pure-Julia backend has no such
> dependency.

## Common physics settings

Shared by both domains unless noted (set in `build_params()`; everything
else uses Yelmo.jl defaults):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dynamics solver | DIVA | `ydyn.solver = "diva"` |
| Thermodynamics | `method = "temp"` | full temperature solver |
| Flow law | Glen, `rf_method = 1` | temperature-dependent rate factor |
| Calving | `vm-l19` | eigencalving (von Mises) |
| Effective pressure | till pressure (`yneff.method = 3`) | |
| Vertical layers | 10 ice + 5 bedrock | `nz_aa = 10`, `nzr_aa = 5` |
| Timestep | adaptive PC (`dt_method = 2`) | HEUN (yelmo) / AB-SAM (mirror) |
| Initialisation | `robin-cold` | from topography, no restart |

---

## Greenland (`initmip-grl/`)

| Property | Value |
|----------|-------|
| Domain / grid | Greenland, GRL-16KM (106 × 181, 16 km) |
| Projection | polar stereographic (EPSG:3413) |
| Shelf basal melt | constant −0.5 m/yr |

| Field | Source | File |
|-------|--------|------|
| Grid / region mask | — | `GRL-16KM_REGIONS.nc` |
| Initial topography | Morlighem et al. 2017 (M17) | `GRL-16KM_TOPO-M17-v5.nc` |
| Drainage basins | NASA | `GRL-16KM_BASINS-nasa.nc` |
| SMB + surface temperature | MARv3.11 / ERA 1961–1990 mean (annual) | `GRL-16KM_MARv3.11-ERA_annmean_1961-1990.nc` |
| Geothermal heat flux | Shapiro & Ritzwoller 2004 (S04) | `GRL-16KM_GHF-S04.nc` |

Forcing conversions (`apply_forcing!`, mirrors Fortran `yelmo_data.f90`):

- `T_srf`: °C → K (`+ 273.15`).
- `smb`: mm w.e./yr → m i.e./yr (`× 1e-3 · ρ_w/ρ_ice`, Fortran
  `conv_mmawe_maie`).
- Ice-free cells get an extra **−2 m/yr** SMB penalty to suppress
  spurious growth outside the present-day margin (Fortran
  `yelmo_initmip.f90:183`).

`GRL-16KM_VEL-J18.nc` is present in `data/` for future use but not
loaded. `summary.jl` produces an optional `summary.json` of high-level
statistics from a completed run.

---

## Antarctica (`initmip-ant/`)

| Property | Value |
|----------|-------|
| Domain / grid | Antarctica, ANT-32KM (191 × 191, 32 km) |
| Projection | polar stereographic (EPSG:3031) |
| Shelf basal melt | constant −0.2 m/yr |

| Field | Source | File |
|-------|--------|------|
| Grid / region mask | — | `ANT-32KM_REGIONS.nc` |
| Initial topography | BedMachine | `ANT-32KM_TOPO-BedMachine.nc` |
| Drainage basins | NASA | `ANT-32KM_BASINS-nasa.nc` |
| SMB + surface temperature | RACMO2.3 / ERA-Interim hybrid 1981–2010 (monthly) | `ANT-32KM_RACMO23-ERAINT-HYBRID_1981-2010.nc` |
| Geothermal heat flux | Shapiro & Ritzwoller 2004 (S04) | `ANT-32KM_GHF-S04.nc` |

Forcing conversions (`apply_forcing!`):

- RACMO fields are **monthly** (`nx, ny, 12`) — averaged to an annual
  mean (Fortran `yelmo_data.f90:317`).
- `smb`: kg m⁻² d⁻¹ (≡ mm w.e./d) → m i.e./yr (`× 1e-3 · 365 · ρ_w/ρ_ice`,
  Fortran `conv_mmdwe_maie`), then `|smb| < 1e-3 → 0`.
- `T_srf`: already in Kelvin (no offset).
- **No** ice-free SMB penalty (unlike Greenland).

A 20-yr run is stable: V_sle holds at ~58 m (the Antarctic sea-level
equivalent), max H steady at ~4757 m.

---

## Outputs

Written under `output/` (or `output-mirror/`), all gitignored:

- `region_domain.nc` — whole-domain time series (volume, area, SLE,
  velocities, …) via the regions API. **`:yelmo` backend only.**
- `snapshots.nc` — 2D snapshots of all tpo/dyn/mat/thrm/bnd/dta fields
  every `snapshot_dt`.
- `restart_final.nc` — full model state at `t_end`; reload with
  `YelmoModel("restart_final.nc", t_end)`.
- `yelmo_timesteps.nc` — adaptive-PC timestep log (`log_timestep`).

Per-outer-step `V_ice`, `V_sle`, and `max_H` are also printed to stdout,
and `yelmo.timing = true` triggers a `print_timings` table at the end
(`:yelmo` backend).

## Notes

- The default run is 20 yr — a functional check, not an equilibrium
  spin-up. For equilibrium, increase to several thousand years (the
  Fortran reference's `equil_method = "opt"` friction optimisation is
  not yet ported to Yelmo.jl).
- The Fortran reference runs two brief equilibration passes before its
  main loop; these are not replicated here — the run starts directly
  from the robin-cold state.
- Shelf basal melt is a conservative constant on both domains (Fortran
  defaults are −1.0 m/yr for Greenland, −0.2 m/yr for Antarctica).
- The DIVA SSA assembly can optionally use the symmetric viscous-energy
  Hessian (`SSASolver(method = :energy_quadratic)`, CG inner solve)
  instead of the default strong-form residual Jacobian; set it in
  `build_params()` via `ydyn = ydyn_params(ssa_solver = SSASolver(method = :energy_quadratic))`.
  Equivalence is unit-tested on the SLAB-S06 fixture in
  [`test/test_yelmo_ssa_energy.jl`](../../test/test_yelmo_ssa_energy.jl).

## References

- Morlighem, M. et al. (2017). BedMachine v3 (Greenland). *GRL*, 44(21),
  11051–11061.
- Morlighem, M. et al. (2020). Deep glacial troughs and stabilizing
  ridges unveiled beneath Antarctica (BedMachine Antarctica). *Nature
  Geoscience*, 13, 132–137.
- Shapiro, N. M., & Ritzwoller, M. H. (2004). Inferring surface heat flux
  distributions guided by a global seismic model. *EPSL*, 223(1–2),
  213–224.
- Fettweis, X. et al. (2017). MAR Greenland SMB reconstructions. *The
  Cryosphere*, 11(2), 1015–1033.
- van Wessem, J. M. et al. (2018). RACMO2.3p2 polar climate. *The
  Cryosphere*, 12, 1479–1498.
- Fortran reference: `yelmo/tests/yelmo_initmip.f90`,
  `yelmo/par/yelmo_initmip.nml`.
