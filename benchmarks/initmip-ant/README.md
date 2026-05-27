# `initmip-ant`

Steady-state Antarctic Ice Sheet simulation under present-day boundary
conditions. It is the Antarctic counterpart of
[`initmip-grl`](../initmip-grl/README.md) and shares the same
**pure-Julia-first design**, run interface (`main(; t_end, dt_outer,
backend, snapshot_dt, outdir)`), backend selection (`:yelmo` /
`:mirror` via `to_mirror`), initialisation flow
(`init_topo_load! → init_masks! → apply_forcing! → init_state!`), and
outputs.

**Read [`initmip-grl/README.md`](../initmip-grl/README.md) first** for
all of that. This page documents only what differs for Antarctica. It
follows the Fortran reference `set_ant_pd` case.

## How to run

```bash
cd benchmarks/initmip-ant
julia --project=. -e 'include("run.jl"); main()'              # 20 yr, yelmo backend
julia --project=. -e 'include("run.jl"); main(t_end=1.0)'     # quick one-step check
```

## Differences from `initmip-grl`

| | `initmip-grl` | `initmip-ant` |
|---|---|---|
| Domain / grid | Greenland, GRL-16KM (106 × 181, 16 km) | Antarctica, ANT-32KM (191 × 191, 32 km) |
| Projection | polar stereographic (EPSG:3413) | polar stereographic (EPSG:3031) |
| Initial topography | Morlighem 2017 (M17), `GRL-16KM_TOPO-M17-v5.nc` | BedMachine, `ANT-32KM_TOPO-BedMachine.nc` |
| Climate forcing | MARv3.11, **annual mean** | RACMO2.3 / ERA-Interim hybrid, **monthly** |
| `smb` units → m i.e./yr | mm w.e./yr · `1e-3·ρ_w/ρ_ice` | kg m⁻² d⁻¹ · `1e-3·365·ρ_w/ρ_ice`, then `|smb|<1e-3 → 0` |
| `T_srf` | °C → K (`+273.15`) | already in Kelvin |
| Ice-free SMB penalty | −2 m/yr | **none** |
| Shelf basal melt | −0.5 m/yr | −0.2 m/yr |

Everything else (physics: DIVA, `method="temp"`, Glen `rf_method=1`,
`vm-l19` calving, till-pressure `N_eff`, 10+5 layers, adaptive PC,
robin-cold init) is identical to `initmip-grl`.

## Forcing and data sources

| Field | Source | File |
|-------|--------|------|
| Grid / region mask | — | `ANT-32KM_REGIONS.nc` |
| Initial topography | BedMachine Antarctica | `ANT-32KM_TOPO-BedMachine.nc` |
| Drainage basins | NASA | `ANT-32KM_BASINS-nasa.nc` |
| SMB + surface temperature | RACMO2.3 / ERA-Interim hybrid 1981–2010 (monthly) | `ANT-32KM_RACMO23-ERAINT-HYBRID_1981-2010.nc` |
| Geothermal heat flux | Shapiro & Ritzwoller 2004 (S04) | `ANT-32KM_GHF-S04.nc` |

RACMO fields are monthly (`nx, ny, 12`); `apply_forcing!` averages them
to an annual mean (Fortran `yelmo_data.f90:317`) before the unit
conversion above.

A 20-yr run is stable: V_sle holds at ~58 m (the Antarctic sea-level
equivalent) and max H stays steady at ~4757 m.

## References

- Morlighem, M. et al. (2020). Deep glacial troughs and stabilizing
  ridges unveiled beneath Antarctica (BedMachine Antarctica). *Nature
  Geoscience*, 13, 132–137.
- van Wessem, J. M. et al. (2018). RACMO2.3p2 polar climate. *The
  Cryosphere*, 12, 1479–1498.
- Shapiro, N. M., & Ritzwoller, M. H. (2004). *EPSL*, 223(1–2), 213–224.
- Fortran reference: `yelmo/tests/yelmo_initmip.f90`,
  `yelmo/par/yelmo_initmip.nml` (`set_ant_pd`).
