# initMIP Antarctica (initmip-ant)

**Run script**: `benchmarks/initmip-ant/run.jl`  
**Backends supported**: `YelmoModel` (default) and `YelmoMirror`

Steady-state Antarctic Ice Sheet simulation under present-day boundary
conditions — the Antarctic counterpart of
[initMIP Greenland](initmip-grl.md). It shares the same
**pure-Julia-first design**, run interface (`main(; t_end, dt_outer,
backend, snapshot_dt, outdir)`), backend selection (`:yelmo` / `:mirror`
via `to_mirror`), initialisation flow
(`init_topo_load! → init_masks! → apply_forcing! → init_state!`), and
outputs.

**See [initMIP Greenland](initmip-grl.md) first** for all of that. This
page covers only the Antarctica differences. It follows the Fortran
reference `set_ant_pd` case.

## Running

```bash
cd benchmarks/initmip-ant
julia --project=. -e 'include("run.jl"); main()'              # 20 yr, yelmo backend
julia --project=. -e 'include("run.jl"); main(t_end=1.0)'     # quick one-step check
```

## Differences from initmip-grl

| | initmip-grl | initmip-ant |
|---|---|---|
| Domain / grid | Greenland, GRL-16KM (106 × 181, 16 km, EPSG:3413) | Antarctica, ANT-32KM (191 × 191, 32 km, EPSG:3031) |
| Initial topography | Morlighem 2017 (M17) | BedMachine Antarctica |
| Climate forcing | MARv3.11, **annual mean** | RACMO2.3 / ERA-Interim hybrid, **monthly** |
| `smb` → m i.e./yr | mm w.e./yr · `1e-3·ρ_w/ρ_ice` | kg m⁻² d⁻¹ · `1e-3·365·ρ_w/ρ_ice`, then `|smb|<1e-3 → 0` |
| `T_srf` | °C → K (`+273.15`) | already in Kelvin |
| Ice-free SMB penalty | −2 m/yr | **none** |
| Shelf basal melt | −0.5 m/yr | −0.2 m/yr |

Everything else — DIVA dynamics, `method="temp"` thermodynamics, Glen
`rf_method=1`, `vm-l19` calving, till-pressure effective pressure,
10 + 5 vertical layers, adaptive PC, robin-cold initialisation, and the
output set — is identical to initmip-grl.

RACMO fields are monthly (`nx, ny, 12`); `apply_forcing!` averages them
to an annual mean before the unit conversion above. A 20-yr run is
stable: V_sle holds at ~58 m (the Antarctic sea-level equivalent) and
max H stays steady at ~4757 m.

For full details see `benchmarks/initmip-ant/README.md`.
