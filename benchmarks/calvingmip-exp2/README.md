# `calvingmip-exp2`

CalvingMIP Experiment 2: oscillating calving front on the circular
bowl. Chained from a `calvingmip-exp1` spin-up — the Exp1 restart
provides the equilibrium ice cap as initial state.

The calving rate law `calvmip_exp2!` sets the net front velocity to
`w = (u/|u|) · wv`, where `wv = −300 sin(2π t / 1000) m/yr` oscillates
with period 1000 yr. By symmetry of the boundary forcing and bed
geometry, the front should advance and retreat radially while
remaining rotationally symmetric.

This benchmark exists primarily as a **symmetry-preservation test**
for the level-set / calving-step pipeline.

## Reference

- CalvingMIP wiki: <https://github.com/JRowanJordan/CalvingMIP/wiki>
- Yelmo Fortran reference: `yelmo/src/physics/calving/calving_ac.f90:484-533`
  (`calvmip_exp2`).

## How to run

```bash
# 1. Generate the Exp1 restart (~minutes):
cd benchmarks/calvingmip-exp1
julia --project=. run.jl

# 2. Run the oscillating-front phase. Stops automatically once the
#    asymmetry metric exceeds the threshold (default 5%):
cd ../calvingmip-exp2
julia --project=. run.jl
julia --project=. summary.jl
```

## Asymmetry metric

Each snapshot, eight cardinal/intercardinal radii are measured from
the domain centre to the first `lsf = 0` crossing along each ray.
The asymmetry metric is

    asym = (max − min) / mean

over the eight radii. The run terminates the moment `asym >
ASYM_THRESHOLD` (default 0.05) — once asymmetry develops in this
benchmark it grows monotonically, so finishing the full 5-cycle
protocol adds no information. The state at the threshold-crossing
moment is saved as `output/restart_at_threshold.nc` for inspection.

## Outputs

- `output/timeseries.nc`        — per-snapshot 8 front radii + asymmetry metric.
- `output/restart_at_threshold.nc` — full 2D state (H_ice, lsf, ux, uy,
  bed/forcing) at the asymmetry threshold; nothing is written if the
  threshold is never reached.
- `summary.json` — peak asymmetry, time of crossing, per-direction
  front-radius range. Committed.

`output/` and `plots/` are gitignored.

## Notes

- The CalvingMIPBenchmark struct does not currently distinguish
  `:exp1` and `:exp2` in any way that affects the in-memory IC — both
  share the same circular-bowl geometry and constant SMB / T_srf /
  Q_geo. The `exp` symbol only selects the per-experiment Fortran
  namelist and lets the calving-law dispatch keep them straight.
- The Yelmo namelist `yelmo_calvingmip_exp2.nml` is committed inside
  this directory so the benchmark is self-contained.
- The 8-direction radii use cell-centre indices nearest the origin
  (cell-centred grid with even `Nx` puts the origin between two
  cells); the choice is symmetric under the 90° rotations.
