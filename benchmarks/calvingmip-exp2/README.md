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

Each snapshot, eight cardinal/intercardinal radii are measured by
ray-casting bilinear-interpolated `lsf` from the true origin
`(0, 0)` and finding the first sign change. Two metrics are
recorded:

- `asym4 = max((max − min)/mean over E/N/W/S,
                (max − min)/mean over NE/NW/SW/SE)`
  Spread *within* each 4-fold orbit. **Stays at machine precision on
  any cap that is rotationally 4-fold symmetric** — i.e. it sees
  through the Cartesian grid's intrinsic cardinal-vs-diagonal
  anisotropy and only triggers on actual symmetry breaking. This
  metric drives the threshold check.

- `asym8 = (max − min)/mean over all 8 radii`
  Spread across all 8 directions. Carries a baseline of a few tenths
  of a percent on any finite-resolution Cartesian grid (the cap is
  "slightly square", with cardinal radii ~2 km longer than diagonals
  on a 25 km grid) and oscillates with the front position over the
  advance/retreat cycle. Reported for diagnostic comparison with the
  pre-fix behaviour.

The run terminates the moment `asym4 > ASYM_THRESHOLD` (default
0.05). The state at the threshold-crossing moment is saved as
`output/restart_at_threshold.nc` for inspection.

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
