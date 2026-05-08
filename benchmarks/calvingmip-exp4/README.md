# `calvingmip-exp4`

CalvingMIP Experiment 4: oscillating calving front on the **Thule domain**.
Chained from a `calvingmip-exp3` spin-up — the Exp3 restart provides the
equilibrium Thule ice cap as initial state.

Same calving rate law as Exp2 (`calvmip_exp2!`):
  w = (u/|u|) · wv,  wv = −300 sin(2π t / 1000) m/yr
with period 1000 yr over the full 5-cycle (5000 yr) protocol.

## Reference

- CalvingMIP wiki: <https://github.com/JRowanJordan/CalvingMIP/wiki>
- Yelmo Fortran reference: `yelmo/src/physics/calving/calving_ac.f90:484-533`
  (`calvmip_exp2`, "Experiment 2 & 4 of CalvMIP").

## How to run

```bash
# 1. Generate the Exp3 restart (~minutes):
cd benchmarks/calvingmip-exp3
julia --project=. run.jl

# 2. Run the oscillating-front phase (full 5-cycle, 5000 yr):
cd ../calvingmip-exp4
julia --project=. run.jl
julia --project=. summary.jl
```

## 8-direction front radii

Eight cardinal/intercardinal radii are measured by ray-casting bilinear-
interpolated `lsf` from the true origin `(0, 0)` and finding the first
sign change. Two metrics are recorded alongside:

- `asym4` — spread within cardinals (E/N/W/S) and within diagonals
  (NE/NW/SW/SE), each normalised by the group mean. On the Thule domain
  this is **not expected to be near zero** — the bed itself breaks
  4-fold symmetry — but it should remain stable across the 5 cycles.

- `asym8` — spread over all 8 directions, normalised by the mean.
  Carries a larger baseline from the Thule bed asymmetry.

No threshold check is applied. The radii are purely diagnostic: they
show how the Thule bed topography modulates the front position during
advance/retreat cycles.

## Outputs

- `output/timeseries.nc`         — per-snapshot 8 front radii + asym4/asym8.
- `output/snapshots_phase4.nc`   — 2D snapshots (H_ice, lsf, ux_bar, uy_bar)
  at `SAMPLE_DT_YR` cadence.
- `output/restart_final.nc`      — full 2D state at end of 5-cycle run.
- `summary.json`                 — peak asym metrics + per-direction radius
  range. Committed.

`output/` and `plots/` are gitignored.
