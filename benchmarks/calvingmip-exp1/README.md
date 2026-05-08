# `calvingmip-exp1`

CalvingMIP Experiment 1: equilibrium calving on the circular bowl
domain. Ice grows from a cold start under constant SMB until the
calving front locks at `r = 750 km`, where `calvmip_exp1!` cancels ice
outflow exactly. The steady-state solution is rotationally symmetric
about the origin.

This benchmark exercises:

- the in-memory `YelmoModel(::CalvingMIPBenchmark, t)` constructor;
- the level-set front-tracking pipeline (`lsf_update!`, redistancing,
  above-SL pin in `calving_step!`);
- the user-supplied calving-law hook (`y.hooks.calv_flt`).

## Reference

- CalvingMIP wiki: <https://github.com/JRowanJordan/CalvingMIP/wiki>
- Cornford, S. L. *et al.* (2020) "Results of the third Marine Ice
  Sheet Model Intercomparison Project (MISMIP+)". *The Cryosphere* 14.
- Yelmo Fortran reference:
  - bed geometry — `yelmo/tests/calving_benchmarks.f90:63-87`
  - exp1 calving law — `yelmo/src/physics/calving/calving_ac.f90:395-482`

## How to run

```bash
cd benchmarks/calvingmip-exp1
julia --project=. run.jl       # ≈ a few minutes; produces output/*.nc
julia --project=. summary.jl   # writes summary.json
```

The first run will instantiate the local project (resolving
`Yelmo` and `IceSheetBenchmarks` via `[sources]` paths in
`Project.toml`).

## Outputs

- `output/timeseries.nc` — 100-yr-cadence snapshots of `max_H`,
  `mean_H`, `total_volume_m3`, ice-covered cell count, and
  ocean-mask cell count.
- `output/restart_final.nc` — full 2D state at `t = T_END_YR`,
  consumed by `benchmarks/calvingmip-exp2/run.jl`.
- `summary.json` — committed high-level statistics + metadata.

`output/` and `plots/` are gitignored; `summary.json` is committed.

## Notes

- Default `T_END_YR = 10 000`; the front pins at 750 km well before
  this. Shorten if you only need a near-equilibrium state.
- The Yelmo namelist `yelmo_calvingmip_exp1.nml` is committed inside
  this directory so the benchmark stays self-contained (no dependency
  on `test/benchmarks/specs/`). Keep the two copies in sync if you
  edit either.
- Symmetry: the steady-state cap is rotationally symmetric within
  numerical tolerance. If you observe asymmetry above ~1 %, that's a
  regression — see the LSF-asymmetry investigation logged against
  Exp2.
