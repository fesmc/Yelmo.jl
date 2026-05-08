# `calvingmip-exp3`

CalvingMIP Experiment 3: equilibrium calving on the **Thule domain** with
the front pinned at r = 750 km. Same calving law as Exp1 (`calvmip_exp1!`)
but on the Thule bed — a parabolic bowl with cosine undulations:

```
l(θ)  = R0 − cos(2θ) · R0/2
z_bed = Ba · cos(3π r / l(θ)) + Bc − (Bc − Bl) · r² / R0²
```

(R0 = 800 km, Bc = 900 m, Bl = −2000 m, Ba = 1100 m.)

The Thule bed is not rotationally symmetric, so the equilibrium ice cap is
expected to be asymmetric. The calving law still pins the front at the
750 km circle in the calving-rate sense, but the actual front position
reflects the bed-driven thickness variations.

This experiment produces a `restart_final.nc` that seeds Exp4.

## Reference

- CalvingMIP wiki: <https://github.com/JRowanJordan/CalvingMIP/wiki>
- Yelmo Fortran reference: `yelmo/src/physics/calving/calving_ac.f90:395-482`
  (`calvmip_exp1`, "Experiment 1 & 3 of CalvMIP").

## How to run

```bash
cd benchmarks/calvingmip-exp3
julia --project=. run.jl
julia --project=. summary.jl
```

## Outputs

- `output/timeseries.nc`   — time, ice volume, max H, ice cell count.
- `output/restart_final.nc` — full 2D state at t = T_END_YR; consumed by
  `calvingmip-exp4/run.jl`.
- `summary.json`           — final-state statistics. Committed.

`output/` and `plots/` are gitignored.
