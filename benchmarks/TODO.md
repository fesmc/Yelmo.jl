# `benchmarks/` — next steps

Tracking items for the runnable-benchmarks effort. Sister doc to the
top-level [TODO.md](../TODO.md) (which tracks broader Yelmo.jl porting
items).

## Next benchmarks to add

In priority order — each follows the existing scaffolder pattern
(`julia benchmarks/new.jl <name>`):

- [ ] **`mismip3d-stnd-att-ramp`** — same setup as `mismip3d-stnd`
  with phased changes to the Glen rate factor. Stretched 3-phase
  variant of `test/benchmarks/test_mismip3d_stnd_att.jl`. The
  pre-benchmarks prototype lives at
  [`examples/mismip3d_att_ramp/run_ramp.jl`](../examples/mismip3d_att_ramp/run_ramp.jl)
  — port that into the benchmarks layout. Should reuse `mismip3d-stnd`'s
  `_build` and add a phase loop.

- [x] **`calvingmip-exp1`** and **`calvingmip-exp2`** — added.
  Exp1 spins up the equilibrium ice cap (`calvmip_exp1!` pinned front);
  Exp2 chains from the Exp1 restart, runs the oscillating-front
  `calvmip_exp2!` law, tracks an 8-direction asymmetry metric, and
  stops on threshold. Spec + bed geometry live in
  [IceSheetBenchmarks.jl](https://github.com/fesmc/IceSheetBenchmarks.jl)
  (`src/calvingmip.jl`); calving-law hooks in the `YelmoBenchmarks`
  package extension.

- [ ] **CalvingMIP-Exp3..5** — remaining CalvingMIP experiments
  (Thule geometry). No reference solution; runnable reproductions for
  paper figures. Will share the bed-geometry helpers already in
  [IceSheetBenchmarks.jl](https://github.com/fesmc/IceSheetBenchmarks.jl)
  (`src/calvingmip.jl`).

## `init_state!` — remaining gaps

`Yelmo.init_state!(y, time; thrm_method="robin")` mirrors Fortran's
`yelmo_init_state` for the temperature + bedrock + initial dyn cycle.
Deferred items:

- [ ] **Restart-aware bypass.** Fortran's `if (dom%par%use_restart)`
  branch reads `yelmo_restart_read` and skips the equilibration
  cycle. Yelmo.jl callers that load from a NetCDF restart should
  currently just not call `init_state!`. Adding a `restart::Bool`
  kwarg (or detecting via a YelmoModel flag set by the file-based
  constructor) would match Fortran more closely. Pending a concrete
  use case.

- [ ] **`yelmo_regions_update`.** Fortran's final step is to refresh
  region-level diagnostics. Yelmo.jl regions are currently read-only
  (no `update_regions!` in `init_state!`). Add when a benchmark or
  test starts asserting on region outputs from t=0.

- [ ] **Numerical validation against Fortran.** The bedrock
  `define_temp_bedrock_3D!` call inside `init_thrm!` produces
  sensible values (`Q_rock = Q_geo`, T_rock spans ~30 K for default
  parameters) but hasn't been bit-for-bit checked against Fortran's
  reference. Add a dedicated unit test once a Fortran-side fixture is
  available.

## `mismip3d-stnd` — open questions

Current 2000-yr trajectory: GL retreats from 392 → 376 km. Pattyn
et al. (2013) reference shows much larger retreat. Plausible
contributors:

- [ ] Run length (2 kyr vs the reference's longer equilibration).
- [ ] dt sensitivity (1 yr forward Euler vs Fortran's adaptive PC).
- [ ] Missing physics in Yelmo.jl's SSA — the test still uses the
  `1000 - 0.9 z_bed` IC override because the literal Fortran 10 m
  IC is rank-deficient under SSA. Once Yelmo.jl gets adaptive
  dt + `dHdt_dyn_lim`, retest with the literal IC.

Investigation can wait — current trajectory is at least physically
sane (no clamped velocities, GL moves correctly).

## Infrastructure

- [x] **`IceSheetBenchmarks/` consolidation with `test/benchmarks/`.**
  Done in #83 — duplicate structs collapsed; test-side uses ISB types
  and adds only Yelmo-Mirror scaffolding.
- [x] **`benchmarks/IceSheetBenchmarks/` → standalone repo.** Lives at
  [fesmc/IceSheetBenchmarks.jl](https://github.com/fesmc/IceSheetBenchmarks.jl);
  pulled in as a git-pinned dependency. Registration in the General
  registry is a future step.
