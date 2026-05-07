# IceSheetBenchmarks.jl

Portable, model-agnostic Julia package providing the standard
ice-sheet benchmark specifications (geometry, forcing, analytical or
empirical reference solutions). Benchmarks are returned as plain Julia
structs and as model-agnostic NamedTuples of fields, so any ice-sheet
model implementation can consume them.

> **Note**: this is a vendored copy under `benchmarks/IceSheetBenchmarks/`
> for use by the runnable benchmarks in `Yelmo.jl/benchmarks/`. The
> intended long-term home is the standalone repo at
> [fesmc/IceSheetBenchmarks.jl](https://github.com/fesmc/IceSheetBenchmarks.jl);
> when that lands, this directory may be replaced with a regular
> registered dependency.

## What's included

- `AbstractBenchmark` — interface contract.
- `state(b, t)` — analytical state at time `t` as a NamedTuple of
  ice-sheet fields (`H_ice`, `z_bed`, `smb_ref`, `T_srf`, `Q_geo`,
  `z_sl`, …).
- `write_fixture!(b, path; times)` — write the analytical state to a
  NetCDF restart file.

### Concrete benchmarks

| Benchmark                  | Status |
| -------------------------- | ------ |
| `EISMINT1MovingBenchmark`  | ✅     |
| `MISMIP3DBenchmark(:Stnd)` | ✅     |

## Yelmo glue

A package extension `YelmoBenchmarks` activates when both
`IceSheetBenchmarks` and `Yelmo` are loaded. It provides
`Yelmo.YelmoModel(b::AbstractBenchmark, t; …)` to build a Yelmo model
in memory directly from a benchmark spec.

## TODO

- Migrate the remaining benchmark structs from
  `Yelmo.jl/test/benchmarks/` (BUELER, MISMIP3D, ISMIP-HOM, trough,
  …) into this package and have `test/benchmarks/` consume them.
