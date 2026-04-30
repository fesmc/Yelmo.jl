# Benchmark fixtures

NetCDF restart files written by YelmoMirror via
[`run_mirror_benchmark!`](../helpers.jl). One file per spec output time;
filename pattern is `<spec_name>__t<time-yr>.nc`.

These are committed binaries. They're tiny (synthetic grids run on small
domains; the BUELER-B smoke fixture is well under 100 KB), but the
provenance discipline matters because nothing in CI can regenerate them
without `libyelmo_c_api.so` available.

## Provenance

Each fixture should be regenerated whenever the Yelmo Fortran source or its
namelist drifts. Re-run [`../regenerate.jl`](../regenerate.jl) locally and
re-commit. NetCDF global attributes carry the Yelmo Fortran git hash and
generation date written by the Fortran `yelmo_restart_write` routine —
inspect via `ncdump -h <fixture>.nc | head` to verify which Fortran build
produced a fixture.

## Current fixtures

| Filename | Spec | Solver | Notes |
|---|---|---|---|
| `bueler_b_smoke__t1000.nc` | `bueler_b_smoke` | SIA | 31×31 at dx=50 km, Halfar IC, 1000-yr decay |
