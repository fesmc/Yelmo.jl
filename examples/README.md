# `examples/`

Development-time scratchpads — **not** part of the test suite. These scripts
were originally under `test/` with a `test_` prefix but never functioned as
CI-runnable tests; they reference machine-local paths or namelist files that
are not included in the repository.

| Script                | What it demonstrates                                                |
| --------------------- | ------------------------------------------------------------------- |
| `test_yelmo.jl`       | End-to-end YelmoMirror smoke run for a Greenland configuration: load namelist, initialize boundary + state, open NetCDF output. Requires a local `Greenland.nml` (or equivalent). |
| `test_yelmo_io.jl`    | Load a Yelmo restart NetCDF directly and compute diagnostic fields. References a hardcoded `yelmox` restart path; edit before running.                                              |

Both pull in `CairoMakie` for plotting. Edit the paths / parameters at the
top of each script before invoking with `julia --project=test <file>`.

For tests that run as part of CI / `Pkg.test()`, see `test/`.
