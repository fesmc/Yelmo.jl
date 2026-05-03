# `examples/`

Development-time scratchpads — **not** part of the test suite. These scripts
were originally under `test/` with a `test_` prefix but never functioned as
CI-runnable tests; they reference machine-local paths or namelist files that
are not included in the repository.

| Script                          | What it demonstrates                                                |
| ------------------------------- | ------------------------------------------------------------------- |
| `test_yelmo.jl`                 | End-to-end YelmoMirror smoke run for a Greenland configuration: load namelist, initialize boundary + state, open NetCDF output. Requires a local `Greenland.nml` (or equivalent). |
| `test_yelmo_io.jl`              | Load a Yelmo restart NetCDF directly and compute diagnostic fields. References a hardcoded `yelmox` restart path; edit before running.                                              |
| `mismip3d_att_ramp/run_ramp.jl` | Full MISMIP3D Stnd + ATT-ramp grounding-line migration run (3 phases × 2000 yr ≈ 5–10 min wall time). Writes a NetCDF time series and a 2-panel `A(t)` / `GL_x(t)` PNG to `<repo>/logs/mismip3d_att_ramp/`. Stretched variant of the CI test `test/benchmarks/test_mismip3d_stnd_att.jl`; no assertions. |

All pull in `CairoMakie` for plotting. Edit the paths / parameters at the
top of each script before invoking with `julia --project=test <file>`.

For tests that run as part of CI / `Pkg.test()`, see `test/`.
