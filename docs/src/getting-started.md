# Getting started

This page walks through installing `Yelmo.jl`, linking it against a
compiled Yelmo Fortran tree (only required for the
[`YelmoMirror`](@ref) backend), and running a short integration
script.

## Installation

`Yelmo.jl` is not yet registered. Clone the repository and add it as a
local development package:

```bash
git clone git@github.com:fesmc/Yelmo.jl.git
cd Yelmo.jl
```

If you intend to use the Fortran-backed
[`YelmoMirror`](api/mirror.md) backend, clone the Yelmo Fortran
source and symlink it into the package root, then build the C
interface:

```bash
ln -s /path/to/yelmo yelmo
cd yelmo
make yelmo-c           # builds libyelmo/include/libyelmo_c_api.so
cd ..
```

Then in Julia:

```julia
using Pkg
Pkg.develop(path="/path/to/Yelmo.jl")
using Yelmo
```

The pure-Julia [`YelmoModel`](@ref) backend has no Fortran dependency
and works without the symlink — useful for development on systems
where the Fortran toolchain isn't available.

## Five-step smoke test (`YelmoModel`)

The script below mirrors `test/test_yelmo_model.jl`. It builds a
`YelmoModel` from a Yelmo Fortran restart, opens a NetCDF output file,
and integrates for five years.

```julia
using Yelmo

# Path to a Yelmo Fortran restart NetCDF (any 2D grid; the constructor
# infers Nx, Ny, vertical zeta-axes, and ice/rock layer counts from
# the file). Replace with a restart from your own runs.
restart = "yelmo_restart.nc"

# Default parameters. Override individual groups via keyword arguments
# to YelmoModelParameters or by reading a Fortran namelist.
p = YelmoModelParameters("demo")

# Build the model. `:dta` is omitted because Fortran restarts do not
# carry a `data` group; `strict=false` allows individual variables in
# the loaded groups to be missing (they fall back to default-allocated
# values).
y = YelmoModel(restart, 0.0;
    alias  = "demo",
    rundir = "./output",
    p      = p,
    groups = (:bnd, :dyn, :mat, :thrm, :tpo),
    strict = false,
)

init_state!(y, 0.0)

# Open an output NetCDF; by default writes every field in every group.
out = init_output(y, "./output/demo.nc")

for k in 1:5
    step!(y, 1.0)        # one-year step: topo + dyn (SIA); mat/thrm pending
    write_output!(out, y)
end

close(out)
```

After running, the file `./output/demo.nc` will contain six time
slices (the initial state plus five steps) with one variable per field
in the loaded groups. Use the [output and NetCDF](usage/io.md) guide
for selecting subsets of variables.

## Five-step smoke test (`YelmoMirror`)

For the Fortran-backed mirror, replace the constructor and add a
namelist file describing the run:

```julia
using Yelmo

p   = read_nml("Yelmo_GRL.nml")          # YelmoParameters from a Fortran nml
ymf = YelmoMirror(p, 0.0; alias="demo", rundir="./output", overwrite=true)
init_state!(ymf, 0.0)

out = init_output(ymf, "./output/mirror.nc")
for k in 1:5
    step!(ymf, 1.0)                      # delegates to libyelmo via ccall
    write_output!(out, ymf)
end
close(out)
```

`YelmoMirror` uses the same `init_output` / `write_output!` API as
`YelmoModel` because both inherit from [`AbstractYelmoModel`](@ref).

## Running the test suite

```bash
cd Yelmo.jl
julia --project=test test/test_yelmo_model.jl
julia --project=test test/test_yelmo_topo.jl
julia --project=test test/test_yelmo_dyn.jl
julia --project=test test/test_yelmo_sia.jl
```

The model test depends on a Fortran-Yelmo restart fixture; the path
is hard-coded near the top of the file and may need adjustment for
your machine. The `test_yelmo_topo.jl` and `test_yelmo_dyn.jl` test
suites are mostly self-contained (synthetic kernels and analytical
references — see the
[grounded fraction page](physics/grounded-fraction.md) for the
convergence benchmark).

## Analytical benchmarks

The `test/benchmarks/` directory holds end-to-end analytical
benchmarks (currently the BUELER-B Halfar dome at `t = 1000 yr`).
They are runnable as scripts and exercise the full
`step!(y, dt)` chain (currently topo + SIA dynamics) against a
known closed-form ice-sheet evolution:

```bash
julia --project=test test/benchmarks/test_smoke.jl
julia --project=test test/benchmarks/test_sia.jl
```

The fixture restart file `bueler_b_t1000.nc` is checked in;
`test/benchmarks/regenerate.jl` rebuilds it from the analytical
Halfar formula if it ever needs refreshing.

## Examples

`examples/` contains scratchpad scripts and worked use cases (these
were previously under `test/`; they were moved to `examples/` to
keep `test/` focused on regression and benchmark tests). Browse
`examples/README.md` for the current list.

## Building this documentation locally

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

This produces a static HTML site under `docs/build/`. Open
`docs/build/index.html` in a browser. Math expressions render via
KaTeX (no extra dependency needed).
