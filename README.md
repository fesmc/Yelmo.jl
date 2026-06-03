# Yelmo.jl

A Julia implementation of the Yelmo ice-sheet model. The pure-Julia
[`YelmoModel`](docs/src/api/) backend is the primary path; an optional
[`YelmoMirror`](docs/src/api/mirror.md) backend wraps the Fortran model
through a C API for side-by-side comparison.

## Installation

```bash
git clone git@github.com:fesmc/Yelmo.jl.git
```

```julia
using Pkg
Pkg.develop(path="/path/to/Yelmo.jl")
using Yelmo
```

The pure-Julia backend has no Fortran dependency.

## Quick-start: Greenland (initmip-grl)

The `benchmarks/initmip-grl/` directory runs a present-day Greenland
configuration from defaults — a `YelmoParameters` value built in code,
topography and forcing loaded from NetCDF, no namelist or restart file.

```bash
cd benchmarks/initmip-grl
julia --project=. -e 'include("run.jl"); main(t_end=1.0)'  # one-year smoke test
julia --project=. -e 'include("run.jl"); main()'           # default 20 yr
```

See [`benchmarks/initmip-grl/README.md`](benchmarks/initmip-grl/README.md)
for the configuration, data sources, and how to adapt it.

### Bare-bones equivalent

The same pattern in a few lines — construct parameters, build the
model, step it forward, write output:

```julia
using Yelmo

p = YelmoParameters("demo";
    yelmo = yelmo_params(domain = "Greenland", grid_name = "GRL-16KM",
                         grid_path = "path/to/GRL-16KM_REGIONS.nc"),
    # ... yelmo_init_topo, yelmo_masks, forcing groups ...
)

y = YelmoModel(p, 0.0)
init_state!(y, 0.0)
out = init_output(y, "demo.nc")

for k in 1:5
    step!(y, 1.0)
    write_output!(out, y)
end

close(out)
```

## Optional: Fortran-backed `YelmoMirror`

To run the same configuration through the Fortran model behind the
same interface, build the C API shared library and select the mirror
backend:

```bash
ln -s /path/to/yelmo yelmo
cd yelmo && make yelmo-c        # builds libyelmo/include/libyelmo_c_api.so
```

```bash
cd benchmarks/initmip-grl
julia --project=. -e 'include("run.jl"); main(backend=:mirror)'
```

`YelmoMirror` requires `libyelmo_c_api.so`; the pure-Julia
`YelmoModel` does not.

## Documentation

Full documentation lives under [`docs/`](docs/) as a Documenter.jl
site, covering installation, both backends, the six-component state
and Arakawa C grid, the topography pipeline (advection, mass balance,
grounded fraction, calving), the API reference, and the canonical
variable tables. Build locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

then open `docs/build/index.html`.
