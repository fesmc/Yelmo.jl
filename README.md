# Yelmo.jl

This package is designed to allow running Yelmo interactively from within the Julia environment.

## Installation

First make sure that Yelmo is downloaded, configured and compiled somewhere on your system.
Then, download Yelmo.jl and add a link to Yelmo.

```bash
git clone git@github.com:fesmc/Yelmo.jl.git
cd Yelmo.jl
ln -s /path/to/yelmo    # link to yelmo installation
cd yelmo
make yelmo-c            # compile c-interface to yelmo
cd ..                   # back to Yelmo.jl
```

Now Yelmo.jl is ready to use from within Julia:

```julia
julia> using Pkg
julia> Pkg.add(path/to/Yelmo.jl)
julia> using Yelmo
```

## Quick-start

```julia
using Yelmo

# Build a YelmoModel from a Yelmo Fortran restart NetCDF.
y = YelmoModel("yelmo_restart.nc", 0.0;
               alias  = "demo",
               groups = (:bnd, :dyn, :mat, :thrm, :tpo),
               strict = false)

init_state!(y, 0.0)
out = init_output(y, "demo.nc")

for k in 1:5
    step!(y, 1.0)
    write_output!(out, y)
end

close(out)
```

A Fortran-backed [`YelmoMirror`] that goes through `libyelmo` exists
behind the same interface — see the documentation linked below.

## Documentation

The full documentation lives under [`docs/`](docs/) as a Documenter.jl
site. Build locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

then open `docs/build/index.html`. The site covers:

- **Getting started** — install, link the Yelmo Fortran tree, run a
  five-step smoke test.
- **Concepts** — the two backends (`YelmoModel` vs `YelmoMirror`),
  the six-component state, the Arakawa C grid, parameters vs constants.
- **Usage** — loading from a restart, stepping the model, NetCDF
  output and selection, lockstepping the two backends.
- **Physics** — the topography step's 21-phase pipeline, advection,
  mass balance (SMB / BMB / FMB / DMB / residual cleanup), the CISM
  subgrid-grounded-fraction algorithm with analytical-limit fixes,
  optional relaxation, and the level-set calving formulation.
- **API reference** — auto-generated from the docstrings, one page
  per public-facing module.
- **Variables** — the canonical variable tables for the six state
  groups (and how to add a new field).