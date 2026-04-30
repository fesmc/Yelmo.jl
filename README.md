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

To do...

## Documentation

- [`docs/topo-step.md`](docs/topo-step.md) — per-phase reference for
  `topo_step!`, the topography component's per-timestep orchestrator
  (advection, mass balance, residual cleanup, diagnostics).
- [`docs/grounded-fraction.md`](docs/grounded-fraction.md) —
  mathematical description of the subgrid grounded-fraction kernel
  (CISM bilinear-interpolation scheme of Leguy et al. 2021), with
  derivation of the analytical-limit stabilisations introduced by
  the Julia port.