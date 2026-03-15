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