# Rules for Claude

- The Fortran Yelmo source code is the definitive guide for how to implement things in terms of the physics or model features/options. Yelmo.jl is meant to port the functionality of Yelmo as faithfully as possible. The Fortran Yelmo source code is available in the linked directory yelmo/.
- Wherever possible, implement solutions that will work with multi-threading on CPU and even a GPU backend, as these are long-term goals to include support for in Yelmo.jl.
