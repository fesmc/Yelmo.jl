# Rules for Claude

- Don't modify any code related to the YelmoMirror object, without explicit instruction and approval from the user. This means YelmoMirrorCoreFields.jl and YelmoMirrorCoreMatrices.jl.
- The Fortran Yelmo source code is the definitive guide for how to implement things in terms of the physics or model features/options. Yelmo.jl is meant to port the functionality of Yelmo as faithfully as possible. The Fortran Yelmo source code is available in the linked directory yelmo/.