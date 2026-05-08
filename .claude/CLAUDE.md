# Rules for Claude

- The Fortran Yelmo source code is the definitive guide for how to implement things in terms of the physics or model features/options. Yelmo.jl is meant to port the functionality of Yelmo as faithfully as possible. The Fortran Yelmo source code is available in the linked directory yelmo/.
- Wherever possible, implement solutions that will work with multi-threading on CPU and even a GPU backend, as these are long-term goals to include support for in Yelmo.jl.
- Do not modify the YelmoMirror interface. YelmoMirror wraps Fortran via the C API and has no hooks, custom fields, or Julia-side callbacks. All calving and physics are handled internally by Fortran when using YelmoMirror. Hooks and custom laws belong on YelmoModel only.
