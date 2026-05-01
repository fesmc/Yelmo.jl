# Yelmo Mirror API

The Fortran-backed backend. Lives in `Yelmo.YelmoMirrorCore` and is
re-exported at the package level. See the
[concepts page](../concepts.md) for the backend split rationale.

The Mirror requires a compiled `libyelmo_c_api.so` — see the
[getting-started guide](../getting-started.md) for the build
instructions.

## Container type

`YelmoMirror` is the concrete subtype of [`AbstractYelmoModel`](@ref)
that wraps the Fortran solver. The struct holds field arrays mirrored
from Fortran (`bnd`, `dta`, `dyn`, `mat`, `thrm`, `tpo`), the model
parameters [`YelmoParameters`](@ref), and the C-aliased name buffer
that identifies this Yelmo instance to `libyelmo`.

Constructor signatures:

```julia
YelmoMirror(filename::String, time::Float64;
            alias="ylmo1", rundir="./", overwrite=false)

YelmoMirror(p::YelmoParameters, time::Float64;
            alias="ylmo1", rundir="./", overwrite=false)

YelmoMirror(g::NamedTuple, p::YelmoParameters, time::Float64;
            alias="ylmo1", rundir="./", overwrite=false)
```

The first form reads a Fortran namelist file as
[`YelmoParameters`](@ref) and hands off to the second. The third form
takes a synthetic-grid named-tuple (see `yelmo_define_grids`) and
initialises Yelmo without reading any topography file — useful for
analytical benchmarks like the BUELER-B Halfar dome.

## Stepping

The Mirror extends the generic `step!` and `init_state!` declared in
`YelmoCore`. Each call is a `ccall` round-trip: push Julia state to
Fortran via `yelmo_sync!`, advance the Fortran solver, pull state
back.

`yelmo_sync!(ylmo)` is exported for users who want to push a manually
edited Julia-side field into the Fortran object before the next
`step!` (or before writing a restart with `yelmo_write_restart!`).

## Restart write-out

`yelmo_write_restart!(ylmo, filename; time=nothing)` writes the
current Fortran state to a NetCDF restart that
[`YelmoModel`](@ref)`(restart_file, time)` can load directly. Useful
for spinning up the pure-Julia backend from a Fortran-Yelmo run.

## Low-level field accessors

The Mirror exposes per-variable `ccall` wrappers for reading and
writing one variable at a time across the C boundary, mostly used
internally by `yelmo_sync!` and the per-step pull-back. Exported for
users who want bulk-edit a single variable without invoking a full
sync:

- `yelmo_get_var2D!(buffer, cname, calias)` /
  `yelmo_get_var3D!(buffer, cname, calias)` — copy Fortran field to
  Julia buffer.
- `yelmo_set_var2D!(buffer, cname, calias)` /
  `yelmo_set_var3D!(buffer, cname, calias)` — copy Julia buffer to
  Fortran field.

`cname` is a null-terminated `Vector{UInt8}` of the variable's
Fortran-side name; `calias` is the model instance's C alias.

## Differences from YelmoModel

- Variable layout comes from `src/variables/mirror/`, which mirrors
  the Fortran NetCDF schema (e.g. `cmb_flt_x`/`cmb_flt_y` rather
  than the staggered-grid suffix `cmb_flt_acx`/`cmb_flt_acy`).
- Parameters are [`YelmoParameters`](@ref) (from `Yelmo.YelmoPar`)
  rather than [`YelmoModelParameters`](@ref).
- All physics is delegated to Fortran — no `<comp>_step!` Julia
  helpers are involved.

The shared API surface ([`init_output`](@ref), [`write_output!`](@ref),
[`compare_state`](@ref), the time-loop pattern) is identical between
the two backends.
