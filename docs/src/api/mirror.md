# Yelmo Mirror

The Fortran-backed backend. `YelmoMirror` wraps a `libyelmo` instance
behind the same `init_state!` / `step!` / output API used by
[`YelmoModel`](@ref). Each `step!` is a `ccall` round-trip: push the
Julia-side mirror state into Fortran via `yelmo_sync!`, advance the
solver, pull every field back into Julia.

The Mirror requires a compiled `libyelmo_c_api.so` — see the
[getting-started guide](../getting-started.md) for the build
instructions.

## Constructors

```julia
YelmoMirror(filename::String, time::Float64;
            alias="ylmo1", rundir="./", overwrite=false)

YelmoMirror(p::YelmoParameters, time::Float64;
            grid=nothing,
            alias="ylmo1", rundir="./", overwrite=false)
```

The first form reads a Fortran namelist file into a
[`YelmoParameters`](@ref) and hands off to the second.

`alias` is the C-side identifier of this Yelmo instance (multiple
mirrors can coexist if their aliases differ). `rundir` is where the
constructor writes the namelist that Fortran reads back; existing
files in `rundir` are preserved unless `overwrite = true`.

## Building from a grid definition

The `grid` keyword controls how Fortran obtains the model grid:

- `grid === nothing` (default): Fortran runs `yelmo_init` with
  `grid_def = "file"`. Grid axes and topography are read from the
  NetCDF file pointed to by `&yelmo_init_topo / init_topo_path`.

- `grid::NamedTuple`: a synthetic grid is constructed in Fortran from
  explicit axes before `yelmo_init(grid_def="none")` runs. Required
  fields are `xc::Vector`, `yc::Vector` (cell-centre coordinates in
  metres). Optional fields are `grid_name::String` (default
  `"synthetic"`), `lon`, `lat` (default zeros), and `area` (default
  `dx * dy`). The namelist must set `init_topo_load = false` so
  Fortran does not attempt to load topography from disk.

```julia
using Yelmo

p = read_nml("yelmo_BUELER.nml")    # YelmoParameters

xc = collect(-500e3:5e3:500e3)      # cell centres in metres
yc = collect(-500e3:5e3:500e3)

ymf = YelmoMirror(p, 0.0;
    grid   = (xc=xc, yc=yc, grid_name="BUELER"),
    alias  = "bueler",
    rundir = "./output",
    overwrite = true,
)
```

## Loading a restart

The Mirror ingests a restart through the Fortran namelist, not a Julia
keyword. Set the `&yelmo_init_topo` block to point at the restart
NetCDF, then construct as usual:

```fortran
&yelmo_init_topo
    init_topo_load  = True
    init_topo_path  = "path/to/restart.nc"
    init_topo_names = "H_ice", "z_bed", "z_bed_sd", "z_srf"
    init_topo_state = 0
/
```

```julia
ymf = YelmoMirror("yelmo_GRL.nml", 0.0; alias="grl", rundir="./output")
init_state!(ymf, 0.0)
```

The same field set ([`YelmoInitTopoParams`](@ref)) is exposed as Julia
keywords on `yelmo_init_topo_params(...)` if you build the parameters
programmatically.

## Saving a restart

```julia
yelmo_write_restart!(ymf, "output/run_5kyr.nc"; time=ymf.time)
```

Writes the current Fortran state as a NetCDF restart with the same
schema Fortran Yelmo writes itself. The file is loadable by:

- another `YelmoMirror` via the namelist mechanism above, or
- a [`YelmoModel`](@ref) directly: `YelmoModel("run_5kyr.nc", t; ...)`.

`time` defaults to `ymf.time`. Parent directories are created if
missing.

## Stepping

```julia
init_state!(ylmo::YelmoMirror, time::Float64; thrm_method="robin-cold")
step!(ylmo::YelmoMirror, dt::Float64)
```

`init_state!` syncs Julia → Fortran, calls `yelmo_init_state` (which
populates thermodynamics from the chosen profile and runs the
predictor/corrector spin-up), then pulls everything back. Call once
after construction, before the time loop.

`step!` advances the Fortran solver by `dt` years. One call is one
full Fortran predictor / corrector step.

## Updating a field before the next step

To inject a Julia-side value into Fortran before stepping, write the
field's interior and then sync:

```julia
# 2D field — Oceananigans Field, write to its interior
fill!(interior(ymf.bnd.smb_ref), 0.5)        # m/yr surface mass balance
interior(ymf.tpo.H_ice)[:, :, 1] .= H_user    # custom thickness IC

yelmo_sync!(ymf)     # push Julia → Fortran
```

`yelmo_sync!` pushes the full `bnd` group plus a fixed subset of state
fields: `tpo.H_ice`, `dyn.cb_ref`, `dyn.N_eff`, `dyn.ux`, `dyn.uy`,
`dyn.uz`, `thrm.T_ice`, `thrm.H_w`. Any other field edited on the
Julia side will be **overwritten** by the next `step!` (which pulls
state back from Fortran).

To push a field that `yelmo_sync!` does not cover, call the low-level
setter directly:

```julia
buf = ymf.buffers.v2D                      # reuse the preallocated buffer
copyto!(buf, interior(some_field))
yelmo_set_var2D!(buf, ymf.v.<group>.<name>.cname, ymf.calias)
```

`yelmo_get_var2D!` / `yelmo_get_var3D!` and `yelmo_set_var2D!` /
`yelmo_set_var3D!` are exported for this purpose. `cname` is the
null-terminated `Vector{UInt8}` of the variable's Fortran name (the
metadata stored in `ymf.v.<group>.<varname>.cname`); `calias` is the
model instance's C alias (`ymf.calias`).

`step!` calls `yelmo_sync!` automatically, so no manual sync is needed
for fields in the synced subset — just edit the Julia field and step.

## Writing output during a time loop

[`init_output`](@ref) and [`write_output!`](@ref) are shared with
`YelmoModel`. See [usage/io.md](../usage/io.md) for the full options
(variable selection, deflate, name collisions). A minimal pattern:

```julia
out = init_output(ymf, "./output/run.nc")

for k in 1:N
    step!(ymf, dt)
    write_output!(out, ymf)
end

close(out)
```

## A complete time-loop example

```julia
using Yelmo

# Parameters from a Fortran namelist (synthetic-grid run, so the
# namelist sets init_topo_load = false).
p = read_nml("yelmo_BUELER.nml")

# Synthetic 5 km grid.
xc = collect(-500e3:5e3:500e3)
yc = collect(-500e3:5e3:500e3)

ymf = YelmoMirror(p, 0.0;
    grid      = (xc=xc, yc=yc, grid_name="BUELER"),
    alias     = "bueler",
    rundir    = "./output",
    overwrite = true,
)

# Set initial conditions on the Julia side and push to Fortran.
H0 = zeros(length(xc), length(yc))
# ... fill H0 with the analytical Halfar profile ...
interior(ymf.tpo.H_ice)[:, :, 1] .= H0
fill!(interior(ymf.bnd.smb_ref), 0.0)
fill!(interior(ymf.bnd.T_srf),   263.15)
yelmo_sync!(ymf)

init_state!(ymf, 0.0)

out = init_output(ymf, "./output/run.nc")

dt    = 1.0
T_end = 1000.0
while ymf.time < T_end - 1e-9
    step!(ymf, dt)
    write_output!(out, ymf)
end

close(out)

# Restart at t = T_end for later reuse.
yelmo_write_restart!(ymf, "./output/run_t1000.nc")
```

## Differences from YelmoModel

- Variable layout comes from `src/variables/mirror/`, which mirrors
  the Fortran NetCDF schema (e.g. `cmb_flt_x` / `cmb_flt_y` rather
  than the staggered-grid suffixes `cmb_flt_acx` / `cmb_flt_acy`).
- Parameters are [`YelmoParameters`](@ref) (from `Yelmo.YelmoPar`)
  rather than [`YelmoModelParameters`](@ref).
- Physics is delegated to Fortran — no `<comp>_step!` Julia helpers
  are involved.

The shared API surface ([`init_output`](@ref), [`write_output!`](@ref),
[`compare_state`](@ref), the time-loop pattern) is identical between
the two backends.
