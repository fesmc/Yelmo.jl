# Output and NetCDF

The output module is backend-agnostic: [`init_output`](@ref) and
[`write_output!`](@ref) both dispatch on
[`AbstractYelmoModel`](@ref), so the same code works against a
`YelmoModel` or a `YelmoMirror`.

## Two-call lifecycle

```julia
out = init_output(y, "run.nc";
                  selection = OutputSelection(),
                  deflate   = 4)

# ... time loop, calling write_output!(out, y) each step ...

close(out)
```

`init_output` opens the NetCDF file, defines all dimensions and
coordinate variables, and pre-defines one variable per field that
passes `selection`. `write_output!` appends one time slice (at
`y.time`) on each call. `close(out)` flushes and closes the dataset.

The output file's variable list is fixed at `init_output` time. If
you need to change the selection mid-run, close and re-open with a
fresh `init_output` call — `write_output!` will not silently start
defining new variables.

## Selecting variables

[`OutputSelection`](@ref) supports four modes:

```julia
# 1. Everything (default).
OutputSelection()

# 2. Restrict to specific groups.
OutputSelection(groups = [:tpo, :bnd])

# 3. Filter by variable-name patterns across all selected groups.
OutputSelection(; include = ["H_ice", "z_srf", r"^smb"])
OutputSelection(; exclude = [r"^jvel", r"^strs", r"^strn"])

# 4. Per-group variable lists (most explicit).
OutputSelection([
    :tpo => ["H_ice", "f_grnd", "z_srf"],
    :dyn => ["ux", "uy", r".*_acx$"],
])
```

Patterns can be plain strings (matched as substrings), regex
literals, or any value accepted by Julia's `occursin(p, ::String)`.
Per-group patterns take priority over `include`/`exclude` when both
are given.

## NetCDF schema

Spatial dimensions are written based on each field's location:

| Yelmo node | NetCDF dim names |
|---|---|
| Cell centre (aa) | `x_c`, `y_c`, (`zeta` if 3D ice, `zeta_rock` if 3D rock) |
| x-face (acx)    | `x_f`, `y_c`, … |
| y-face (acy)    | `x_c`, `y_f`, … |
| z-face          | `x_c`, `y_c`, `zeta_ac` |

`time` is the unlimited dimension. All variables are written as
`Float32` with `_FillValue = NaN` and the requested deflate level
(0–9, default 4). Coordinates are written as `Float64` with `units`
and `long_name` attributes.

## Name-collision handling

Some field names appear in more than one group — `tau_relax` is in
both `bnd` and `tpo`, for example. NetCDF uses a flat namespace, so
the second occurrence cannot be written under the same name. The
output module detects collisions during `init_output` and
disambiguates by group prefix: `bnd_tau_relax` and `tpo_tau_relax`.
Names that occur in only one selected group are written without a
prefix.

The disambiguation map is stored on the `YelmoOutput` object that
[`init_output`](@ref) returns (field `out.nc_names`) for inspection.

## Example: minimal output for tracking margins

```julia
sel = OutputSelection([
    :tpo => ["H_ice", "f_grnd", "z_srf", "mb_net", "dHidt"],
    :bnd => ["smb_ref", "z_bed"],
])

out = init_output(y, "run.nc"; selection=sel, deflate=6)
for k in 1:N
    step!(y, dt)
    write_output!(out, y)
end
close(out)
```

This writes seven 2D fields per time slice — useful for tracking
margin evolution and the closure of the mass balance over a long
run without paying for the full state.

## Reading the output back

Use any NetCDF reader. From Julia:

```julia
using NCDatasets
ds = NCDataset("run.nc")
H = ds["H_ice"][:, :, end]    # last time slice
t = ds["time"][:]
close(ds)
```

The same restart-loader functions (`load_grids_from_restart`,
`load_field_from_dataset_2D`/`_3D`, `load_fields_from_restart`) work
on output files too — useful for restarting a run from a previous
output, or for spinning up a comparison `YelmoModel` from another
model's state.
