# Input / output API

The output module is in `Yelmo.YelmoIO`; restart-loading helpers live
in `Yelmo.YelmoCore` and are documented on the
[core API page](core.md#loading-from-a-netcdf-restart). Both surfaces
dispatch on [`AbstractYelmoModel`](@ref) so [`YelmoModel`](@ref) and
[`YelmoMirror`](@ref) share the same output API.

See the [output guide](../usage/io.md) for the narrative.

## Output lifecycle

```@docs
init_output
write_output!
```

Closing the file is via the standard `close(out)` — `Base.close` is
extended for the internal `YelmoOutput` struct returned by
[`init_output`](@ref).

## Variable selection

```@docs
OutputSelection
```

Patterns are matched against variable names via the `matches_patterns`
helper (a thin `any(p -> occursin(p, name), …)` wrapper). Patterns can
be plain strings (substring match), regex literals, or any type
accepted by `occursin(p, name::String)`.

## Variable metadata

The variable tables under `src/variables/{model,mirror}/` are parsed
into `VariableMeta` entries by [`parse_variable_table`](@ref) at
constructor time. Field allocation, NetCDF dimension names, and per-
group disambiguation all key off these entries.

```@docs
parse_variable_table
```
