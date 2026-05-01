module YelmoIO

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields

using NCDatasets
using ..YelmoMeta
using ..YelmoCore: AbstractYelmoModel, matches_patterns

export init_output
export OutputSelection
export write_output!

# NetCDF input-side helpers (load_grids_from_restart,
# load_field_from_dataset_2D/3D, load_fields_from_restart) now live in
# YelmoCore — they are construction helpers, not output. YelmoIO is
# output-only.

## Writing output ##

# ---------------------------------------------------------------------------
# NetCDF dimension helpers
# ---------------------------------------------------------------------------

struct OutputSelection
    groups    :: Union{Nothing, Vector{Symbol}}
    include   :: Union{Nothing, Vector}
    exclude   :: Union{Nothing, Vector}
    per_group :: Union{Nothing, Dict{Symbol, Vector}}  # group => name/regex patterns
end

OutputSelection(; groups=nothing, include=nothing, exclude=nothing) =
    OutputSelection(groups, include, exclude, nothing)

"""
    OutputSelection(group_vars)

Select specific variables per group, e.g.:

    OutputSelection([
        :tpo => ["H_ice", "z_srf"],
        :dyn => ["ux", "uy", r".*_acx\$"],
    ])
"""
OutputSelection(group_vars::Vector{Pair{Symbol, T}} where T) =
    OutputSelection(
        Symbol[k for (k, _) in group_vars],
        nothing,
        nothing,
        Dict{Symbol, Vector}(k => v for (k, v) in group_vars),
    )

function _selected(name::String, gname::Symbol, sel::OutputSelection)
    # per-group patterns take priority
    if sel.per_group !== nothing
        patterns = get(sel.per_group, gname, nothing)
        patterns === nothing && return false
        return matches_patterns(name, patterns)
    end
    sel.include !== nothing && !matches_patterns(name, sel.include) && return false
    sel.exclude !== nothing &&  matches_patterns(name, sel.exclude) && return false
    return true
end

# ---------------------------------------------------------------------------
# Location → NetCDF dimension name
# ---------------------------------------------------------------------------

_nc_x(::Type{Center}) = "x_c"
_nc_x(::Type{Face})   = "x_f"
_nc_y(::Type{Center}) = "y_c"
_nc_y(::Type{Face})   = "y_f"
_nc_z(::Type{Center}, ::Val{:ice})  = "zeta"
_nc_z(::Type{Face},   ::Val{:ice})  = "zeta_ac"
_nc_z(::Type{Center}, ::Val{:rock}) = "zeta_rock"
_nc_z(::Type{Face},   ::Val{:rock}) = "zeta_rock_ac"

_oc_loc(::Type{Center}) = Center()
_oc_loc(::Type{Face})   = Face()

function _zgrid_kind(grid, ylmo)
    grid === ylmo.gt && return Val(:ice)
    grid === ylmo.gr && return Val(:rock)
    return nothing
end

"""
Return a `Tuple{Vararg{String}}` of spatial NetCDF dimension names for
the given field, or `nothing` if the vertical grid is not recognised.
No heap allocation — branches on type parameters and grid identity only.
"""
function _spatial_dims(field::Field{X,Y,Z}, ylmo) where {X,Y,Z}
    grid  = field.grid
    has_x = topology(grid, 1) != Flat
    has_y = topology(grid, 2) != Flat
    has_z = topology(grid, 3) != Flat

    if has_z
        zk = _zgrid_kind(grid, ylmo)
        zk === nothing && return nothing
        zname = _nc_z(Z, zk)
        return has_x && has_y ? (_nc_x(X), _nc_y(Y), zname) :
               has_x           ? (_nc_x(X), zname)           :
               has_y           ? (_nc_y(Y), zname)           :
                                 (zname,)
    else
        return has_x && has_y ? (_nc_x(X), _nc_y(Y)) :
               has_x           ? (_nc_x(X),)          :
               has_y           ? (_nc_y(Y),)           :
                                 ()
    end
end

# ---------------------------------------------------------------------------
# Data extraction — no allocation
# ---------------------------------------------------------------------------

"""
Return a halo-free view of `field.data`, with any trailing singleton
dimensions (from Flat grid axes) dropped. No allocation.
"""
function _get_data(field::Field)
    arr = interior(field)
    while ndims(arr) > 1 && size(arr, ndims(arr)) == 1
        arr = dropdims(arr; dims=ndims(arr))
    end
    return arr
end

# ---------------------------------------------------------------------------
# YelmoOutput
# ---------------------------------------------------------------------------

struct YelmoOutput
    ds        :: NCDataset
    selection :: OutputSelection
    groups    :: Vector{Symbol}
    # NetCDF variable name per (group, fname). Mostly identical to
    # `String(fname)`, but a field name that appears in more than one
    # selected group is disambiguated as `<group>_<fname>` (e.g.
    # `bnd_tau_relax`, `tpo_tau_relax`). NetCDF uses a flat namespace
    # so the prefix is the only way to write both.
    nc_names  :: Dict{Tuple{Symbol, Symbol}, String}
    # Scratch entries (e.g. `dyn.scratch.sia_tau_xz`), populated only
    # when `init_output(...; include_scratch=true)`. Keyed by
    # (parent_group, scratch_subfname). NetCDF variable name is always
    # `<group>_scratch_<subfname>`, no collision-detection needed.
    scratch_nc_names :: Dict{Tuple{Symbol, Symbol}, String}
end

Base.close(out::YelmoOutput) = close(out.ds)

# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------

function _defcoord(ds, name, FT, dims, data, units, long_name)
    v = defVar(ds, name, FT, dims)
    v[:] = data
    v.attrib["units"]     = units
    v.attrib["long_name"] = long_name
    return v
end

# ---------------------------------------------------------------------------
# init_output
# ---------------------------------------------------------------------------

const _ALL_GROUPS = [:bnd, :dta, :dyn, :mat, :thrm, :tpo]

"""
    init_output(ylmo, path; selection, deflate, include_scratch) -> YelmoOutput

Create a NetCDF file at `path`. Defines all dimensions, coordinate variables,
and one NetCDF variable per field that passes `selection` and has a
recognised grid. Returns a `YelmoOutput` to be passed to `write_output!`.

Set `include_scratch=true` to also serialize solver-internal scratch
buffers (e.g. `dyn.scratch.sia_tau_xz`) — useful for debugging the SIA
solver's intermediate state. Off by default so production runs don't
bloat NetCDFs with non-state fields. Scratch fields belong to their
parent group, so `OutputSelection`'s group filter still applies; their
NetCDF variable names are prefixed with `<group>_scratch_` (e.g.
`dyn_scratch_sia_tau_xz`).
"""
function init_output(ylmo::AbstractYelmoModel, path::String;
                     selection::OutputSelection = OutputSelection(),
                     deflate::Int = 4,
                     include_scratch::Bool = false)

    active_groups = selection.groups === nothing ? _ALL_GROUPS : selection.groups

    ds = NCDataset(path, "c")

    # ---- dimensions -------------------------------------------------------
    # Face dimension sizes depend on the horizontal-axis topology:
    #   - Bounded: Nx+1 / Ny+1 face slots (boundary endpoints stored).
    #   - Periodic: Nx / Ny (the wrap-around face is the 1st face and is
    #     not stored separately).
    Tx_g = topology(ylmo.g, 1)
    Ty_g = topology(ylmo.g, 2)
    x_f_size = Tx_g === Periodic ? ylmo.g.Nx : ylmo.g.Nx + 1
    y_f_size = Ty_g === Periodic ? ylmo.g.Ny : ylmo.g.Ny + 1
    defDim(ds, "x_c",         ylmo.g.Nx)
    defDim(ds, "x_f",         x_f_size)
    defDim(ds, "y_c",         ylmo.g.Ny)
    defDim(ds, "y_f",         y_f_size)
    defDim(ds, "zeta",         ylmo.gt.Nz)
    defDim(ds, "zeta_ac",      ylmo.gt.Nz + 1)
    defDim(ds, "zeta_rock",    ylmo.gr.Nz)
    defDim(ds, "zeta_rock_ac", ylmo.gr.Nz + 1)
    defDim(ds, "time",         Inf)

    # ---- coordinate variables ---------------------------------------------
    _defcoord(ds, "x_c", Float64, ("x_c",), xnodes(ylmo.g, Center()), "m", "x-coordinate, center")
    _defcoord(ds, "x_f", Float64, ("x_f",), xnodes(ylmo.g, Face()),   "m", "x-coordinate, face")
    _defcoord(ds, "y_c", Float64, ("y_c",), ynodes(ylmo.g, Center()), "m", "y-coordinate, center")
    _defcoord(ds, "y_f", Float64, ("y_f",), ynodes(ylmo.g, Face()),   "m", "y-coordinate, face")
    _defcoord(ds, "zeta",         Float64, ("zeta",),
              znodes(ylmo.gt, Center()), "1", "normalised ice layer midpoint")
    _defcoord(ds, "zeta_ac",      Float64, ("zeta_ac",),
              znodes(ylmo.gt, Face()),   "1", "normalised ice layer interface")
    _defcoord(ds, "zeta_rock",    Float64, ("zeta_rock",),
              znodes(ylmo.gr, Center()), "1", "normalised bedrock layer midpoint")
    _defcoord(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",),
              znodes(ylmo.gr, Face()),   "1", "normalised bedrock layer interface")

    tv = defVar(ds, "time", Float64, ("time",))
    tv.attrib["units"]     = "yr"
    tv.attrib["long_name"] = "model time"

    # ---- field variables --------------------------------------------------
    # First pass: enumerate every selected (group, fname) pair that has
    # a recognised grid, count name occurrences across groups so we can
    # disambiguate collisions (e.g. `tau_relax` exists in both `bnd`
    # and `tpo`).
    selected = Vector{Tuple{Symbol, Symbol, NTuple}}()
    name_count = Dict{String, Int}()
    for gname in active_groups
        group_nt = getfield(ylmo, gname)
        for fname in keys(group_nt)
            # Skip the scratch substruct explicitly, plus any other
            # non-Field entries defensively.
            fname === :scratch && continue
            group_nt[fname] isa Field || continue
            name = String(fname)
            _selected(name, gname, selection) || continue
            dims = _spatial_dims(group_nt[fname], ylmo)
            dims === nothing && continue
            push!(selected, (gname, fname, dims))
            name_count[name] = get(name_count, name, 0) + 1
        end
    end

    nc_names = Dict{Tuple{Symbol, Symbol}, String}()
    for (gname, fname, dims) in selected
        name = String(fname)
        nc_name = name_count[name] > 1 ? "$(gname)_$(name)" : name
        nc_names[(gname, fname)] = nc_name
        defVar(ds, nc_name, Float32, (dims..., "time");
               deflatelevel = deflate,
               fillvalue    = Float32(NaN))
    end

    # ---- scratch entries (opt-in via `include_scratch=true`) -------------
    # Walk the `:scratch` substruct of every active group whose parent
    # group passes `selection.groups`. Per-field selection still applies
    # — patterns are matched against the bare scratch sub-fname (e.g.
    # `"sia_tau_xz"`), so users can `include` / `exclude` them like any
    # other field. NetCDF variable name is always `<group>_scratch_<sub>`,
    # which is unambiguous so no collision-counting is needed.
    scratch_nc_names = Dict{Tuple{Symbol, Symbol}, String}()
    if include_scratch
        for gname in active_groups
            group_nt = getfield(ylmo, gname)
            haskey(group_nt, :scratch) || continue
            scratch_nt = group_nt[:scratch]
            scratch_nt isa NamedTuple || continue
            for sname in keys(scratch_nt)
                scratch_nt[sname] isa Field || continue
                _selected(String(sname), gname, selection) || continue
                dims = _spatial_dims(scratch_nt[sname], ylmo)
                dims === nothing && continue
                nc_name = "$(gname)_scratch_$(sname)"
                scratch_nc_names[(gname, sname)] = nc_name
                defVar(ds, nc_name, Float32, (dims..., "time");
                       deflatelevel = deflate,
                       fillvalue    = Float32(NaN))
            end
        end
    end

    return YelmoOutput(ds, selection, active_groups, nc_names, scratch_nc_names)
end

# ---------------------------------------------------------------------------
# write_output!
# ---------------------------------------------------------------------------

"""
    write_output!(out::YelmoOutput, ylmo::AbstractYelmoModel; include_scratch)

Append one time slice (at `ylmo.time`) to the open output file.

`include_scratch` defaults to whatever `init_output` set up — i.e. `true`
iff scratch entries were registered. Passing `include_scratch=false`
explicitly suppresses scratch writes for a particular slice (the
NetCDF variables remain, just unwritten at that time index).
"""
function write_output!(out::YelmoOutput, ylmo::AbstractYelmoModel;
                       include_scratch::Bool = !isempty(out.scratch_nc_names))
    ds    = out.ds
    t_idx = length(ds["time"]) + 1
    ds["time"][t_idx] = ylmo.time

    for gname in out.groups
        group_nt = getfield(ylmo, gname)

        for fname in keys(group_nt)
            # Skip the scratch substruct explicitly, plus any other
            # non-Field entries defensively.
            fname === :scratch && continue
            group_nt[fname] isa Field || continue
            nc_name = get(out.nc_names, (gname, fname), nothing)
            nc_name === nothing && continue
            haskey(ds, nc_name) || continue

            data = _get_data(group_nt[fname])
            sz   = size(data)

            if ndims(data) == 2
                ds[nc_name][1:sz[1], 1:sz[2], t_idx] = data
            elseif ndims(data) == 3
                ds[nc_name][1:sz[1], 1:sz[2], 1:sz[3], t_idx] = data
            end
        end
    end

    if include_scratch
        for ((gname, sname), nc_name) in out.scratch_nc_names
            haskey(ds, nc_name) || continue
            group_nt   = getfield(ylmo, gname)
            haskey(group_nt, :scratch) || continue
            scratch_nt = group_nt[:scratch]
            haskey(scratch_nt, sname) || continue
            field = scratch_nt[sname]
            field isa Field || continue

            data = _get_data(field)
            sz   = size(data)

            if ndims(data) == 2
                ds[nc_name][1:sz[1], 1:sz[2], t_idx] = data
            elseif ndims(data) == 3
                ds[nc_name][1:sz[1], 1:sz[2], 1:sz[3], t_idx] = data
            end
        end
    end

    return nothing
end

end # Module