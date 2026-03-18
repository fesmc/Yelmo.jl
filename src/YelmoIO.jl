module YelmoIO

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields

using NCDatasets
using ..YelmoCore
using ..YelmoMeta
export load_grids_from_restart
export load_field_from_dataset_2D
export load_field_from_dataset_3D
export load_fields_from_restart
export init_output
export OutputSelection
export write_output!

function matches_patterns(name, patterns)
    any(p -> occursin(p, name), patterns)
end

function load_grids_from_restart(filename)

    # Open the NetCDF file
    ds = NCDataset(filename)

    xc = ds["xc"][:]
    yc = ds["yc"][:]
    
    Nx = length(xc)
    Ny = length(yc)
    xlims = extrema(xc)
    ylims = extrema(yc)

    # Expand x- and ylims so that nodes represent centers of cells
    dx = xc[2]-xc[1]
    dy = yc[2]-yc[1]
    xlims = (xlims[1] - dx/2, xlims[2] + dx/2)
    ylims = (ylims[1] - dy/2, ylims[2] + dy/2)
    
    # First, create your grid
    grid2d = RectilinearGrid(
        size = (Nx, Ny),
        x = xlims,
        y = ylims,
        topology = (Bounded, Bounded,Flat)
    )

    # Create 3D grid if zeta exists
    grid3d = nothing
    if haskey(ds, "zeta")
        zeta = ds["zeta"][:]
        zeta_ac = ds["zeta_ac"][:]
        Nz = length(zeta)
        zlims = extrema(zeta)
        
        grid3d = RectilinearGrid(
            size = (Nx, Ny, Nz),
            x = xlims,
            y = ylims,
            #z = zlims,
            z = zeta_ac,
            topology = (Bounded, Bounded, Bounded)
        )
    end

    close(ds)

    return grid2d, grid3d
end

function load_field_from_dataset_2D(ds::NCDataset, varname::Union{AbstractString,Symbol}, grid2d)

    varname = String(varname)

    xface_variables = ["ux_s","ux_b", r".*_acx$"]
    yface_variables = ["uy_s","uy_b", r".*_acy$"]
    zface_variables = []

    # Create the field and store in interior of field
    if matches_patterns(varname, xface_variables)
        field = XFaceField(grid2d)
        interior(field)[2:end,:] .= ds[varname][:,:]
        interior(field)[1,:] .= ds[varname][1,:]
    elseif matches_patterns(varname, yface_variables)
        field = YFaceField(grid2d)
        interior(field)[:,2:end] .= ds[varname][:,:]
        interior(field)[:,1] .= ds[varname][:,1]
    elseif matches_patterns(varname, zface_variables)
        # Maybe this case doesn't exist in 2d?
        field = ZFaceField(grid2d)
    else
        field = CenterField(grid2d)
        interior(field) .= ds[varname][:,:]
    end

    #interior(field) .= ds[varname][:,:]

    return (field)
end

function load_field_from_dataset_3D(ds::NCDataset, varname::Union{AbstractString,Symbol}, grid3d)

    varname = String(varname)

    xface_variables = ["ux", r".*_acx$"]
    yface_variables = ["uy", r".*_acy$"]
    zface_variables = ["uz","uz_star","jvel_dzx","jvel_dzy","jvel_dzz"]

    # Create the field and store in interior of field
    if matches_patterns(varname, xface_variables)
        field = XFaceField(grid3d)
        interior(field)[2:end,:,:] .= ds[varname][:,:,:]
        interior(field)[1,:,:] .= ds[varname][1,:,:]
    elseif matches_patterns(varname, yface_variables)
        field = YFaceField(grid3d)
        interior(field)[:,2:end,:] .= ds[varname][:,:,:]
        interior(field)[:,1,:] .= ds[varname][:,1,:]
    elseif matches_patterns(varname, zface_variables)
        field = ZFaceField(grid3d)
        interior(field)[:,:,:] .= ds[varname][:,:,:]
    else
        field = CenterField(grid3d)
        interior(field) .= ds[varname][:,:,:]
    end

    return (field)
end

function load_field_from_dataset_2D(filename, varname::Union{AbstractString,Symbol}, grid2d)
    ds = NCDataset(filename)
    field = load_field_from_dataset_2D(ds,varname,grid2d)
    close(ds)
    return field
end

function load_field_from_dataset_3D(filename::AbstractString, varname::Union{AbstractString,Symbol}, grid3d)
    ds = NCDataset(filename)
    field = load_field_from_dataset_3D(ds,varname,grid3d)
    close(ds)
    return field
end

function load_fields_from_restart(filename,grid2d,grid3d)

    # Open the NetCDF file
    ds = NCDataset(filename)

    # Load variables into dictionary
    dat = Dict{String, Field}()
    dat = Dict{String, Field}()
    
    # Skip several variables for now
    variables_to_skip = [
        r"^jvel.*",
        r"^strs.*",
        r"^strn.*",
    ]

    # Also limit 3D variables to the following...
    variables_to_load_3d = [
        r"^ux.*",
        r"^uy.*",
        r"^uz.*",
        r"^T.*",
    ]
            
    for varname in keys(ds)

        # Skip variables as needed
        if matches_patterns(varname,variables_to_skip)
            continue
        end

        # Get current dimension names
        dimnames_now = dimnames(ds[varname])

        # Skip variables that do not at least contain 2D information
        if length(dimnames_now) < 2
            continue
        end

        if dimnames_now[1:2] != ("xc", "yc")
            continue
        end
        
        dims = size(ds[varname])
        
        println("$varname : $dims")

        if length(dims) == 2 || dimnames_now[3] == "time"
            # 2D variable
            dat[varname] = load_field_from_dataset_2D(ds,varname,grid2d)

        elseif (dimnames_now[3] == "zeta" || dimnames_now[3] == "zeta_ac") && grid3d !== nothing
            # Only load some variables to avoid segfault...

            if matches_patterns(varname,variables_to_load_3d)
                # 3D variable on zeta grid
                dat[varname] = load_field_from_dataset_3D(ds,varname,grid3d)
            end
        end

    end
    
    close(ds)

    return dat
end
## Writing output ##

# ---------------------------------------------------------------------------
# NetCDF dimension helpers
# ---------------------------------------------------------------------------

struct OutputSelection
    groups  :: Union{Nothing, Vector{Symbol}}
    include :: Union{Nothing, Vector}
    exclude :: Union{Nothing, Vector}
end

OutputSelection(; groups=nothing, include=nothing, exclude=nothing) =
    OutputSelection(groups, include, exclude)

function _name_matches(name::String, patterns)
    for p in patterns
        if p isa Regex
            occursin(p, name) && return true
        else
            name == p && return true
        end
    end
    return false
end

function _selected(name::String, sel::OutputSelection)
    sel.include !== nothing && !_name_matches(name, sel.include) && return false
    sel.exclude !== nothing &&  _name_matches(name, sel.exclude) && return false
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
    init_output(ylmo, path; selection, deflate) -> YelmoOutput

Create a NetCDF file at `path`. Defines all dimensions, coordinate variables,
and one NetCDF variable per field that passes `selection` and has a
recognised grid. Returns a `YelmoOutput` to be passed to `write_output!`.
"""
function init_output(ylmo::YelmoMirror, path::String;
                     selection::OutputSelection = OutputSelection(),
                     deflate::Int = 4)

    active_groups = selection.groups === nothing ? _ALL_GROUPS : selection.groups

    ds = NCDataset(path, "c")

    # ---- dimensions -------------------------------------------------------
    defDim(ds, "x_c",         ylmo.g.Nx)
    defDim(ds, "x_f",         ylmo.g.Nx + 1)
    defDim(ds, "y_c",         ylmo.g.Ny)
    defDim(ds, "y_f",         ylmo.g.Ny + 1)
    defDim(ds, "zeta",         ylmo.gt.Nz)
    defDim(ds, "zeta_ac",      ylmo.gt.Nz + 1)
    defDim(ds, "zeta_rock",    ylmo.gr.Nz)
    defDim(ds, "zeta_rock_ac", ylmo.gr.Nz + 1)
    defDim(ds, "time",         Inf)

    # ---- coordinate variables ---------------------------------------------
    _defcoord(ds, "x_c", Float64, ("x_c",), xnodes(ylmo.g, Center()), "km", "x cell centre")
    _defcoord(ds, "x_f", Float64, ("x_f",), xnodes(ylmo.g, Face()),   "km", "x face")
    _defcoord(ds, "y_c", Float64, ("y_c",), ynodes(ylmo.g, Center()), "km", "y cell centre")
    _defcoord(ds, "y_f", Float64, ("y_f",), ynodes(ylmo.g, Face()),   "km", "y face")
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
    for gname in active_groups
        group_nt = getfield(ylmo, gname)

        for fname in keys(group_nt)
            name  = String(fname)
            _selected(name, selection) || continue

            field = group_nt[fname]
            dims  = _spatial_dims(field, ylmo)
            dims === nothing && continue

            defVar(ds, name, Float32, (dims..., "time");
                   deflatelevel = deflate,
                   fillvalue    = Float32(NaN))
        end
    end

    return YelmoOutput(ds, selection, active_groups)
end

# ---------------------------------------------------------------------------
# write_output!
# ---------------------------------------------------------------------------

"""
    write_output!(out::YelmoOutput, ylmo::YelmoMirror)

Append one time slice (at `ylmo.time`) to the open output file.
"""
function write_output!(out::YelmoOutput, ylmo::YelmoMirror)
    ds    = out.ds
    t_idx = length(ds["time"]) + 1
    ds["time"][t_idx] = ylmo.time

    for gname in out.groups
        group_nt = getfield(ylmo, gname)

        for fname in keys(group_nt)
            name = String(fname)
            _selected(name, out.selection) || continue
            haskey(ds, name)               || continue

            data = _get_data(group_nt[fname])
            sz   = size(data)

            if ndims(data) == 2
                ds[name][1:sz[1], 1:sz[2], t_idx] = data
            elseif ndims(data) == 3
                ds[name][1:sz[1], 1:sz[2], 1:sz[3], t_idx] = data
            end
        end
    end

    return nothing
end

end # Module