module YelmoCore

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using NCDatasets

using ..YelmoMeta: VariableMeta, parse_variable_table

export AbstractYelmoModel, YelmoModel
export init_state!, step!, load_state!
export load_grids_from_restart, load_fields_from_restart
export load_field_from_dataset_2D, load_field_from_dataset_3D
export make_field, matches_patterns, yelmo_define_grids
export XFACE_VARIABLES, YFACE_VARIABLES, ZFACE_VARIABLES, VERTICAL_DIMS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const VERTICAL_DIMS = (:zeta, :zeta_ac, :zeta_rock, :zeta_rock_ac)

const XFACE_VARIABLES = ["ux_s", "ux_b", "ux", r".*_acx$"]
const YFACE_VARIABLES = ["uy_s", "uy_b", "uy", r".*_acy$"]
const ZFACE_VARIABLES = ["uz", "uz_star", "jvel_dzx", "jvel_dzy", "jvel_dzz"]

# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

matches_patterns(name, patterns) = any(p -> occursin(p, name), patterns)

# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

yelmo_define_grids(g::NamedTuple) =
    yelmo_define_grids(g.xc, g.yc, g.zeta_ac, g.zeta_r_ac)

function yelmo_define_grids(xc, yc, zeta_ac, zeta_r_ac)
    Nx, Ny = length(xc), length(yc)
    dx, dy = xc[2] - xc[1], yc[2] - yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)

    grid2d = RectilinearGrid(size=(Nx, Ny),
                             x=xlims, y=ylims,
                             topology=(Bounded, Bounded, Flat))

    grid3d_ice = RectilinearGrid(size=(Nx, Ny, length(zeta_ac) - 1),
                                 x=xlims, y=ylims, z=zeta_ac,
                                 topology=(Bounded, Bounded, Bounded))

    grid3d_rock = RectilinearGrid(size=(Nx, Ny, length(zeta_r_ac) - 1),
                                  x=xlims, y=ylims, z=zeta_r_ac,
                                  topology=(Bounded, Bounded, Bounded))

    return grid2d, grid3d_ice, grid3d_rock
end

# ---------------------------------------------------------------------------
# Field allocation
# ---------------------------------------------------------------------------

function make_field(varname::Union{AbstractString,Symbol}, grid::RectilinearGrid)
    varname = String(varname)
    if matches_patterns(varname, XFACE_VARIABLES)
        return XFaceField(grid)
    elseif matches_patterns(varname, YFACE_VARIABLES)
        return YFaceField(grid)
    elseif matches_patterns(varname, ZFACE_VARIABLES)
        return ZFaceField(grid)
    else
        return CenterField(grid)
    end
end

function _alloc_field(meta::VariableMeta, g2d, g3d, g3r)
    dims = meta.dimensions
    if any(d -> d in (:zeta_rock, :zeta_rock_ac), dims)
        return make_field(meta.name, g3r)
    elseif any(d -> d in (:zeta, :zeta_ac), dims)
        return make_field(meta.name, g3d)
    else
        return make_field(meta.name, g2d)
    end
end

_alloc_group(vlist, g2d, g3d, g3r) =
    NamedTuple{keys(vlist)}(_alloc_field(vlist[k], g2d, g3d, g3r) for k in keys(vlist))

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

"""
    AbstractYelmoModel

Common supertype for any concrete Yelmo state container. Concrete subtypes
must expose the fields `alias`, `rundir`, `time`, `p`, `g`, `gt`, `gr`, `v`,
`bnd`, `dta`, `dyn`, `mat`, `thrm`, `tpo`, and provide methods for
`init_state!(y, time; kwargs...)` and `step!(y, dt)`.
"""
abstract type AbstractYelmoModel end

# ---------------------------------------------------------------------------
# NetCDF restart loading
# ---------------------------------------------------------------------------

"""
    load_grids_from_restart(filename) -> (grid2d, grid3d_ice, grid3d_rock)

Read coordinate arrays from a NetCDF restart and build the 2D and 3D
Oceananigans `RectilinearGrid`s. `grid3d_ice` and `grid3d_rock` may be
`nothing` if the file does not contain the corresponding vertical axes.
"""
function load_grids_from_restart(filename::AbstractString)
    ds = NCDataset(filename)

    xc = ds["xc"][:]
    yc = ds["yc"][:]

    Nx = length(xc)
    Ny = length(yc)
    dx = xc[2] - xc[1]
    dy = yc[2] - yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)

    grid2d = RectilinearGrid(size=(Nx, Ny),
                             x=xlims, y=ylims,
                             topology=(Bounded, Bounded, Flat))

    grid3d_ice = nothing
    if haskey(ds, "zeta_ac")
        zeta_ac = ds["zeta_ac"][:]
        grid3d_ice = RectilinearGrid(size=(Nx, Ny, length(zeta_ac) - 1),
                                     x=xlims, y=ylims, z=zeta_ac,
                                     topology=(Bounded, Bounded, Bounded))
    end

    grid3d_rock = nothing
    if haskey(ds, "zeta_rock_ac")
        zeta_r_ac = ds["zeta_rock_ac"][:]
        grid3d_rock = RectilinearGrid(size=(Nx, Ny, length(zeta_r_ac) - 1),
                                      x=xlims, y=ylims, z=zeta_r_ac,
                                      topology=(Bounded, Bounded, Bounded))
    end

    close(ds)
    return grid2d, grid3d_ice, grid3d_rock
end

function load_field_from_dataset_2D(ds::NCDataset, varname::Union{AbstractString,Symbol}, grid2d)
    varname = String(varname)
    xface = ["ux_s", "ux_b", r".*_acx$"]
    yface = ["uy_s", "uy_b", r".*_acy$"]
    zface = String[]

    if matches_patterns(varname, xface)
        field = XFaceField(grid2d)
        interior(field)[2:end, :] .= ds[varname][:, :]
        interior(field)[1, :]     .= ds[varname][1, :]
    elseif matches_patterns(varname, yface)
        field = YFaceField(grid2d)
        interior(field)[:, 2:end] .= ds[varname][:, :]
        interior(field)[:, 1]     .= ds[varname][:, 1]
    elseif matches_patterns(varname, zface)
        field = ZFaceField(grid2d)
    else
        field = CenterField(grid2d)
        interior(field) .= ds[varname][:, :]
    end
    return field
end

function load_field_from_dataset_3D(ds::NCDataset, varname::Union{AbstractString,Symbol}, grid3d)
    varname = String(varname)
    xface = ["ux", r".*_acx$"]
    yface = ["uy", r".*_acy$"]
    zface = ["uz", "uz_star", "jvel_dzx", "jvel_dzy", "jvel_dzz"]

    if matches_patterns(varname, xface)
        field = XFaceField(grid3d)
        interior(field)[2:end, :, :] .= ds[varname][:, :, :]
        interior(field)[1, :, :]     .= ds[varname][1, :, :]
    elseif matches_patterns(varname, yface)
        field = YFaceField(grid3d)
        interior(field)[:, 2:end, :] .= ds[varname][:, :, :]
        interior(field)[:, 1, :]     .= ds[varname][:, 1, :]
    elseif matches_patterns(varname, zface)
        field = ZFaceField(grid3d)
        interior(field)[:, :, :] .= ds[varname][:, :, :]
    else
        field = CenterField(grid3d)
        interior(field) .= ds[varname][:, :, :]
    end
    return field
end

function load_field_from_dataset_2D(filename::AbstractString, varname, grid2d)
    ds = NCDataset(filename)
    field = load_field_from_dataset_2D(ds, varname, grid2d)
    close(ds)
    return field
end

function load_field_from_dataset_3D(filename::AbstractString, varname, grid3d)
    ds = NCDataset(filename)
    field = load_field_from_dataset_3D(ds, varname, grid3d)
    close(ds)
    return field
end

"""
    load_fields_from_restart(filename, grid2d, grid3d) -> Dict{String, Field}

Ad-hoc loader that reads every recognised 2D or 3D variable from a restart
into a `Field` and returns them keyed by name. Skips variables matching
`jvel*`, `strs*`, `strn*`, and limits 3D loading to velocity and temperature.
Useful for exploration; `YelmoModel` itself uses the structured per-group
loader inside `load_state!`.
"""
function load_fields_from_restart(filename, grid2d, grid3d)
    ds = NCDataset(filename)

    dat = Dict{String, Field}()

    variables_to_skip = [r"^jvel.*", r"^strs.*", r"^strn.*"]
    variables_to_load_3d = [r"^ux.*", r"^uy.*", r"^uz.*", r"^T.*"]

    for varname in keys(ds)
        if matches_patterns(varname, variables_to_skip)
            continue
        end

        dimnames_now = dimnames(ds[varname])
        if length(dimnames_now) < 2
            continue
        end
        if dimnames_now[1:2] != ("xc", "yc")
            continue
        end

        dims = size(ds[varname])

        if length(dims) == 2 || dimnames_now[3] == "time"
            dat[varname] = load_field_from_dataset_2D(ds, varname, grid2d)
        elseif (dimnames_now[3] == "zeta" || dimnames_now[3] == "zeta_ac") && grid3d !== nothing
            if matches_patterns(varname, variables_to_load_3d)
                dat[varname] = load_field_from_dataset_3D(ds, varname, grid3d)
            end
        end
    end

    close(ds)
    return dat
end

# ---------------------------------------------------------------------------
# YelmoModel — pure-Julia state container
# ---------------------------------------------------------------------------

mutable struct YelmoModel{P, B, DT, DY, M, TH, TP} <: AbstractYelmoModel
    alias::String
    rundir::String
    time::Float64
    p::P
    g::RectilinearGrid
    gt::RectilinearGrid
    gr::RectilinearGrid
    v::NamedTuple
    bnd::B
    dta::DT
    dyn::DY
    mat::M
    thrm::TH
    tpo::TP
end

"""
    YelmoModel(restart_file, time; alias, rundir, p)

Construct a `YelmoModel` whose grids and field values are read from a NetCDF
restart file. Variable layout is taken from `src/variables/model/` markdown
tables. The model parameters `p` are passed through verbatim and may be
`nothing`, a `YelmoModelParameters`, or any user object.
"""
function YelmoModel(restart_file::String, time::Float64;
                    alias::String = "ymodel1",
                    rundir::String = "./",
                    p = nothing)

    g, gt, gr = load_grids_from_restart(restart_file)
    gt === nothing && error("Restart file $(restart_file) has no zeta_ac axis; cannot build ice grid.")
    gr === nothing && error("Restart file $(restart_file) has no zeta_rock_ac axis; cannot build rock grid.")

    vdir = joinpath(@__DIR__, "variables", "model")
    v_meta = (
        bnd  = parse_variable_table(joinpath(vdir, "yelmo-variables-ybound.md"), "bnd"),
        dta  = parse_variable_table(joinpath(vdir, "yelmo-variables-ydata.md"),  "dta"),
        dyn  = parse_variable_table(joinpath(vdir, "yelmo-variables-ydyn.md"),   "dyn"),
        mat  = parse_variable_table(joinpath(vdir, "yelmo-variables-ymat.md"),   "mat"),
        thrm = parse_variable_table(joinpath(vdir, "yelmo-variables-ytherm.md"), "thrm"),
        tpo  = parse_variable_table(joinpath(vdir, "yelmo-variables-ytopo.md"),  "tpo"),
    )

    bnd  = _alloc_group(v_meta.bnd,  g, gt, gr)
    dta  = _alloc_group(v_meta.dta,  g, gt, gr)
    dyn  = _alloc_group(v_meta.dyn,  g, gt, gr)
    mat  = _alloc_group(v_meta.mat,  g, gt, gr)
    thrm = _alloc_group(v_meta.thrm, g, gt, gr)
    tpo  = _alloc_group(v_meta.tpo,  g, gt, gr)

    y = YelmoModel(alias, rundir, time, p, g, gt, gr, v_meta, bnd, dta, dyn, mat, thrm, tpo)

    load_state!(y, restart_file)

    return y
end

# ---------------------------------------------------------------------------
# load_state! — populate field values from a restart file
# ---------------------------------------------------------------------------

"""
    load_state!(y::YelmoModel, restart_file) -> y

Populate every field in `y`'s component groups from variables of the same
name in `restart_file`. Errors if any variable named in `y.v` is missing
from the file (strict mode). Grids are not re-read; the file's grid is
assumed to match `y.g`/`y.gt`/`y.gr`.
"""
function load_state!(y::YelmoModel, restart_file::AbstractString)
    ds = NCDataset(restart_file)
    try
        for gname in (:bnd, :dta, :dyn, :mat, :thrm, :tpo)
            group_nt = getfield(y, gname)
            metas    = getfield(y.v, gname)
            for k in keys(metas)
                meta = metas[k]
                name_str = String(meta.name)
                haskey(ds, name_str) || error(
                    "Variable `$(name_str)` (group `$(gname)`) not found in restart " *
                    "file $(restart_file). Strict mode: every variable in the model " *
                    "variable table must be present.")
                _load_into_field!(group_nt[k], ds[name_str])
            end
        end
    finally
        close(ds)
    end
    return y
end

# Field-shape-aware copy from NetCDF variable to a pre-allocated Field's
# interior. Mirrors the staggered-grid handling in YelmoMirrorCore's
# _get_var! family.

function _load_into_field!(field::Field{Face, Center, Center}, ncvar) where {Face, Center}
    if ndims(ncvar) == 3
        interior(field)[2:end, :, :] .= ncvar[:, :, :]
        interior(field)[1, :, :]     .= ncvar[1, :, :]
    else
        interior(field)[2:end, :] .= ncvar[:, :]
        interior(field)[1, :]     .= ncvar[1, :]
    end
    return field
end

function _load_into_field!(field::Field{Center, Face, Center}, ncvar) where {Face, Center}
    if ndims(ncvar) == 3
        interior(field)[:, 2:end, :] .= ncvar[:, :, :]
        interior(field)[:, 1, :]     .= ncvar[:, 1, :]
    else
        interior(field)[:, 2:end] .= ncvar[:, :]
        interior(field)[:, 1]     .= ncvar[:, 1]
    end
    return field
end

function _load_into_field!(field::Field{Center, Center, Face}, ncvar) where {Face, Center}
    if ndims(ncvar) == 3
        interior(field)[:, :, :] .= ncvar[:, :, :]
    else
        interior(field)[:, :] .= ncvar[:, :]
    end
    return field
end

function _load_into_field!(field::Field{Center, Center, Center}, ncvar) where {Center}
    if ndims(ncvar) == 3
        interior(field)[:, :, :] .= ncvar[:, :, :]
    else
        interior(field)[:, :] .= ncvar[:, :]
    end
    return field
end

# ---------------------------------------------------------------------------
# init_state! and step! — abstract-interface methods for YelmoModel
# ---------------------------------------------------------------------------

function init_state!(y::YelmoModel, time::Float64; kwargs...)
    y.time = time
    return y
end

function step!(y::YelmoModel, dt::Float64)
    y.time += dt
    return y
end

end # module YelmoCore
