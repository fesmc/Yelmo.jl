module YelmoCore

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using NCDatasets

using ..YelmoMeta: VariableMeta, parse_variable_table
using ..YelmoModelPar: YelmoModelParameters

export AbstractYelmoModel, YelmoModel
export init_state!, step!, load_state!
export load_grids_from_restart, load_fields_from_restart
export load_field_from_dataset_2D, load_field_from_dataset_3D
export make_field, matches_patterns, yelmo_define_grids
export XFACE_VARIABLES, YFACE_VARIABLES, ZFACE_VARIABLES, VERTICAL_DIMS
export MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC
export compare_state, StateComparison

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const VERTICAL_DIMS = (:zeta, :zeta_ac, :zeta_rock, :zeta_rock_ac)

const XFACE_VARIABLES = ["ux_s", "ux_b", "ux", r".*_acx$"]
const YFACE_VARIABLES = ["uy_s", "uy_b", "uy", r".*_acy$"]
const ZFACE_VARIABLES = ["uz", "uz_star", "jvel_dzx", "jvel_dzy", "jvel_dzz"]

# Per-cell ice evolution mask values (`bnd.mask_ice`).
# Stored as Float64 in the field so they round-trip through Oceananigans
# CenterField storage; the values themselves are integers.
const MASK_ICE_NONE    = 0  # H_ice forced to 0
const MASK_ICE_FIXED   = 1  # H_ice held at its current value
const MASK_ICE_DYNAMIC = 2  # H_ice evolves freely

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

"""
Construct a 2D `CenterField` on `grid` with Dirichlet `value` boundary
conditions on every non-Flat horizontal face (east/west/south/north).
Used for ice-sheet fields like `H_ice` whose physical boundary
condition is "no ice past the domain edge."
"""
function _dirichlet_2d_field(grid::RectilinearGrid, value::Real)
    bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center());
        east  = ValueBoundaryCondition(value),
        west  = ValueBoundaryCondition(value),
        south = ValueBoundaryCondition(value),
        north = ValueBoundaryCondition(value),
    )
    return CenterField(grid; boundary_conditions=bcs)
end

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

    xc = Vector{Float64}(ds["xc"][:])
    yc = Vector{Float64}(ds["yc"][:])

    # Yelmo's NetCDF convention stores horizontal coordinates in
    # kilometres while velocities and thickness are in metres / m·yr⁻¹.
    # Convert to metres on load so the grid spacing is consistent with
    # the prognostic fields and CFL/advection arithmetic produces
    # physical results without per-call rescaling.
    x_units = lowercase(strip(get(ds["xc"].attrib, "units", "")))
    y_units = lowercase(strip(get(ds["yc"].attrib, "units", "")))
    if x_units == "km"
        xc .*= 1000.0
    end
    if y_units == "km"
        yc .*= 1000.0
    end

    Nx = length(xc)
    Ny = length(yc)
    dx = xc[2] - xc[1]
    dy = yc[2] - yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)

    grid2d = RectilinearGrid(size=(Nx, Ny),
                             x=xlims, y=ylims,
                             topology=(Bounded, Bounded, Flat))

    grid3d_ice = _build_3d_grid(ds, "zeta", "zeta_ac", Nx, Ny, xlims, ylims)
    grid3d_rock = _build_3d_grid(ds, "zeta_rock", "zeta_rock_ac", Nx, Ny, xlims, ylims)

    close(ds)
    return grid2d, grid3d_ice, grid3d_rock
end

# Build a 3D RectilinearGrid using face-coordinates from `face_var` if
# present, otherwise reconstruct them from cell-center coordinates in
# `center_var` (`zeta_ac[i] = (zeta[i-1] + zeta[i]) / 2` for interior
# faces, with the boundary faces clamped to the first/last center).
function _build_3d_grid(ds, center_var, face_var, Nx, Ny, xlims, ylims)
    if haskey(ds, face_var)
        z_face = Vector{Float64}(ds[face_var][:])
    elseif haskey(ds, center_var)
        z_center = Vector{Float64}(ds[center_var][:])
        z_face = _faces_from_centers(z_center)
    else
        return nothing
    end
    return RectilinearGrid(size=(Nx, Ny, length(z_face) - 1),
                           x=xlims, y=ylims, z=z_face,
                           topology=(Bounded, Bounded, Bounded))
end

function _faces_from_centers(zc::AbstractVector)
    N = length(zc)
    zf = Vector{Float64}(undef, N + 1)
    zf[1] = zc[1]
    zf[end] = zc[end]
    for i in 2:N
        zf[i] = 0.5 * (zc[i-1] + zc[i])
    end
    return zf
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

const _ALL_MODEL_GROUPS = (:bnd, :dta, :dyn, :mat, :thrm, :tpo)

"""
    YelmoModel(restart_file, time; alias, rundir, p, groups, strict)

Construct a `YelmoModel` whose grids and field values are read from a NetCDF
restart file. Variable layout is taken from `src/variables/model/` markdown
tables. The model parameters `p` are passed through verbatim and may be
`nothing`, a `YelmoModelParameters`, or any user object.

`groups` selects which component groups to load from the restart (default:
all six). `strict=true` (default) errors if a variable in a loaded group
is missing from the restart; `strict=false` silently skips missing
variables, leaving the corresponding field at its default-allocated value.
"""
function YelmoModel(restart_file::String, time::Float64;
                    alias::String = "ymodel1",
                    rundir::String = "./",
                    p = nothing,
                    groups::NTuple{N,Symbol} where N = _ALL_MODEL_GROUPS,
                    strict::Bool = true)

    if p === nothing
        @warn "No parameters supplied to YelmoModel; constructing YelmoModelParameters(\"$(alias)\") with defaults."
        p = YelmoModelParameters(alias)
    end

    g, gt, gr = load_grids_from_restart(restart_file)
    gt === nothing && error("Restart file $(restart_file) has no ice vertical axis; cannot build ice grid.")
    gr === nothing && error("Restart file $(restart_file) has no rock vertical axis; cannot build rock grid.")

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

    # Replace H_ice with a CenterField that carries Dirichlet H_ice = 0
    # boundary conditions on the domain edge. The upwind advection
    # operator reads these via Oceananigans' standard halo machinery
    # without per-cell branching in the kernel.
    haskey(tpo, :H_ice) && (tpo = merge(tpo, (H_ice = _dirichlet_2d_field(g, 0.0),)))

    y = YelmoModel(alias, rundir, time, p, g, gt, gr, v_meta, bnd, dta, dyn, mat, thrm, tpo)

    # Default mask_ice to all-dynamic before load_state!. If the restart
    # file carries `mask_ice`, load_state! overwrites this; otherwise the
    # post-load inference may overwrite based on `ice_allowed`.
    fill!(interior(y.bnd.mask_ice), Float64(MASK_ICE_DYNAMIC))

    load_state!(y, restart_file; groups=groups, strict=strict)

    _infer_mask_ice!(y, restart_file)

    return y
end

# Fill `bnd.mask_ice` based on what is actually in the restart file.
#  - If the restart contains `mask_ice`, do nothing (load_state! has
#    already placed its values into `y.bnd.mask_ice`).
#  - Else if it contains `ice_allowed`, derive: allowed → DYNAMIC,
#    not-allowed → NONE. Read directly from the file so this works
#    even when `:bnd` was not in the loaded `groups`.
#  - Else leave the all-dynamic default established before load_state!.
function _infer_mask_ice!(y::YelmoModel, restart_file::AbstractString)
    NCDataset(restart_file) do ds
        haskey(ds, "mask_ice") && return  # already loaded
        haskey(ds, "ice_allowed") || return  # nothing to infer from

        ia = _read_nc_2d(ds["ice_allowed"])
        m = interior(y.bnd.mask_ice)
        @inbounds for j in axes(m, 2), i in axes(m, 1)
            m[i, j, 1] = ia[i, j] != 0 ? Float64(MASK_ICE_DYNAMIC) : Float64(MASK_ICE_NONE)
        end
    end
    return y
end

# ---------------------------------------------------------------------------
# load_state! — populate field values from a restart file
# ---------------------------------------------------------------------------

"""
    load_state!(y::YelmoModel, restart_file; groups, strict) -> y

Populate fields in `y`'s component groups from variables of the same name
in `restart_file`. `groups` selects which groups to load (default: all six);
`strict=true` (default) errors on any missing variable in a loaded group,
`strict=false` skips missing variables. Grids are not re-read.
"""
function load_state!(y::YelmoModel, restart_file::AbstractString;
                     groups::NTuple{N,Symbol} where N = _ALL_MODEL_GROUPS,
                     strict::Bool = true)
    ds = NCDataset(restart_file)
    try
        for gname in groups
            group_nt = getfield(y, gname)
            metas    = getfield(y.v, gname)
            for k in keys(metas)
                meta = metas[k]
                name_str = String(meta.name)
                if !haskey(ds, name_str)
                    strict && error(
                        "Variable `$(name_str)` (group `$(gname)`) not found in restart " *
                        "file $(restart_file). Pass `strict=false` to skip missing variables.")
                    continue
                end
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
# _get_var! family. Restart variables may carry a trailing singleton
# `time` dimension; `_read_nc_*` strips it.
#
# Note: Oceananigans 2D fields on a Flat-z grid still have a 3D interior
# array (with size 1 in the third dim), so we branch on that singleton,
# not on `ndims`.

_read_nc_2d(ncvar) = ndims(ncvar) == 2 ? ncvar[:, :]    : ncvar[:, :, 1]
_read_nc_3d(ncvar) = ndims(ncvar) == 3 ? ncvar[:, :, :] : ncvar[:, :, :, 1]

@inline _is_3d_field(int::AbstractArray) = ndims(int) == 3 && size(int, 3) > 1

function _load_into_field!(field::Field{Face, Center, Center}, ncvar) where {Face, Center}
    int = interior(field)
    if _is_3d_field(int)
        data = _read_nc_3d(ncvar)
        int[2:end, :, :] .= data
        int[1, :, :]     .= @view data[1, :, :]
    else
        data = _read_nc_2d(ncvar)
        int[2:end, :, 1] .= data
        int[1, :, 1]     .= @view data[1, :]
    end
    return field
end

function _load_into_field!(field::Field{Center, Face, Center}, ncvar) where {Face, Center}
    int = interior(field)
    if _is_3d_field(int)
        data = _read_nc_3d(ncvar)
        int[:, 2:end, :] .= data
        int[:, 1, :]     .= @view data[:, 1, :]
    else
        data = _read_nc_2d(ncvar)
        int[:, 2:end, 1] .= data
        int[:, 1, 1]     .= @view data[:, 1]
    end
    return field
end

function _load_into_field!(field::Field{Center, Center, Face}, ncvar) where {Face, Center}
    int = interior(field)
    if _is_3d_field(int)
        int[:, :, :] .= _read_nc_3d(ncvar)
    else
        int[:, :, 1] .= _read_nc_2d(ncvar)
    end
    return field
end

function _load_into_field!(field::Field{Center, Center, Center}, ncvar) where {Center}
    int = interior(field)
    if _is_3d_field(int)
        int[:, :, :] .= _read_nc_3d(ncvar)
    else
        int[:, :, 1] .= _read_nc_2d(ncvar)
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

# step!(::YelmoModel, dt) is provided by YelmoModelTopo, which orchestrates
# the per-component physics chain. The generic function is declared here so
# downstream modules (YelmoMirrorCore, YelmoModelTopo) can extend it via
# `import ..YelmoCore: step!`.
function step! end

# ---------------------------------------------------------------------------
# compare_state — backend-agnostic field-wise diff for regression tests
# ---------------------------------------------------------------------------

const _COMPARE_GROUPS = (:bnd, :dta, :dyn, :mat, :thrm, :tpo)

"""
    StateComparison

Result of `compare_state(a, b; tol)`. `passes` is `true` iff every
field present in both models agreed within `tol` (relative L∞).
`failures` lists the per-field violations as
`(group, name, max_abs_diff, rel_linf)` named tuples. `n_compared`
counts fields actually compared; `n_skipped` counts fields skipped
because they were absent from one side or had mismatched shapes.
"""
struct StateComparison
    tol        :: Float64
    failures   :: Vector{NamedTuple{(:group, :name, :max_abs_diff, :rel_linf),
                                    Tuple{Symbol, Symbol, Float64, Float64}}}
    n_compared :: Int
    n_skipped  :: Int
    passes     :: Bool
end

function Base.show(io::IO, c::StateComparison)
    if c.passes
        print(io, "StateComparison: passed ($(c.n_compared) fields, ",
                  "$(c.n_skipped) skipped, tol=$(c.tol))")
    else
        print(io, "StateComparison: FAILED — $(length(c.failures)) of ",
                  "$(c.n_compared) fields exceed tol=$(c.tol)")
        for f in c.failures[1:min(5, length(c.failures))]
            print(io, "\n  $(f.group).$(f.name): rel_linf=", f.rel_linf,
                      " (max_abs_diff=", f.max_abs_diff, ")")
        end
        length(c.failures) > 5 &&
            print(io, "\n  ...$(length(c.failures) - 5) more")
    end
end

"""
    compare_state(a::AbstractYelmoModel, b::AbstractYelmoModel; tol=1e-3)
        -> StateComparison

Field-wise diff between two model states. Iterates over the six
component groups (`bnd`, `dta`, `dyn`, `mat`, `thrm`, `tpo`); for
each field present in *both* `a.<group>` and `b.<group>` with the
same shape, computes `max|a - b| / max|b|` (or `max|a - b|` when the
reference field is identically zero). A field passes if the result
is `≤ tol`. Fields present only on one side, or with mismatched
shapes, are skipped (not failures).

Used to lockstep-validate `YelmoModel` against `YelmoMirror` (or
against another `YelmoModel`).
"""
function compare_state(a::AbstractYelmoModel, b::AbstractYelmoModel;
                       tol::Float64 = 1e-3)
    failures   = NamedTuple{(:group, :name, :max_abs_diff, :rel_linf),
                            Tuple{Symbol, Symbol, Float64, Float64}}[]
    n_compared = 0
    n_skipped  = 0

    for gname in _COMPARE_GROUPS
        a_grp = getfield(a, gname)
        b_grp = getfield(b, gname)
        for k in keys(a_grp)
            if !haskey(b_grp, k)
                n_skipped += 1
                continue
            end
            af = interior(a_grp[k])
            bf = interior(b_grp[k])
            if size(af) != size(bf)
                n_skipped += 1
                continue
            end
            n_compared += 1

            max_abs_diff = Float64(maximum(abs.(af .- bf)))
            max_ref      = Float64(maximum(abs.(bf)))
            rel_linf     = max_ref > 0 ? max_abs_diff / max_ref : max_abs_diff

            if rel_linf > tol
                push!(failures, (group=gname, name=k,
                                 max_abs_diff=max_abs_diff, rel_linf=rel_linf))
            end
        end
    end

    return StateComparison(tol, failures, n_compared, n_skipped, isempty(failures))
end

end # module YelmoCore
