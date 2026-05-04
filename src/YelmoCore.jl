module YelmoCore

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.Grids: topology
using Oceananigans.BoundaryConditions: FieldBoundaryConditions,
                                       ValueBoundaryCondition,
                                       GradientBoundaryCondition,
                                       fill_halo_regions!
using NCDatasets
using Krylov: BicgstabWorkspace

using ..YelmoMeta: VariableMeta, parse_variable_table
using ..YelmoConst: YelmoConstants,
                    MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC,
                    MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN,
                    MASK_BED_STREAM, MASK_BED_GRLINE, MASK_BED_FLOAT,
                    MASK_BED_ISLAND, MASK_BED_PARTIAL
using ..YelmoModelPar: YelmoModelParameters
using ..YelmoTiming: YelmoTimer, @timed_section

export AbstractYelmoModel, YelmoModel
export init_state!, step!, load_state!
export topo_step!, dyn_step!, mat_step!, therm_step!
export load_grids_from_restart, load_fields_from_restart
export load_field_from_dataset_2D, load_field_from_dataset_3D
export make_field, matches_patterns, yelmo_define_grids
export resolve_boundaries, neumann_2d_field, dirichlet_2d_field
export fill_halo_regions!, fill_corner_halos!
export XFACE_VARIABLES, YFACE_VARIABLES, ZFACE_VARIABLES, VERTICAL_DIMS
export MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC
export MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN, MASK_BED_STREAM,
       MASK_BED_GRLINE, MASK_BED_FLOAT, MASK_BED_ISLAND, MASK_BED_PARTIAL
export compare_state, StateComparison

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const VERTICAL_DIMS = (:zeta, :zeta_ac, :zeta_rock, :zeta_rock_ac)

# Substring patterns that map a variable name to an XFace / YFace /
# ZFace allocation. `matches_patterns` uses `occursin`, so e.g. `"ux"`
# matches `"ux_bar"`, `"ux_b"`, `"duxdz"`, etc. — the storage convention
# from Yelmo Fortran is that derivatives live on the same staggered
# grid as the parent variable (`duxdz` is acx-staggered like `ux`),
# so the substring rule does the right thing for those.
const XFACE_VARIABLES = ["ux", r".*_acx$"]
const YFACE_VARIABLES = ["uy", r".*_acy$"]
const ZFACE_VARIABLES = ["uz", "uz_star", "jvel_dzx", "jvel_dzy", "jvel_dzz"]

# Exceptions to the substring rule: names that overlap a face pattern
# but physically live at aa-cell centres. The override is checked
# *before* the face patterns in `make_field`, so a match here wins.
#
#   - `"uxy"` catches the horizontal-velocity *magnitude* family
#     (`uxy`, `uxy_b`, `uxy_s`, `uxy_bar`, `uxy_i_bar`) and its
#     time derivative `duxydt`, plus the corresponding observations
#     `pd_uxy_s` and `pd_err_uxy_s`. These are scalar magnitudes on
#     aa-cells, not staggered components.
#   - `r"^uz_b$"` / `r"^uz_s$"` catch the 2D slices of the 3D `uz`
#     field. `uz` itself stays on z-faces (per the Yelmo Fortran
#     layout for vertical velocity), but its bottom (`uz_b`) and top
#     (`uz_s`) 2D slices are emitted at aa-cell centres in the
#     schema. The patterns are anchored regex to avoid a false
#     substring match against `uz_star` (which is a genuine 3D
#     ZFace field and *should* keep its face allocation).
const CENTER_OVERRIDES = ["uxy", r"^uz_b$", r"^uz_s$"]

# Per-cell ice evolution mask values (`bnd.mask_ice`) — defined in
# YelmoConst and re-exported here for back-compat with existing call
# sites that import from YelmoCore.

# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

matches_patterns(name, patterns) = any(p -> occursin(p, name), patterns)

# ---------------------------------------------------------------------------
# Grid construction and boundary resolution
# ---------------------------------------------------------------------------

# Convert a user-friendly `boundaries` argument to an Oceananigans
# horizontal-axis topology pair `(X, Y)`. Accepts:
#
#   - Symbol shortcut: `:bounded`, `:periodic`, `:periodic_x`, `:periodic_y`
#   - 2-tuple of Symbols: `(:periodic, :bounded)` etc.
#   - Direct Oceananigans tuple of types: `(Periodic, Bounded)`,
#     optionally with a trailing 3rd entry which is ignored.
#
# The vertical (`z`) topology is always inferred from the calling
# context (Flat for 2D, Bounded for 3D), so this helper only resolves
# the horizontal pair.
"""
    resolve_boundaries(boundaries) -> (X, Y)

Resolve a user-friendly `boundaries` specifier to a pair of
Oceananigans horizontal topology types. Returns `(X, Y)` where each
entry is `Bounded` or `Periodic`. Used by the grid constructors and
by callers that need to inspect the active topology.

Supported forms: `:bounded`, `:periodic`, `:periodic_x`,
`:periodic_y`, `(:periodic, :bounded)`, `(Bounded, Periodic)`, etc.
"""
function resolve_boundaries(boundaries)
    boundaries === :bounded     && return (Bounded,  Bounded)
    boundaries === :periodic    && return (Periodic, Periodic)
    boundaries === :periodic_x  && return (Periodic, Bounded)
    boundaries === :periodic_y  && return (Bounded,  Periodic)

    if boundaries isa Tuple && length(boundaries) >= 2
        x = _topology_from(boundaries[1])
        y = _topology_from(boundaries[2])
        return (x, y)
    end

    error("resolve_boundaries: unrecognised boundaries=$(boundaries). " *
          "Use a Symbol (:bounded, :periodic, :periodic_x, :periodic_y), " *
          "a 2-tuple of Symbols (e.g. (:periodic, :bounded)), or a tuple " *
          "of Oceananigans topology types (e.g. (Bounded, Periodic)).")
end

@inline _topology_from(t::Symbol) = t === :periodic ? Periodic :
                                    t === :bounded  ? Bounded  :
                                    error("Unknown topology symbol :$(t). " *
                                          "Use :bounded or :periodic.")
@inline _topology_from(::Type{Periodic}) = Periodic
@inline _topology_from(::Type{Bounded})  = Bounded
@inline _topology_from(t) = error("Cannot interpret topology entry $(t).")

yelmo_define_grids(g::NamedTuple; kwargs...) =
    yelmo_define_grids(g.xc, g.yc, g.zeta_ac, g.zeta_r_ac; kwargs...)

function yelmo_define_grids(xc, yc, zeta_ac, zeta_r_ac;
                            boundaries = :bounded)
    Nx, Ny = length(xc), length(yc)
    dx, dy = xc[2] - xc[1], yc[2] - yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)

    Tx, Ty = resolve_boundaries(boundaries)

    grid2d = RectilinearGrid(size=(Nx, Ny),
                             x=xlims, y=ylims,
                             topology=(Tx, Ty, Flat))

    grid3d_ice = RectilinearGrid(size=(Nx, Ny, length(zeta_ac) - 1),
                                 x=xlims, y=ylims, z=zeta_ac,
                                 topology=(Tx, Ty, Bounded))

    grid3d_rock = RectilinearGrid(size=(Nx, Ny, length(zeta_r_ac) - 1),
                                  x=xlims, y=ylims, z=zeta_r_ac,
                                  topology=(Tx, Ty, Bounded))

    return grid2d, grid3d_ice, grid3d_rock
end

# ---------------------------------------------------------------------------
# Field allocation
# ---------------------------------------------------------------------------

function make_field(varname::Union{AbstractString,Symbol}, grid::RectilinearGrid)
    varname = String(varname)
    # Override list wins: a name on `CENTER_OVERRIDES` allocates as a
    # CenterField even if it matches one of the face patterns by
    # substring (see the documentation alongside `CENTER_OVERRIDES`).
    if matches_patterns(varname, CENTER_OVERRIDES)
        return neumann_2d_field(grid)
    elseif matches_patterns(varname, XFACE_VARIABLES)
        return XFaceField(grid)
    elseif matches_patterns(varname, YFACE_VARIABLES)
        return YFaceField(grid)
    elseif matches_patterns(varname, ZFACE_VARIABLES)
        return ZFaceField(grid)
    else
        # Default Center fields get Neumann-zero (clamp / zero-gradient)
        # on Bounded sides. Halo reads after `fill_halo_regions!` then
        # return the first interior cell — semantically "extend the
        # field across the boundary" — matching the Fortran `infinite`
        # boundary-code default. Periodic sides wrap automatically via
        # the grid topology.
        return neumann_2d_field(grid)
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
    neumann_2d_field(grid; value=0)

Construct a 2D `CenterField` on `grid` whose horizontal boundary
conditions are zero-gradient (Neumann) on the four sides of any
`Bounded` axis. This populates the halo with the first interior cell
after `fill_halo_regions!`, the natural choice for diagnostic /
state fields where the physically meaningful interpretation of "outside
the domain" is "extend the value at the boundary." Periodic axes
wrap automatically and ignore the per-side BC entries.

This is the default boundary-condition palette for `tpo`/`thrm`/`mat`/
`dyn`/`bnd`/`dta` group fields allocated through `make_field`. Use
`dirichlet_2d_field` instead when the field's physical interpretation
demands a specific *face value* outside the domain (e.g. `H_ice = 0`
for advection mass conservation).
"""
function neumann_2d_field(grid::RectilinearGrid; value::Real = 0.0)
    # Oceananigans rejects per-side BCs on Periodic axes — the wrap
    # already determines halo values. Only attach the Gradient BC on
    # axes that are Bounded.
    Tx = topology(grid, 1)
    Ty = topology(grid, 2)
    east_bc  = Tx === Bounded ? GradientBoundaryCondition(value) : nothing
    west_bc  = Tx === Bounded ? GradientBoundaryCondition(value) : nothing
    south_bc = Ty === Bounded ? GradientBoundaryCondition(value) : nothing
    north_bc = Ty === Bounded ? GradientBoundaryCondition(value) : nothing
    kwargs = Dict{Symbol,Any}()
    east_bc  !== nothing && (kwargs[:east]  = east_bc)
    west_bc  !== nothing && (kwargs[:west]  = west_bc)
    south_bc !== nothing && (kwargs[:south] = south_bc)
    north_bc !== nothing && (kwargs[:north] = north_bc)
    bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()); kwargs...)
    return CenterField(grid; boundary_conditions=bcs)
end

"""
    fill_corner_halos!(field) -> field

Fill the four corner halo regions of a 2D `field` consistently with
the grid's per-axis topology. `Oceananigans.fill_halo_regions!` only
populates the four orthogonal sides of the halo for Bounded axes —
it leaves the (i ≤ 0, j ≤ 0) etc. corner halo cells unwritten because
the standard finite-volume operators don't read them. Kernels that
*do* need diagonal halo reads (chamfer distance transforms, the
sub-grid-grounded-fraction quad stencil, ...) must call this helper
*after* `fill_halo_regions!` to populate the corners.

Per axis:

  - `Periodic`: corner halo wraps to the opposite side of the
    interior (e.g. SW corner halo cell `(1-hi, 1-hj)` = interior
    `(Nx-hi+1, Ny-hj+1)`).
  - `Bounded`: corner halo clamps to the nearest interior corner
    (e.g. SW corner halo = interior `(1, 1)`).

Mixed topologies (e.g. periodic-x + bounded-y) compose component-wise.
The helper is a no-op for fields whose grids are `Flat` in either
horizontal axis (no horizontal halo to fill).
"""
function fill_corner_halos!(field)
    grid = field.grid
    Tx = topology(grid, 1)
    Ty = topology(grid, 2)

    (Tx === Flat || Ty === Flat) && return field

    Nx = size(field, 1)
    Ny = size(field, 2)
    Hx = grid.Hx
    Hy = grid.Hy

    @inbounds for k in axes(parent(field), 3),
                  hj in 1:Hy, hi in 1:Hx
        # SW corner: (i, j) = (1 - hi, 1 - hj).
        ix_sw = Tx === Periodic ? Nx - hi + 1 : 1
        jy_sw = Ty === Periodic ? Ny - hj + 1 : 1
        field[1 - hi, 1 - hj, k] = field[ix_sw, jy_sw, k]

        # SE corner: (Nx + hi, 1 - hj).
        ix_se = Tx === Periodic ? hi : Nx
        jy_se = Ty === Periodic ? Ny - hj + 1 : 1
        field[Nx + hi, 1 - hj, k] = field[ix_se, jy_se, k]

        # NW corner: (1 - hi, Ny + hj).
        ix_nw = Tx === Periodic ? Nx - hi + 1 : 1
        jy_nw = Ty === Periodic ? hj : Ny
        field[1 - hi, Ny + hj, k] = field[ix_nw, jy_nw, k]

        # NE corner: (Nx + hi, Ny + hj).
        ix_ne = Tx === Periodic ? hi : Nx
        jy_ne = Ty === Periodic ? hj : Ny
        field[Nx + hi, Ny + hj, k] = field[ix_ne, jy_ne, k]
    end
    return field
end

"""
    dirichlet_2d_field(grid, value)

Construct a 2D `CenterField` on `grid` with Dirichlet `value` boundary
conditions on every non-Flat horizontal face (east/west/south/north).
Used for ice-sheet fields like `H_ice` whose physical boundary
condition is "no ice past the domain edge" — the upwind advection
operator reads the face values directly via Oceananigans' standard
halo machinery without per-cell branching.
"""
function dirichlet_2d_field(grid::RectilinearGrid, value::Real)
    # Oceananigans rejects per-side BCs on Periodic axes; only attach
    # the Value BC on Bounded axes.
    Tx = topology(grid, 1)
    Ty = topology(grid, 2)
    kwargs = Dict{Symbol,Any}()
    Tx === Bounded && (kwargs[:east]  = ValueBoundaryCondition(value);
                       kwargs[:west]  = ValueBoundaryCondition(value))
    Ty === Bounded && (kwargs[:south] = ValueBoundaryCondition(value);
                       kwargs[:north] = ValueBoundaryCondition(value))
    bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()); kwargs...)
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
function load_grids_from_restart(filename::AbstractString;
                                 boundaries = :bounded)
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

    Tx, Ty = resolve_boundaries(boundaries)

    grid2d = RectilinearGrid(size=(Nx, Ny),
                             x=xlims, y=ylims,
                             topology=(Tx, Ty, Flat))

    grid3d_ice = _build_3d_grid(ds, "zeta", "zeta_ac", Nx, Ny, xlims, ylims, Tx, Ty)
    grid3d_rock = _build_3d_grid(ds, "zeta_rock", "zeta_rock_ac", Nx, Ny, xlims, ylims, Tx, Ty)

    close(ds)
    return grid2d, grid3d_ice, grid3d_rock
end

# Build a 3D RectilinearGrid using face-coordinates from `face_var` if
# present, otherwise reconstruct them from cell-center coordinates in
# `center_var` (`zeta_ac[i] = (zeta[i-1] + zeta[i]) / 2` for interior
# faces, with the boundary faces clamped to the first/last center).
function _build_3d_grid(ds, center_var, face_var, Nx, Ny, xlims, ylims,
                        Tx::DataType, Ty::DataType)
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
                           topology=(Tx, Ty, Bounded))
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
    c::YelmoConstants
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
    # Per-section wall-clock accumulator populated by `@timed_section`.
    # Always present; `timer.enabled` controls whether call sites
    # actually measure (set via `yelmo_params(timing = true)`).
    # See `src/timing.jl`.
    timer::YelmoTimer
end

const _ALL_MODEL_GROUPS = (:bnd, :dta, :dyn, :mat, :thrm, :tpo)

# Load the per-group `VariableMeta` tables from `src/variables/model/`.
# Shared by every `YelmoModel` constructor so the schema lookup is in one
# place; returns a NamedTuple with one entry per component group.
function _load_yelmo_variable_meta()
    vdir = joinpath(@__DIR__, "variables", "model")
    return (
        bnd  = parse_variable_table(joinpath(vdir, "yelmo-variables-ybound.md"), "bnd"),
        dta  = parse_variable_table(joinpath(vdir, "yelmo-variables-ydata.md"),  "dta"),
        dyn  = parse_variable_table(joinpath(vdir, "yelmo-variables-ydyn.md"),   "dyn"),
        mat  = parse_variable_table(joinpath(vdir, "yelmo-variables-ymat.md"),   "mat"),
        thrm = parse_variable_table(joinpath(vdir, "yelmo-variables-ytherm.md"), "thrm"),
        tpo  = parse_variable_table(joinpath(vdir, "yelmo-variables-ytopo.md"),  "tpo"),
    )
end

# Build the six component-group NamedTuples (`bnd`, `dta`, `dyn`, `mat`,
# `thrm`, `tpo`) for a `YelmoModel`, including:
#
#   - per-field allocation via `_alloc_group`,
#   - `dyn.scratch` substruct holding SIA-only scratch buffers (see
#     `src/dyn/velocity_sia.jl`), and
#   - replacement of `tpo.H_ice` with a Dirichlet-zero `CenterField`
#     so the upwind advection kernels see `H_ice = 0` on the domain
#     edges via Oceananigans' standard halo machinery.
#
# Both the file-based `YelmoModel(restart_file, time; …)` constructor and
# the in-memory `YelmoModel(::AbstractBenchmark, t; …)` constructor in
# `test/benchmarks/benchmarks.jl` go through this helper, so the
# allocation is identical regardless of whether the field values come
# from a NetCDF file or from a benchmark's analytical state.
function _alloc_yelmo_groups(g, gt, gr, v_meta)
    bnd  = _alloc_group(v_meta.bnd,  g, gt, gr)
    dta  = _alloc_group(v_meta.dta,  g, gt, gr)
    dyn  = _alloc_group(v_meta.dyn,  g, gt, gr)
    mat  = _alloc_group(v_meta.mat,  g, gt, gr)
    thrm = _alloc_group(v_meta.thrm, g, gt, gr)
    tpo  = _alloc_group(v_meta.tpo,  g, gt, gr)

    # SIA / SSA solver scratch buffers; not in the dyn schema because
    # they are recomputed every `dyn_step!` and not part of the model
    # state. Exposed as `y.dyn.scratch.<name>`. See
    # `src/dyn/velocity_sia.jl` (SIA) and `src/dyn/viscosity.jl` (SSA).
    #
    #   - `sia_tau_xz` / `sia_tau_yz` (3D, on `gt`): per-layer SIA
    #     vertical shear stresses at Center positions.
    #   - `ux_i_s` / `uy_i_s` (2D, on `g`): SIA shear velocity at the
    #     ice surface (zeta = 1), computed by the wrapper since the
    #     3D `ux_i` / `uy_i` Center stagger does NOT include the
    #     surface endpoint under Option C. Read by `dyn_step!` to
    #     assemble `ux_s = ux_i_s + ux_b` (and analogously for uy_s).
    #   - `ssa_n_aa_ab` (2D, on `g`): corner-staggered viscosity cache
    #     `Field((Face(), Face(), Center()), g)`. Output of
    #     `stagger_visc_aa_ab!`; consumed by the SSA matrix assembly
    #     (PR-A.2). Allocated here so the buffer is ready when the SSA
    #     solver wires up.
    #   - `ssa_I_idx` / `ssa_J_idx` / `ssa_vals` (PR-A.2): COO triplet
    #     buffers for the SSA stiffness matrix. Pre-allocated to a
    #     generous size (2 * 9 * Nx * Ny) covering the 9-point inner
    #     stencil for both ux and uy block rows. The actual non-zero
    #     count is tracked in `ssa_nnz`.
    #   - `ssa_b_vec` / `ssa_x_vec` (PR-A.2): RHS and solution buffers,
    #     length 2 * Nx * Ny. The matrix-assembly kernel writes
    #     `ssa_b_vec` and the (PR-B) Krylov solver writes `ssa_x_vec`.
    #   - `ssa_iter_now` (PR-A.2): `Ref{Int}` for the Picard iteration
    #     counter (PR-B's driver writes here).
    Nx_int, Ny_int = size(g, 1), size(g, 2)
    N_cells = Nx_int * Ny_int
    N_rows = 2 * N_cells
    N_nz_max = 2 * 9 * N_cells

    sia_scratch = (sia_tau_xz  = XFaceField(gt), sia_tau_yz  = YFaceField(gt),
                   ux_i_s      = XFaceField(g),  uy_i_s      = YFaceField(g),
                   ssa_n_aa_ab = Field((Face(), Face(), Center()), g))
    # PR-B additions: Krylov workspace + AMG-cache + Picard "n minus 1"
    # snapshots + per-iteration L2 residual history. The Krylov workspace
    # is allocated once at this size (2*Nx*Ny rows). The AMG cache is
    # rebuilt every Picard iteration since the matrix coefficients
    # change with viscosity / beta — store a `Ref{Any}` that the driver
    # populates lazily.
    ssa_scratch = (
        ssa_I_idx                  = Vector{Int}(undef, N_nz_max),
        ssa_J_idx                  = Vector{Int}(undef, N_nz_max),
        ssa_vals                   = Vector{Float64}(undef, N_nz_max),
        ssa_nnz                    = Ref{Int}(0),
        ssa_b_vec                  = Vector{Float64}(undef, N_rows),
        ssa_x_vec                  = Vector{Float64}(undef, N_rows),
        ssa_iter_now               = Ref{Int}(0),
        ssa_solver_workspace       = BicgstabWorkspace(N_rows, N_rows, Vector{Float64}),
        ssa_amg_cache              = Ref{Any}(nothing),
        ssa_picard_visc_eff_nm1    = CenterField(gt),
        ssa_picard_ux_b_nm1        = XFaceField(g),
        ssa_picard_uy_b_nm1        = YFaceField(g),
        ssa_residuals              = Vector{Float64}(undef, 100),
        # Bed (zeta = 0) / surface (zeta = 1) visc boundary fields for
        # the Option C trapezoidal-with-boundary depth-integration in
        # `calc_visc_eff_int!`. Filled per Picard iteration in
        # `calc_velocity_ssa!` from the nearest-Center value.
        ssa_visc_eff_b             = CenterField(g),
        ssa_visc_eff_s             = CenterField(g),
    )
    # Adaptive predictor-corrector scratch (src/timestepping.jl).
    # `Ref{Any}` so we can lazily allocate a `PCScratch` on the first
    # call to `step!` with `dt_method != 0` — the type lives in a
    # later-included module, so eagerly typing it here would create
    # an awkward forward-reference. Mirror the `ssa_amg_cache` pattern.
    pc_scratch = (pc_scratch = Ref{Any}(nothing),)

    # DIVA scratch (src/dyn/velocity_diva.jl). The Picard outer loop
    # there mirrors SSA's structure but operates on the depth-averaged
    # `ux_bar / uy_bar` (matrix unknown for the depth-integrated
    # momentum balance) rather than `ux_b / uy_b`. F1 is 3D Center
    # (cumulative bed-to-layer integral). beta_eff substitutes for
    # beta in the matrix-kernel inputs.
    diva_scratch = (
        diva_F2                 = CenterField(g),                 # 2D depth-integrated 1/η · (1−ζ)²
        diva_F1_3D              = CenterField(gt),                # 3D cumulative 1/η · (1−ζ)
        diva_beta_eff           = CenterField(g),                 # Goldberg-2011 β_eff at aa-cells
        diva_beta_eff_acx       = XFaceField(g),                  # face-staggered β_eff
        diva_beta_eff_acy       = YFaceField(g),
        diva_picard_ux_bar_nm1  = XFaceField(g),                  # Picard "n minus 1" snapshots
        diva_picard_uy_bar_nm1  = YFaceField(g),                  # for the depth-averaged unknown
    )

    dyn = merge(dyn, (scratch = merge(sia_scratch, ssa_scratch, pc_scratch, diva_scratch),))

    # Replace H_ice with a CenterField that carries Dirichlet H_ice = 0
    # boundary conditions on the domain edge. The upwind advection
    # operator reads these via Oceananigans' standard halo machinery
    # without per-cell branching in the kernel.
    haskey(tpo, :H_ice) && (tpo = merge(tpo, (H_ice = dirichlet_2d_field(g, 0.0),)))

    # Implicit advection scratch (src/topo/advection.jl).
    # `Ref{Any}` is filled lazily on the first
    # `advect_tracer!(...; scheme=:upwind_implicit)` call with an
    # `ImplicitAdvectionCache(grid)`. The concrete type lives in the
    # later-included YelmoModelTopo module, so we mirror the
    # `pc_scratch` / `ssa_amg_cache` lazy pattern rather than
    # eagerly typing it here.
    tpo_scratch = (adv_cache = Ref{Any}(nothing),)
    tpo = merge(tpo, (scratch = tpo_scratch,))

    return bnd, dta, dyn, mat, thrm, tpo
end

"""
    YelmoModel(restart_file, time; alias, rundir, p, c, groups, strict)

Construct a `YelmoModel` whose grids and field values are read from a NetCDF
restart file. Variable layout is taken from `src/variables/model/` markdown
tables. The model parameters `p` are passed through verbatim and may be
`nothing`, a `YelmoModelParameters`, or any user object.

The physical constants `c` default to `YelmoConstants()` (Yelmo Fortran
defaults). The struct is immutable, so the same `c` instance can be safely
shared across multiple `YelmoModel`s when the physics is identical.

`groups` selects which component groups to load from the restart (default:
all six). `strict=true` (default) errors if a variable in a loaded group
is missing from the restart; `strict=false` silently skips missing
variables, leaving the corresponding field at its default-allocated value.
"""
# Read the `yelmo.timing` flag from a YelmoModelParameters, defaulting
# to `false` when no parameter object is available. Used by both
# `YelmoModel` constructors to initialise the per-model `YelmoTimer`.
_resolve_timing_enabled(p) = p === nothing ? false : p.yelmo.timing

function YelmoModel(restart_file::String, time::Float64;
                    alias::String = "ymodel1",
                    rundir::String = "./",
                    p = nothing,
                    c::YelmoConstants = YelmoConstants(),
                    boundaries = :bounded,
                    groups::NTuple{N,Symbol} where N = _ALL_MODEL_GROUPS,
                    strict::Bool = true)

    if p === nothing
        @warn "No parameters supplied to YelmoModel; constructing YelmoModelParameters(\"$(alias)\") with defaults."
        p = YelmoModelParameters(alias)
    end

    g, gt, gr = load_grids_from_restart(restart_file; boundaries=boundaries)
    gt === nothing && error("Restart file $(restart_file) has no ice vertical axis; cannot build ice grid.")
    gr === nothing && error("Restart file $(restart_file) has no rock vertical axis; cannot build rock grid.")

    v_meta = _load_yelmo_variable_meta()
    bnd, dta, dyn, mat, thrm, tpo = _alloc_yelmo_groups(g, gt, gr, v_meta)

    timer = YelmoTimer(enabled = _resolve_timing_enabled(p))

    y = YelmoModel(alias, rundir, time, p, c, g, gt, gr, v_meta, bnd, dta, dyn, mat, thrm, tpo, timer)

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
        mask_ice = interior(y.bnd.mask_ice)
        @inbounds for j in axes(mask_ice, 2), i in axes(mask_ice, 1)
            mask_ice[i, j, 1] = ia[i, j] != 0 ? Float64(MASK_ICE_DYNAMIC) : Float64(MASK_ICE_NONE)
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
    # XFaceField loader. Interior shape depends on x-axis topology:
    #   - Bounded-x:  `(Nx+1, Ny[, Nz])` — Ny cells × (Nx+1) face slots.
    #     Fixture stores face data at `(Nx, Ny[, Nz])` (Fortran cell-centred
    #     convention); we write `data` into slots `[2:end, :, …]` and
    #     replicate the leading slot from the first written column to
    #     match the convention used by `load_field_from_dataset_2D`.
    #   - Periodic-x: `(Nx, Ny[, Nz])` — incoming data shape matches
    #     interior shape directly; write in place, no replicate slot.
    #
    # The `Tx` dispatch matches the pattern used by `_ip1_modular` and
    # related helpers in `src/dyn/topology_helpers.jl`.
    Tx, _Ty, _Tz = topology(field.grid)
    int = interior(field)
    if _is_3d_field(int)
        data = _read_nc_3d(ncvar)
        if Tx === Bounded
            int[2:end, :, :] .= data
            int[1, :, :]     .= @view data[1, :, :]
        elseif Tx === Periodic
            int[:, :, :] .= data
        else
            error("_load_into_field!(XFaceField, 3D): unsupported x-topology $Tx")
        end
    else
        data = _read_nc_2d(ncvar)
        if Tx === Bounded
            int[2:end, :, 1] .= data
            int[1, :, 1]     .= @view data[1, :]
        elseif Tx === Periodic
            int[:, :, 1] .= data
        else
            error("_load_into_field!(XFaceField, 2D): unsupported x-topology $Tx")
        end
    end
    return field
end

function _load_into_field!(field::Field{Center, Face, Center}, ncvar) where {Face, Center}
    # YFaceField loader. Symmetric to the XFaceField method above —
    # interior shape depends on y-axis topology:
    #   - Bounded-y:  `(Nx, Ny+1[, Nz])`. Fixture stores `(Nx, Ny[, Nz])`,
    #     written into slots `[:, 2:end, …]` with the first slot
    #     replicated.
    #   - Periodic-y: `(Nx, Ny[, Nz])`. Direct copy — no replicate slot,
    #     no extra face row (the `Ny+1`-th face is the `1`-st by wrap
    #     and is not stored).
    _Tx, Ty, _Tz = topology(field.grid)
    int = interior(field)
    if _is_3d_field(int)
        data = _read_nc_3d(ncvar)
        if Ty === Bounded
            int[:, 2:end, :] .= data
            int[:, 1, :]     .= @view data[:, 1, :]
        elseif Ty === Periodic
            int[:, :, :] .= data
        else
            error("_load_into_field!(YFaceField, 3D): unsupported y-topology $Ty")
        end
    else
        data = _read_nc_2d(ncvar)
        if Ty === Bounded
            int[:, 2:end, 1] .= data
            int[:, 1, 1]     .= @view data[:, 1]
        elseif Ty === Periodic
            int[:, :, 1] .= data
        else
            error("_load_into_field!(YFaceField, 2D): unsupported y-topology $Ty")
        end
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

# ---------------------------------------------------------------------------
# Per-component time-stepping generics
# ---------------------------------------------------------------------------
#
# Each phase of the model has its own `<comp>_step!(y, dt)` function declared
# here as a forward-only generic. The phase modules (YelmoModelTopo,
# eventually YelmoModelDyn / YelmoModelMat / YelmoModelTherm) `import` the
# matching generic and add the actual method body. Forward declaration here
# lets the orchestrator below dispatch to the per-phase methods at runtime
# without YelmoCore needing to depend on the phase modules.
function topo_step!  end
function dyn_step!   end
function mat_step!   end
function therm_step! end

# `step!(::YelmoModel, dt)` orchestrates the per-component physics chain in
# a fixed phase order (tpo → dyn → mat → therm). Phase order matches the
# Fortran per-step loop at `yelmo_ice.f90:268-286` (predictor topo →
# `calc_ydyn` → `calc_ymat` → `calc_ytherm` → corrector topo). `dyn_step!`
# reads the previous step's `mat.ATT`; `mat_step!` then computes a fresh
# `ATT` (and viscosity, stress, ...) from the just-solved velocity field
# for the next step. Methods for the `<comp>_step!` calls below are added
# by the corresponding phase modules at load time. `YelmoMirror` overrides
# `step!` in `YelmoMirrorCore` to call the C API instead of the per-phase
# chain.
function step!(y::YelmoModel, dt::Float64)
    # Backend dispatch on `y.p.yelmo.dt_method`:
    #   0 = fixed forward Euler (this body, current default).
    #   2 = adaptive predictor-corrector (delegates to
    #       `Yelmo._select_step!` from src/timestepping.jl, which
    #       handles snapshot/restore + PC + PI controller).
    # When `y.p === nothing`, fall back to fixed FE for backwards
    # compatibility with simple in-memory benchmark constructions
    # that don't pass parameters.
    method = y.p === nothing ? 0 : Int(y.p.yelmo.dt_method)
    if method == 0
        # Per-section timing wraps live inside `_step_fe!` (defined in
        # src/timestepping.jl) so both the fixed-FE and the adaptive PC
        # paths share the same instrumented call sites.
        return _step_fe!(y, dt)
    else
        # `_select_step!` is defined in src/timestepping.jl, which is
        # included after the topo + dyn modules so it can call them.
        return _select_step!(y, dt)
    end
end

# Forward declaration: body lives in src/timestepping.jl. Defining the
# symbol here lets `step!` reference it before timestepping.jl is
# included.
function _step_fe! end

# Forward declaration — body lives in src/timestepping.jl. Defining
# the symbol here keeps `step!` self-contained even though the
# adaptive backend is added by a later include.
function _select_step! end

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
            # Skip the scratch substruct explicitly, plus any other
            # non-Field entries defensively. The scratch sub-NamedTuple
            # under `dyn.scratch` holds SIA solver buffers (recomputed
            # every step, not part of model state) — exclude by name
            # first, then fall back to a type check that catches any
            # future non-Field entry under another name.
            k === :scratch && continue
            a_grp[k] isa AbstractField || (n_skipped += 1; continue)
            b_grp[k] isa AbstractField || (n_skipped += 1; continue)
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
