module YelmoCore

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields
using Oceananigans.Grids: topology
using Oceananigans.BoundaryConditions: FieldBoundaryConditions,
                                       ValueBoundaryCondition,
                                       GradientBoundaryCondition,
                                       fill_halo_regions!
using NCDatasets
using Krylov: BicgstabWorkspace, CgWorkspace

using ..YelmoMeta: VariableMeta, parse_variable_table
using ..YelmoConst: YelmoConstants,
                    MASK_ICE_NONE, MASK_ICE_FIXED, MASK_ICE_DYNAMIC,
                    MASK_BED_OCEAN, MASK_BED_LAND, MASK_BED_FROZEN,
                    MASK_BED_STREAM, MASK_BED_GRLINE, MASK_BED_FLOAT,
                    MASK_BED_ISLAND, MASK_BED_PARTIAL
using ..YelmoModelPar: YelmoModelParameters
using ..YelmoTiming: YelmoTimer, @timed_section
using ..YelmoUtils: map_scrip_field, map_scrip_load, gen_map_filename

export AbstractYelmoModel, YelmoModel
export init_state!, step!, load_state!
export topo_step!, dyn_step!, mat_step!, therm_step!
export load_grids_from_restart, load_grids_with_regrid, load_fields_from_restart
export load_field_from_dataset_2D, load_field_from_dataset_3D
export make_field, matches_patterns, yelmo_define_grids
export resolve_boundaries, neumann_2d_field, dirichlet_2d_field
export fill_halo_regions!, fill_corner_halos!
export XFACE_VARIABLES, YFACE_VARIABLES, ZFACE_VARIABLES, VERTICAL_DIMS
export PATH_B_REGISTRY_ICE, is_path_b_registered, path_b_slice_kind, path_b_unified_name
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
# Path B boundary-fields registry
# ---------------------------------------------------------------------------
#
# Under the Path B vertical convention (commit 1+ of the vert-split
# refactor), the file format on disk stays unchanged: a 3D ice
# variable like `T_ice` is stored as a length-`Nz_file` slab whose
# first and last z-slices are the basal (z=0) and surface (z=1)
# boundary values respectively, and whose interior `[:, :, 2:end-1]`
# is the cell-centred interior.
#
# Inside Yelmo, that single file slab decomposes into three fields:
#   - `<name>_b` — 2D, basal boundary value (z=0).
#   - `<name>`   — 3D, length-Nz interior (Nz = Nz_file − 2).
#   - `<name>_s` — 2D, surface boundary value (z=1).
#
# This registry is the single source of truth for the basal /
# interior / surface decomposition. It drives:
#   - restart load (`load_state!`): split the unified slab on read.
#   - output write (`init_output` / `write_output!`): glue the
#     three fields back into a unified slab on write so the file
#     format remains Mirror-compatible.
#   - YelmoMirror C-API marshalling (`yelmo_get_var3D!` /
#     `yelmo_set_var3D!`): same glue/split when bridging Yelmo
#     fields to the Fortran-side length-`Nz_file` buffers.
#
# Only ice-grid fields are registered in commit 2. Bedrock fields
# (`T_rock`, `enth_rock`) remain on the legacy grid convention (commit
# 5c defers the bedrock Path B grid switch). `T_rock_b` is written as
# a diagnostic from the deepest bedrock layer in `therm_step!`.
# When the bedrock grid switches to Path B, `T_rock_s` will be aliased
# to `T_ice_b` — encoded via a special-case entry rather than storage.
#
# Registry shape: a `NamedTuple` keyed by the unified-name `Symbol`,
# each entry is `(b = <basal sym>, s = <surface sym>)`. The
# unified key itself is the interior-field symbol.
const PATH_B_REGISTRY_ICE = (
    T_ice   = (b = :T_ice_b,   s = :T_ice_s),
    enth    = (b = :enth_b,    s = :enth_s),
    T_pmp   = (b = :T_pmp_b,   s = :T_pmp_s),
    T_prime = (b = :T_prime_b, s = :T_prime_s),
    visc    = (b = :visc_b,    s = :visc_s),
    ATT     = (b = :ATT_b,     s = :ATT_s),
    enh     = (b = :enh_b,     s = :enh_s),
    # 3D dyn velocity components are added with commit 3 (the dyn
    # refactor). The existing 2D `ux_b` / `ux_s` / `uy_b` / `uy_s`
    # fields are now true boundary storage rather than redundant
    # slice-1 / slice-end copies of the 3D fields. On read the file's
    # unified `ux` / `uy` slab decomposes into `_b` (slice 1),
    # interior (slice 2:end-1), and `_s` (slice end). On write the
    # three Yelmo fields glue back into one unified slab.
    ux      = (b = :ux_b,      s = :ux_s),
    uy      = (b = :uy_b,      s = :uy_s),
)

# Reverse-lookup tables built once at module load. Map every
# registered symbol (interior, _b, _s) to:
#   - its slice kind (`:basal`, `:interior`, `:surface`)
#   - the unified file-variable name (always the interior sym as
#     a `String`).
const _PATH_B_KIND      = let d = Dict{Symbol, Symbol}()
    for (interior_sym, bs) in pairs(PATH_B_REGISTRY_ICE)
        d[interior_sym] = :interior
        d[bs.b]         = :basal
        d[bs.s]         = :surface
    end
    d
end
const _PATH_B_UNIFIED   = let d = Dict{Symbol, String}()
    for (interior_sym, bs) in pairs(PATH_B_REGISTRY_ICE)
        d[interior_sym] = String(interior_sym)
        d[bs.b]         = String(interior_sym)
        d[bs.s]         = String(interior_sym)
    end
    d
end

"""
    is_path_b_registered(sym::Symbol) -> Bool

True iff `sym` is one of the interior, basal, or surface symbols of
a Path B-registered field.
"""
@inline is_path_b_registered(sym::Symbol) = haskey(_PATH_B_KIND, sym)

"""
    path_b_slice_kind(sym::Symbol) -> Symbol

Return `:basal`, `:interior`, or `:surface` for a registered `sym`.
Throws `KeyError` if `sym` is not registered — guard with
`is_path_b_registered`.
"""
@inline path_b_slice_kind(sym::Symbol) = _PATH_B_KIND[sym]

"""
    path_b_unified_name(sym::Symbol) -> String

Return the unified disk / Fortran variable name for a registered
`sym` (i.e. the name of the file slab that combines basal + interior
+ surface). Throws `KeyError` if `sym` is not registered.
"""
@inline path_b_unified_name(sym::Symbol) = _PATH_B_UNIFIED[sym]

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
# RMSEStats: scalar comparison metrics for the `dta` group
# ---------------------------------------------------------------------------
#
# Lives in `YelmoCore` (rather than `YelmoModelData`) so the type can
# be referenced by `_alloc_yelmo_groups` below without a forward
# reference into a later-loaded module. `data_compare!` (in
# `src/data/YelmoModelData.jl`) is the only writer; readers just
# `getfield` the four scalars from `y.dta.rmse`.
#
# Mirrors Fortran `dta%pd%rmse_H/_zsrf/_uxy/_loguxy`. `rmse_iso` is
# omitted — isochrone comparison is deferred until the corresponding
# `mat` diagnostics land.
mutable struct RMSEStats
    H::Float64
    zsrf::Float64
    uxy::Float64
    loguxy::Float64
end
RMSEStats() = RMSEStats(NaN, NaN, NaN, NaN)
export RMSEStats

# ---------------------------------------------------------------------------
# NetCDF restart loading
# ---------------------------------------------------------------------------

"""
    load_grids_with_regrid(restart_file, target_grid_file; boundaries)
        -> (grid2d, grid3d_ice, grid3d_rock)

Variant of `load_grids_from_restart` for the regridding path: build
the target horizontal grid from `target_grid_file` (typically a
`<grid_name>_REGIONS.nc` or any NetCDF with `xc` / `yc` axes), and
keep the vertical axes from `restart_file`. The returned `grid2d`
has the target horizontal extent; the 3D grids share that extent
plus the restart's vertical discretisation.

This is the kw-driven entry-point for in-line restart regridding —
`YelmoModel(restart_file, time; target_grid_file=...)` calls this
when a target grid file is supplied.
"""
function load_grids_with_regrid(restart_file::AbstractString,
                                target_grid_file::AbstractString;
                                boundaries = :bounded)
    # Horizontal coords from target.
    ds_tgt = NCDataset(target_grid_file)
    xc = Vector{Float64}(ds_tgt["xc"][:])
    yc = Vector{Float64}(ds_tgt["yc"][:])
    x_units = lowercase(strip(get(ds_tgt["xc"].attrib, "units", "")))
    y_units = lowercase(strip(get(ds_tgt["yc"].attrib, "units", "")))
    close(ds_tgt)
    if x_units == "km"
        xc .*= 1000.0
    end
    if y_units == "km"
        yc .*= 1000.0
    end

    Nx = length(xc); Ny = length(yc)
    dx = xc[2] - xc[1]; dy = yc[2] - yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)
    Tx, Ty = resolve_boundaries(boundaries)

    grid2d = RectilinearGrid(size = (Nx, Ny),
                             x = xlims, y = ylims,
                             topology = (Tx, Ty, Flat))

    # Vertical from restart.
    ds_src = NCDataset(restart_file)
    grid3d_ice  = _build_3d_grid(ds_src, "zeta",      "zeta_ac",
                                 Nx, Ny, xlims, ylims, Tx, Ty; path_b = true)
    grid3d_rock = _build_3d_grid(ds_src, "zeta_rock", "zeta_rock_ac",
                                 Nx, Ny, xlims, ylims, Tx, Ty; path_b = false)
    close(ds_src)
    return grid2d, grid3d_ice, grid3d_rock
end

# Read the `grid_name` global attribute from a NetCDF, returning
# `nothing` if not present.
function _read_grid_name_attr(filename::AbstractString)
    NCDataset(filename) do ds
        return haskey(ds.attrib, "grid_name") ?
               String(ds.attrib["grid_name"]) : nothing
    end
end

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

    # Path B vertical convention applies to the ICE grid only (commit
    # 1 of the vertical-split refactor). Bedrock continues using the
    # legacy convention until the bedrock thrm refactor (commit 5).
    grid3d_ice  = _build_3d_grid(ds, "zeta",      "zeta_ac",      Nx, Ny, xlims, ylims, Tx, Ty; path_b=true)
    grid3d_rock = _build_3d_grid(ds, "zeta_rock", "zeta_rock_ac", Nx, Ny, xlims, ylims, Tx, Ty; path_b=false)

    close(ds)
    return grid2d, grid3d_ice, grid3d_rock
end

# Build a 3D RectilinearGrid for one of the vertical axes in `ds`.
#
# Two conventions are supported via `path_b`:
#
#   - Legacy (`path_b=false`): file `zeta` is treated as the Yelmo
#     `Center` axis (length Nz) and `zeta_ac` as the `Face` axis
#     (length Nz+1). Yelmo's grid Nz equals the file's `zeta` length.
#     This is the historical Yelmo.jl behaviour and remains in force
#     for the bedrock grid until the bedrock refactor.
#
#   - Path B (`path_b=true`): file `zeta` is interpreted as
#     `[0; centers...; 1]` (length Nz_file) — i.e. the basal and
#     surface boundary endpoints are stored alongside the interior
#     centers. Yelmo's interior grid takes file `zeta[2:end-1]` as the
#     `Center` axis (length Nz = Nz_file - 2). Yelmo's `Face` axis is
#     constructed as midpoints between consecutive interior centers,
#     with endpoints clamped to {0, 1}.
#
# Under Path B the file's `zeta_ac` is *not* authoritative for the
# Yelmo grid; it is only consulted at I/O time for the boundary
# slices via the boundary-fields registry (commit 2).
function _build_3d_grid(ds, center_var, face_var, Nx, Ny, xlims, ylims,
                        Tx::DataType, Ty::DataType; path_b::Bool=false)
    if path_b
        haskey(ds, center_var) || return nothing
        z_center_file = Vector{Float64}(ds[center_var][:])
        z_face = _path_b_faces_from_centers(z_center_file)
    elseif haskey(ds, face_var)
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

# Path B Face construction: file `zeta` is `[0; interior_centers...; 1]`.
# Yelmo Nz = length(zeta) - 2, with `zeta_aa = zeta[2:end-1]`. The
# returned `Face` axis (length Nz + 1) takes `Face[1] = 0`,
# `Face[end] = 1`, and interior `Face[k] = (zeta_aa[k-1] + zeta_aa[k])/2`.
#
# Note (over-determined system): there is no Face configuration that
# simultaneously hits {0, 1} at the endpoints AND has every
# Oceananigans-derived Center (which is the midpoint of consecutive
# Faces) equal the file's interior `zeta` value. This construction
# preserves the endpoints exactly; Oceananigans-derived Centers will
# differ from the file's interior `zeta` by up to a half-cell at the
# basal and surface cells. Yelmo solvers that need exact center
# positions should consult a separately-stored `zeta_aa` (introduced
# with the thrm scratch struct in commit 5), not `znodes(grid, Center())`.
function _path_b_faces_from_centers(zeta_file::AbstractVector)
    N = length(zeta_file)
    N >= 3 || error("_path_b_faces_from_centers: need length ≥ 3, got $N. " *
                    "File `zeta` must include both the z=0 and z=1 endpoints.")
    Nz = N - 2
    zf = Vector{Float64}(undef, Nz + 1)
    zf[1]   = 0.0
    zf[end] = 1.0
    @inbounds for k in 2:Nz
        zf[k] = 0.5 * (zeta_file[k] + zeta_file[k+1])
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
        # CG workspace for `method = :energy_quadratic` (Hessian-of-energy
        # assembly produces a symmetric positive-definite matrix). Allocated
        # eagerly alongside the BiCGStab workspace; both share the same
        # `N_rows`-shaped buffers and remain alive for the lifetime of the
        # model. Only one is actually used per Picard iter, dispatched on
        # `resolve_linear_method(ssa_solver)` in `_solve_ssa_linear!`.
        ssa_cg_workspace           = CgWorkspace(N_rows, N_rows, Vector{Float64}),
        ssa_amg_cache              = Ref{Any}(nothing),
        # Reusable diagonal-inverse buffer for the `:jacobi`
        # preconditioner — `_build_ssa_precond` refreshes
        # `d_inv = 1.0 ./ diag(A)` into this buffer in-place each Picard
        # iteration (was a fresh `Vector{Float64}` allocation per iter
        # before).
        ssa_jacobi_d_inv           = Vector{Float64}(undef, N_rows),
        # CSC cache for the SSA stiffness matrix. The COO triplets
        # written by `_assemble_ssa_matrix!` change values across
        # Picard iterations but the *structure* (set of unique
        # (row, col) pairs) is invariant within one `dyn_step!` call.
        # On the first Picard iter we build the CSC via `sparse(...)`
        # and cache the COO→CSC permutation `ssa_coo_to_csc`; on
        # subsequent iters we just refresh the cached CSC's `nzval`
        # via the permutation, avoiding the colptr/rowval rebuild
        # and most allocations. See
        # `_build_or_refresh_ssa_csc!` in `src/dyn/velocity_ssa.jl`.
        ssa_csc                    = Ref{Any}(nothing),
        ssa_coo_to_csc             = Vector{Int}(undef, N_nz_max),
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
    # `timestep_log` slot follows the same lazy pattern; populated only
    # when `y.p.yelmo.log_timestep = true` (src/timestep_log.jl).
    pc_scratch = (pc_scratch   = Ref{Any}(nothing),
                  timestep_log = Ref{Any}(nothing),)

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

    # thrm scratch (Path B commit 5a, vert-split refactor). `zeta_aa`
    # and `zeta_ac` are concrete `Vector{Float64}` snapshots of the
    # ice grid's Center / Face axes, materialised once at YelmoModel
    # construction so per-`therm_step!` calls no longer
    # `collect(znodes(...))`. Roughly 340 KB / step of allocator
    # pressure removed (each 1D vector is small but the Field-aware
    # solver helpers gate on a concrete `Vector{Float64}` type, so
    # the calls happened multiple times per step).
    thrm_scratch = (
        zeta_aa = collect(Float64, znodes(gt, Center())),
        zeta_ac = collect(Float64, znodes(gt, Face())),
    )
    thrm = merge(thrm, (scratch = thrm_scratch,))

    # Replace H_ice with a CenterField that carries Dirichlet H_ice = 0
    # boundary conditions on the domain edge. The upwind advection
    # operator reads these via Oceananigans' standard halo machinery
    # without per-cell branching in the kernel.
    haskey(tpo, :H_ice) && (tpo = merge(tpo, (H_ice = dirichlet_2d_field(g, 0.0),)))

    # Advection scratch (src/topo/advection.jl).
    # `Ref{Any}` is filled lazily on the first `advect_tracer!` call
    # with an `AdvectionCache(grid)`. The concrete type lives in the
    # later-included YelmoModelTopo module, so we mirror the
    # `pc_scratch` / `ssa_amg_cache` lazy pattern rather than
    # eagerly typing it here. The same cache serves both schemes:
    # the implicit path uses the sparse operator + GMRES workspace,
    # the explicit path uses the preallocated `tend` buffer.
    tpo_scratch = (adv_cache = Ref{Any}(nothing),)
    tpo = merge(tpo, (scratch = tpo_scratch,))

    # `dta.rmse`: scalar comparison metrics filled by `data_compare!`
    # (src/data/YelmoModelData.jl). Initialised to NaN so absence of a
    # comparison call shows up cleanly downstream rather than reading as
    # zero. See `RMSEStats` above for field layout.
    dta = merge(dta, (rmse = RMSEStats(),))

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
                    strict::Bool = true,
                    target_grid_file::Union{Nothing,AbstractString} = nothing,
                    maps_dir::AbstractString = "maps",
                    regrid_method::AbstractString = "con")

    if p === nothing
        @warn "No parameters supplied to YelmoModel; constructing YelmoModelParameters(\"$(alias)\") with defaults."
        p = YelmoModelParameters(alias)
    end

    # Build grids: if a target_grid_file is provided, the model lives
    # on the target horizontal grid (vertical axis from restart) and
    # data is regridded on read via a SCRIP map.
    mps = nothing
    if target_grid_file !== nothing
        src_grid_name = _read_grid_name_attr(restart_file)
        dst_grid_name = _read_grid_name_attr(target_grid_file)
        src_grid_name === nothing && error(
            "YelmoModel(...; target_grid_file=...): restart file " *
            "$(restart_file) has no `grid_name` global attribute; cannot " *
            "construct SCRIP map filename.")
        dst_grid_name === nothing && error(
            "YelmoModel(...; target_grid_file=...): target grid file " *
            "$(target_grid_file) has no `grid_name` global attribute.")
        if src_grid_name == dst_grid_name
            @warn "target_grid_file grid_name matches restart grid_name (\"$(src_grid_name)\"); skipping regrid."
            g, gt, gr = load_grids_from_restart(restart_file; boundaries = boundaries)
        else
            g, gt, gr = load_grids_with_regrid(restart_file, target_grid_file;
                                               boundaries = boundaries)
            mps = map_scrip_load(src_grid_name, dst_grid_name, maps_dir;
                                 method = regrid_method)
        end
    else
        g, gt, gr = load_grids_from_restart(restart_file; boundaries = boundaries)
    end
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

    load_state!(y, restart_file; groups=groups, strict=strict, mps=mps)

    _infer_mask_ice!(y, restart_file; mps=mps)

    return y
end

# Fill `bnd.mask_ice` based on what is actually in the restart file.
#  - If the restart contains `mask_ice`, do nothing (load_state! has
#    already placed its values into `y.bnd.mask_ice`).
#  - Else if it contains `ice_allowed`, derive: allowed → DYNAMIC,
#    not-allowed → NONE. Read directly from the file so this works
#    even when `:bnd` was not in the loaded `groups`.
#  - Else leave the all-dynamic default established before load_state!.
function _infer_mask_ice!(y::YelmoModel, restart_file::AbstractString;
                          mps = nothing)
    NCDataset(restart_file) do ds
        haskey(ds, "mask_ice") && return  # already loaded
        haskey(ds, "ice_allowed") || return  # nothing to infer from

        ia = _read_nc_2d_mps(ds["ice_allowed"], mps, "ice_allowed")
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

Path B vertical convention (Path B-registered fields like `T_ice`,
`enth`, `T_pmp`, `T_prime`, `visc`, `ATT`, `enh`): the unified file
slab is read once per group and decomposed into the basal `_b` 2D
field (slice 1), the interior 3D field (slice 2:end-1), and the
surface `_s` 2D field (slice end). Non-registered fields are loaded
by direct name as before.
"""
function load_state!(y::YelmoModel, restart_file::AbstractString;
                     groups::NTuple{N,Symbol} where N = _ALL_MODEL_GROUPS,
                     strict::Bool = true,
                     mps = nothing)
    ds = NCDataset(restart_file)
    try
        for gname in groups
            group_nt = getfield(y, gname)
            metas    = getfield(y.v, gname)
            # Cache the unified slab read for each Path B-registered
            # interior name so we don't read the same NetCDF variable
            # three times (once each for _b, interior, _s).
            slab_cache = Dict{String, Array{Float64,3}}()
            for k in keys(metas)
                meta = metas[k]
                if is_path_b_registered(meta.name)
                    unified = path_b_unified_name(meta.name)
                    if !haskey(ds, unified)
                        strict && error(
                            "Path B-registered variable `$(meta.name)` (group " *
                            "`$(gname)`) requires unified slab `$(unified)` in restart " *
                            "file $(restart_file); not found. Pass `strict=false` to skip.")
                        continue
                    end
                    slab = get!(slab_cache, unified) do
                        Array{Float64,3}(_read_nc_3d_mps(ds[unified], mps, unified))
                    end
                    _apply_path_b_slice!(group_nt[k], slab, path_b_slice_kind(meta.name))
                else
                    name_str = String(meta.name)
                    if !haskey(ds, name_str)
                        strict && error(
                            "Variable `$(name_str)` (group `$(gname)`) not found in restart " *
                            "file $(restart_file). Pass `strict=false` to skip missing variables.")
                        continue
                    end
                    _load_into_field!(group_nt[k], ds[name_str];
                                      mps = mps, varname = name_str)
                end
            end
        end
    finally
        close(ds)
    end
    return y
end

# Apply a Path B slice (`:basal`, `:interior`, `:surface`) of a
# unified file slab into a Yelmo field. `slab` has shape
# `(Nx, Ny, Nz_file)` with `Nz_file = Nz_yelmo + 2`. Center fields
# absorb the slice directly; XFace / YFace fields (e.g. `ux`) write
# into `[2:end, ...]` slots and replicate the leading slot, matching
# the convention used by `_load_into_field!`.
function _apply_path_b_slice!(field, slab::AbstractArray{<:Real,3}, kind::Symbol)
    nz_file = size(slab, 3)
    nz_file >= 3 || error("_apply_path_b_slice!: slab z-dim $nz_file < 3 (need ≥ 3 to split _b/interior/_s).")
    if kind === :basal
        slice = view(slab, :, :, 1)
        _apply_path_b_2d_slice!(field, slice)
    elseif kind === :surface
        slice = view(slab, :, :, nz_file)
        _apply_path_b_2d_slice!(field, slice)
    elseif kind === :interior
        slice = view(slab, :, :, 2:nz_file - 1)
        _apply_path_b_3d_interior!(field, slice)
    else
        error("_apply_path_b_slice!: unknown kind $kind.")
    end
    return field
end

# 2D slice copy into a field. Mirrors `_load_into_field!` 2D handling
# for XFace / YFace / Center fields, plus a Yelmo-written-file shape
# branch: the writer emits XFace / YFace fields at the full
# `(Nx+1, Ny)` / `(Nx, Ny+1)` interior shape under Bounded topology,
# so the loader must accept both that and the Mirror `(Nx, Ny)`
# convention (the latter writes into `[2:end, …]` with a leading
# slot replicate).
function _apply_path_b_2d_slice!(field::Field{Face, Center, Center}, slice::AbstractArray{<:Real,2}) where {Face, Center}
    Tx, _Ty, _Tz = topology(field.grid)
    int = interior(field)
    Nx_int = size(int, 1)
    if Tx === Bounded
        if size(slice, 1) == Nx_int
            int[:, :, 1] .= slice
        elseif size(slice, 1) == Nx_int - 1
            int[2:end, :, 1] .= slice
            int[1,     :, 1] .= @view slice[1, :]
        else
            error("_apply_path_b_2d_slice!(XFaceField): slab x-dim $(size(slice, 1)) " *
                  "matches neither $Nx_int (Yelmo) nor $(Nx_int - 1) (Mirror).")
        end
    elseif Tx === Periodic
        int[:, :, 1] .= slice
    else
        error("_apply_path_b_2d_slice!(XFaceField): unsupported x-topology $Tx")
    end
    return field
end

function _apply_path_b_2d_slice!(field::Field{Center, Face, Center}, slice::AbstractArray{<:Real,2}) where {Face, Center}
    _Tx, Ty, _Tz = topology(field.grid)
    int = interior(field)
    Ny_int = size(int, 2)
    if Ty === Bounded
        if size(slice, 2) == Ny_int
            int[:, :, 1] .= slice
        elseif size(slice, 2) == Ny_int - 1
            int[:, 2:end, 1] .= slice
            int[:, 1,     1] .= @view slice[:, 1]
        else
            error("_apply_path_b_2d_slice!(YFaceField): slab y-dim $(size(slice, 2)) " *
                  "matches neither $Ny_int (Yelmo) nor $(Ny_int - 1) (Mirror).")
        end
    elseif Ty === Periodic
        int[:, :, 1] .= slice
    else
        error("_apply_path_b_2d_slice!(YFaceField): unsupported y-topology $Ty")
    end
    return field
end

function _apply_path_b_2d_slice!(field::Field{Center, Center, Center}, slice::AbstractArray{<:Real,2}) where {Center}
    int = interior(field)
    int[:, :, 1] .= slice
    return field
end

# 3D interior copy into a field. Z-dim of slice already trimmed to
# the interior. Same Yelmo / Mirror dual-shape handling as the 2D
# variant above.
function _apply_path_b_3d_interior!(field::Field{Face, Center, Center}, slice::AbstractArray{<:Real,3}) where {Face, Center}
    Tx, _Ty, _Tz = topology(field.grid)
    int = interior(field)
    Nx_int = size(int, 1)
    if Tx === Bounded
        if size(slice, 1) == Nx_int
            int[:, :, :] .= slice
        elseif size(slice, 1) == Nx_int - 1
            int[2:end, :, :] .= slice
            int[1,     :, :] .= @view slice[1, :, :]
        else
            error("_apply_path_b_3d_interior!(XFaceField): slab x-dim $(size(slice, 1)) " *
                  "matches neither $Nx_int (Yelmo) nor $(Nx_int - 1) (Mirror).")
        end
    elseif Tx === Periodic
        int[:, :, :] .= slice
    else
        error("_apply_path_b_3d_interior!(XFaceField): unsupported x-topology $Tx")
    end
    return field
end

function _apply_path_b_3d_interior!(field::Field{Center, Face, Center}, slice::AbstractArray{<:Real,3}) where {Face, Center}
    _Tx, Ty, _Tz = topology(field.grid)
    int = interior(field)
    Ny_int = size(int, 2)
    if Ty === Bounded
        if size(slice, 2) == Ny_int
            int[:, :, :] .= slice
        elseif size(slice, 2) == Ny_int - 1
            int[:, 2:end, :] .= slice
            int[:, 1,     :] .= @view slice[:, 1, :]
        else
            error("_apply_path_b_3d_interior!(YFaceField): slab y-dim $(size(slice, 2)) " *
                  "matches neither $Ny_int (Yelmo) nor $(Ny_int - 1) (Mirror).")
        end
    elseif Ty === Periodic
        int[:, :, :] .= slice
    else
        error("_apply_path_b_3d_interior!(YFaceField): unsupported y-topology $Ty")
    end
    return field
end

function _apply_path_b_3d_interior!(field::Field{Center, Center, Center}, slice::AbstractArray{<:Real,3}) where {Center}
    interior(field) .= slice
    return field
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

# Maybe-regrid wrappers. When `mps === nothing` they're identity.
# When an mps Dict is supplied (constructed by `map_scrip_load`),
# the raw NetCDF read is regridded onto the target grid via the
# vendored `map_scrip_field` (`src/utils/scrip_map.jl`). NaN cells in
# the regridded output (no source contribution) are replaced by
# `nan_replacement` (default 0.0) so prognostic fields stay numerically
# clean — boundary cells outside the source domain become zeros.
@inline function _read_nc_2d_mps(ncvar, mps, varname::AbstractString;
                                 nan_replacement::Float64 = 0.0)
    raw = _read_nc_2d(ncvar)
    mps === nothing && return raw
    src = Float64.(raw)
    _, dst = map_scrip_field(mps, varname, src)
    @inbounds for i in eachindex(dst)
        if isnan(dst[i])
            dst[i] = nan_replacement
        end
    end
    return dst
end

@inline function _read_nc_3d_mps(ncvar, mps, varname::AbstractString;
                                 nan_replacement::Float64 = 0.0)
    raw = _read_nc_3d(ncvar)
    mps === nothing && return raw
    src = Float64.(raw)
    Nx_dst = mps["dst_grid_dims"][1]
    Ny_dst = mps["dst_grid_dims"][2]
    Nz     = size(src, 3)
    dst = Array{Float64,3}(undef, Nx_dst, Ny_dst, Nz)
    for k in 1:Nz
        _, layer = map_scrip_field(mps, varname,
                                                    @view src[:, :, k])
        @inbounds for i in eachindex(layer)
            if isnan(layer[i])
                layer[i] = nan_replacement
            end
        end
        dst[:, :, k] .= layer
    end
    return dst
end

@inline _is_3d_field(int::AbstractArray) = ndims(int) == 3 && size(int, 3) > 1

# Transitional Path B vertical-slice helper.
#
# Under Path B (commit 1) the ice grid `Nz` drops to `Nz_file − 2`.
# Until the boundary-fields registry lands (commit 2), restart files
# whose 3D ice slabs still carry the basal (z=0) and surface (z=1)
# endpoint slices need to have those endpoints stripped on load so
# the interior of the field can absorb the centred values. Boundary
# `_b` / `_s` 2D fields remain at their constructed default until the
# registry properly populates them.
#
# The helper is a no-op on the legacy-convention rock grid (whose
# field `Nz` equals the file slab `Nz_file`).
@inline function _path_b_interior_slice_3d(data::AbstractArray{T,3}, field_nz::Integer) where {T}
    nz_file = size(data, 3)
    if nz_file == field_nz
        return data
    elseif nz_file == field_nz + 2
        return view(data, :, :, 2:nz_file - 1)
    else
        error("_path_b_interior_slice_3d: file z-dim $nz_file does not match " *
              "field z-dim $field_nz (legacy) or $field_nz + 2 (Path B with " *
              "boundary endpoints). Likely a grid / restart-file mismatch.")
    end
end

function _load_into_field!(field::Field{Face, Center, Center}, ncvar;
                           mps = nothing, varname::AbstractString = "") where {Face, Center}
    # XFaceField loader. Interior shape depends on x-axis topology:
    #   - Bounded-x:  `(Nx+1, Ny[, Nz])` — Ny cells × (Nx+1) face slots.
    #     File slab x-dim may be either:
    #       * `Nx` (Mirror / Fortran cell-centred convention): write
    #         data into `[2:end, …]` and replicate the leading slot.
    #       * `Nx+1` (Yelmo writer's full-shape convention): write
    #         data directly with no replicate.
    #   - Periodic-x: `(Nx, Ny[, Nz])` — incoming data shape matches
    #     interior shape directly; write in place, no replicate slot.
    #
    # The `Tx` dispatch matches the pattern used by `_ip1_modular` and
    # related helpers in `src/dyn/topology_helpers.jl`.
    Tx, _Ty, _Tz = topology(field.grid)
    int = interior(field)
    Nx_int = size(int, 1)
    if _is_3d_field(int)
        data_full = _read_nc_3d_mps(ncvar, mps, varname)
        data = _path_b_interior_slice_3d(data_full, size(int, 3))
        if Tx === Bounded
            if size(data, 1) == Nx_int
                int[:, :, :] .= data
            elseif size(data, 1) == Nx_int - 1
                int[2:end, :, :] .= data
                int[1, :, :]     .= @view data[1, :, :]
            else
                error("_load_into_field!(XFaceField, 3D): file x-dim $(size(data, 1)) " *
                      "matches neither $Nx_int (Yelmo) nor $(Nx_int - 1) (Mirror).")
            end
        elseif Tx === Periodic
            int[:, :, :] .= data
        else
            error("_load_into_field!(XFaceField, 3D): unsupported x-topology $Tx")
        end
    else
        data = _read_nc_2d_mps(ncvar, mps, varname)
        if Tx === Bounded
            if size(data, 1) == Nx_int
                int[:, :, 1] .= data
            elseif size(data, 1) == Nx_int - 1
                int[2:end, :, 1] .= data
                int[1, :, 1]     .= @view data[1, :]
            else
                error("_load_into_field!(XFaceField, 2D): file x-dim $(size(data, 1)) " *
                      "matches neither $Nx_int (Yelmo) nor $(Nx_int - 1) (Mirror).")
            end
        elseif Tx === Periodic
            int[:, :, 1] .= data
        else
            error("_load_into_field!(XFaceField, 2D): unsupported x-topology $Tx")
        end
    end
    return field
end

function _load_into_field!(field::Field{Center, Face, Center}, ncvar;
                           mps = nothing, varname::AbstractString = "") where {Face, Center}
    # YFaceField loader. Symmetric to the XFaceField method above —
    # interior shape depends on y-axis topology:
    #   - Bounded-y:  `(Nx, Ny+1[, Nz])`. File slab y-dim may be
    #     either `Ny` (Mirror) or `Ny+1` (Yelmo writer); the loader
    #     handles both cases.
    #   - Periodic-y: `(Nx, Ny[, Nz])`. Direct copy — no replicate slot,
    #     no extra face row (the `Ny+1`-th face is the `1`-st by wrap
    #     and is not stored).
    _Tx, Ty, _Tz = topology(field.grid)
    int = interior(field)
    Ny_int = size(int, 2)
    if _is_3d_field(int)
        data_full = _read_nc_3d_mps(ncvar, mps, varname)
        data = _path_b_interior_slice_3d(data_full, size(int, 3))
        if Ty === Bounded
            if size(data, 2) == Ny_int
                int[:, :, :] .= data
            elseif size(data, 2) == Ny_int - 1
                int[:, 2:end, :] .= data
                int[:, 1, :]     .= @view data[:, 1, :]
            else
                error("_load_into_field!(YFaceField, 3D): file y-dim $(size(data, 2)) " *
                      "matches neither $Ny_int (Yelmo) nor $(Ny_int - 1) (Mirror).")
            end
        elseif Ty === Periodic
            int[:, :, :] .= data
        else
            error("_load_into_field!(YFaceField, 3D): unsupported y-topology $Ty")
        end
    else
        data = _read_nc_2d_mps(ncvar, mps, varname)
        if Ty === Bounded
            if size(data, 2) == Ny_int
                int[:, :, 1] .= data
            elseif size(data, 2) == Ny_int - 1
                int[:, 2:end, 1] .= data
                int[:, 1, 1]     .= @view data[:, 1]
            else
                error("_load_into_field!(YFaceField, 2D): file y-dim $(size(data, 2)) " *
                      "matches neither $Ny_int (Yelmo) nor $(Ny_int - 1) (Mirror).")
            end
        elseif Ty === Periodic
            int[:, :, 1] .= data
        else
            error("_load_into_field!(YFaceField, 2D): unsupported y-topology $Ty")
        end
    end
    return field
end

function _load_into_field!(field::Field{Center, Center, Face}, ncvar;
                           mps = nothing, varname::AbstractString = "") where {Face, Center}
    int = interior(field)
    if _is_3d_field(int)
        data = _read_nc_3d_mps(ncvar, mps, varname)
        nz_file  = size(data, 3)
        nz_field = size(int, 3)
        if nz_file == nz_field
            int[:, :, :] .= data
        elseif nz_file == nz_field + 2
            # Mirror-format file: Nz_file+1 = Nz+3 face levels. The 2 extra
            # slots are the extrapolated boundary faces written by
            # `_zface_extend` (commit 2.5). Strip them to recover the
            # Yelmo-internal Nz+1 = Nz_file-1 values.
            # Note: ZFace fields (uz, uz_star, ...) are recomputed by the
            # velocity solver at init_state! — this load only seeds the
            # initial guess, so the 1-face boundary replication used on write
            # is a safe approximation.
            int[:, :, :] .= view(data, :, :, 2:nz_file - 1)
        else
            error("_load_into_field!(ZFaceField, 3D): unexpected z-dim mismatch " *
                  "(file=$nz_file, field=$nz_field).")
        end
    else
        int[:, :, 1] .= _read_nc_2d_mps(ncvar, mps, varname)
    end
    return field
end

function _load_into_field!(field::Field{Center, Center, Center}, ncvar;
                           mps = nothing, varname::AbstractString = "") where {Center}
    int = interior(field)
    if _is_3d_field(int)
        data_full = _read_nc_3d_mps(ncvar, mps, varname)
        data = _path_b_interior_slice_3d(data_full, size(int, 3))
        int[:, :, :] .= data
    else
        int[:, :, 1] .= _read_nc_2d_mps(ncvar, mps, varname)
    end
    return field
end

# ---------------------------------------------------------------------------
# init_state! and step! — abstract-interface methods for YelmoModel
# ---------------------------------------------------------------------------

# Forward declarations for `update_diagnostics!` (defined in
# YelmoModelTopo) and `init_thrm!` (defined in YelmoModelThrm). YelmoCore
# can't depend on either phase module, so we declare the generics here
# and the phase modules `import ..YelmoCore` to add their methods.
function update_diagnostics! end
function init_thrm! end

"""
    init_state!(y::YelmoModel, time::Float64;
                 thrm_method::AbstractString = "robin") -> y

Initialise a `YelmoModel`'s state — the Fortran-faithful port of
`yelmo_init_state` (yelmo_ice.f90:1262-1362), called after topography
and boundary fields are externally prescribed and before the time
loop. Drives a deterministic equilibration cycle:

  1. `update_diagnostics!`           — sync masks / `f_ice` / `f_grnd` /
                                        `z_srf` / etc. from the loaded
                                        H_ice + z_bed + z_sl.
  2. `init_thrm!`                    — analytic temperature
                                        initialisation via `thrm_method`
                                        (writes `T_ice`, `T_ice_b`,
                                        `T_ice_s`, derived diagnostics).
  3. `mat_step!(y, 0.0)`             — material properties (stress
                                        tensor, enhancement, ATT,
                                        viscosity) from the new T_ice.
  4. β safety net                    — if both `dyn.beta` and
                                        `dyn.cb_ref` are zero, seed
                                        `cb_ref = 1`, `c_bed = 1e5`,
                                        `beta = 1e5` (mirrors Fortran
                                        yelmo_ice.f90:1332-1338).
  5. `dyn_step!(y, 0.0)`             — initial velocity solve.
  6. `mat_step!(y, 0.0)`             — material refresh with new
                                        velocities (strain rates feed
                                        the enh / visc kernels).
  7. `update_diagnostics!`           — final sync.

Sets `y.time = time`. Mirrors Fortran's `thrm_method ∈ {"linear",
"robin", "robin-cold"}` validation; the analytic-only constraint
ensures the initial temperature field is set explicitly rather than
left at the Field-allocation default. Default `"robin"` matches the
Fortran convention used by every Yelmo benchmark program.

Restart-file callers should typically skip this — `T_ice`, velocity,
and material state are already prescribed by the loaded snapshot.
"""
function init_state!(y::YelmoModel, time::Float64;
                       thrm_method::AbstractString = "robin")
    y.time = time

    # 1. Topography sync (Fortran calc_ytopo_pc(...,topo_fixed=.TRUE.,
    #    pc_step="none")). Refreshes `f_ice`, `f_grnd`, `z_srf`, masks,
    #    etc. from the externally loaded `H_ice` / `z_bed` / `z_sl`
    #    without advancing time.
    update_diagnostics!(y)

    # 2. Thermal initialisation via the chosen analytic solver.
    init_thrm!(y; thrm_method = thrm_method)

    # 3. First material pass (no dynamics yet) — populates the
    #    deviatoric stress tensor and rate factor / viscosity from the
    #    freshly initialised T_ice. Mirrors Fortran calc_ymat call
    #    at yelmo_ice.f90:1324.
    mat_step!(y, 0.0)

    # 4. β safety net (Fortran yelmo_ice.f90:1332-1338). Only kicks in
    #    if dyn.beta and dyn.cb_ref are both still zero — i.e. nothing
    #    upstream supplied a friction coefficient. Seeds a high-β state
    #    so the SSA solver doesn't see a fully-frictionless system on
    #    the first call. Will be overwritten by `dyn_step!`'s normal
    #    cb_ref / beta calculation when those are method-driven.
    _init_state_beta_safety_net!(y)

    # 5. Initial dynamic state. Mirrors Fortran calc_ydyn call at
    #    yelmo_ice.f90:1340. With `dt = 0.0` the SSA Picard loop
    #    just produces a steady-state velocity for the current
    #    geometry; the `duxydt` post-pass is guarded against `dt = 0`.
    dyn_step!(y, 0.0)

    # 6. Material refresh with the newly-solved velocities — strain
    #    rates and stress tensor are now populated, so enhancement
    #    and viscosity reflect the actual flow regime. Mirrors
    #    Fortran calc_ymat call at yelmo_ice.f90:1344.
    mat_step!(y, 0.0)

    # 7. Final topography sync. Mirrors Fortran calc_ytopo_pc call at
    #    yelmo_ice.f90:1355.
    update_diagnostics!(y)

    return y
end

# β safety net helper. Inlined into init_state! call site; kept
# separate so `init_state!` reads as a clear sequence of phase calls.
@inline function _init_state_beta_safety_net!(y::YelmoModel)
    beta_int   = interior(y.dyn.beta)
    cb_ref_int = interior(y.dyn.cb_ref)
    c_bed_int  = interior(y.dyn.c_bed)
    if maximum(beta_int) == 0.0
        if maximum(cb_ref_int) == 0.0
            fill!(cb_ref_int, 1.0)
        end
        @. c_bed_int = cb_ref_int * 1e5
        @. beta_int  = c_bed_int
    end
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
    # Backend dispatch:
    #
    #   - `y.p === nothing` (parameter-less benchmark constructions):
    #     fall through to a plain forward-Euler chain via `_step_fe!`,
    #     so simple in-memory test setups work without parameters.
    #   - Otherwise route through `_select_step!` (src/timestepping.jl)
    #     which dispatches on `y.p.yelmo.dt_method`. Both `dt_method=0`
    #     (fixed-dt Heun, no controller) and `dt_method=2` (adaptive
    #     Heun + PI42) run the PC machinery so `eta` is always
    #     available as a diagnostic.
    if y.p === nothing
        return _step_fe!(y, dt)
    else
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
