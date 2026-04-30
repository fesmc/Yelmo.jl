module YelmoMirrorCore

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields

using ..YelmoMeta: VariableMeta, parse_variable_table
using ..YelmoPar: YelmoParameters, write_nml
using ..YelmoCore: AbstractYelmoModel, _alloc_field, yelmo_define_grids,
                   XFACE_VARIABLES, YFACE_VARIABLES, ZFACE_VARIABLES, VERTICAL_DIMS
import ..YelmoCore: init_state!, step!

export YelmoMirror, init_state!, step!, yelmo_sync!, yelmo_write_restart!
export yelmo_get_var2D, yelmo_get_var2D!
export yelmo_get_var3D, yelmo_get_var3D!
export yelmo_set_var2D!, yelmo_set_var3D!

# ---------------------------------------------------------------------------
const yelmopath = joinpath(@__DIR__, "..", "yelmo")
const yelmolib = joinpath(@__DIR__, "..", "yelmo", "libyelmo", "include", "libyelmo_c_api.so")

# ---------------------------------------------------------------------------
mutable struct YelmoMirror{B, DT, DY, M, TH, TP} <: AbstractYelmoModel
    alias::String
    calias::Vector{UInt8}
    rundir::String
    time::Float64
    p::YelmoParameters
    g::RectilinearGrid   # 2D Grid
    gt::RectilinearGrid  # 3D Ice Grid
    gr::RectilinearGrid  # 3D Rock Grid
    v::NamedTuple        # Metadata
    bnd::B               # NamedTuple of Fields
    dta::DT
    dyn::DY
    mat::M
    thrm::TH
    tpo::TP
    buffers::NamedTuple # Stores (v2D=..., v3D=..., v3Dr=...)
end

# --- Constructors ---

function YelmoMirror(filename::String, time::Float64; 
    alias::String="ylmo1", rundir::String="./", overwrite::Bool=false)
    p = YelmoParameters(filename)
    return YelmoMirror(p, time; alias, rundir, overwrite)
end

"""
    YelmoMirror(p, time; grid=nothing, alias, rundir, overwrite)

Construct a `YelmoMirror` from a `YelmoParameters` set. The optional
`grid` keyword switches grid-construction strategy:

  - `grid === nothing` (default): the underlying Fortran `yelmo_init`
    runs with `grid_def="file"` and reads grid + topography from a
    NetCDF file identified by the namelist's `&yelmo_masks` /
    `&yelmo_init_topo` blocks. Required for real-domain runs (e.g.
    Greenland-16km) where the grid lives on disk.

  - `grid` is a NamedTuple with at least `xc::Vector` and `yc::Vector`
    (cell centres in metres) plus optional `grid_name::String` (default
    `"synthetic"`), `lon`, `lat`, `area`: the grid is constructed
    synthetically via `yelmo_init_grid_fromaxes` (C API
    `:yelmo_init_grid`) and `yelmo_init` then runs with
    `grid_def="none"` to skip its own grid setup. Used for
    Fortran-internally-generated benchmarks (BUELER, EISMINT,
    ISMIP-HOM, slab, trough, MISMIP+, CalvingMIP) where no grid file
    exists. The namelist must set `init_topo_load = .false.` so
    `yelmo_init` doesn't try to load topography from a missing file.

For lon/lat/area defaults: lon and lat default to zero arrays (a
flat Cartesian grid; lon/lat aren't physically meaningful for
benchmarks), area defaults to `dx · dy` (uniform spacing inferred
from xc/yc).
"""
function YelmoMirror(p::YelmoParameters, time::Float64;
    grid::Union{Nothing,NamedTuple}=nothing,
    alias::String="ylmo1", rundir::String="./", overwrite::Bool=false)

    calias = Vector{UInt8}("$(alias)\0")

    # 1. Fortran Init. Two paths:
    #    - default (grid=nothing): yelmo_init(grid_def="file") reads
    #      everything from disk.
    #    - synthetic (grid=NamedTuple): construct the grid via
    #      yelmo_init_grid_fromaxes first, then yelmo_init(grid_def="none").
    filename = joinpath(rundir, p.name * ".nml")
    write_nml(filename, p; overwrite)
    if grid === nothing
        _init_yelmomirror(filename, time, calias)
    else
        _yelmo_init_grid_fromaxes(grid, calias)
        _init_yelmomirror(filename, time, calias; grid_def="none")
    end

    # 2. Grid Setup
    ginfo = yelmo_get_grid_info(calias)
    g, gt, gr = yelmo_define_grids(ginfo)

    # Define buffer fields for loading and setting
    v2D = Array{Float64}(undef, g.Nx, g.Ny)
    v3D = Array{Float64}(undef, gt.Nx, gt.Ny, gt.Nz+1)
    v3Dr = Array{Float64}(undef, gr.Nx, gr.Ny, gr.Nz+1)
    buffers = (v2D=v2D, v3D=v3D, v3Dr=v3Dr)

    # 3. Metadata
    vdir = joinpath(@__DIR__, "variables", "mirror")
    v_meta = (
        bnd  = parse_variable_table(joinpath(vdir, "yelmo-variables-ybound.md"), "bnd"),
        dta  = parse_variable_table(joinpath(vdir, "yelmo-variables-ydata.md"),  "dta"),
        dyn  = parse_variable_table(joinpath(vdir, "yelmo-variables-ydyn.md"),   "dyn"),
        mat  = parse_variable_table(joinpath(vdir, "yelmo-variables-ymat.md"),   "mat"),
        thrm = parse_variable_table(joinpath(vdir, "yelmo-variables-ytherm.md"), "thrm"),
        tpo  = parse_variable_table(joinpath(vdir, "yelmo-variables-ytopo.md"),  "tpo"),
    )

    # 4. Allocate and Fill Fields
    bnd  = yelmo_get_variable_group(v_meta.bnd,  g, gt, gr, buffers, calias)
    dta  = yelmo_get_variable_group(v_meta.dta,  g, gt, gr, buffers, calias)
    dyn  = yelmo_get_variable_group(v_meta.dyn,  g, gt, gr, buffers, calias)
    mat  = yelmo_get_variable_group(v_meta.mat,  g, gt, gr, buffers, calias)
    thrm = yelmo_get_variable_group(v_meta.thrm, g, gt, gr, buffers, calias)
    tpo  = yelmo_get_variable_group(v_meta.tpo,  g, gt, gr, buffers, calias)
    
    return YelmoMirror(alias, calias, rundir, time, p, g, gt, gr, v_meta, bnd, dta, dyn, mat, thrm, tpo, buffers)
end

# --- Grid Logic ---

function yelmo_get_grid_info(calias::Vector{UInt8})

    # Step 1: get sizes
    nx    = Ref{Cint}(0)
    ny    = Ref{Cint}(0)
    nz_aa = Ref{Cint}(0)
    nz_ac = Ref{Cint}(0)
    nzr_aa = Ref{Cint}(0)
    nzr_ac = Ref{Cint}(0)

    ccall((:yelmo_get_grid_sizes, yelmolib), Cvoid,
          (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint}, Ptr{UInt8}),
          nx, ny, nz_aa, nz_ac, nzr_aa, nzr_ac, calias)

    # Step 2: allocate buffers
    xc      = Vector{Cdouble}(undef, nx[])
    yc      = Vector{Cdouble}(undef, ny[])
    zeta_aa = Vector{Cdouble}(undef, nz_aa[])
    zeta_ac = Vector{Cdouble}(undef, nz_ac[])
    zeta_r_aa = Vector{Cdouble}(undef, nzr_aa[])
    zeta_r_ac = Vector{Cdouble}(undef, nzr_ac[])

    # Step 3: fill buffers
    ccall((:yelmo_get_grid_info, yelmolib), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{UInt8}),
          xc, yc, zeta_aa, zeta_ac, zeta_r_aa, zeta_r_ac, calias)

    return (
        nx    = Int(nx[]),
        ny    = Int(ny[]),
        nz_aa = Int(nz_aa[]),
        nz_ac = Int(nz_ac[]),
        nzr_aa = Int(nzr_aa[]),
        nzr_ac = Int(nzr_ac[]),
        xc      = xc,
        yc      = yc,
        zeta_aa = zeta_aa,
        zeta_ac = zeta_ac,
        zeta_r_aa = zeta_r_aa,
        zeta_r_ac = zeta_r_ac,
    )

end

# --- Simulations ---

function init_state!(ylmo::YelmoMirror, time::Float64; thrm_method::String="robin-cold")

    # Sync yelmo to fortran
    yelmo_sync!(ylmo)

    # call yelmo_init_state in fortran
    ccall((:yelmo_init_state, yelmolib), Cvoid,
        (Float64, Ptr{UInt8}, Ptr{UInt8}),
        time, thrm_method * "\0", ylmo.calias)

    # Update yelmo in julia
    yelmo_get_variables!(ylmo)

    # Update time
    ylmo.time = time

    return ylmo
end

function step!(ylmo::YelmoMirror, dt::Float64)

    # Sync yelmo to fortran
    yelmo_sync!(ylmo)

    # Update time
    ylmo.time += dt
    
    # Call yelmo_step in fortran
    ccall((:yelmo_step, yelmolib), Cvoid, (Float64, Ptr{UInt8}), ylmo.time, ylmo.calias)

    # Update yelmo in julia
    yelmo_get_variables!(ylmo)

    return ylmo
end

# --- Variable Management ---

function yelmo_get_variables!(ylmo)

    yelmo_get_variable_group!(ylmo.bnd, ylmo.v.bnd,   ylmo.buffers, ylmo.calias)
    yelmo_get_variable_group!(ylmo.dta, ylmo.v.dta,   ylmo.buffers, ylmo.calias)
    yelmo_get_variable_group!(ylmo.dyn, ylmo.v.dyn,   ylmo.buffers, ylmo.calias)
    yelmo_get_variable_group!(ylmo.mat, ylmo.v.mat,   ylmo.buffers, ylmo.calias)
    yelmo_get_variable_group!(ylmo.thrm, ylmo.v.thrm, ylmo.buffers, ylmo.calias)
    yelmo_get_variable_group!(ylmo.tpo, ylmo.v.tpo,   ylmo.buffers, ylmo.calias)
    
    return ylmo
end

function yelmo_get_variable_group!(fields, vlist, buffers, calias::Vector{UInt8})
    for k in keys(vlist)
        _get_var!(fields[k], vlist[k], buffers, calias)
    end
    return fields
end

function yelmo_get_variable_group(vlist, g2d, g3d, g3r, buffers, calias::Vector{UInt8})
    fields = NamedTuple{keys(vlist)}(
        _alloc_field(vlist[k], g2d, g3d, g3r) for k in keys(vlist)
    )
    yelmo_get_variable_group!(fields, vlist, buffers, calias)
    return fields
end

function _get_var!(buffer::Union{Array{Float64},SubArray{Float64}}, meta, calias::Vector{UInt8})
    if ndims(buffer) == 3
        yelmo_get_var3D!(buffer, meta.cname, calias)
    else
        yelmo_get_var2D!(buffer, meta.cname, calias)
    end
    return buffer
end

function yelmo_sync!(ylmo::YelmoMirror)

    # --- 1. Push Boundary fields (sync the whole set of variables) ---
    sync_group!(ylmo, :bnd)

    # --- 2. Push Specific variables (Manually specified for clarity/control) ---
    # Topo
    _set_var!(ylmo.tpo.H_ice, ylmo.v.tpo.H_ice, ylmo.buffers, ylmo.calias)

    # Dyn
    _set_var!(ylmo.dyn.cb_ref, ylmo.v.dyn.cb_ref, ylmo.buffers, ylmo.calias)
    _set_var!(ylmo.dyn.N_eff,  ylmo.v.dyn.N_eff,  ylmo.buffers, ylmo.calias)
    _set_var!(ylmo.dyn.ux,     ylmo.v.dyn.ux,     ylmo.buffers, ylmo.calias)
    _set_var!(ylmo.dyn.uy,     ylmo.v.dyn.uy,     ylmo.buffers, ylmo.calias)
    _set_var!(ylmo.dyn.uz,     ylmo.v.dyn.uz,     ylmo.buffers, ylmo.calias)

    # Thrm
    _set_var!(ylmo.thrm.T_ice, ylmo.v.thrm.T_ice, ylmo.buffers, ylmo.calias)
    _set_var!(ylmo.thrm.H_w,   ylmo.v.thrm.H_w,   ylmo.buffers, ylmo.calias)

    return nothing
end

# Useful to sync a whole set of variables to Yelmo-fortran. But, currently, it'S
# not possible to update every variable on the Fortran side, so use with caution.
function sync_group!(ylmo, set_name::Symbol)
    fields = getfield(ylmo, set_name)
    metas  = getfield(ylmo.v, set_name)
    for k in keys(metas)
        _set_var!(fields[k], metas[k], ylmo.buffers, ylmo.calias)
    end
end

function _set_var!(buffer::Union{Array{Float64},SubArray{Float64}}, meta, calias)
    #name = "$(meta.set)_$(meta.name)"
    if ndims(buffer) == 3
        yelmo_set_var3D!(buffer, meta.cname, calias)
    else
        yelmo_set_var2D!(buffer, meta.cname, calias)
    end
    return buffer
end

function _get_buffer(buffers::NamedTuple, meta)
    dims = meta.dimensions
    if _has_dim(dims, :zeta_rock)
        return view(buffers.v3Dr, :, :, 1:size(buffers.v3Dr, 3)-1)
    elseif _has_dim(dims, :zeta_rock_ac)
        return buffers.v3Dr
    elseif _has_dim(dims, :zeta)
        return view(buffers.v3D, :, :, 1:size(buffers.v3D, 3)-1)
    elseif _has_dim(dims, :zeta_ac)
        return buffers.v3D
    else
        return buffers.v2D
    end
end

@inline _has_dim(dims, d) = d in dims

# ── _set_var!: Field interior → C buffer (Julia ⇒ Fortran) ───────────────────

function _set_var!(field::Field{Face, Center, Center}, meta, buffers, calias) where {Face, Center}
    buffer = _get_buffer(buffers, meta)
    if ndims(buffer) == 3
        copyto!(buffer, interior(field)[2:end, :, :])
    else
        copyto!(buffer, interior(field)[2:end, :])
    end
    _set_var!(buffer, meta, calias)
    return field
end

function _set_var!(field::Field{Center, Face, Center}, meta, buffers, calias) where {Face, Center}
    buffer = _get_buffer(buffers, meta)
    if ndims(buffer) == 3
        copyto!(buffer, interior(field)[:, 2:end, :])
    else
        copyto!(buffer, interior(field)[:, 2:end])
    end
    _set_var!(buffer, meta, calias)
    return field
end

function _set_var!(field::Field{Center, Center, Face}, meta, buffers, calias) where {Face, Center}
    buffer = _get_buffer(buffers, meta)
    copyto!(buffer, interior(field))
    _set_var!(buffer, meta, calias)
    return field
end

function _set_var!(field::Field{Center, Center, Center}, meta, buffers, calias) where {Center}
    buffer = _get_buffer(buffers, meta)
    copyto!(buffer, interior(field))
    _set_var!(buffer, meta, calias)
    return field
end

# ── _get_var!: C buffer → Field interior (Fortran ⇒ Julia) ───────────────────

function _get_var!(field::Field{Face, Center, Center}, meta, buffers, calias::Vector{UInt8}) where {Face, Center}
    buffer = _get_buffer(buffers, meta)
    _get_var!(buffer, meta, calias)
    if ndims(buffer) == 3
        interior(field)[2:end, :, :] .= buffer              # fill non-halo indices
        interior(field)[1,     :, :] .= buffer[1, :, :]     # duplicate first slice
    else
        interior(field)[2:end, :] .= buffer          # fill non-halo indices
        interior(field)[1,     :] .= buffer[1, :]    # duplicate first slice
    end
    return field
end

function _get_var!(field::Field{Center, Face, Center}, meta, buffers, calias::Vector{UInt8}) where {Face, Center}
    buffer = _get_buffer(buffers, meta)
    _get_var!(buffer, meta, calias)
    if ndims(buffer) == 3
        interior(field)[:, 2:end, :] .= buffer
        interior(field)[:, 1,     :] .= buffer[:, 1, :]
    else
        interior(field)[:, 2:end] .= buffer
        interior(field)[:, 1,   ] .= buffer[:, 1]
    end
    return field
end

function _get_var!(field::Field{Center, Center, Face}, meta, buffers, calias::Vector{UInt8}) where {Face, Center}
    buffer = _get_buffer(buffers, meta)
    _get_var!(buffer, meta, calias)
    copyto!(interior(field), buffer)
    return field
end

function _get_var!(field::Field{Center, Center, Center}, meta, buffers, calias::Vector{UInt8}) where {Center}
    buffer = _get_buffer(buffers, meta)
    _get_var!(buffer, meta, calias)
    copyto!(interior(field), buffer)
    return field
end

# --- C Wrappers (Getters) ---

function yelmo_get_var2D!(buffer::Union{Array{Float64},SubArray{Float64}}, cname::Vector{UInt8}, calias::Vector{UInt8})
    nx, ny = size(buffer)
    ccall((:yelmo_get_var2D, yelmolib), Cvoid,
        (Ptr{Float64}, Cint, Cint, Ptr{UInt8}, Ptr{UInt8}),
        buffer, Cint(nx), Cint(ny), cname, calias)
    return buffer
end

function yelmo_get_var3D!(buffer::Union{Array{Float64},SubArray{Float64}}, cname::Vector{UInt8}, calias::Vector{UInt8})
    nx, ny, nz = size(buffer)
    ccall((:yelmo_get_var3D, yelmolib), Cvoid,
        (Ptr{Float64}, Cint, Cint, Cint, Ptr{UInt8}, Ptr{UInt8}),
        buffer, Cint(nx), Cint(ny), Cint(nz), cname, calias)
    return buffer
end

# --- C Wrappers (Setters) ---

function yelmo_set_var2D!(buffer::Union{Array{Float64},SubArray{Float64}}, cname::Vector{UInt8}, calias::Vector{UInt8})
    nx, ny = size(buffer)
    ccall((:yelmo_set_var2D, yelmolib), Cvoid,
        (Ptr{Float64}, Cint, Cint, Ptr{UInt8}, Ptr{UInt8}),
        buffer, Cint(nx), Cint(ny), cname, calias)
    return buffer
end

function yelmo_set_var3D!(buffer::Union{Array{Float64},SubArray{Float64}}, cname::Vector{UInt8}, calias::Vector{UInt8})
    nx, ny, nz = size(buffer)
    ccall((:yelmo_set_var3D, yelmolib), Cvoid,
        (Ptr{Float64}, Cint, Cint, Cint, Ptr{UInt8}, Ptr{UInt8}),
        buffer, Cint(nx), Cint(ny), Cint(nz), cname, calias)
    return buffer
end

# --- Internal Helper for Init ---

function _init_yelmomirror(filename::String, time::Float64, calias::Vector{UInt8};
                           grid_def::String="file")
    ccall((:yelmo_init, yelmolib), Cvoid,
        (Ptr{UInt8}, Ptr{UInt8}, Float64, Ptr{UInt8}),
        filename * "\0", grid_def * "\0", time, calias)
end

# Synthetic-grid path. Wraps the C API entry name "yelmo_init_grid"
# (declared in yelmo_c_api.f90 as `yelmo_init_grid_fromaxes_wrapper`),
# which sets the persistent `ylmo%grd` from explicit axes before
# `yelmo_init(grid_def="none")` runs. `grid` requires `xc` and `yc`
# (Vectors of cell-centre coordinates in metres); `grid_name`, `lon`,
# `lat`, and `area` are optional and have sensible synthetic-Cartesian
# defaults.
function _yelmo_init_grid_fromaxes(grid::NamedTuple, calias::Vector{UInt8})
    haskey(grid, :xc) && haskey(grid, :yc) ||
        error("YelmoMirror synthetic grid requires `grid.xc` and `grid.yc` (cell centres in metres).")

    xc = collect(Float64, grid.xc)
    yc = collect(Float64, grid.yc)
    nx, ny = length(xc), length(yc)

    grid_name = String(get(grid, :grid_name, "synthetic"))

    # lon/lat default to zero (no geographic meaning on a Cartesian
    # synthetic grid; consumers like the benchmark drivers don't read
    # them). area defaults to dx·dy from the xc/yc spacing.
    lon  = collect(Float64, get(grid, :lon,  zeros(nx, ny)))
    lat  = collect(Float64, get(grid, :lat,  zeros(nx, ny)))
    area = if haskey(grid, :area)
        collect(Float64, grid.area)
    else
        dx = nx >= 2 ? abs(xc[2] - xc[1]) : 1.0
        dy = ny >= 2 ? abs(yc[2] - yc[1]) : 1.0
        fill(dx * dy, nx, ny)
    end

    size(lon)  == (nx, ny) || error("grid.lon must have shape ($nx, $ny); got $(size(lon))")
    size(lat)  == (nx, ny) || error("grid.lat must have shape ($nx, $ny); got $(size(lat))")
    size(area) == (nx, ny) || error("grid.area must have shape ($nx, $ny); got $(size(area))")

    # The C-side signature is:
    #   yelmo_init_grid(char* grid_name, int nx, int ny,
    #                   double* xc, double* yc,
    #                   double* lon, double* lat, double* area,
    #                   char* alias)
    ccall((:yelmo_init_grid, yelmolib), Cvoid,
        (Ptr{UInt8}, Cint, Cint,
         Ptr{Cdouble}, Ptr{Cdouble},
         Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
         Ptr{UInt8}),
        grid_name * "\0", Cint(nx), Cint(ny),
        xc, yc, lon, lat, area,
        calias)
    return nothing
end

# Restart-write — wraps the C API entry name "yelmo_restart_write"
# (declared in yelmo_c_api.f90, paired with `yelmo_restart_write` in
# yelmo_io.f90). Writes the full Fortran state to a NetCDF that
# `YelmoModel(restart_file, time; ...)` can load directly.
function yelmo_write_restart!(ylmo::YelmoMirror, filename::String;
                              time::Union{Nothing,Float64}=nothing)
    t = time === nothing ? ylmo.time : time
    mkpath(dirname(filename))
    ccall((:yelmo_restart_write, yelmolib), Cvoid,
        (Ptr{UInt8}, Float64, Ptr{UInt8}),
        filename * "\0", t, ylmo.calias)
    return filename
end

end # module