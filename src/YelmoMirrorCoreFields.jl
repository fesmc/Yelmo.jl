module YelmoMirrorCore

using Oceananigans, Oceananigans.Grids, Oceananigans.Fields

using ..YelmoMeta: VariableMeta, parse_variable_table
using ..YelmoPar: YelmoParameters, write_nml

export YelmoMirror, init_state!, step!, yelmo_sync!
export yelmo_get_var2D, yelmo_get_var2D!
export yelmo_get_var3D, yelmo_get_var3D!
export yelmo_set_var2D!, yelmo_set_var3D!

# ---------------------------------------------------------------------------
const yelmopath = joinpath(@__DIR__, "..", "yelmo")
const yelmolib = joinpath(@__DIR__, "..", "yelmo", "libyelmo", "include", "libyelmo_c_api.so")

const VERTICAL_DIMS = (:zeta, :zeta_ac, :zeta_rock, :zeta_rock_ac)

const XFACE_VARIABLES = ["ux_s","ux_b", "ux", r".*_acx$"]
const YFACE_VARIABLES = ["uy_s","uy_b", "uy", r".*_acy$"]
const ZFACE_VARIABLES = ["uz","uz_star","jvel_dzx","jvel_dzy","jvel_dzz"]

# ---------------------------------------------------------------------------
mutable struct YelmoMirror{B, DT, DY, M, TH, TP}
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

function YelmoMirror(p::YelmoParameters, time::Float64; 
    alias::String="ylmo1", rundir::String="./", overwrite::Bool=false)

    calias = Vector{UInt8}("$(alias)\0")

    # 1. Fortran Init
    filename = joinpath(rundir, p.name * ".nml")
    write_nml(filename, p; overwrite)
    _init_yelmomirror(filename, time, calias)

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

function yelmo_define_grids(g::NamedTuple)
    return yelmo_define_grids(g.xc, g.yc, g.zeta_ac, g.zeta_r_ac)
end

function yelmo_define_grids(xc, yc, zeta_ac, zeta_r_ac)
    Nx, Ny = length(xc), length(yc)
    dx, dy = xc[2]-xc[1], yc[2]-yc[1]
    xlims = (xc[1] - dx/2, xc[end] + dx/2)
    ylims = (yc[1] - dy/2, yc[end] + dy/2)
    
    grid2d = RectilinearGrid(size=(Nx, Ny), x=xlims, y=ylims, topology=(Bounded, Bounded, Flat))
    
    grid3d_ice = RectilinearGrid(size=(Nx, Ny, length(zeta_ac)-1), 
                                 x=xlims, y=ylims, z=zeta_ac, topology=(Bounded, Bounded, Bounded))

    grid3d_rock = RectilinearGrid(size=(Nx, Ny, length(zeta_r_ac)-1), 
                                  x=xlims, y=ylims, z=zeta_r_ac, topology=(Bounded, Bounded, Bounded))
    
    return grid2d, grid3d_ice, grid3d_rock
end

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
    #name = "$(meta.set)_$(meta.name)"
    if ndims(buffer) == 3
        yelmo_get_var3D!(buffer, meta.cname, calias)
    else
        yelmo_get_var2D!(buffer, meta.cname, calias)
    end
    return buffer
end

function _alloc_field(meta, g2d, g3d, g3r)
    dims = meta.dimensions
    if any(d -> d in (:zeta_rock, :zeta_rock_ac), dims)
        return make_field(meta.name, g3r)
    elseif any(d -> d in (:zeta, :zeta_ac), dims)
        return make_field(meta.name, g3d)
    else
        return make_field(meta.name, g2d)
    end
end

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

function matches_patterns(name, patterns)
    any(p -> occursin(p, name), patterns)
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

function _init_yelmomirror(filename::String, time::Float64, calias::Vector{UInt8})
    grid_def = "file"
    ccall((:yelmo_init, yelmolib), Cvoid,
        (Ptr{UInt8}, Ptr{UInt8}, Float64, Ptr{UInt8}),
        filename * "\0", grid_def * "\0", time, calias)
end

end # module