module YelmoMirrorCore

using Oceananigans: Grids, Fields

using ..YelmoMeta: VariableMeta, parse_variable_table
using ..YelmoPar: YelmoParameters, write_nml

export YelmoMirror, init_state!, time_step!, sync!
export yelmo_get_var2D, yelmo_get_var2D!
export yelmo_get_var3D, yelmo_get_var3D!
export yelmo_set_var2D!, yelmo_set_var3D!

# ---------------------------------------------------------------------------
#const yelmolib = "../libyelmo/include/libyelmo_c_api.so"
const yelmopath = joinpath(@__DIR__, "..", "yelmo")
const yelmolib = joinpath(@__DIR__, "..", "yelmo", "libyelmo", "include", "libyelmo_c_api.so")

const VERTICAL_DIMS = (:zeta, :zeta_ac, :zeta_rock, :zeta_rock_ac)

# ---------------------------------------------------------------------------
mutable struct YelmoMirror
    alias::String
    rundir::String
    time::Float64
    p::YelmoParameters
    g::NamedTuple
    v::NamedTuple
    bnd::NamedTuple
    dta::NamedTuple
    dyn::NamedTuple
    mat::NamedTuple
    thrm::NamedTuple
    tpo::NamedTuple
end

function YelmoMirror(filename::String, time::Float64; 
    alias::String="ylmo1", 
    rundir::String="./",
    overwrite::Bool=false
    )

    # Load parameters from source parameter file
    p = YelmoParameters(filename)

    return YelmoMirror(p, time; alias, rundir, overwrite)
end

function YelmoMirror(p::YelmoParameters, time::Float64; 
    alias::String="ylmo1",
    rundir::String="./",
    overwrite::Bool=false
    )

    # Write a namelist file with current parameters for Yelmo
    filename = joinpath(rundir,p.name*".nml")
    write_nml(filename,p;overwrite)
    
    # First initialize the Yelmo mirror in fortran
    _init_yelmomirror(filename, time, alias)

    # Populate Julia version of Yelmo object with info from fortran
    g = yelmo_get_grid_info(alias)

    # Load variable meta information
    v = (
        bnd  = parse_variable_table("input/yelmo-variables-ybound.md","bnd"),
        dta  = parse_variable_table("input/yelmo-variables-ydata.md","dta"),
        dyn  = parse_variable_table("input/yelmo-variables-ydyn.md","dyn"),
        mat  = parse_variable_table("input/yelmo-variables-ymat.md","mat"),
        thrm = parse_variable_table("input/yelmo-variables-ytherm.md","thrm"),
        tpo  = parse_variable_table("input/yelmo-variables-ytopo.md","tpo"),
    )

    bnd = yelmo_get_variable_set(v.bnd,"bnd",g.nx,g.ny,g.nz_aa,g.nz_ac,g.nzr_aa,g.nzr_ac,alias)
    dta = yelmo_get_variable_set(v.dta,"dta",g.nx,g.ny,g.nz_aa,g.nz_ac,g.nzr_aa,g.nzr_ac,alias)
    dyn = yelmo_get_variable_set(v.dyn,"dyn",g.nx,g.ny,g.nz_aa,g.nz_ac,g.nzr_aa,g.nzr_ac,alias)
    mat = yelmo_get_variable_set(v.mat,"mat",g.nx,g.ny,g.nz_aa,g.nz_ac,g.nzr_aa,g.nzr_ac,alias)
    thrm = yelmo_get_variable_set(v.thrm,"thrm",g.nx,g.ny,g.nz_aa,g.nz_ac,g.nzr_aa,g.nzr_ac,alias)
    tpo = yelmo_get_variable_set(v.tpo,"tpo",g.nx,g.ny,g.nz_aa,g.nz_ac,g.nzr_aa,g.nzr_ac,alias)
    
    return YelmoMirror(alias,rundir,time,p,g,v,bnd,dta,dyn,mat,thrm,tpo)
end

function _init_yelmomirror(filename::String, time::Float64, alias::String)

    # Option needed for Fortran call
    grid_def = "file"

    # First call yelmo init to initialize model in fortran
    ccall((:yelmo_init, yelmolib), Cvoid,
        (Ptr{UInt8}, Ptr{UInt8}, Float64, Ptr{UInt8}),
        filename * "\0", grid_def * "\0", time, alias * "\0")

    return
end

function init_state!(ylmo::YelmoMirror, time::Float64, thrm_method::String)
    
    # Sync yelmo to fortran
    sync!(ylmo)

    # call yelmo_init_state in fortran
    ccall((:yelmo_init_state, yelmolib), Cvoid,
        (Float64, Ptr{UInt8}, Ptr{UInt8}),
        time, thrm_method * "\0", ylmo.alias * "\0")
    
    # Update yelmo in julia
    yelmo_get_variables!(ylmo)

    # Update time
    ylmo.time = time

    return ylmo
end

function time_step!(ylmo::YelmoMirror, dt::Float64)

    # Sync yelmo to fortran
    sync!(ylmo)

    # Update time
    ylmo.time += dt
    
    # Call yelmo_step in fortran
    ccall((:yelmo_step, yelmolib), Cvoid, (Float64, Ptr{UInt8}), ylmo.time, ylmo.alias * "\0")

    # Update yelmo in julia
    yelmo_get_variables!(ylmo)

    return ylmo
end

function yelmo_get_grid_info(alias::String)

    # Step 1: get sizes
    nx    = Ref{Cint}(0)
    ny    = Ref{Cint}(0)
    nz_aa = Ref{Cint}(0)
    nz_ac = Ref{Cint}(0)
    nzr_aa = Ref{Cint}(0)
    nzr_ac = Ref{Cint}(0)

    ccall((:yelmo_get_grid_sizes, yelmolib), Cvoid,
          (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint}, Ptr{UInt8}),
          nx, ny, nz_aa, nz_ac, nzr_aa, nzr_ac, alias * "\0")

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
          xc, yc, zeta_aa, zeta_ac, zeta_r_aa, zeta_r_ac, alias * "\0")

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

function yelmo_get_variables!(ylmo)

    yelmo_get_variable_set!(ylmo.bnd, ylmo.v.bnd, "bnd", ylmo.alias)
    yelmo_get_variable_set!(ylmo.dta, ylmo.v.dta, "dta", ylmo.alias)
    yelmo_get_variable_set!(ylmo.dyn, ylmo.v.dyn, "dyn", ylmo.alias)
    yelmo_get_variable_set!(ylmo.mat, ylmo.v.mat, "mat", ylmo.alias)
    yelmo_get_variable_set!(ylmo.thrm, ylmo.v.thrm, "thrm", ylmo.alias)
    yelmo_get_variable_set!(ylmo.tpo, ylmo.v.tpo, "tpo", ylmo.alias)
    
    return ylmo
end

function yelmo_get_variable_set!(dat, vlist, prefix, alias::String)
    for k in keys(vlist)
        _get_var!(dat[k], vlist[k], prefix, k, alias)
    end
    return dat
end

function _get_var!(arr, meta, prefix, k, alias::String)
    varname = "$(prefix)_$(k)"
    dims = meta.dimensions
    is3D = any(d -> d in VERTICAL_DIMS, dims)
    if is3D
        yelmo_get_var3D!(arr, varname, alias)
    else
        yelmo_get_var2D!(arr, varname, alias)
    end
end

function yelmo_get_variable_set(vlist, prefix, nx, ny, nz_aa, nz_ac, nzr_aa, nzr_ac, alias::String)
    dat = NamedTuple{keys(vlist)}(
        _alloc_var(vlist[k], nx, ny, nz_aa, nz_ac, nzr_aa, nzr_ac) for k in keys(vlist)
    )
    yelmo_get_variable_set!(dat, vlist, prefix, alias)
    return dat
end

function _alloc_var(meta, nx, ny, nz_aa, nz_ac, nzr_aa, nzr_ac)
    dims = meta.dimensions
    nz = nothing
    for (d, n) in ((:zeta, nz_aa), (:zeta_ac, nz_ac), (:zeta_rock, nzr_aa), (:zeta_rock_ac, nzr_ac))
        d in dims && (nz = n; break)
    end
    return isnothing(nz) ? zeros(Float64, nx, ny) : zeros(Float64, nx, ny, nz)
end

function yelmo_get_var2D!(v2D::Array{Float64,2}, name::String, alias::String)
    nx, ny = size(v2D)
    ccall((:yelmo_get_var2D, yelmolib), Cvoid,
        (Ptr{Float64}, Int32, Int32, Ptr{UInt8}, Ptr{UInt8}),
        v2D, Int32(nx), Int32(ny), name * "\0", alias * "\0")
    return v2D
end

function yelmo_get_var2D(nx::Int, ny::Int, name::String, alias::String)
    v2D = Matrix{Float64}(undef, nx, ny)
    yelmo_get_var2D!(v2D, name, alias)
    return v2D
end

function yelmo_get_var3D!(v3D::Array{Float64,3}, name::String, alias::String)
    nx, ny, nz = size(v3D)
    ccall((:yelmo_get_var3D, yelmolib), Cvoid,
        (Ptr{Float64}, Int32, Int32, Int32, Ptr{UInt8}, Ptr{UInt8}),
        v3D, Int32(nx), Int32(ny), Int32(nz), name * "\0", alias * "\0")
    return v3D
end

function yelmo_get_var3D(nx::Int, ny::Int, nz::Int, name::String, alias::String)
    v3D = Array{Float64}(undef, nx, ny, nz)
    yelmo_get_var3D!(v3D, name, alias)
    return v3D
end

function sync!(ylmo)
    # Push values from Julia to fortran

    # All boundary fields
    for k in keys(ylmo.v.bnd)
        yelmo_set_var2D!("bnd_$(k)", ylmo.bnd[k], ylmo.alias)
    end

    # tpo variables
    yelmo_set_var2D!("tpo_H_ice", ylmo.tpo.H_ice, ylmo.alias)

    # dyn variables
    yelmo_set_var2D!("dyn_cb_ref", ylmo.dyn.cb_ref, ylmo.alias)
    yelmo_set_var2D!("dyn_N_eff",  ylmo.dyn.N_eff,  ylmo.alias)
    yelmo_set_var3D!("dyn_ux", ylmo.dyn.ux, ylmo.alias)
    yelmo_set_var3D!("dyn_uy", ylmo.dyn.uy, ylmo.alias)
    yelmo_set_var3D!("dyn_uz", ylmo.dyn.uz, ylmo.alias)

    # thrm variables
    yelmo_set_var3D!("thrm_T_ice", ylmo.thrm.T_ice, ylmo.alias)
    yelmo_set_var2D!("thrm_H_w", ylmo.thrm.H_w, ylmo.alias)

end

function yelmo_set_var2D!(name::String, v2D::Array{Float64,2}, alias::String)

    nx, ny = size(v2D)

    ccall((:yelmo_set_var2D, yelmolib), Cvoid,
          (Ptr{Cdouble}, Cint, Cint, Cstring, Ptr{UInt8}),
          v2D, nx, ny, name, alias)

    return nothing

end

function yelmo_set_var3D!(name::String, v3D::Array{Float64,3}, alias::String)

    nx, ny, nz = size(v3D)

    ccall((:yelmo_set_var3D, yelmolib), Cvoid,
          (Ptr{Cdouble}, Cint, Cint, Cint, Cstring, Ptr{UInt8}),
          v3D, nx, ny, nz, name, alias)

    return nothing

end

end # Module
