"""
    YelmoPar

Module for constructing and serializing Yelmo namelist parameter files.

Each namelist group is represented by a dedicated struct. Constructor functions
provide default values matching the reference namelist. A `write_namelist`
routine serializes any `YelmoPar` object to a `.nml` file.

Usage:
    using .YelmoPar

    p = YelmoPar.yelmo_params(
        ctrl    = YelmoPar.ctrl_params(time_end=50e3, dtt=5.0),
        yelmo   = YelmoPar.yelmo_params(domain="Antarctica"),
    )

    YelmoPar.write_namelist("run.nml", p)
"""
module YelmoPar

using Printf

export YelmoParameters
export yelmo_params, ytopo_params, ycalv_params, ydyn_params,
       ytill_params, yneff_params, ymat_params, ytherm_params,
       yelmo_masks_params, yelmo_init_topo_params, yelmo_data_params
export write_nml

# ---------------------------------------------------------------------------
# &yelmo  (top-level Yelmo group)
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoParams
    domain           ::String  = "Greenland"
    grid_name        ::String  = "GRL-16KM"
    grid_path        ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_REGIONS.nc"
    phys_const       ::String  = "Earth"
    experiment       ::String  = "None"
    nml_ytopo        ::String  = "ytopo"
    nml_ycalv        ::String  = "ycalv"
    nml_ydyn         ::String  = "ydyn"
    nml_ytill        ::String  = "ytill"
    nml_yneff        ::String  = "yneff"
    nml_ymat         ::String  = "ymat"
    nml_ytherm       ::String  = "ytherm"
    nml_masks        ::String  = "yelmo_masks"
    nml_init_topo    ::String  = "yelmo_init_topo"
    nml_data         ::String  = "yelmo_data"
    restart          ::String  = "None"
    restart_z_bed    ::Bool    = false
    restart_H_ice    ::Bool    = false
    restart_relax    ::Float64 = 1e3
    log_timestep     ::Bool    = false
    disable_kill     ::Bool    = false
    zeta_scale       ::String  = "exp"
    zeta_exp         ::Float64 = 2.0
    nz_aa            ::Int     = 10
    dt_method        ::Int     = 2
    dt_min           ::Float64 = 0.1
    cfl_max          ::Float64 = 0.1
    cfl_diff_max     ::Float64 = 0.12
    pc_method        ::String  = "AB-SAM"
    pc_controller    ::String  = "PI42"
    pc_use_H_pred    ::Bool    = true
    pc_filter_vel    ::Bool    = true
    pc_corr_vel      ::Bool    = false
    pc_n_redo        ::Int     = 5
    pc_tol           ::Float64 = 5.0
    pc_eps           ::Float64 = 1.0
end
yelmo_params(; kwargs...) = YelmoParams(; kwargs...)

# ---------------------------------------------------------------------------
# &ytopo
# ---------------------------------------------------------------------------
Base.@kwdef struct YtopoParams
    solver              ::String  = "impl-lis"
    surf_gl_method      ::Int     = 0
    grad_lim            ::Float64 = 0.5
    grad_lim_zb         ::Float64 = 0.5
    dHdt_dyn_lim        ::Float64 = 100.0
    margin2nd           ::Bool    = false
    margin_flt_subgrid  ::Bool    = false
    use_bmb             ::Bool    = true
    topo_fixed          ::Bool    = false
    topo_rel            ::Int     = 0
    topo_rel_tau        ::Float64 = 10.0
    topo_rel_field      ::String  = "H_ref"
    bmb_gl_method       ::String  = "pmp"
    gl_sep              ::Int     = 1
    gz_nx               ::Int     = 15
    dist_grz            ::Float64 = 200.0
    gz_Hg0              ::Float64 = 0.0
    gz_Hg1              ::Float64 = 0.0
    dmb_method          ::Int     = 0
    dmb_alpha_max       ::Float64 = 60.0
    dmb_tau             ::Float64 = 100.0
    dmb_sigma_ref       ::Float64 = 300.0
    dmb_m_d             ::Float64 = 3.0
    dmb_m_r             ::Float64 = 1.0
    fmb_method          ::Int     = 0
    fmb_scale           ::Float64 = 1.0
end
ytopo_params(; kwargs...) = YtopoParams(; kwargs...)

# ---------------------------------------------------------------------------
# &ycalv
# ---------------------------------------------------------------------------
Base.@kwdef struct YcalvParams
    use_lsf         ::Bool    = false
    dt_lsf          ::Float64 = -1.0
    calv_flt_method ::String  = "vm-l19"
    calv_grnd_method::String  = "zero"
    H_min_grnd      ::Float64 = 5.0
    H_min_flt       ::Float64 = 75.0
    sd_min          ::Float64 = 100.0
    sd_max          ::Float64 = 500.0
    calv_grnd_max   ::Float64 = 0.0
    calv_tau        ::Float64 = 1.0
    calv_thin       ::Float64 = 30.0
    k2              ::Float64 = 3.2e9
    w2              ::Float64 = 0.0
    kt_ref          ::Float64 = 0.0025
    kt_deep         ::Float64 = 0.1
    tau_ice         ::Float64 = 250.0e3
    Hc_ref_flt      ::Float64 = 200.0
    Hc_ref_grnd     ::Float64 = 200.0
    Hc_ref_thin     ::Float64 = 50.0
    Hc_deep         ::Float64 = 500.0
    zb_deep_0       ::Float64 = -1000.0
    zb_deep_1       ::Float64 = -1500.0
    zb_sigma        ::Float64 = 0.0
end
ycalv_params(; kwargs...) = YcalvParams(; kwargs...)

# ---------------------------------------------------------------------------
# &ydyn
# ---------------------------------------------------------------------------
Base.@kwdef struct YdynParams
    solver          ::String  = "diva"
    uz_method       ::Int     = 3
    visc_method     ::Int     = 1
    visc_const      ::Float64 = 1e7
    beta_method     ::Int     = 1
    beta_const      ::Float64 = 1e3
    beta_q          ::Float64 = 1.0
    beta_u0         ::Float64 = 100.0
    beta_gl_scale   ::Int     = 0
    beta_gl_stag    ::Int     = 1
    beta_gl_f       ::Float64 = 1.0
    taud_gl_method  ::Int     = 0
    H_grnd_lim      ::Float64 = 500.0
    beta_min        ::Float64 = 100.0
    eps_0           ::Float64 = 1e-6
    scale_T         ::Int     = 1
    T_frz           ::Float64 = -3.0
    ssa_lis_opt     ::String  = "-i minres -p jacobi -maxiter 100 -tol 1.0e-2 -initx_zeros false"
    ssa_lat_bc      ::String  = "floating"
    ssa_beta_max    ::Float64 = 1e20
    ssa_vel_max     ::Float64 = 5000.0
    ssa_iter_max    ::Int     = 20
    ssa_iter_rel    ::Float64 = 0.7
    ssa_iter_conv   ::Float64 = 1e-2
    taud_lim        ::Float64 = 2e5
    cb_sia          ::Float64 = 0.0
end
ydyn_params(; kwargs...) = YdynParams(; kwargs...)

# ---------------------------------------------------------------------------
# &ytill
# ---------------------------------------------------------------------------
Base.@kwdef struct YtillParams
    method    ::Int     = 1
    scale_zb  ::Int     = 1
    scale_sed ::Int     = 0
    is_angle  ::Bool    = false
    n_sd      ::Int     = 10
    f_sed     ::Float64 = 0.01
    sed_min   ::Float64 = 5.0
    sed_max   ::Float64 = 15.0
    z0        ::Float64 = -300.0
    z1        ::Float64 = 200.0
    cf_min    ::Float64 = 0.1
    cf_ref    ::Float64 = 0.8
end
ytill_params(; kwargs...) = YtillParams(; kwargs...)

# ---------------------------------------------------------------------------
# &yneff
# ---------------------------------------------------------------------------
Base.@kwdef struct YneffParams
    method  ::Int     = 3
    nxi     ::Int     = 0
    const_  ::Float64 = 1e7          # note: 'const' is a Julia keyword, stored as const_
    p       ::Float64 = 0.0
    H_w_max ::Float64 = -1.0
    N0      ::Float64 = 1000.0
    delta   ::Float64 = 0.04
    e0      ::Float64 = 0.69
    Cc      ::Float64 = 0.12
    s_const ::Float64 = 0.5
end
yneff_params(; kwargs...) = YneffParams(; kwargs...)

# ---------------------------------------------------------------------------
# &ymat
# ---------------------------------------------------------------------------
Base.@kwdef struct YmatParams
    flow_law            ::String  = "glen"
    rf_method           ::Int     = 1
    rf_const            ::Float64 = 1e-18
    rf_use_eismint2     ::Bool    = false
    rf_with_water       ::Bool    = false
    n_glen              ::Float64 = 3.0
    visc_min            ::Float64 = 1e3
    de_max              ::Float64 = 2.0
    enh_method          ::String  = "shear3D"
    enh_shear           ::Float64 = 3.0
    enh_stream          ::Float64 = 3.0
    enh_shlf            ::Float64 = 0.7
    enh_umin            ::Float64 = 50.0
    enh_umax            ::Float64 = 500.0
    calc_age            ::Bool    = false
    age_iso             ::Vector{Float64} = [11.7, 29.0, 57.0, 115.0]
    tracer_method       ::String  = "expl"
    tracer_impl_kappa   ::Float64 = 1.5
end
ymat_params(; kwargs...) = YmatParams(; kwargs...)

# ---------------------------------------------------------------------------
# &ytherm
# ---------------------------------------------------------------------------
Base.@kwdef struct YthermParams
    method          ::String  = "temp"
    qb_method       ::Int     = 2
    dt_method       ::String  = "FE"
    solver_advec    ::String  = "impl-upwind"
    gamma           ::Float64 = 1.0
    use_strain_sia  ::Bool    = false
    use_const_cp    ::Bool    = false
    const_cp        ::Float64 = 2009.0
    use_const_kt    ::Bool    = false
    const_kt        ::Float64 = 6.62e7
    enth_cr         ::Float64 = 1e-3
    omega_max       ::Float64 = 0.01
    till_rate       ::Float64 = 0.001
    H_w_max         ::Float64 = 2.0
    rock_method     ::String  = "equil"
    nzr_aa          ::Int     = 5
    zeta_scale_rock ::String  = "exp-inv"
    zeta_exp_rock   ::Float64 = 2.0
    H_rock          ::Float64 = 2000.0
    cp_rock         ::Float64 = 1000.0
    kt_rock         ::Float64 = 6.3e7
end
ytherm_params(; kwargs...) = YthermParams(; kwargs...)

# ---------------------------------------------------------------------------
# &yelmo_masks
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoMasksParams
    basins_load   ::Bool   = true
    basins_path   ::String = "ice_data/{domain}/{grid_name}/{grid_name}_BASINS-nasa.nc"
    basins_nms    ::Vector{String} = ["basin", "basin_mask"]
    regions_load  ::Bool   = true
    regions_path  ::String = "ice_data/{domain}/{grid_name}/{grid_name}_REGIONS.nc"
    regions_nms   ::Vector{String} = ["mask", "None"]
end
yelmo_masks_params(; kwargs...) = YelmoMasksParams(; kwargs...)

# ---------------------------------------------------------------------------
# &yelmo_init_topo
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoInitTopoParams
    init_topo_load  ::Bool    = true
    init_topo_path  ::String  = "ice_data/{domain}/{grid_name}/{grid_name}_TOPO-M17.nc"
    init_topo_names ::Vector{String} = ["H_ice", "z_bed", "z_bed_sd", "z_srf"]
    init_topo_state ::Int     = 0
    z_bed_f_sd      ::Float64 = -1.0
    smooth_H_ice    ::Float64 = 0.0
    smooth_z_bed    ::Float64 = 0.0
end
yelmo_init_topo_params(; kwargs...) = YelmoInitTopoParams(; kwargs...)

# ---------------------------------------------------------------------------
# &yelmo_data
# ---------------------------------------------------------------------------
Base.@kwdef struct YelmoDataParams
    pd_topo_load    ::Bool   = true
    pd_topo_path    ::String = "ice_data/{domain}/{grid_name}/{grid_name}_TOPO-M17.nc"
    pd_topo_names   ::Vector{String} = ["H_ice", "z_bed", "z_bed_sd", "z_srf"]
    pd_tsrf_load    ::Bool   = true
    pd_tsrf_path    ::String = "ice_data/{domain}/{grid_name}/{grid_name}_MARv3.11-ERA_annmean_1961-1990.nc"
    pd_tsrf_name    ::String = "T_srf"
    pd_tsrf_monthly ::Bool   = false
    pd_smb_load     ::Bool   = true
    pd_smb_path     ::String = "ice_data/{domain}/{grid_name}/{grid_name}_MARv3.11-ERA_annmean_1961-1990.nc"
    pd_smb_name     ::String = "smb"
    pd_smb_monthly  ::Bool   = false
    pd_vel_load     ::Bool   = true
    pd_vel_path     ::String = "ice_data/{domain}/{grid_name}/{grid_name}_VEL-J18.nc"
    pd_vel_names    ::Vector{String} = ["ux_srf", "uy_srf"]
    pd_age_load     ::Bool   = false
    pd_age_path     ::String = "ice_data/{domain}/{grid_name}/{grid_name}_STRAT-M15.nc"
    pd_age_names    ::Vector{String} = ["age_iso", "depth_iso"]
end
yelmo_data_params(; kwargs...) = YelmoDataParams(; kwargs...)

# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------
"""
    YelmoParameters

Top-level container holding one struct per namelist group. Construct via
`yelmo_params(; kwargs...)` to override individual groups.
"""
struct YelmoParameters
    name            ::String
    yelmo           ::YelmoParams
    ytopo           ::YtopoParams
    ycalv           ::YcalvParams
    ydyn            ::YdynParams
    ytill           ::YtillParams
    yneff           ::YneffParams
    ymat            ::YmatParams
    ytherm          ::YthermParams
    yelmo_masks     ::YelmoMasksParams
    yelmo_init_topo ::YelmoInitTopoParams
    yelmo_data      ::YelmoDataParams
end

"""
    YelmoParameters(name; yelmo, ytopo, ...) -> YelmoParameters

Construct a `YelmoParameters` object. Any group can be supplied as a keyword
argument; omitted groups are filled with defaults. `name` is provided as an
alias to the parameter set.

# Example
```julia
p = YelmoParameters("experiment1";
    ydyn  = ydyn_params(solver="ssa"),
)
write_namelist("run.nml", p)
```
"""
function YelmoParameters(name;
    yelmo           = yelmo_params(),
    ytopo           = ytopo_params(),
    ycalv           = ycalv_params(),
    ydyn            = ydyn_params(),
    ytill           = ytill_params(),
    yneff           = yneff_params(),
    ymat            = ymat_params(),
    ytherm          = ytherm_params(),
    yelmo_masks     = yelmo_masks_params(),
    yelmo_init_topo = yelmo_init_topo_params(),
    yelmo_data      = yelmo_data_params(),
)
    return YelmoParameters(
        name, yelmo, ytopo, ycalv, ydyn, ytill, yneff, ymat, ytherm,
        yelmo_masks, yelmo_init_topo, yelmo_data,
    )
end

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

"""
    format_value(v) -> String

Format a Julia value for Fortran namelist syntax.
"""
format_value(v::Bool)              = v ? "True" : "False"
format_value(v::AbstractString)    = "\"$(v)\""
format_value(v::Int)               = string(v)
format_value(v::Float64)           = _fmt_float(v)
format_value(v::AbstractVector{<:AbstractString}) = join(["\"$s\"" for s in v], " ")
format_value(v::AbstractVector{<:Real})            = join(format_value.(v), ", ")

"""
    _fmt_float(x) -> String

Produce a clean, compact float representation. Uses exponential notation
when the magnitude is very large or very small, otherwise decimal.
"""
function _fmt_float(x::Float64)
    x == 0.0 && return "0.0"
    a = abs(x)
    if a >= 1e5 || (a < 1e-3 && a > 0.0)
        # Use exponential form, stripping trailing zeros
        s = @sprintf("%.6e", x)
        # Convert 1.500000e+03 → 1.5e3
        m = match(r"^(-?)(\d+\.\d*?)0*(e[+-]?)0*(\d+)$", s)
        if m !== nothing
            mantissa = endswith(m[2], ".") ? m[2] * "0" : m[2]
            exp_sign = replace(m[3], "e+" => "e", "e-" => "e-")
            exp_dig  = m[4]
            return "$(m[1])$(mantissa)$(exp_sign)$(exp_dig)"
        end
        return s
    else
        s = @sprintf("%.10g", x)
        # Ensure at least one decimal point so Fortran reads it as real
        occursin('.', s) || (s *= ".0")
        return s
    end
end

using Printf

"""
    write_group(io, group_name, s)

Write one namelist group to `io` from struct `s`.
The field named `const_` is written as `const` (Julia keyword workaround).
"""
function write_group(io::IO, group_name::AbstractString, s)
    println(io, "&$(group_name)")
    for fname in fieldnames(typeof(s))
        nml_name = fname == :const_ ? "const" : string(fname)
        val = getfield(s, fname)
        println(io, "    $(rpad(nml_name, 20)) = $(format_value(val))")
    end
    println(io, "/\n")
end

"""
    write_namelist(filename, p::YelmoParameters)

Write a complete Yelmo namelist file from `p`.
"""
function write_nml(filename::AbstractString, p::YelmoParameters)
    open(filename, "w") do io
        write_group(io, "yelmo",           p.yelmo)
        write_group(io, "ytopo",           p.ytopo)
        write_group(io, "ycalv",           p.ycalv)
        write_group(io, "ydyn",            p.ydyn)
        write_group(io, "ytill",           p.ytill)
        write_group(io, "yneff",           p.yneff)
        write_group(io, "ymat",            p.ymat)
        write_group(io, "ytherm",          p.ytherm)
        write_group(io, "yelmo_masks",     p.yelmo_masks)
        write_group(io, "yelmo_init_topo", p.yelmo_init_topo)
        write_group(io, "yelmo_data",      p.yelmo_data)
    end
    @info "Namelist written to $(filename)"
    return nothing
end

function write_nml(p::YelmoParameters;rundir="")
    filename = joinpath(rundir,p.name*".nml")
    write_nml(filename,p)
    return nothing
end

end # module YelmoPar