"""
    YelmoModelPar

Parameter module for the pure-Julia `YelmoModel`. Structurally identical to
`YelmoPar` in v0; will evolve independently as `YelmoModel` grows new
parameters that the Fortran-backed Mirror does not need.

`write_nml`, `read_nml`, and `compare` are imported from `YelmoPar` and
extended with methods for `YelmoModelParameters`, so users see a single
generic function across both backends.

Usage:
    using .YelmoModelPar
    p = YelmoModelParameters("experiment1";
        ydyn = YelmoModelPar.ydyn_params(solver="ssa"),
    )
    write_nml("run.nml", p)
"""
module YelmoModelPar

using Printf

import ..YelmoPar: write_nml, compare
using ..YelmoSolvers: Solver, SSASolver

# Note on read_nml: YelmoPar.read_nml and YelmoModelPar.read_nml share the
# same signature `read_nml(::AbstractString)` but return different types,
# so they cannot be a single generic function. They live as separate
# functions in their respective module namespaces — call as
# `YelmoModelPar.read_nml(...)` to get a `YelmoModelParameters`.

export YelmoModelParameters

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
    # Fortran default is `dt_method = 2` (adaptive PC). Yelmo.jl
    # defaults to `0` (fixed forward Euler) so existing tests
    # written before adaptive landed keep their original semantics.
    # Set `dt_method = 2` explicitly to opt into adaptive PC.
    dt_method        ::Int     = 0
    dt_min           ::Float64 = 0.1
    cfl_max          ::Float64 = 0.1
    cfl_diff_max     ::Float64 = 0.12
    # Fortran namelist default is "AB-SAM"; Yelmo.jl currently
    # implements "HEUN" and "FE-SBE" only. Default to "FE-SBE" so
    # `dt_method = 2` works out of the box (Fortran-namelist users
    # passing "AB-SAM" will get a clear error from the resolver).
    pc_method        ::String  = "FE-SBE"
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
    H_min_grnd          ::Float64 = 0.0
    H_min_flt           ::Float64 = 0.0
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
    # Periodic-wrap offsets for surface-gradient kernels. Used only for
    # benchmarks with a known additively non-periodic axis (uniform-slope
    # surface across a periodic axis), where finite-differencing across
    # the wrap face would otherwise read the raw periodic image and
    # produce a spurious gradient. Set to the *signed surface change*
    # `Δz_srf` going one full periodic image distance in +x or +y, e.g.
    # for HOM-C (`z_srf = -x · tan α`),
    # `dzsdx_periodic_offset = -tan(α) · Lx_m`. Default 0.0 — production
    # ice-sheet configs use Bounded lateral axes and the offset is a
    # no-op there. Threaded through `_update_diagnostics!` to the
    # `dzsdx`/`dzsdy`/`dzbdx`/`dzbdy` gradient calls (the `z_base`
    # gradient sees the same offset since `z_base = z_srf - H_ice` and
    # `H_ice` is periodic by construction in these benchmarks).
    #
    # IMPORTANT — *config-time constant*: this is set ONCE at
    # `YelmoModelParameters` construction and is NOT recomputed during
    # the simulation. The mechanism is correct for benchmarks where the
    # uniform-slope component of the surface is static — typically those
    # using `topo_fixed = true` (HOM-C, MISMIP3D Stnd, ISMIP-HOM family).
    # For prognostic runs that simultaneously have (a) a periodic axis,
    # (b) a uniform-slope surface component, AND (c) topographic evolution
    # that changes that slope (e.g. a slab thinning under load), this
    # mechanism would silently desynchronise from the true surface slope.
    # Such configurations are rare in practice (real ice sheets are
    # Bounded, and benchmarks with periodic geometry are typically
    # diagnostic / fixed-geometry). If a future use case requires a
    # dynamically-evolving slope under periodic BC, the offset should be
    # promoted to a per-step recomputed quantity (e.g. derived from a
    # least-squares fit of the slope component of `z_srf` each step).
    dzsdx_periodic_offset ::Float64 = 0.0
    dzsdy_periodic_offset ::Float64 = 0.0
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
    ssa_solver      ::SSASolver = SSASolver()
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
# Note: physical constants (rho_ice, rho_sw, g, ...) used to live here as
# `PhysParams` / `phys_params(...)`. They've moved to the `YelmoConst` module
# (`YelmoConstants`), which lives separately so a single instance can be
# shared across multi-domain runs without going through model parameters.
# Read them from `y.c` on a constructed `YelmoModel`.

# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------
"""
    YelmoModelParameters
Top-level container holding one struct per namelist group. Construct via
`YelmoModelParameters(name; kwargs...)` to override individual groups.
"""
struct YelmoModelParameters
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
    YelmoModelParameters(name; yelmo, ytopo, ...) -> YelmoModelParameters
Construct a `YelmoModelParameters` object. Any group can be supplied as a keyword
argument; omitted groups are filled with defaults. `name` is a label for the
parameter set and the stem of the output filename.
# Example
```julia
p = YelmoModelParameters("experiment1";
    ydyn = ydyn_params(solver="ssa"),
)
write_nml("run.nml", p)
```
"""
function YelmoModelParameters(name;
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
    return YelmoModelParameters(
        name, yelmo, ytopo, ycalv, ydyn, ytill, yneff, ymat, ytherm,
        yelmo_masks, yelmo_init_topo, yelmo_data,
    )
end

function YelmoModelParameters(filename, name)
    p = read_nml(filename)
    return YelmoModelParameters(
        name, p.yelmo, p.ytopo, p.ycalv, p.ydyn, p.ytill, p.yneff,
        p.ymat, p.ytherm, p.yelmo_masks, p.yelmo_init_topo, p.yelmo_data,
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
        s = @sprintf("%.6e", x)
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
        occursin('.', s) || (s *= ".0")
        return s
    end
end
"""
    write_group(io, group_name, s)
Write one namelist group to `io` from struct `s`.
The field named `const_` is written as `const` (Julia keyword workaround).
Fields whose type is not a Fortran-namelist primitive (e.g. nested
struct configs like `SSASolver`) are skipped — those are Julia-native
configuration objects that have no namelist representation.
"""
function write_group(io::IO, group_name::AbstractString, s)
    println(io, "&$(group_name)")
    for fname in fieldnames(typeof(s))
        nml_name = fname == :const_ ? "const" : string(fname)
        val = getfield(s, fname)
        # Skip nested-struct fields (no Fortran namelist analogue).
        _is_nml_primitive(val) || continue
        println(io, "    $(rpad(nml_name, 20)) = $(format_value(val))")
    end
    println(io, "/\n")
end

# Predicate: is `v` a Fortran namelist primitive (or vector of primitives)?
# Used by `write_group` to skip nested-struct fields like `SSASolver`.
_is_nml_primitive(v::Bool)              = true
_is_nml_primitive(v::AbstractString)    = true
_is_nml_primitive(v::Integer)           = true
_is_nml_primitive(v::AbstractFloat)     = true
_is_nml_primitive(v::AbstractVector{<:AbstractString}) = true
_is_nml_primitive(v::AbstractVector{<:Real})           = true
_is_nml_primitive(v)                    = false
"""
    write_nml(filename, p::YelmoModelParameters)
Write a complete Yelmo namelist file from `p`.
"""
function write_nml(filename::AbstractString, p::YelmoModelParameters; overwrite::Bool=false)
    if isfile(filename) && !overwrite
        error("File already exists: $(filename). Use overwrite=true to overwrite.")
    end
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
function write_nml(p::YelmoModelParameters; rundir::String="", overwrite::Bool=false)
    filename = joinpath(rundir, p.name * ".nml")
    write_nml(filename, p; overwrite)
    return nothing
end

### READING NML FILES ###

# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

"""
    parse_nml_file(filename) -> Dict{String, Dict{String, String}}

Low-level parser. Returns a two-level dict:
    group_name => (field_name => raw_value_string)
Handles line continuation, inline comments, and multi-line values.
"""
function parse_nml_file(filename::AbstractString)
    groups = Dict{String, Dict{String, String}}()
    current_group = nothing
    current_key   = nothing
    current_val   = nothing

    for raw_line in eachline(filename)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, '!') && continue          # comment line

        # Strip inline comments (outside of quoted strings)
        line = _strip_inline_comment(line)
        isempty(line) && continue

        # &group_name
        if startswith(line, '&')
            # Flush any pending continuation
            if current_group !== nothing && current_key !== nothing
                groups[current_group][current_key] = strip(current_val)
                current_key = current_val = nothing
            end
            current_group = lowercase(strip(line[2:end]))
            groups[current_group] = Dict{String, String}()
            continue
        end

        # End-of-group marker
        if line == "/" || line == "&end" || startswith(line, "/")
            if current_group !== nothing && current_key !== nothing
                groups[current_group][current_key] = strip(current_val)
                current_key = current_val = nothing
            end
            current_group = nothing
            continue
        end

        current_group === nothing && continue

        # key = value  (possibly continued on next line via trailing comma)
        if occursin('=', line)
            # Flush previous key if any
            if current_key !== nothing
                groups[current_group][current_key] = strip(current_val)
            end
            idx = findfirst('=', line)
            current_key = strip(line[1:idx-1])
            current_val = strip(line[idx+1:end])
        else
            # Continuation line: append to current value
            current_key !== nothing && (current_val *= " " * line)
        end
    end
    # Flush final key
    if current_group !== nothing && current_key !== nothing
        groups[current_group][current_key] = strip(current_val)
    end

    return groups
end

"""
    _strip_inline_comment(line) -> String

Remove everything after an unquoted `!` character.
"""
function _strip_inline_comment(line::AbstractString)
    in_quote = false
    for (i, c) in enumerate(line)
        c == '"' && (in_quote = !in_quote)
        !in_quote && c == '!' && return strip(line[1:i-1])
    end
    return line
end

# ---------------------------------------------------------------------------
# Type-directed value parsing
# ---------------------------------------------------------------------------

"""
    parse_nml_value(::Type{T}, s) -> T

Parse a raw namelist string `s` into Julia type `T`.
"""
parse_nml_value(::Type{Bool}, s::AbstractString) =
    lowercase(strip(s)) in ("true", ".true.", "t", "1")

parse_nml_value(::Type{Int}, s::AbstractString) =
    parse(Int, strip(s))

parse_nml_value(::Type{Float64}, s::AbstractString) =
    parse(Float64, replace(strip(s), r"[dD]" => "e"))  # Fortran D-exponent

parse_nml_value(::Type{String}, s::AbstractString) =
    strip(s, [' ', '"', '\''])

function parse_nml_value(::Type{Vector{Float64}}, s::AbstractString)
    parts = split(strip(s), r"[\s,]+"; keepempty=false)
    return parse.(Float64, replace.(parts, r"[dD]" => "e"))
end

function parse_nml_value(::Type{Vector{String}}, s::AbstractString)
    # Match all quoted tokens
    ms = collect(eachmatch(r"\"([^\"]*)\"|'([^']*)'", s))
    isempty(ms) && return String[]
    return [something(m[1], m[2]) for m in ms]
end

# Fallback for unexpected types
parse_nml_value(::Type{T}, s::AbstractString) where {T} = parse(T, strip(s))

# ---------------------------------------------------------------------------
# Struct reconstruction from a flat Dict{String,String}
# ---------------------------------------------------------------------------

"""
    struct_from_dict(::Type{S}, d) -> S

Reconstruct struct `S` from a `Dict{String,String}` of raw namelist values.
Fields absent from `d` keep the default value from `S`'s `@kwdef` constructor.
The Fortran field `const` is mapped back to Julia field `const_`.
"""
function struct_from_dict(::Type{S}, d::Dict{String,String}) where {S}
    kwargs = Dict{Symbol,Any}()
    defaults = S()   # zero-arg @kwdef constructor gives us all defaults
    for fname in fieldnames(S)
        nml_name = fname == :const_ ? "const" : string(fname)
        FT = fieldtype(S, fname)
        if haskey(d, nml_name) && _is_nml_primitive(getfield(defaults, fname))
            kwargs[fname] = parse_nml_value(FT, d[nml_name])
        else
            # Either field absent from namelist, or field type is a
            # nested Julia-native config struct (no namelist analogue).
            # Fall back to the default value either way.
            kwargs[fname] = getfield(defaults, fname)
        end
    end
    return S(; kwargs...)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    read_nml(filename) -> YelmoModelParameters

Read a Yelmo namelist file and return a fully populated `YelmoModelParameters`.
Groups or fields absent from the file fall back to the struct defaults.

# Example
```julia
p = read_nml("run.nml")
println(p.ydyn.solver)   # "diva"
```
"""
function read_nml(filename::AbstractString)
    raw = parse_nml_file(filename)
    get_group(name) = get(raw, name, Dict{String,String}())

    return YelmoModelParameters(
        splitext(basename(filename))[1];   # name = stem of filename
        yelmo           = struct_from_dict(YelmoParams,          get_group("yelmo")),
        ytopo           = struct_from_dict(YtopoParams,          get_group("ytopo")),
        ycalv           = struct_from_dict(YcalvParams,          get_group("ycalv")),
        ydyn            = struct_from_dict(YdynParams,           get_group("ydyn")),
        ytill           = struct_from_dict(YtillParams,          get_group("ytill")),
        yneff           = struct_from_dict(YneffParams,          get_group("yneff")),
        ymat            = struct_from_dict(YmatParams,           get_group("ymat")),
        ytherm          = struct_from_dict(YthermParams,         get_group("ytherm")),
        yelmo_masks     = struct_from_dict(YelmoMasksParams,     get_group("yelmo_masks")),
        yelmo_init_topo = struct_from_dict(YelmoInitTopoParams,  get_group("yelmo_init_topo")),
        yelmo_data      = struct_from_dict(YelmoDataParams,      get_group("yelmo_data")),
    )
end


## Comparison

# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

function Base.:(==)(a::YelmoModelParameters, b::YelmoModelParameters)
    for fname in fieldnames(YelmoModelParameters)
        fname == :name && continue
        getfield(a, fname) == getfield(b, fname) || return false
    end
    return true
end

for S in (YelmoParams, YtopoParams, YcalvParams, YdynParams, YtillParams,
          YneffParams, YmatParams, YthermParams, YelmoMasksParams,
          YelmoInitTopoParams, YelmoDataParams)
    @eval function Base.:(==)(a::$S, b::$S)
        for fname in fieldnames($S)
            getfield(a, fname) == getfield(b, fname) || return false
        end
        return true
    end
end

# For each sub-struct, fall back to the auto-generated field-wise ==
# (this works because all leaf types are Bool/Int/Float64/String/Vector,
#  which already have == defined)

# ---------------------------------------------------------------------------
# Diff printing
# ---------------------------------------------------------------------------

"""
    diff_nml([io,] p1, p2; include_name=false)

Print all fields that differ between `p1` and `p2`, grouped by namelist group.
Identical groups are skipped entirely.
"""
function compare(io::IO, p1::YelmoModelParameters, p2::YelmoModelParameters; include_name=false)
    any_diff = false
    for fname in fieldnames(YelmoModelParameters)
        fname == :name && !include_name && continue
        g1, g2 = getfield(p1, fname), getfield(p2, fname)
        g1 == g2 && continue

        # Group header
        any_diff = true
        println(io, "&$(fname)")

        if fname == :name
            println(io, "  $(rpad("name", 24))  \"$(g1)\"  =>  \"$(g2)\"")
        else
            for sfield in fieldnames(typeof(g1))
                v1, v2 = getfield(g1, sfield), getfield(g2, sfield)
                v1 == v2 && continue
                label = sfield == :const_ ? "const" : string(sfield)
                if _is_nml_primitive(v1) && _is_nml_primitive(v2)
                    println(io, "  $(rpad(label, 24))  $(format_value(v1))  =>  $(format_value(v2))")
                else
                    # Nested-struct field (e.g. SSASolver) — fall back to
                    # repr() since format_value isn't defined for them.
                    println(io, "  $(rpad(label, 24))  $(repr(v1))  =>  $(repr(v2))")
                end
            end
        end
        println(io, "/\n")
    end
    any_diff || println(io, "(no differences)")
    return nothing
end

compare(p1::YelmoModelParameters, p2::YelmoModelParameters; kw...) = compare(stdout, p1, p2; kw...)

end # module YelmoPar

