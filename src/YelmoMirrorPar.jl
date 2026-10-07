"""
    YelmoMirrorPar
Parameter module for the Fortran-backed `YelmoMirror`.

A `YelmoMirrorParameters` holds only the namelist values that differ from the
Fortran defaults. The schema (groups, keys and default values) is Fortran's own
`yelmo/input/yelmo_defaults.nml`, which Fortran `yelmo_init` reads anyway: every
key not set here takes its value from that file. Keys are checked against the
schema when parameters are built, so a removed or renamed key fails in Julia
before Fortran is called.

Groups that are not Yelmo groups (e.g. a driver's `&ctrl`) are kept verbatim when
a namelist is read and written back unchanged.

The generic `write_nml`/`compare` functions are owned by `YelmoPar` (the primary
pure-Julia parameter module) and extended here with methods for
`YelmoMirrorParameters`. `read_nml` stays module-local (same signature as
`YelmoPar.read_nml` but returns `YelmoMirrorParameters`) — call as
`YelmoMirrorPar.read_nml(...)`.

Usage:
    using .YelmoMirrorPar
    p = YelmoMirrorParameters("experiment1";
        ydyn = (solver = "ssa", ssa_vel_max = 5e4),
    )
    p.ydyn.solver        # "ssa"
    p.ydyn.beta_method   # Fortran default
    YelmoMirrorPar.write_nml("run.nml", p)
"""
module YelmoMirrorPar

# write_nml/compare are owned by the primary `YelmoPar` module; extend them here
# with methods for `YelmoMirrorParameters`.
import ..YelmoPar: write_nml, compare
import ..YelmoPar

export YelmoMirrorParameters
export to_mirror, MIRROR_DIVERGENT_YELMO

# ---------------------------------------------------------------------------
# Fortran schema: yelmo/input/yelmo_defaults.nml and yelmo_phys_const.nml
# ---------------------------------------------------------------------------

const YELMO_FORTRAN_DIR = joinpath(@__DIR__, "..", "yelmo")

defaults_file()   = joinpath(YELMO_FORTRAN_DIR, "input", "yelmo_defaults.nml")
phys_const_file() = joinpath(YELMO_FORTRAN_DIR, "input", "yelmo_phys_const.nml")

# Ordered raw namelist: group => [key => raw value string]
const RawGroups = Vector{Pair{String, Vector{Pair{String,String}}}}

# Parsed schema: group order, key order per group, typed default per key.
struct Schema
    groups   :: Vector{String}
    keys     :: Dict{String, Vector{String}}
    defaults :: Dict{String, Dict{String, Any}}
end

const _SCHEMA = Ref{Union{Nothing,Schema}}(nothing)

"""
    schema() -> Schema

The Fortran parameter schema, parsed once from `yelmo_defaults.nml`.
"""
function schema()
    if _SCHEMA[] === nothing
        f = defaults_file()
        isfile(f) || error("YelmoMirrorPar: Fortran defaults file not found: $(f). " *
                           "YelmoMirror needs the Fortran yelmo tree at $(YELMO_FORTRAN_DIR).")
        raw = parse_nml_file(f)
        groups = String[]
        keys = Dict{String, Vector{String}}()
        defaults = Dict{String, Dict{String, Any}}()
        for (g, kvs) in raw
            push!(groups, g)
            keys[g] = first.(kvs)
            defaults[g] = Dict{String,Any}(k => parse_nml_value(v) for (k, v) in kvs)
        end
        _SCHEMA[] = Schema(groups, keys, defaults)
    end
    return _SCHEMA[]
end

is_yelmo_group(g::AbstractString) = haskey(schema().defaults, g)

# Value kinds used to check an override against its default.
_kind(::Bool) = :bool
_kind(::AbstractString) = :string
_kind(::AbstractVector{<:AbstractString}) = :string
_kind(::Real) = :num
_kind(::AbstractVector{<:Real}) = :num
_kind(x) = error("YelmoMirrorPar: unsupported namelist value type $(typeof(x))")

function _check_key(group::String, key::String, value)
    s = schema()
    haskey(s.defaults, group) ||
        error("YelmoMirrorPar: unknown Yelmo namelist group &$(group). " *
              "Groups in $(defaults_file()): $(join(s.groups, ", "))")
    d = s.defaults[group]
    haskey(d, key) ||
        error("YelmoMirrorPar: &$(group) has no parameter `$(key)` in $(defaults_file()).")
    _kind(value) == _kind(d[key]) ||
        error("YelmoMirrorPar: &$(group).$(key) = $(repr(value)) does not match the type " *
              "of its default $(repr(d[key])).")
    return nothing
end

"""
    phys_constants(name="Earth") -> NamedTuple

Physical constants of group `&name` in Fortran's `input/yelmo_phys_const.nml`,
the file Fortran reads for `yelmo.phys_const`.
"""
function phys_constants(name::AbstractString="Earth")
    raw = parse_nml_file(phys_const_file())
    i = findfirst(gp -> gp.first == lowercase(name), raw)
    i === nothing && error("YelmoMirrorPar: no group &$(name) in $(phys_const_file()).")
    kvs = raw[i].second
    return NamedTuple(Symbol(k) => parse_nml_value(v) for (k, v) in kvs)
end

# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
"""
    YelmoMirrorParameters

Namelist for a `YelmoMirror`: the values that differ from Fortran's
`yelmo_defaults.nml` (`overrides`, group => key => value) plus any non-Yelmo
groups carried verbatim (`extra`, e.g. a driver's `&ctrl`).

Group access falls back to the Fortran defaults: `p.ydyn.solver` returns the
override if set, else the default. `p.phys` returns the constants of
`p.yelmo.phys_const` from Fortran's `yelmo_phys_const.nml`.
"""
struct YelmoMirrorParameters
    name      :: String
    overrides :: Dict{String, Dict{String, Any}}
    extra     :: RawGroups
end

"""
    YelmoMirrorParameters(name; extra=[], group=(key=value, ...), ...)

Build a parameter set from overrides, one keyword per Yelmo group. Each group is
a `NamedTuple` or `AbstractDict` of key => value. Unknown groups or keys, and
values whose type does not match the Fortran default, are errors.

# Example
```julia
p = YelmoMirrorParameters("experiment1";
    yelmo = (domain = "Greenland", grid_name = "GRL-16KM"),
    ydyn  = (solver = "ssa",),
)
write_nml("run.nml", p)
```
"""
function YelmoMirrorParameters(name::AbstractString; extra::RawGroups=RawGroups(), groups...)
    p = YelmoMirrorParameters(String(name), Dict{String, Dict{String, Any}}(), extra)
    for (g, kv) in groups
        _set!(p, String(g), kv)
    end
    return p
end

"""
    YelmoMirrorParameters(p::YelmoMirrorParameters; name=p.name, group=(...), ...)

Copy of `p` with further overrides applied.
"""
function YelmoMirrorParameters(p::YelmoMirrorParameters; name::AbstractString=p.name, groups...)
    q = YelmoMirrorParameters(String(name),
                              Dict(g => copy(d) for (g, d) in p.overrides),
                              copy(p.extra))
    for (g, kv) in groups
        _set!(q, String(g), kv)
    end
    return q
end

"""
    YelmoMirrorParameters(filename, name)

Read the namelist `filename` (see [`read_nml`](@ref)) and label it `name`.
"""
YelmoMirrorParameters(filename::AbstractString, name::AbstractString) =
    YelmoMirrorParameters(read_nml(filename); name)

_pairs(kv::NamedTuple) = pairs(kv)
_pairs(kv) = kv

function _set!(p::YelmoMirrorParameters, group::String, kv)
    d = get!(p.overrides, group, Dict{String,Any}())
    for (k, v) in _pairs(kv)
        key = String(k)
        _check_key(group, key, v)
        d[key] = v
    end
    return p
end

# Group view with fallback to the Fortran defaults.
struct MirrorGroup
    group     :: String
    overrides :: Dict{String, Any}
end

function Base.getproperty(p::YelmoMirrorParameters, s::Symbol)
    s in fieldnames(YelmoMirrorParameters) && return getfield(p, s)
    s === :phys && return phys_constants(p.yelmo.phys_const)
    g = String(s)
    is_yelmo_group(g) || error("YelmoMirrorParameters has no group &$(g).")
    return MirrorGroup(g, get(getfield(p, :overrides), g, Dict{String,Any}()))
end

Base.propertynames(::YelmoMirrorParameters) =
    (fieldnames(YelmoMirrorParameters)..., :phys, Symbol.(schema().groups)...)

function Base.getproperty(g::MirrorGroup, s::Symbol)
    s in fieldnames(MirrorGroup) && return getfield(g, s)
    key = String(s)
    ov = getfield(g, :overrides)
    haskey(ov, key) && return ov[key]
    d = schema().defaults[getfield(g, :group)]
    haskey(d, key) || error("&$(getfield(g, :group)) has no parameter `$(key)`.")
    return d[key]
end

Base.propertynames(g::MirrorGroup) = Tuple(Symbol.(schema().keys[getfield(g, :group)]))

"""
    effective(p, group) -> Dict{String,Any}

All parameters of `group`: Fortran defaults overlaid with the overrides of `p`.
"""
effective(p::YelmoMirrorParameters, group::String) =
    merge(schema().defaults[group], get(p.overrides, group, Dict{String,Any}()))

function Base.show(io::IO, ::MIME"text/plain", p::YelmoMirrorParameters)
    println(io, "YelmoMirrorParameters \"$(p.name)\" (overrides of $(defaults_file()))")
    for g in schema().groups
        ov = get(p.overrides, g, nothing)
        (ov === nothing || isempty(ov)) && continue
        println(io, "  &$(g)")
        for k in schema().keys[g]
            haskey(ov, k) && println(io, "    $(rpad(k, 20)) = $(format_value(ov[k]))")
        end
    end
    isempty(p.extra) || println(io, "  extra groups: ", join(first.(p.extra), ", "))
end

# ---------------------------------------------------------------------------
# Translation from the pure-Julia YelmoParameters
# ---------------------------------------------------------------------------

"""
    MIRROR_DIVERGENT_YELMO

`&yelmo` parameters whose meaning or valid options differ between the
pure-Julia `YelmoModel` timestepping and the Fortran (Mirror)
timestepping. `to_mirror` never copies these from a `YelmoParameters`
into the generated Mirror configuration — the Mirror keeps the Fortran
defaults (e.g. `pc_method = "AB-SAM"`, since Julia's
"HEUN" ≠ Fortran's "HEUN" despite the shared name). The shared adaptive
controls (`dt_method`, `dt_min`, `cfl_max`) are NOT in
this set — they have identical meaning on both backends and are copied.

Setting any parameter listed here to a non-default value on the Julia
side and then requesting a Mirror translation is an error: the intent
cannot be honored on the Mirror backend, so it must be configured
explicitly on the Mirror side instead.
"""
const MIRROR_DIVERGENT_YELMO = (
    :pc_method, :pc_controller, :pc_use_H_pred, :pc_filter_vel,
    :pc_n_redo, :pc_tol, :pc_eps,
)

"""
    to_mirror(p::YelmoParameters) -> YelmoMirrorParameters

Translate a pure-Julia `YelmoParameters` (the canonical configuration
for a `YelmoModel`) into a `YelmoMirrorParameters` for the Fortran
backend. Every namelist entry of a `YelmoParameters` group (as written
by `write_nml`) that is a parameter of the Fortran group of the same
name is copied — `ydyn.ssa_solver::SSASolver` becomes Fortran's
`ssa_solver` string; Julia-only keys (`YelmoPar.JULIA_ONLY_KEYS`) are
skipped, and backend-divergent timestepping options
(`MIRROR_DIVERGENT_YELMO`) are left at the Fortran defaults.

Errors if any divergent parameter was changed away from its
`YelmoParameters` default, since that intent cannot be carried to the
Mirror backend — configure it on the Mirror side explicitly instead.
"""
function to_mirror(p::YelmoPar.YelmoParameters)
    jdef = YelmoPar.YelmoParameters("")   # all-default reference
    violations = Symbol[]
    for f in MIRROR_DIVERGENT_YELMO
        if getfield(p.yelmo, f) != getfield(jdef.yelmo, f)
            push!(violations, f)
        end
    end
    isempty(violations) || error(
        "to_mirror: these &yelmo parameters are backend-divergent " *
        "(pure-Julia vs Fortran timestepping) and cannot be translated " *
        "to the Mirror backend: $(join(violations, ", ")). Leave them at " *
        "their YelmoParameters defaults — the Mirror uses its own " *
        "Fortran-native values. See `MIRROR_DIVERGENT_YELMO`.")

    s = schema()
    overrides = Dict{String, Dict{String, Any}}()
    for gname in fieldnames(YelmoPar.YelmoParameters)
        g = String(gname)
        haskey(s.defaults, g) || continue
        jgroup = getfield(p, gname)
        d = Dict{String, Any}()
        for f in fieldnames(typeof(jgroup))
            g == "yelmo" && f in MIRROR_DIVERGENT_YELMO && continue
            for (key, v) in YelmoPar._nml_entries(f, getfield(jgroup, f))
                haskey(s.defaults[g], key) || continue
                _check_key(g, key, v)
                d[key] = v
            end
        end
        isempty(d) || (overrides[g] = d)
    end
    return YelmoMirrorParameters(p.name, overrides, RawGroups())
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
format_value(v::Integer)           = string(v)
format_value(v::AbstractFloat)     = _fmt_float(Float64(v))
format_value(v::AbstractVector{<:AbstractString}) = join(["\"$s\"" for s in v], " ")
format_value(v::AbstractVector{<:Real})            = join(format_value.(v), ", ")
"""
    _fmt_float(x) -> String
Shortest representation that reads back to exactly `x` (Julia's `repr`,
e.g. `0.1`, `1.0e7`, `3.168808781402895e-11`); Fortran reads all of these.
"""
_fmt_float(x::Float64) = repr(x)

# An integer-valued float for an integer parameter is written as an integer, which
# Fortran's integer read requires (e.g. a Julia Float64 field copied by `to_mirror`).
_as_default_type(::Integer, v::AbstractFloat) = isinteger(v) ? Int(v) : v
_as_default_type(default, v) = v

function _write_group(io::IO, group::AbstractString, kvs)
    println(io, "&$(group)")
    for (k, v) in kvs
        println(io, "    $(rpad(k, 20)) = $(v)")
    end
    println(io, "/\n")
end

"""
    write_nml(filename, p::YelmoMirrorParameters; overwrite=false)

Write the namelist of `p`: the Yelmo groups that have overrides (keys in the
order of `yelmo_defaults.nml`), then the extra groups verbatim. Fortran takes
every key not written here from `yelmo_defaults.nml`.
"""
function write_nml(filename::AbstractString, p::YelmoMirrorParameters; overwrite::Bool=false)
    if isfile(filename) && !overwrite
        error("File already exists: $(filename). Use overwrite=true to overwrite.")
    end
    s = schema()
    open(filename, "w") do io
        println(io, "! Overrides of $(abspath(defaults_file()))\n")
        for g in s.groups
            ov = get(p.overrides, g, nothing)
            (ov === nothing || isempty(ov)) && continue
            d = s.defaults[g]
            _write_group(io, g, (k => format_value(_as_default_type(d[k], ov[k]))
                                 for k in s.keys[g] if haskey(ov, k)))
        end
        for (g, kvs) in p.extra
            _write_group(io, g, kvs)
        end
    end
    @info "Namelist written to $(filename)"
    return nothing
end
function write_nml(p::YelmoMirrorParameters; rundir::String="", overwrite::Bool=false)
    filename = joinpath(rundir, p.name * ".nml")
    write_nml(filename, p; overwrite)
    return nothing
end

### READING NML FILES ###

"""
    parse_nml_file(filename) -> Vector{Pair{String, Vector{Pair{String,String}}}}

Low-level parser. Returns the groups in file order, each with its
`key => raw value string` pairs in file order. Group names are lowercased.
Handles line continuation, inline comments, and multi-line values.
"""
function parse_nml_file(filename::AbstractString)
    groups = RawGroups()
    current = nothing          # Vector{Pair{String,String}} of the open group
    current_key = nothing
    current_val = nothing

    flush!() = if current !== nothing && current_key !== nothing
        push!(current, current_key => String(strip(current_val)))
        current_key = current_val = nothing
    end

    for raw_line in eachline(filename)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, '!') && continue          # comment line

        # Strip inline comments (outside of quoted strings)
        line = _strip_inline_comment(line)
        isempty(line) && continue

        # &group_name
        if startswith(line, '&')
            flush!()
            current = Pair{String,String}[]
            push!(groups, lowercase(strip(line[2:end])) => current)
            continue
        end

        # End-of-group marker
        if line == "/" || line == "&end" || startswith(line, "/")
            flush!()
            current = nothing
            continue
        end

        current === nothing && continue

        # key = value  (possibly continued on next line via trailing comma)
        if occursin('=', line)
            flush!()
            idx = findfirst('=', line)
            current_key = String(strip(line[1:idx-1]))
            current_val = strip(line[idx+1:end])
        else
            # Continuation line: append to current value
            current_key !== nothing && (current_val *= " " * line)
        end
    end
    flush!()

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

"""
    parse_nml_value(s) -> Bool | Int | Float64 | String | Vector

Parse a raw namelist value. Quoted values are strings (several quoted tokens: a
`Vector{String}`), `True`/`False` (or `.true.`/`.false.`) are `Bool`, other
values are numbers (several tokens: a vector).
"""
function parse_nml_value(s::AbstractString)
    s = strip(s)
    if startswith(s, '"') || startswith(s, '\'')
        toks = [something(m[1], m[2]) for m in eachmatch(r"\"([^\"]*)\"|'([^']*)'", s)]
        return length(toks) == 1 ? String(toks[1]) : String.(toks)
    end
    lowercase(s) in ("true", ".true.", "t") && return true
    lowercase(s) in ("false", ".false.", "f") && return false
    toks = split(s, r"[\s,]+"; keepempty=false)
    nums = [_parse_number(t) for t in toks]
    return length(nums) == 1 ? nums[1] : [nums...]
end

function _parse_number(t::AbstractString)
    i = tryparse(Int, t)
    i !== nothing && return i
    x = tryparse(Float64, replace(t, r"[dD]" => "e"))   # Fortran D-exponent
    x === nothing && error("YelmoMirrorPar: cannot parse namelist value `$(t)`.")
    return x
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    read_nml(filename) -> YelmoMirrorParameters

Read a Yelmo namelist file. Every key of a Yelmo group becomes an override and
is checked against `yelmo_defaults.nml` (unknown keys are an error, as in
Fortran's `nml_validate`); non-Yelmo groups are kept verbatim in `extra`.

# Example
```julia
p = read_nml("run.nml")
println(p.ydyn.solver)
```
"""
function read_nml(filename::AbstractString)
    p = YelmoMirrorParameters(splitext(basename(filename))[1])   # name = stem of filename
    for (g, kvs) in parse_nml_file(filename)
        if is_yelmo_group(g)
            _set!(p, g, (k => parse_nml_value(v) for (k, v) in kvs))
        else
            push!(p.extra, g => kvs)
        end
    end
    return p
end


## Comparison

function Base.:(==)(a::YelmoMirrorParameters, b::YelmoMirrorParameters)
    all(effective(a, g) == effective(b, g) for g in schema().groups) || return false
    return a.extra == b.extra
end

"""
    compare([io,] p1, p2; include_name=false)

Print all parameters that differ between `p1` and `p2` (with Fortran defaults
filled in), grouped by namelist group. Identical groups are skipped entirely.
"""
function compare(io::IO, p1::YelmoMirrorParameters, p2::YelmoMirrorParameters; include_name=false)
    any_diff = false
    if include_name && p1.name != p2.name
        any_diff = true
        println(io, "  $(rpad("name", 24))  \"$(p1.name)\"  =>  \"$(p2.name)\"\n")
    end
    s = schema()
    for g in s.groups
        e1, e2 = effective(p1, g), effective(p2, g)
        e1 == e2 && continue
        any_diff = true
        println(io, "&$(g)")
        for k in s.keys[g]
            e1[k] == e2[k] && continue
            println(io, "  $(rpad(k, 24))  $(format_value(e1[k]))  =>  $(format_value(e2[k]))")
        end
        println(io, "/\n")
    end
    if p1.extra != p2.extra
        any_diff = true
        println(io, "extra groups differ: ", join(first.(p1.extra), ", "), "  =>  ",
                join(first.(p2.extra), ", "))
    end
    any_diff || println(io, "(no differences)")
    return nothing
end

compare(p1::YelmoMirrorParameters, p2::YelmoMirrorParameters; kw...) = compare(stdout, p1, p2; kw...)

end # module YelmoMirrorPar
