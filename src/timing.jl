# ----------------------------------------------------------------------
# Timing scaffold for Yelmo.jl
#
# A minimal, toggleable per-section wall-clock accumulator. Instrumentation
# is opt-in via `yelmo_params(timing = true)`. When the flag is `false`
# (the default), each `@timed_section` call site expands to a literal
# pass-through with no timer interaction, so production runs pay nothing.
#
# Public API:
#
#   - `YelmoTimer` (struct, see field docs below)
#   - `@timed_section y :section_name expr`
#   - `reset_timings!(y)`
#   - `print_timings(y; io=stdout)`
#
# Sections are referred to by `Symbol`s. Names containing an underscore
# are treated as sub-sections of the parent named by the leading prefix
# (e.g. `:dyn_jacobian_uxy` is a subsection of `:dyn`); this affects
# table indentation and how the totals row is computed.
#
# Sections are independent — wrapping a sub-section is decoupled from
# wrapping its parent. The parent's recorded time will include the
# sub-section's time only because both timers see the same elapsed
# wall-clock (the parent's section spans the inner call).
# ----------------------------------------------------------------------

"""
    YelmoTiming

Module hosting the per-section wall-clock timing scaffold: the
`YelmoTimer` accumulator, the `@timed_section` macro, and the
`print_timings` / `reset_timings!` helpers. See the user guide page
"Timing" for usage.
"""
module YelmoTiming

using Printf

export YelmoTimer, @timed_section, reset_timings!, print_timings

"""
    YelmoTimer(; enabled=false)

Per-model wall-clock accumulator for tagged code sections, populated by
[`@timed_section`](@ref). Stored on the model as `y.timer`.

When `enabled` is `false` the timer is created but no `@timed_section`
call site touches it. Toggle from a parameter file with
`yelmo_params(timing = true)` (or call the constructor directly).

Per-section state, keyed by `Symbol`:

  - `counts[s]`   — number of times the section was entered
  - `total_ns[s]` — cumulative elapsed time in nanoseconds
  - `max_ns[s]`   — slowest single call (nanoseconds)
  - `last_ns[s]`  — most recent single call (nanoseconds)

See also: [`reset_timings!`](@ref), [`print_timings`](@ref).
"""
mutable struct YelmoTimer
    enabled::Bool
    counts::Dict{Symbol, Int}
    total_ns::Dict{Symbol, Int}
    max_ns::Dict{Symbol, Int}
    last_ns::Dict{Symbol, Int}
end

YelmoTimer(; enabled::Bool=false) = YelmoTimer(
    enabled,
    Dict{Symbol,Int}(),
    Dict{Symbol,Int}(),
    Dict{Symbol,Int}(),
    Dict{Symbol,Int}(),
)

# Internal — write a single `(section, dt_ns)` sample into the timer.
# Always called inside the `enabled` branch of `@timed_section`.
@inline function _record!(t::YelmoTimer, section::Symbol, dt_ns::Integer)
    dt = Int(dt_ns)
    t.counts[section]   = get(t.counts, section, 0) + 1
    t.total_ns[section] = get(t.total_ns, section, 0) + dt
    t.max_ns[section]   = max(get(t.max_ns, section, 0), dt)
    t.last_ns[section]  = dt
    return nothing
end

"""
    reset_timings!(y)
    reset_timings!(t::YelmoTimer) -> t

Zero all section counters and totals on the timer. The `enabled` flag
is preserved.

Typical usage: call after model spinup so the timed window covers only
the production phase.
"""
function reset_timings!(t::YelmoTimer)
    empty!(t.counts)
    empty!(t.total_ns)
    empty!(t.max_ns)
    empty!(t.last_ns)
    return t
end
reset_timings!(y) = reset_timings!(y.timer)

"""
    @timed_section y :section expr

Record the wall time of `expr` under `:section` on `y.timer`, then
return the value of `expr`. When `y.timer.enabled` is `false`, the
expansion is just `expr` — no clock reads, no dict touches.

`section` must be a literal `Symbol` (i.e. `:foo`), not a runtime
expression. This keeps the section name a compile-time constant so the
disabled-branch elision is reliable.

Example:

    @timed_section y :dyn dyn_step!(y, dt)
"""
macro timed_section(y, section, expr)
    section isa QuoteNode ||
        error("@timed_section: section must be a Symbol literal " *
              "(e.g. `:dyn`); got `$(section)`")
    sec_value = section.value
    sec_value isa Symbol ||
        error("@timed_section: section literal must be a Symbol; " *
              "got `:$(sec_value)` of type $(typeof(sec_value))")

    record_fn = GlobalRef(@__MODULE__, :_record!)
    return quote
        let _y = $(esc(y))
            if _y.timer.enabled
                local _t0 = Base.time_ns()
                local _r  = $(esc(expr))
                $(record_fn)(_y.timer, $(QuoteNode(sec_value)),
                             Base.time_ns() - _t0)
                _r
            else
                $(esc(expr))
            end
        end
    end
end

# Heuristic: a section name with an underscore is rendered as a
# sub-section under the prefix before the first underscore.
_section_parent(s::Symbol) = begin
    str = String(s)
    idx = findfirst('_', str)
    idx === nothing ? nothing : Symbol(str[1:idx-1])
end
_is_subsection(s::Symbol) = _section_parent(s) !== nothing

"""
    print_timings(y; io=stdout)
    print_timings(t::YelmoTimer; io=stdout)

Pretty-print accumulated section timings as a table sorted by total
time. Sub-sections (those with an underscore in the name) are indented
under their prefix-matched parent.

Percentages are computed against the sum of top-level (non-sub-section)
totals — this avoids the double-counting that would happen if a parent
and its sub-sections were summed.
"""
function print_timings(t::YelmoTimer; io::IO=stdout)
    if !t.enabled
        println(io, "Timing is disabled. Set yelmo_params(timing = true) " *
                    "to record sections.")
        return
    end
    if isempty(t.counts)
        println(io, "No timing sections recorded.")
        return
    end

    # Top-level total ≡ sum of sections whose name has no underscore.
    # (Sub-sections nest inside their parent's wall time, so summing
    # everything would double-count.)
    grand_ns = sum(t.total_ns[s] for s in keys(t.counts) if !_is_subsection(s);
                   init = 0)

    # Order: sort top-level by total descending, then place each
    # parent's sub-sections directly underneath, also sorted by total.
    sections_top = sort([s for s in keys(t.counts) if !_is_subsection(s)];
                         by = s -> -t.total_ns[s])
    sections_sub = Dict{Symbol, Vector{Symbol}}()
    for s in keys(t.counts)
        _is_subsection(s) || continue
        parent = _section_parent(s)
        push!(get!(sections_sub, parent, Symbol[]), s)
    end
    for v in values(sections_sub)
        sort!(v; by = s -> -t.total_ns[s])
    end

    # Sub-sections whose parent is not itself a wrapped top-level
    # section (e.g. someone instruments only `:foo_bar` without `:foo`).
    # Show these as their own group at the bottom.
    orphans = Symbol[]
    for parent in keys(sections_sub)
        parent in sections_top && continue
        append!(orphans, sections_sub[parent])
    end

    @printf(io, "%-26s %8s %10s %10s %10s  %5s\n",
            "section", "calls", "total[s]", "mean[ms]", "max[ms]", "%tot")
    println(io, "─"^80)

    function _row(s, indent)
        n   = t.counts[s]
        tot = t.total_ns[s]
        mean_ms = (tot / n) / 1e6
        max_ms  = t.max_ns[s] / 1e6
        pct     = grand_ns > 0 ? 100 * tot / grand_ns : 0.0
        @printf(io, "%-26s %8d %10.4f %10.3f %10.3f  %5.1f\n",
                indent * String(s), n, tot / 1e9, mean_ms, max_ms, pct)
    end

    for s in sections_top
        _row(s, "")
        for sub in get(sections_sub, s, Symbol[])
            _row(sub, "  ")
        end
    end
    for s in orphans
        _row(s, "  ")
    end

    println(io, "─"^80)
    @printf(io, "%-26s %8s %10.4f\n", "total (top-level)", "", grand_ns / 1e9)
    return nothing
end

print_timings(y; io::IO=stdout) = print_timings(y.timer; io=io)

end # module YelmoTiming
