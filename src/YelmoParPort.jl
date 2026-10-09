# Port status of Fortran Yelmo options in YelmoModel (included in module YelmoPar).
#
# `YelmoParameters` carries the full Fortran schema with Fortran defaults
# (yelmo dev, eda5462f). Options that YelmoModel does not implement yet are
# listed here; `check_ported` runs when a YelmoModel is built and rejects
# them, so a run never silently does something other than what its
# parameters say. Remove an entry when the option is ported.

"""
    PORTED_CHOICES

Fortran choice parameters whose options are only partly ported:
`(group, key) => (supported values, condition)`. `condition(p)` says when the
key is in effect (e.g. only with `use_lsf = true`); a value outside the
supported set is an error then. The first supported value is the one
`with_ported_options` picks.
"""
const PORTED_CHOICES = Dict{Tuple{Symbol,Symbol}, Tuple{Tuple, Function}}(
    (:yelmo,  :log_mb_check)       => ((false,),         p -> true),
    (:yelmo,  :write_metrics)      => ((false,),         p -> true),
    (:ytopo,  :front_subgrid)      => (("none",),        p -> true),
    (:ytopo,  :fmb_method)         => ((0, 1, 2),        p -> true),
    (:ycalv,  :lsf_method)         => (("redist",),      p -> p.ycalv.use_lsf),
    (:ydyn,   :uz_method)          => ((3,),             p -> true),
    (:ydyn,   :frz_scale)          => ((false,),         p -> true),
    (:ydyn,   :ssa_vel_lim_method) => (("clip",),        p -> true),
    (:ydyn,   :neff_nxi)           => ((0,),             p -> true),
    (:yhyd,   :method_til)         => ((1,),             p -> true),
    (:yhyd,   :method_transport)   => ((0,),             p -> true),
    # The Julia bucket saturates floating cells and their grounded margin ring.
    (:yhyd,   :bkt_floating_mode)  => ((1,),             p -> p.yhyd.method_til == 1),
    (:ytrc,   :use_tracer)         => ((false,),         p -> true),
    (:ytrc,   :use_elsa)           => ((false,),         p -> true),
    (:ytrc,   :t_dep_source)       => (("euler",),       p -> true),
    (:ytherm, :qb_method)          => ((4, 3),           p -> true),
    (:ytherm, :advecxy_order)      => ((1,),             p -> p.ytherm.method in ("enth", "temp")),
    (:ytherm, :basal_bc_method)    => (("wtil",),        p -> p.ytherm.method in ("enth", "temp")),
    (:ytherm, :gl_temperate)       => ((false,),         p -> p.ytherm.method in ("enth", "temp")),
)

"""
    NOT_PORTED_KNOBS

Fortran parameters that YelmoModel does not use yet and that have no setting
reproducing the current Julia behaviour (tuning of unported schemes).
`check_ported` warns once that they are ignored.
"""
const NOT_PORTED_KNOBS = Dict{Tuple{Symbol,Symbol}, String}(
    (:yelmo,  :mask_border)   => "border ice mask is set by the boundary conditions",
    (:ycalv,  :H_min_tau)     => "margin ice below H_min_* is removed in one step",
    (:ytherm, :advecxy_cfl)   => "no horizontal-advection sub-stepping",
    (:ytherm, :advecxy_nmax)  => "no horizontal-advection sub-stepping",
    (:ytherm, :enth_cp_method)=> "enthalpy uses cp(T)·T (neither \"const\" nor \"integral\")",
)

"""
    check_ported(p::YelmoParameters)

Error if `p` selects a Fortran option that YelmoModel does not implement
(`PORTED_CHOICES`); warn once about ignored parameters (`NOT_PORTED_KNOBS`).
Called when a `YelmoModel` is built.
"""
function check_ported(p::YelmoParameters)
    bad = String[]
    for ((g, k), (ok, active)) in PORTED_CHOICES
        active(p) || continue
        v = getfield(getfield(p, g), k)
        v in ok || push!(bad, "$(g).$(k) = $(repr(v)) (supported: $(join(repr.(ok), ", ")))")
    end
    isempty(bad) || error("YelmoModel: these Fortran Yelmo options are not ported to " *
                          "YelmoModel yet:\n  " * join(sort(bad), "\n  ") *
                          "\nSet the supported values (see YelmoPar.PORTED_CHOICES).")
    ignored = sort(["$(g).$(k): $(why)" for ((g, k), why) in NOT_PORTED_KNOBS])
    @info "YelmoModel ignores these Fortran parameters (not ported):\n  " *
          join(ignored, "\n  ") maxlog=1
    return nothing
end

"""
    with_ported_options(p::YelmoParameters) -> YelmoParameters

Copy of `p` with every option that `check_ported` rejects set to the first
supported value of `PORTED_CHOICES` (the current YelmoModel behaviour), and
the changes logged. For runs that need YelmoModel before those options are
ported; the result says what is actually simulated.
"""
function with_ported_options(p::YelmoParameters)
    changes = Dict{Symbol, Vector{Pair{Symbol,Any}}}()
    for ((g, k), (ok, active)) in PORTED_CHOICES
        active(p) || continue
        getfield(getfield(p, g), k) in ok && continue
        push!(get!(changes, g, Pair{Symbol,Any}[]), k => first(ok))
    end
    isempty(changes) && return p
    groups = Dict{Symbol,Any}(g => getfield(p, g) for g in GROUPS)
    for (g, kv) in changes
        old = groups[g]
        groups[g] = typeof(old)(; (f => getfield(old, f) for f in fieldnames(typeof(old)))..., kv...)
    end
    @info "with_ported_options: set " * join(sort(["$(g).$(k) = $(repr(v))" for (g, kv) in changes for (k, v) in kv]), ", ")
    q = YelmoParameters(p.name; groups...)
    # An option can become active through another change (none at present);
    # check that the result is complete.
    check_ported(q)
    return q
end
