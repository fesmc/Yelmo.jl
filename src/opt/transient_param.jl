"""
    optimize_transient_param(time, time1, time2, p1, p2, m) -> Float64

Interpolate a parameter as a function of time: `p1` before `time1`, `p2`
after `time2`, and a `m`-th power interpolation in between. Used to ramp
e.g. a relaxation timescale (`rel_tau1 -> rel_tau2` over
`rel_time1 -> rel_time2`) during a spin-up.

Port of `optimize_set_transient_param` in `ice_optimization.f90`.
"""
function optimize_transient_param(time::Real, time1::Real, time2::Real,
                                   p1::Real, p2::Real, m::Real)
    if time <= time1
        return Float64(p1)
    elseif time >= time2
        return Float64(p2)
    else
        return p1 + (p2 - p1) * ((time - time1) / (time2 - time1))^m
    end
end
