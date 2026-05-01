# Stepping the model

Both backends advance via a uniform interface:

```julia
init_state!(y::AbstractYelmoModel, time::Float64; kwargs...)
step!(y::AbstractYelmoModel, dt::Float64)
```

`init_state!` performs any one-time initialisation (e.g. populating
thermodynamics from a Robin profile in the mirror) and sets
`y.time = time`. `step!` advances the model by `dt` years and updates
`y.time`. Both functions return the model instance for chaining.

## Backend-specific behaviour

### `YelmoModel`

`step!(y::YelmoModel, dt)` calls each component's per-step function
in fixed phase order (currently only the topography component is
ported):

```julia
function step!(y::YelmoModel, dt::Float64)
    topo_step!(y, dt)
    # dyn_step!(y, dt)   — future milestone
    # mat_step!(y, dt)   — future milestone
    # therm_step!(y, dt) — future milestone
    return y
end
```

The topography step is described in detail on the
[topography page](../physics/topography.md). It is a 21-phase
predictor / corrector body that mirrors Fortran's `calc_ytopo_pc`,
applying each contribution to `mb_net` individually so the per-phase
realised tendency is recorded.

`init_state!(y::YelmoModel, time)` is currently a thin wrapper that
sets `y.time = time`; per-component initialisation will land as
component physics ports complete.

### `YelmoMirror`

`step!(y::YelmoMirror, dt)` is a `ccall` round-trip:

1. Push the Julia-side mirror state into the Fortran object via
   `yelmo_sync!(y)`.
2. Bump `y.time += dt`.
3. Call `yelmo_step` in Fortran, which advances the Fortran solver
   to the new time.
4. Pull all fields back into Julia.

So one `step!` is one full Fortran predictor / corrector step.

`init_state!(y::YelmoMirror, time; thrm_method="robin-cold")` syncs,
calls `yelmo_init_state` in Fortran (with the chosen thermodynamic
initialisation), and pulls the result back.

## A complete time loop

```julia
using Yelmo

y = YelmoModel("yelmo_restart.nc", 0.0;
    alias  = "demo",
    p      = YelmoModelParameters("demo"),
    groups = (:bnd, :dyn, :mat, :thrm, :tpo),
    strict = false,
)

init_state!(y, 0.0)
out = init_output(y, "demo.nc")

# 100-year integration with annual output.
dt = 1.0
T_end = 100.0
while y.time < T_end - 1e-9
    step!(y, dt)
    write_output!(out, y)
end

close(out)
```

The exact same loop body works against a `YelmoMirror` once the model
is built — no changes required.

## Mass-balance accounting

After each `step!(y::YelmoModel, dt)` finishes, the topography group
holds a complete record of the per-phase mass-balance contributions:

| Field | Meaning |
|---|---|
| `tpo.smb`      | Surface mass balance applied this step (m/yr) |
| `tpo.bmb`      | Combined basal mass balance |
| `tpo.fmb`      | Frontal mass balance at marine margins |
| `tpo.dmb`      | Subgrid discharge mass balance |
| `tpo.cmb`      | Calving mass balance |
| `tpo.mb_relax` | Relaxation tendency |
| `tpo.mb_resid` | Residual / cleanup tendency |
| `tpo.mb_net`   | Sum of the above |
| `tpo.dHidt`    | Total `(H - H_prev) / dt` |
| `tpo.dHidt_dyn`| Dynamic contribution (post-advection, pre-mass-balance) |

The closure `dHidt = dHidt_dyn + mb_net` holds to within
`apply_tendency!`'s tolerance and is verified in the integration
tests (slab conservation tests in `test_yelmo_topo.jl`).

## CFL and timestep choice

The pure-Julia advection kernel is **internally CFL-aware**: each
call to [`advect_tracer!`](@ref) sub-steps until the requested outer
`dt` is reached, with each sub-step bounded by

```math
\Delta t_\mathrm{sub} \le \mathrm{cfl\_safety} \cdot
\min\!\left(\frac{\Delta x}{|u|_\mathrm{max}},
            \frac{\Delta y}{|v|_\mathrm{max}}\right).
```

`cfl_safety` defaults to `y.p.yelmo.cfl_max` (Fortran's
`ytopo.cfl_max`, default `0.1`). So the caller can pass any outer
`dt` without thinking about stability — but a bigger outer `dt` means
more sub-steps, so the stable per-step throughput scales as
`1 / dt_outer`-cleared. There's no advantage to running with
`dt = 0.01` years over `dt = 1.0`; the latter just performs the
sub-stepping internally.

The Yelmo Fortran predictor / corrector wrapping is **not** yet ported
to the Julia side. Production runs that need PC adaptive time-stepping
should currently use the [`YelmoMirror`](@ref) backend.

## Logging time progress

`Yelmo.jl` does not emit per-step log lines by default — your loop is
responsible for any progress reporting. A typical pattern:

```julia
for k in 1:N
    step!(y, dt)
    write_output!(out, y)
    if k % 10 == 0
        @info "step $k" time=y.time max_H=maximum(interior(y.tpo.H_ice))
    end
end
```

If you need a deeper trace of the per-phase mass balance, inspect
`tpo.smb`, `tpo.bmb`, … directly between `step!` calls — they are
overwritten on every step but live in the model state until the next
`step!`.
