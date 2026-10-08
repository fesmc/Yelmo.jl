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

`step!(y::YelmoModel, dt)` runs the predictor-corrector time loop of
Fortran `yelmo_update` (`src/timestepping.jl`). Each internal step:

```julia
topo_step!(y, dt_now, PCPredictor(); β1, β2)   # H_pred
dyn_step!(y, dt_now)                           # velocity at H_pred
topo_step!(y, dt_now, PCCorrector(); β3, β4)   # H_corr (state back to H_n)
# truncation error from H_corr − H_pred; redo with a smaller dt if too large
mat_step!(y, dt_now)
therm_step!(y, dt_now)
topo_step!(y, dt_now, PCAdvance(); use_H_pred) # H_{n+1}
```

The β coefficients come from `yelmo.pc_method` (`"AB-SAM"`, `"HEUN"`,
`"FE-SBE"`). `yelmo.dt_method = 0` takes the whole `dt` as one step (more
if a step is redone); `dt_method = 2` chooses the steps with the PI
controller (`pc_controller`, `pc_eps`, `pc_tol`, `pc_n_redo`). The first
step of a model is `dt_min` (cold start), unless the model was loaded from
a file with the controller history `pc_dt`, `pc_eta` (see
[Restarts](io.md#restarts)): it then continues the trajectory. The
topography stages are described on the
[topography page](../physics/topography.md).

With `yelmo.log_timestep = true`, the model buffers one row per step and
`close(y.dyn.scratch.timestep_log[])` writes them to
`<rundir>/yelmo_timesteps.nc`, with the variables of the Fortran log:
`dt_now`, `dt_adv` (Courant), `dt_pi` (controller), `pc_eta`, `iter_redo`
(redos of the step), `speed`, `speed_tpo`, `speed_dyn` [kyr/hr],
`ssa_iter`, `ssa_lin_iter`/`ssa_lin_fail` (linear solves of the velocity
solve), `ssa_lim_n` (faces at `ssa_vel_max`) and
`adv_lin_iter`/`adv_lin_fail` (implicit advection solves). The first row
holds the controller state at the start.

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
    p      = YelmoParameters("demo"),
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
| `tpo.mb_net`   | Sum of the above except `cmb` |
| `tpo.dHidt`    | Total `(H - H_prev) / dt` |
| `tpo.dHidt_dyn`| Applied transport rate |
| `tpo.mb_clip`  | Clip of negative thickness after transport |
| `tpo.mb_err`   | Budget residual `dHidt − (dHidt_dyn + mb_clip + mb_net + cmb)` |

As in Fortran, the budget closes to round-off (`mb_err ≈ 0`), since
every tendency is applied with `apply_tendency!(…; adjust_mb = true)`;
the slab conservation tests in `test_yelmo_topo.jl` check it.

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
