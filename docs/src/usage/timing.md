# Timing

Yelmo.jl ships with a lightweight, opt-in **per-section wall-clock
timer** that lets you measure where wall time is being spent inside
`step!` without modifying source code. It is the recommended starting
point when investigating performance — slower than expected, sudden
regressions, or deciding which kernel to optimise next.

## Quick start

Enable timing on the parameter object:

```julia
using Yelmo
using Yelmo.YelmoModelPar: YelmoModelParameters, yelmo_params

p = YelmoModelParameters("my_run";
    yelmo = yelmo_params(timing = true, dt_method = 2, pc_method = "HEUN", ...),
    # ...other groups...
)

y = YelmoModel(restart_file, 0.0; p = p)
```

Step the model as usual:

```julia
for _ in 1:n
    step!(y, dt)
end
```

Print the breakdown:

```julia
print_timings(y)
```

Sample output (EISMINT-1 moving-margin, HEUN+PI42, 5 outer × 100 yr):

```
section                       calls   total[s]   mean[ms]    max[ms]   %tot
────────────────────────────────────────────────────────────────────────────────
dyn                              30     0.9660     32.201     34.320   82.7
  dyn_jacobian_uxy               30     0.3922     13.075     15.507   33.6
  dyn_sia                        30     0.2531      8.436      9.263   21.7
  dyn_uz                         30     0.1110      3.699      4.603    9.5
  dyn_jacobian_uz                30     0.0800      2.668      3.541    6.9
  dyn_strain                     30     0.0474      1.580      2.522    4.1
topo                             30     0.1300      4.335      5.484   11.1
mat                              30     0.0716      2.388      3.421    6.1
  pc_corrector                   15     0.5883     39.218     40.695   50.4
  pc_predictor                   15     0.5795     38.630     40.169   49.6
────────────────────────────────────────────────────────────────────────────────
total (top-level)                       1.1677
```

## What gets measured

When `timing = true`, the model carries a `YelmoTimer` (accessible as
`y.timer`) and the following call sites populate it on every step:

| Section            | What it wraps                                                  |
|--------------------|----------------------------------------------------------------|
| `:topo`            | One full `topo_step!`                                          |
| `:dyn`             | One full `dyn_step!`                                           |
| `:mat`             | One full `mat_step!`                                           |
| `:pc_predictor`    | The HEUN predictor stage (one full FE pipeline)                |
| `:pc_corrector`    | The HEUN corrector stage (one full FE pipeline)                |
| `:dyn_sia`         | `calc_velocity_sia!` (SIA / hybrid solver branch)              |
| `:dyn_jacobian_uxy`| `calc_jacobian_vel_3D_uxyterms!` (Jacobian Step 1)             |
| `:dyn_uz`          | `calc_uz_3D_jac!` (vertical velocity from continuity)          |
| `:dyn_jacobian_uz` | `calc_jacobian_vel_3D_uzterms!` (Jacobian Step 2)              |
| `:dyn_strain`      | `calc_strain_rate_tensor_jac_quad3D!` (strain-rate tensor)     |

The list grows as new components are instrumented (`:therm` and
SSA-internal labels are planned). To add your own, see
[Extending the scaffold](#extending-the-scaffold) below.

### Reading the table

- **calls**: number of times the wrapped block was entered.
- **total[s]**: cumulative wall time spent inside the block.
- **mean[ms]**: total / calls.
- **max[ms]**: slowest single call.
- **%tot**: percentage of `total (top-level)`.
- **`total (top-level)`**: sum over sections that have no underscore in
  their name (`:topo`, `:dyn`, `:mat`, `:therm`). This is the total
  wall time spent inside instrumented top-level phases — i.e. the
  pure-physics budget of `step!`.

A few subtleties:

- **Sub-sections nest within their parent.** A row indented under
  `dyn` represents time *inside* `dyn`, so its `total[s]` is part of
  `dyn`'s `total[s]`. Don't sum sub-section totals with the parent.
  `dyn`'s "missing" % is unaccounted-for time inside `dyn` (e.g.
  driving stress, lateral BC, drag chain) — wrap those if you need
  to see them.
- **`pc_predictor` and `pc_corrector` are wrappers**, not phases —
  each one wraps a *whole* `_step_fe!` call. Their `%tot` is computed
  against the same top-level total, so they will appear larger than
  any single phase. They are listed separately at the bottom of the
  table to make this visible.

## When to enable it

- **Always-on** in long batch runs is fine — the per-section overhead
  is one `time_ns()` call plus a few `Dict` ops per wrapped block, on
  the order of a microsecond. Negligible against millisecond-scale
  phases.
- **For micro-benchmarks** (sub-millisecond kernels, hot inner
  loops), turn it off — the timer overhead becomes a meaningful
  fraction of the measurement.
- **Default is off**, so production runs that don't ask for it pay
  literally nothing: the `@timed_section` macro expands to the bare
  wrapped expression when `y.timer.enabled` is false.

## Reset

After spinup, you usually want to discard the warm-up timings before
measuring the production phase:

```julia
# Spin up — timings here are skewed by compilation, JIT, and IC
# transients.
for _ in 1:10
    step!(y, 100.0)
end

# Drop everything, keeping `timer.enabled = true`.
reset_timings!(y)

# Now the production phase.
for _ in 1:n
    step!(y, dt)
end

print_timings(y)
```

## Programmatic access

`y.timer` is a `YelmoTimer` with four `Dict{Symbol, Int}` fields
keyed by section name:

| Field      | Meaning                                                       |
|------------|---------------------------------------------------------------|
| `counts`   | number of calls                                               |
| `total_ns` | cumulative time in nanoseconds                                |
| `max_ns`   | slowest single call (ns)                                      |
| `last_ns`  | most recent call (ns)                                         |

Useful for in-run printing, custom roll-ups, or feeding numbers into
NetCDF / HDF5 output:

```julia
mean_dyn_ms = y.timer.total_ns[:dyn] / y.timer.counts[:dyn] / 1e6
@printf "  dyn (mean): %.2f ms\n" mean_dyn_ms
```

## Extending the scaffold

The macro `@timed_section y :section_name expr` records the wall time
of `expr` under `section_name` on `y.timer`. It expects a literal
`Symbol` for the section name — that keeps the disabled-branch
elision reliable.

To time a new region, add a wrap at the call site:

```julia
using Yelmo: @timed_section

@timed_section y :my_new_section do_something_expensive!(args...)
```

Section names that contain an underscore (e.g. `:dyn_my_new_section`)
are rendered as a sub-section under the prefix-matched parent
(`:dyn`) in `print_timings`. Use this to nest related instrumentation
under an existing phase.

### Recommendations

- Wrap **stable, well-named regions** — function calls, not arbitrary
  loop bodies. The section name is the contract you publish to
  whoever reads `print_timings` output.
- Avoid wrapping inside tight inner loops. The macro's branch is
  cheap but not free, and a wrap that fires 10⁶ times per
  step will distort the very thing you're trying to measure.
- Prefer one wrap per logical phase. If you need finer-grained data,
  drill in temporarily with extra wraps for a single profiling run
  rather than committing them.

## Caveats

- **First call includes compilation.** Whether timing is on or off,
  Julia compiles the relevant code paths the first time they are
  hit. Either run a warm-up step, or call `reset_timings!(y)` after
  the first `step!` before measuring.
- **`time_ns()` is wall clock.** It includes any GC pauses or OS
  preemption that occurred during the section. This is what you
  want for end-to-end performance work but it is not pure CPU time.
- **`max_ns` is per-call**, not per-step. A worst-case Picard
  iteration that re-runs SSA Krylov twice will inflate `max_ns` for
  `:dyn` even if the average is fine.
- **No threadsafety guarantees.** Sections are not designed to be
  entered from multiple threads in parallel. The model is
  single-threaded today; revisit when CPU-thread parallelism lands.

## API reference

```@docs
YelmoTimer
@timed_section
print_timings
reset_timings!
```
