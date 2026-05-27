# InitMIP — Antarctica (`initmip-ant`)

This benchmark is documented together with its Greenland counterpart on
the single **InitMIP** page: see
[`../initmip-grl/README.md`](../initmip-grl/README.md), which covers the
shared pure-Julia-first design, how to run, backend selection, outputs,
and the Antarctica-specific domain/forcing section.

Quick start:

```bash
cd benchmarks/initmip-ant
julia --project=. -e 'include("run.jl"); main()'              # 20 yr, yelmo backend
julia --project=. -e 'include("run.jl"); main(t_end=1.0)'     # quick one-step check
```
