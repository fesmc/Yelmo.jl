# Greenland 16 km restart fixture of the unit tests (initmip-grl at t = 0,
# initialised by YelmoMirror on yelmo dev) and the YelmoParameters of that
# run. Regenerate with test/benchmarks/regen_grl16km_restart.jl.
const RESTART_PATH = joinpath(@__DIR__, "benchmarks", "fixtures", "grl16km_t0_restart.nc")
const NML_PATH     = joinpath(@__DIR__, "benchmarks", "fixtures", "grl16km_t0.nml")
