# Regenerate the Greenland 16 km restart fixture used by the unit tests
# (test/test_fixtures.jl): the initmip-grl benchmark (M17 topography, MAR
# forcing, robin-cold thermodynamics) initialised by YelmoMirror at t = 0
# and written with Fortran `yelmo_restart_write`. Output:
#
#   test/benchmarks/fixtures/grl16km_t0_restart.nc   (deflated)
#   test/benchmarks/fixtures/grl16km_t0.nml          (YelmoParameters of the run)
#
# Needs `libyelmo_c_api.so` and the initmip-grl data (and its `input`
# link to yelmo/input). Run from the repository root:
#   julia --project=test test/benchmarks/regen_grl16km_restart.jl

using Yelmo
using NCDatasets

const FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))
const RESTART_OUT  = joinpath(FIXTURES_DIR, "grl16km_t0_restart.nc")
const NML_OUT      = joinpath(FIXTURES_DIR, "grl16km_t0.nml")
const INITMIP_DIR  = abspath(joinpath(@__DIR__, "..", "..", "benchmarks", "initmip-grl"))

# Copy `src` to `dst` with every variable deflated (the Fortran restart is
# written uncompressed).
function deflate_copy(src::AbstractString, dst::AbstractString; level::Int = 4)
    NCDataset(src) do s
        NCDataset(dst, "c") do d
            for (k, v) in s.attrib
                d.attrib[k] = v
            end
            for (name, n) in s.dim
                defDim(d, name, name in unlimited(s.dim) ? Inf : n)
            end
            for name in keys(s)
                v = s[name]
                data = Array(v.var)
                attrib = [k => a for (k, a) in v.attrib if k != "_FillValue"]
                fill = get(v.attrib, "_FillValue", nothing)
                kw = fill === nothing ? (;) : (; fillvalue = fill)
                # Scalars (e.g. the grid mapping) cannot be compressed.
                isempty(dimnames(v)) || (kw = (; kw..., deflatelevel = level, shuffle = true))
                dv = defVar(d, name, eltype(v.var), dimnames(v); attrib = attrib, kw...)
                # Raw values (no CF transform); explicit ranges extend `time`.
                ndims(data) == 0 ? (dv.var[] = data[]) : (dv.var[axes(data)...] = data)
            end
        end
    end
    return dst
end

function main()
    cd(INITMIP_DIR)
    include(joinpath(INITMIP_DIR, "run.jl"))   # build_params, build_mirror
    p = Base.invokelatest(build_params, :mirror)
    rundir = mktempdir(; prefix = "grl16km_regen_")
    y = Base.invokelatest(build_mirror, p, rundir)

    raw = yelmo_write_restart!(y, joinpath(rundir, "yelmo_restart.nc"); time = 0.0)
    mkpath(FIXTURES_DIR)
    isfile(RESTART_OUT) && rm(RESTART_OUT)
    deflate_copy(raw, RESTART_OUT)
    write_nml(NML_OUT, p; overwrite = true)
    @info "fixture written" RESTART_OUT raw_MB = filesize(raw) / 2^20 MB = filesize(RESTART_OUT) / 2^20 NML_OUT
end

main()
