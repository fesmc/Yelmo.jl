## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# YelmoParameters vs Fortran yelmo's `input/yelmo_defaults.nml` (via the
# `yelmo` symlink): same groups, same keys, same defaults. Julia-only keys
# must be declared in `YelmoPar.JULIA_ONLY_KEYS`. Fails when Fortran yelmo
# adds, removes, renames or changes the default of a parameter, so the
# Julia schema is updated together with the Fortran reference.

using Test
using Yelmo
const YP = Yelmo.YelmoPar
const MP = Yelmo.YelmoMirrorPar

# Julia write -> Fortran-style value, for comparison with the defaults file.
_nml_value(v) = MP.parse_nml_value(YP.format_value(v))

@testset "YelmoParameters schema = yelmo_defaults.nml" begin
    isfile(MP.defaults_file()) || error("Fortran defaults file not found: $(MP.defaults_file())")
    s = MP.schema()
    p = YelmoParameters("defaults")

    @test collect(String.(YP.GROUPS)) == s.groups

    for g in YP.GROUPS
        G = String(g)
        julia = Dict{String,Any}()
        for f in fieldnames(typeof(getfield(p, g)))
            for (k, v) in YP._nml_entries(f, getfield(getfield(p, g), f))
                julia[k] = v
            end
        end
        only_jl = keys(get(YP.JULIA_ONLY_KEYS, G, Dict()))
        @testset "&$G" begin
            missing_jl = setdiff(s.keys[G], keys(julia))
            @test isempty(missing_jl) || (@info "&$G: Fortran keys missing in Julia" missing_jl; false)
            undeclared = setdiff(keys(julia), s.keys[G], only_jl)
            @test isempty(undeclared) || (@info "&$G: Julia-only keys not in JULIA_ONLY_KEYS" undeclared; false)
            stale = setdiff(only_jl, keys(julia))
            @test isempty(stale) || (@info "&$G: JULIA_ONLY_KEYS entries without a field" stale; false)
            for k in intersect(s.keys[G], keys(julia))
                jv, fv = _nml_value(julia[k]), s.defaults[G][k]
                ok = jv isa Real && fv isa Real ? isapprox(jv, fv; rtol=1e-12) : isequal(jv, fv)
                @test ok || (@info "&$G.$k default differs" julia=jv fortran=fv; false)
            end
        end
    end
end

@testset "write_defaults_nml round trip" begin
    f = joinpath(mktempdir(), "yelmo_defaults.nml")
    write_defaults_nml(f)
    @test YP.read_nml(f) == YelmoParameters("yelmo_defaults")
    @test_throws ErrorException begin
        open(f, "a") do io; println(io, "&ydyn\n    no_such_key = 1\n/"); end
        YP.read_nml(f)
    end
end

@testset "check_ported" begin
    @test_throws ErrorException check_ported(YelmoParameters("defaults"))
end
