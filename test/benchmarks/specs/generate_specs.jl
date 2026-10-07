# Generate the benchmark spec namelists in this directory from the Fortran
# yelmo par files (yelmo/par/) plus the per-benchmark changes below, which set
# up the Julia-side lockstep (initial state from a Julia callback, fixed dt,
# solver settings matching YelmoModel). Rerun after the Fortran par files
# change; every Yelmo key is checked against yelmo/input/yelmo_defaults.nml.
#
#   julia --project=. test/benchmarks/specs/generate_specs.jl

using Yelmo
const MP = Yelmo.YelmoMirrorPar

const PARDIR  = joinpath(MP.YELMO_FORTRAN_DIR, "par")
const SPECDIR = @__DIR__

# Changes relative to the Fortran par file, group => (key => value).
# The CalvingMIP and EISMINT-moving changes date from the v1.15 specs; keys
# that yelmo dev renamed are translated (ssa_lis_opt -> ssa_solver="residual" +
# ssa_lis_opt_residual, &yneff const -> &yhyd bkt_N_closure=0 + const_N).
const LIS_TIGHT = "-i bicgsafe -p jacobi -maxiter 1000 -tol 1.0e-6 -initx_zeros false"

const CALVINGMIP = Dict(
    "yelmo" => Dict("domain" => "CALVINGMIP", "grid_name" => "CALVINGMIP",
                    "dt_method" => 0, "dt_min" => 0.1, "log_timestep" => false,
                    "nz_aa" => 5, "zeta_scale" => "linear", "pc_tol" => 5.0, "pc_eps" => 1.0),
    "ydyn"  => Dict("beta_method" => 4, "beta_q" => 0.3333333, "beta_gl_stag" => 3,
                    "taud_lim" => 1e6, "ssa_iter_max" => 20, "ssa_iter_conv" => 1e-3,
                    "ssa_solver" => "residual", "ssa_lis_opt_residual" => LIS_TIGHT),
    "ytill" => Dict("cf_min" => 3.165176e4, "cf_ref" => 3.165176e4),
    "ymat"  => Dict("de_max" => 0.5, "rf_const" => 3.1536e-18),
    "ytherm"=> Dict("method" => "fixed"),
)

const SPECS = [
    (spec = "yelmo_TROUGH.nml", par = "yelmo_TROUGH-F17.nml",
     header = """
     TROUGH-F17 benchmark, used by `TroughBenchmark` via `generate_fixture!`.
     Initial state from the Julia `_setup_trough_initial_state!` callback.""",
     changes = Dict("ctrl" => Dict("time_end" => 1000.0))),

    (spec = "yelmo_MISMIP3D.nml", par = "yelmo_MISMIP3D.nml",
     header = """
     MISMIP3D Stnd benchmark, used by `MISMIP3DBenchmark` via `generate_fixture!`.
     Initial state from the Julia `_setup_mismip3d_initial_state!` callback.
     Fixed dt = 1 yr (dt_method = 0) for lockstep with YelmoModel; residual SSA
     solver with a tight Lis tolerance.""",
     changes = Dict(
         "ctrl"  => Dict("experiment" => "Stnd", "dx" => 16.0, "dtt" => 1.0, "time_end" => 500.0),
         "yelmo" => Dict("dt_method" => 0),
         "ydyn"  => Dict("ssa_iter_conv" => 1e-3, "ssa_solver" => "residual",
                         "ssa_lis_opt_residual" => LIS_TIGHT))),

    (spec = "yelmo_EISMINT_moving.nml", par = "yelmo_EISMINT_moving.nml",
     header = """
     EISMINT-1 moving-margin benchmark, used by `EISMINT1MovingBenchmark` via
     `generate_fixture!`. Initial state from the Julia
     `_setup_eismint_moving_initial_state!` callback. HEUN timestepping (as
     YelmoModel), fixed temperature (ATT = rf_const). No sliding: the SIA
     solver has none (the v1.15 spec's ytill.method = -1 with cb_ref = 0 is
     rejected by yelmo dev).""",
     changes = Dict(
         "ctrl"  => Dict("time_end" => 25000.0),
         "yelmo" => Dict("zeta_scale" => "linear", "log_timestep" => false, "cfl_max" => 0.5,
                         "pc_method" => "HEUN", "pc_n_redo" => 5, "pc_tol" => 5.0, "pc_eps" => 1.0),
         "ycalv" => Dict("use_lsf" => false),
         "yhyd"  => Dict("bkt_N_closure" => 0, "const_N" => 1.0),
         "ymat"  => Dict("de_max" => 0.5),
         "ytherm"=> Dict("method" => "fixed"))),

    (spec = "yelmo_CalvingMIP_exp1.nml", par = "yelmo_calvingmip.nml",
     header = """
     CalvingMIP Exp1 (circular domain) benchmark, used by `CalvingMIPBenchmark`
     via `generate_fixture!`. Initial state from a Julia callback. Fixed dt
     (dt_method = 0) for lockstep with YelmoModel.""",
     changes = CALVINGMIP),

    (spec = "yelmo_CalvingMIP_exp2.nml", par = "yelmo_calvingmip.nml",
     header = """
     CalvingMIP Exp2 (oscillating calving front) benchmark. As Exp1, but the
     calving law is a Julia hook (calvmip_exp2!), so Fortran calving is off.""",
     changes = merge(CALVINGMIP, Dict("ycalv" => Dict("calv_flt_method" => "zero")))),
]

# Index of the first `!` outside a quoted string, or nothing.
function comment_start(line)
    inq = false
    for i in eachindex(line)
        line[i] == '"' && (inq = !inq)
        !inq && line[i] == '!' && return i
    end
    return nothing
end

# Replace the value of `key = value ! comment` lines, keeping the comment.
function apply_changes(lines, changes)
    out = String[]
    group = nothing
    todo = Dict(g => Set(keys(kv)) for (g, kv) in changes)
    for line in lines
        s = strip(line)
        if startswith(s, '&')
            group = lowercase(strip(s[2:end]))
        elseif startswith(s, '/') && group !== nothing
            for k in sort(collect(get(todo, group, Set{String}())))   # keys missing in the par file
                push!(out, "    $(rpad(k, 16)) = $(MP.format_value(changes[group][k]))")
            end
            delete!(todo, group)
            group = nothing
        elseif group !== nothing && haskey(changes, group) && occursin('=', s) && !startswith(s, '!')
            key = String(strip(split(s, '=')[1]))
            if haskey(changes[group], key)
                ic = comment_start(line)
                comment = ic === nothing ? "" : line[ic:end]
                eq = findfirst('=', line)
                line = rstrip(line[1:eq] * " " * MP.format_value(changes[group][key]) *
                              (isempty(comment) ? "" : "    " * comment))
                delete!(todo[group], key)
            end
        end
        push!(out, line)
    end
    leftover = filter(p -> !isempty(p.second), todo)
    isempty(leftover) || error("groups not found in par file: $(keys(leftover))")
    return out
end

for s in SPECS
    src = joinpath(PARDIR, s.par)
    lines = apply_changes(readlines(src), s.changes)
    hdr = ["! " * l for l in split(s.header, '\n')]
    append!(hdr, ["!", "! GENERATED by generate_specs.jl from yelmo/par/$(s.par) — do not edit;",
                  "! change the par file or the changes in generate_specs.jl and rerun.", ""])
    path = joinpath(SPECDIR, s.spec)
    write(path, join(vcat(hdr, lines), '\n') * '\n')
    MP.read_nml(path)   # validates every Yelmo key against yelmo_defaults.nml
    println("wrote $(path)")
end
