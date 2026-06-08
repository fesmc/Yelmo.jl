# ----------------------------------------------------------------------
# TroughBenchmark — Yelmo-side glue.
#
# The spec struct (`TroughBenchmark`) and the F17 closed-form bed
# (`_trough_f17_zbed`) live in IceSheetBenchmarks. This file adds:
#
#   - `_spec_name` / `_trough_fixture_path` — fixture-naming convention.
#   - `state(b, t)` — read the committed YelmoMirror fixture from disk.
#   - `apply_trough_f17_ic!` — fill `.bnd.*` / `.tpo.*` Fields on any
#     model object that exposes the canonical Yelmo paths (works for
#     both `YelmoMirror` and `YelmoModel`).
#   - `_setup_trough_initial_state!` — the YelmoMirror IC callback
#     consumed by `BenchmarkSpec.setup_initial_state!`.
#   - `write_fixture!(b, …)` — drive YelmoMirror through the F17 IC
#     setup to produce the reference NetCDF fixture.
# ----------------------------------------------------------------------

using IceSheetBenchmarks: IceSheetBenchmarks, TroughBenchmark,
                           _trough_f17_zbed

# Re-export the ISB-resident name so existing tests doing
# `using .YelmoBenchmarkHarness` still see `TroughBenchmark`.
export TroughBenchmark

# Yelmo Fortran namelist that YelmoMirror reads. Disables all file
# loads so the initial state comes from the Julia
# `_setup_trough_initial_state!` callback.
const _DEFAULT_TROUGH_NAMELIST = abspath(joinpath(@__DIR__, "specs",
                                                   "yelmo_TROUGH.nml"))
_trough_namelist_path(::TroughBenchmark) = _DEFAULT_TROUGH_NAMELIST

# Fixture-filename convention used by regenerate.jl.
_spec_name(b::TroughBenchmark) = "trough_$(lowercase(string(b.variant)))"

const _TROUGH_FIXTURES_DIR = abspath(joinpath(@__DIR__, "fixtures"))

function _trough_fixture_path(b::TroughBenchmark, t::Real;
                              fixtures_dir::AbstractString = _TROUGH_FIXTURES_DIR)
    return joinpath(fixtures_dir,
                    "$(_spec_name(b))_t$(Int(round(Float64(t)))).nc")
end

"""
    state(b::TroughBenchmark, t::Real) -> NamedTuple

Read the committed YelmoMirror fixture for `b` at time `t` and return
a NamedTuple whose keys map onto Yelmo schema variables (and are
routed into the appropriate component group by the generic
`YelmoModel(::AbstractBenchmark, t)` constructor provided by the
`IceSheetBenchmarks.YelmoBenchmarks` package extension).

Errors with a hint to run `regenerate.jl` if the fixture is missing.

Returned keys: `xc`, `yc`, plus whichever schema-Center fields the
fixture carries (`H_ice`, `z_bed`, `f_grnd`, `smb_ref`, `T_srf`,
`Q_geo`).

Face-staggered fields (`ux_b`, `uy_b`, `ux_bar`, `uy_bar`, `ATT`)
are *deliberately not* returned by `state()`: their on-disk layout
matches Yelmo Fortran's `(xc, yc)` cell-centred convention, but the
in-memory `YelmoModel` allocates them as `XFaceField`/`YFaceField`
with halo-included sizes `(Nx+1, Ny)` / `(Nx, Ny+1)`. Loading them
through `_assign_field!` would shape-mismatch. The regression test
loads the file-based `YelmoModel(restart_file, time)` for
face-staggered comparisons and uses `state(b, t)` only for the
Center-aligned state.

Yelmo Fortran restarts store Float32 — values are promoted to
Float64 here to match Yelmo.jl's Field eltypes.
"""
function state(b::TroughBenchmark, t::Real)
    path = _trough_fixture_path(b, t)
    isfile(path) || error(
        "TroughBenchmark.state: fixture missing at $path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl " *
        "$(_spec_name(b)) --overwrite` first.")

    NCDataset(path, "r") do ds
        out = Dict{Symbol,Any}(:xc => b.xc, :yc => b.yc)
        for name in ("H_ice", "z_bed", "f_grnd",
                     "smb_ref", "T_srf", "Q_geo")
            if haskey(ds, name)
                raw = ds[name][:, :, :]
                arr2d = ndims(raw) == 3 ? raw[:, :, 1] : raw
                out[Symbol(name)] = Array{Float64}(arr2d)
            end
        end
        return NamedTuple(out)
    end
end

"""
    apply_trough_f17_ic!(model, b::TroughBenchmark) -> model

Apply the Feldmann-Levermann F17 cold-start initial state to `model`'s
`.bnd` and `.tpo` subfields:

  - `z_bed`     ← F17 trough formula (`_trough_f17_zbed`).
  - `H_ice`     ← `50` m for `x ≤ x_cf`, `0` beyond the calving front.
  - `calv_mask` ← `1` where `x ≥ x_cf`, else `0`.
  - `z_sl`, `bmb_shlf`, `H_sed` ← `0`.
  - `T_shlf` ← `T0 = 273.15` K. `T_srf` ← `T0 + b.Tsrf_const`.
  - `smb_ref` ← `b.smb_const` (m/yr). `Q_geo` ← `b.Qgeo_const` (mW/m²).

Works on any model object that exposes Oceananigans `Field`s under the
canonical `.bnd.*` / `.tpo.*` paths — i.e. both YelmoMirror (used by
`_setup_trough_initial_state!` below as the BenchmarkSpec callback)
and a Yelmo.jl `YelmoModel` (used by `bench_diva_trough.jl` to drive
the cold-start trajectory). Caller responsibilities:

  - For `YelmoMirror`: call `yelmo_sync!` afterward so `init_state!`
    sees the new topography (handled by `_setup_trough_initial_state!`).
  - For `YelmoModel`: call `Yelmo.update_diagnostics!(y)` afterward to
    refresh `f_ice / f_grnd / mask_frnt` from the new H_ice / z_bed,
    and zero any pre-existing velocity history (`ux_b`, `uy_b`,
    `ux_bar`, `uy_bar`, `ux`, `uy`) since this helper does not touch
    velocity fields.

Mirrors the Fortran initial-state block at
`yelmo_trough.f90:206-211 + 112-119 + 237`.
"""
function apply_trough_f17_ic!(model, b::TroughBenchmark)
    Nx = length(b.xc)
    Ny = length(b.yc)

    z_bed     = zeros(Nx, Ny)
    H_ice     = zeros(Nx, Ny)
    calv_mask = zeros(Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        x_km = b.xc[i] * 1e-3
        y_km = b.yc[j] * 1e-3
        z_bed[i, j]     = _trough_f17_zbed(x_km, y_km,
                                            b.fc_km, b.dc_m, b.wc_km)
        H_ice[i, j]     = (x_km <= b.x_cf_km) ? 50.0 : 0.0
        calv_mask[i, j] = (x_km >= b.x_cf_km) ? 1.0  : 0.0
    end

    _assign_field!(model.bnd.z_bed,    z_bed)
    _assign_field!(model.tpo.H_ice,    H_ice)
    _assign_field!(model.bnd.calv_mask, calv_mask)

    fill!(interior(model.bnd.z_sl),     0.0)
    fill!(interior(model.bnd.bmb_shlf), 0.0)
    fill!(interior(model.bnd.H_sed),    0.0)

    T0 = 273.15
    fill!(interior(model.bnd.T_shlf),  T0)
    fill!(interior(model.bnd.T_srf),   T0 + b.Tsrf_const)
    fill!(interior(model.bnd.smb_ref), b.smb_const)
    fill!(interior(model.bnd.Q_geo),   b.Qgeo_const)

    return model
end

# YelmoMirror IC callback consumed by `BenchmarkSpec.setup_initial_state!`.
function _setup_trough_initial_state!(ymirror, b::TroughBenchmark, time::Real)
    apply_trough_f17_ic!(ymirror, b)
    yelmo_sync!(ymirror)
    return ymirror
end

"""
    write_fixture!(b::TroughBenchmark, path::AbstractString;
                   times = [1000.0]) -> Vector{String}

Drive YelmoMirror through the F17 initial-state setup and a single
end-time integration, writing the resulting restart NetCDF to `path`.

Single-time only — multi-time fixtures (a `time` dimension with
multiple snapshots) are deferred to a future milestone.

Returns a 1-element `Vector{String}` containing `path`.
"""
function write_fixture!(b::TroughBenchmark, path::AbstractString;
                        times::AbstractVector{<:Real} = [1000.0])
    length(times) == 1 ||
        error("write_fixture!(TroughBenchmark, …): multi-time fixtures " *
              "deferred to a future milestone (got $(length(times)) times).")
    t_out = Float64(first(times))

    namelist_path = _trough_namelist_path(b)
    isfile(namelist_path) || error(
        "TroughBenchmark.write_fixture!: namelist not found at " *
        "$(namelist_path).")

    spec = BenchmarkSpec(
        name           = _spec_name(b),
        namelist_path  = namelist_path,
        grid           = (xc = b.xc, yc = b.yc,
                          grid_name = "TROUGH-F17"),
        time_init      = 0.0,
        end_time       = t_out,
        output_times   = [t_out],
        dt             = 5.0,
        setup_initial_state! = (ymirror, t) ->
            _setup_trough_initial_state!(ymirror, b, t),
    )

    fixtures_dir = dirname(path)
    mkpath(fixtures_dir)
    isfile(path) && rm(path)

    paths = generate_fixture!(spec; fixtures_dir = fixtures_dir,
                                  overwrite = true)
    src = paths[1]
    if src != path
        mv(src, path; force = true)
        paths = [path]
    end
    return paths
end
