## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# MISMIP3D Stnd + ATT-ramp qualitative grounding-line migration test.
#
# Companion to `test_mismip3d_stnd.jl` and `test_mismip3d_stnd_lockstep.jl`.
# The Stnd test exercises a 500-yr buildup. This test extends with a
# rate-factor (Glen `A`) up-and-down ramp, compressed from the literal
# Pattyn-2017 protocol implemented in
# `yelmo/tests/yelmo_mismip_new.f90:148-247` so it stays CI-friendly:
#
#   - **Phase 0 (Stnd equilibration)**: t = 0  → 500   at A_0   (baseline,
#     `rf_const = 3.1536e-18` Pa^-3 yr^-1, matching the existing Stnd
#     tests).
#   - **Phase 1 (perturbation)**:        t = 500 → 1000 at A_low (10×
#     stiffer, `rf_const = 3.1536e-19`). Stiffer ice → slower SSA shelf
#     flow → less ice exit → grounded margin **advances** downstream.
#   - **Phase 2 (recovery)**:            t = 1000 → 1500 at A_0 again.
#     The over-extended margin from Phase 1 should **retreat** back
#     toward the baseline equilibrium under restored softer-ice flow.
#
# Compared to the literal Fortran 3-step protocol (15 kyr Stnd
# equilibration + three 2-kyr ATT phases at A ∈ [1e-16, 1e-17, 1e-16]
# Pa^-3 yr^-1), this test:
#   - reuses the established 500-yr Stnd window (well validated by
#     `test_mismip3d_stnd.jl` and PR #31's lockstep)
#   - perturbs around the existing baseline `rf_const = 3.1536e-18`
#     rather than the Pattyn `1e-16` (which would require its own
#     equilibration phase)
#   - uses 500 yr per ramp phase (vs Fortran's 2000 yr) and skips the
#     Fortran `dHidt_rms < 1e-2 m/yr` convergence gate
#
# All these compressions are scoped at the **test** layer; the
# benchmark struct (`MISMIP3DBenchmark(:Stnd)`), boundary forcing,
# friction setup, and SSA solver wiring are all unchanged.
#
# Phase mutation mechanism: between time loops, refill the model's
# 3D `mat.ATT` field with the new constant `rf_const`. With `rf_method
# = 0` (constant rate factor) Yelmo.jl never recomputes ATT internally,
# so the refill survives untouched.
#
# Assertions (qualitative, no YelmoMirror lockstep yet — that lands in
# a follow-up commit):
#   1. GL position **advances** in Phase 1 (A ↓) vs end-of-Stnd.
#   2. GL position **retreats** in Phase 2 (A ↑) vs end-of-Phase-1 —
#      the unambiguous A-dynamics signal, since under steady Stnd
#      buildup the GL only ever advances. Retreat means the over-
#      extended Phase 1 state is correcting back to the baseline-A
#      equilibrium.
#   3. All H_ice, ux_b values finite throughout.
#   4. Floating-shelf cell count > 0 in each phase (catches a
#      regression of the Phase 5 H_min_flt bug we just fixed).

using Test
using Statistics
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params

const _SPEC_ATT = MISMIP3DBenchmark(:Stnd; dx_km=16.0)

# Same model parameters as `test_mismip3d_stnd_lockstep.jl::_mismip3d_lockstep_params`
# — using a different baseline would invalidate the qualitative comparison
# against the established Stnd state at t=500.
function _mismip3d_att_params()
    return YelmoModelParameters("mismip3d_stnd_att";
        ydyn = ydyn_params(
            solver         = "ssa",
            visc_method    = 1,
            beta_method    = 4,
            beta_q         = 1.0/3.0,
            beta_u0        = 1.0,
            beta_gl_scale  = 0,
            beta_gl_stag   = 3,
            beta_min       = 0.0,
            ssa_lat_bc     = "floating",
            eps_0          = 1e-6,
            taud_lim       = 1e6,
            ssa_solver     = SSASolver(precond         = :jacobi,
                                       picard_tol      = 1e-3,
                                       picard_iter_max = 20,
                                       picard_relax    = 0.7,
                                       rtol            = 1e-6,
                                       itmax           = 500),
        ),
        yneff = yneff_params(method = 0, const_ = 1.0),
        ytill = ytill_params(method = -1),
        ymat  = ymat_params(
            n_glen     = 3.0,
            rf_const   = 3.1536e-18,
            visc_min   = 1e3,
            de_max     = 0.5,
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
        ytopo = ytopo_params(),
    )
end

# Build the model from the analytical t=0 IC plus the Fortran-style
# thicker-grounded override (matches the lockstep test exactly).
function _build_mismip3d_att_model(b::MISMIP3DBenchmark, p)
    Nx, Ny = length(b.xc), length(b.yc)
    y = YelmoModel(b, 0.0; p=p, boundaries = :periodic_y)

    fill!(interior(y.mat.ATT),    p.ymat.rf_const)
    fill!(interior(y.dyn.cb_ref), b.cf_ref)

    H_int     = interior(y.tpo.H_ice)
    z_bed_int = interior(y.bnd.z_bed)
    @inbounds for j in 1:Ny, i in 1:Nx
        zb = z_bed_int[i, j, 1]
        H_int[i, j, 1] = (zb < b.z_bed_floor) ? 0.0 : max(0.0, 1000.0 - 0.9 * zb)
    end
    Yelmo.update_diagnostics!(y)
    return y
end

# Return the easternmost grounded-cell index along centerline j. The
# domain is grounded west, ocean east, so we look for the largest i
# where `f_grnd > 0.5`. Returns 0 if no grounded cells (sanity guard).
function _gl_index_centerline(y; j::Int=4)
    Fg = interior(y.tpo.f_grnd)[:, :, 1]
    Nx = size(Fg, 1)
    last_grnd = 0
    @inbounds for i in 1:Nx
        Fg[i, j] > 0.5 && (last_grnd = i)
    end
    return last_grnd
end

# Run `n_steps` of dt=1 forward Euler holding the current model state.
function _run_phase!(y, n_steps::Int; dt::Float64=1.0)
    for _ in 1:n_steps
        Yelmo.step!(y, dt)
    end
    return y
end

# One-line state summary at the end of a phase.
function _phase_summary(y; j::Int=4, label::AbstractString="")
    H  = interior(y.tpo.H_ice)[:, :, 1]
    Fg = interior(y.tpo.f_grnd)[:, :, 1]
    Ux = interior(y.dyn.ux_b)[:, :, 1]
    n_grnd  = count(>(0.5),  Fg)
    n_shelf = count(i -> H[i] > 0 && Fg[i] <= 0.5, eachindex(H))
    return (
        label    = label,
        time     = y.time,
        gl_idx   = _gl_index_centerline(y; j=j),
        max_H    = maximum(H),
        mean_H   = mean(H),
        max_ux   = maximum(abs, Ux),
        n_grnd   = n_grnd,
        n_shelf  = n_shelf,
        all_finite = all(isfinite, H) && all(isfinite, Ux),
    )
end

@testset "benchmarks: MISMIP3D Stnd + ATT-ramp qualitative GL migration" begin
    b = _SPEC_ATT
    p = _mismip3d_att_params()

    rf_baseline = p.ymat.rf_const          # 3.1536e-18  Pa^-3 yr^-1
    rf_low      = p.ymat.rf_const * 0.1    # 3.1536e-19  (10× stiffer)

    # =====================================================================
    # Phase 0 — Stnd equilibration (0 → 500 yr at baseline A).
    # =====================================================================
    y = _build_mismip3d_att_model(b, p)
    _run_phase!(y, 500)
    s0 = _phase_summary(y; label = "Phase 0 (Stnd, A_0)")
    @info "ATT-ramp Phase 0 end-state: t=$(s0.time) gl_idx=$(s0.gl_idx) " *
          "max_H=$(round(s0.max_H; digits=1)) mean_H=$(round(s0.mean_H; digits=1)) " *
          "max_ux=$(round(s0.max_ux; digits=2)) n_grnd=$(s0.n_grnd) n_shelf=$(s0.n_shelf)"

    @test s0.all_finite
    @test s0.gl_idx > 0
    @test s0.n_shelf > 0    # regression guard for the Phase 5 H_min_flt bug

    # =====================================================================
    # Phase 1 — A reduced 10× (500 → 1000 yr).
    # Stiffer ice → slower SSA shelf flow → GL advances downstream.
    # =====================================================================
    fill!(interior(y.mat.ATT), rf_low)
    _run_phase!(y, 500)
    s1 = _phase_summary(y; label = "Phase 1 (A_low)")
    @info "ATT-ramp Phase 1 end-state: t=$(s1.time) gl_idx=$(s1.gl_idx) " *
          "max_H=$(round(s1.max_H; digits=1)) mean_H=$(round(s1.mean_H; digits=1)) " *
          "max_ux=$(round(s1.max_ux; digits=2)) n_grnd=$(s1.n_grnd) n_shelf=$(s1.n_shelf)"

    @test s1.all_finite
    # GL must have advanced by at least one cell. Stronger thresholds
    # are sensitive to the exact SSA solver state; one cell is the
    # smallest discrete signal we can reliably resolve at dx = 16 km.
    @test s1.gl_idx > s0.gl_idx
    @test s1.n_shelf > 0

    # =====================================================================
    # Phase 2 — A restored to baseline (1000 → 1500 yr).
    # Recovery from Phase 1 over-extension. Under steady Stnd buildup
    # the GL only ever advances; a *retreat* here is the unambiguous
    # A-dynamics signal.
    # =====================================================================
    fill!(interior(y.mat.ATT), rf_baseline)
    _run_phase!(y, 500)
    s2 = _phase_summary(y; label = "Phase 2 (A_0 recovery)")
    @info "ATT-ramp Phase 2 end-state: t=$(s2.time) gl_idx=$(s2.gl_idx) " *
          "max_H=$(round(s2.max_H; digits=1)) mean_H=$(round(s2.mean_H; digits=1)) " *
          "max_ux=$(round(s2.max_ux; digits=2)) n_grnd=$(s2.n_grnd) n_shelf=$(s2.n_shelf)"

    @test s2.all_finite
    # The unambiguous test: GL retreats in Phase 2 vs Phase 1.
    @test s2.gl_idx < s1.gl_idx
    @test s2.n_shelf > 0

    @info "ATT-ramp GL trajectory (centerline j=4): " *
          "t=500 → idx=$(s0.gl_idx); t=1000 (A↓) → idx=$(s1.gl_idx) " *
          "(advanced by $(s1.gl_idx - s0.gl_idx) cells); " *
          "t=1500 (A↑) → idx=$(s2.gl_idx) " *
          "(retreated by $(s1.gl_idx - s2.gl_idx) cells)"

    # =====================================================================
    # Lockstep against YelmoMirror reference at t=1500.
    #
    # Fixture is produced by `regenerate.jl mismip3d_stnd_att --overwrite`,
    # which drives YelmoMirror through the same 3-phase rate-factor
    # ramp (see test/benchmarks/regenerate.jl::MISMIP3D_ATT_RF). Same
    # 50% smoke tolerance as the Stnd lockstep (PR #31, then PR #32).
    # =====================================================================
    fixture_path = abspath(joinpath(@__DIR__, "fixtures",
                                    "mismip3d_stnd_att_t1500.nc"))
    @assert isfile(fixture_path) "MISMIP3D ATT fixture missing: $fixture_path. " *
        "Run `julia --project=test test/benchmarks/regenerate.jl mismip3d_stnd_att --overwrite` first."

    y_ref = YelmoModel(fixture_path, 1500.0;
                       alias  = "mismip3d_stnd_att_ref",
                       p      = p,
                       boundaries = :periodic_y,
                       groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                       strict = false)

    @test y_ref isa AbstractYelmoModel
    @test y_ref.time == 1500.0

    Nx, Ny = length(b.xc), length(b.yc)
    @test size(interior(y_ref.tpo.H_ice))   == (Nx, Ny, 1)
    @test size(interior(y_ref.dyn.ux_b))    == (Nx + 1, Ny, 1)

    ref_H  = interior(y_ref.tpo.H_ice)[:, :, 1]
    jl_H   = interior(y.tpo.H_ice)[:, :, 1]
    ref_fg = interior(y_ref.tpo.f_grnd)[:, :, 1]
    jl_fg  = interior(y.tpo.f_grnd)[:, :, 1]
    ref_ux = interior(y_ref.dyn.ux_b)[:, :, 1]
    jl_ux  = interior(y.dyn.ux_b)[:, :, 1]

    rel_H  = maximum(abs.(jl_H  .- ref_H))  / max(maximum(abs.(ref_H)),  eps())
    abs_fg = maximum(abs.(jl_fg .- ref_fg))
    rel_ux = maximum(abs.(jl_ux .- ref_ux)) / max(maximum(abs.(ref_ux)), eps())

    @info "MISMIP3D Stnd ATT-ramp YelmoMirror lockstep at t=1500: " *
          "rel_H=$(round(rel_H; digits=4)) " *
          "abs_fg=$(round(abs_fg; digits=4)) " *
          "rel_ux=$(round(rel_ux; digits=4)) " *
          "ref max(|ux_b|)=$(round(maximum(abs.(ref_ux)); digits=2)) " *
          "jl  max(|ux_b|)=$(round(maximum(abs.(jl_ux));  digits=2)) " *
          "ref max(H)=$(round(maximum(ref_H); digits=2)) " *
          "jl  max(H)=$(round(maximum(jl_H);  digits=2)) " *
          "ref mean(f_grnd)=$(round(mean(ref_fg); digits=4)) " *
          "jl  mean(f_grnd)=$(round(mean(jl_fg);  digits=4))"

    @test rel_H  < 0.5
    @test abs_fg < 0.5
    @test rel_ux < 0.5
end
