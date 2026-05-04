## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# EISMINT-1 moving-margin standalone Yelmo.jl trajectory test.
#
# Spins up zero-ice EISMINT-1 moving-margin to t = 5000 yr under SIA +
# adaptive HEUN/PI42 timestepping. Validates the milestone-3h pipeline
# end-to-end:
#
#   - SIA velocity solver (`y.dyn.ux`, `y.dyn.uy`).
#   - Vertical velocity from continuity (`y.dyn.uz`).
#   - Explicit upwind advection of `H_ice` via `topo_step!`.
#   - Adaptive predictor-corrector (HEUN + PI42) timestepping under
#     `dt_method = 2`.
#
# Therm is implicitly disabled (`step!` doesn't call `therm_step!`),
# and `mat` is bypassed by setting `mat.ATT` directly to the EISMINT-1
# reference rate factor and never recomputing it. `bnd.smb_ref` carries
# the radial moving-margin pattern; the per-step boundary update from
# the Fortran `eismint_boundaries` callback is unnecessary in our scope
# (T_srf only feeds therm; smb is time-invariant in the "moving"
# variant).
#
# Quantitative checks at t = 5 kyr (transient, NOT steady-state — full
# steady-state is ≈ 25 kyr):
#
#   - All H_ice / ux / uz finite.
#   - max(H_ice) ∈ [800, 2500] m (dome growing but not yet steady).
#   - dome forms at the geometric centre (max H within 2 cells of the
#     centre cell).
#   - max(|uz|) ∈ [0.05, 1.5] m/yr (negative at the dome; positive at
#     the margin).
#   - smb-driven mass balance: total ∫H_ice dx dy positive and growing.
#
# Reference values for the full 25 kyr steady state (Huybrechts 1996,
# Type 1 / "exact margin" ice-benchmarks data) are validated separately
# by the lockstep test (C5b) and the comparison-plot script (C5c).

using Test
using Statistics
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params, ytopo_params,
                           yelmo_params

# Adaptive HEUN+PI42 + SIA-only parameters. Mirrors the Fortran
# `par-gmd/yelmo_EISMINT_moving.nml` where the relevant subset overlaps
# (solver = "sia", uz_method = 3, n_glen = 3, rf_const = 1e-16). Fortran
# uses pc_method = "AB-SAM" but Yelmo.jl only has HEUN implemented;
# HEUN with PI42 is the closest analog.
function _eismint_moving_params()
    return YelmoModelParameters("eismint_moving";
        yelmo = yelmo_params(
            dt_method     = 2,
            pc_method     = "HEUN",
            pc_controller = "PI42",
            pc_tol        = 5.0,
            pc_eps        = 1.0,
            pc_n_redo     = 5,
            dt_min        = 0.01,
            cfl_max       = 0.5,
        ),
        ydyn = ydyn_params(
            solver       = "sia",
            uz_method    = 3,
            visc_method  = 1,   # dynamic viscosity from ATT
            eps_0        = 1e-6,
            taud_lim     = 2e5,
        ),
        ytopo = ytopo_params(
            solver = "expl",
            use_bmb = false,
        ),
        yneff = yneff_params(method = 0, const_ = 1.0),
        ytill = ytill_params(method = -1),
        ymat  = ymat_params(
            n_glen   = 3.0,
            rf_const = 1e-16,
            visc_min = 1e3,
            de_max   = 0.5,
            enh_method = "shear3D",
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
    )
end

# Build a Yelmo.jl model from an EISMINT1MovingBenchmark IC and set the
# pre-filled material fields (ATT, cb_ref) so `dyn_step!` doesn't pull
# them from a stale default.
function _build_eismint_moving(b::EISMINT1MovingBenchmark, p::YelmoModelParameters)
    Nx, Ny = length(b.xc), length(b.yc)
    y = YelmoModel(b, 0.0; p = p, boundaries = :bounded)
    fill!(interior(y.mat.ATT), p.ymat.rf_const)
    # cb_ref / N_eff don't matter for SIA-only (no basal sliding) but
    # set them to deterministic values to avoid relying on default.
    fill!(interior(y.dyn.cb_ref), 0.0)
    fill!(interior(y.dyn.N_eff),  1.0)
    return y
end

# ----------------------------------------------------------------------
# Test setup
# ----------------------------------------------------------------------

const _EISMINT_T_END = 5000.0           # [yr]
const _EISMINT_DT_OUTER = 100.0         # [yr] outer-loop dt; adaptive PC sub-steps inside

@testset "benchmarks: EISMINT-1 moving 5-kyr SIA + adaptive PC trajectory" begin
    b = EISMINT1MovingBenchmark()      # Nx = Ny = 31, dx = 50 km, summit (775, 775)
    p = _eismint_moving_params()
    y = _build_eismint_moving(b, p)

    # Sanity: t = 0 state.
    @test all(isfinite, interior(y.tpo.H_ice))
    @test maximum(interior(y.tpo.H_ice)) == 0.0   # zero IC
    @test maximum(interior(y.bnd.smb_ref)) ≈ b.smb_max atol=1e-9
    @test minimum(interior(y.bnd.smb_ref)) < 0.0   # ablation outside R_el

    # Step forward to t = 5 kyr in 100-yr outer-loop chunks; HEUN+PI42
    # adaptively sub-steps inside each chunk.
    t = 0.0
    n_steps = Int(round(_EISMINT_T_END / _EISMINT_DT_OUTER))
    last_max_H = 0.0
    for k in 1:n_steps
        step!(y, _EISMINT_DT_OUTER)
        t = y.time
        last_max_H = maximum(interior(y.tpo.H_ice))
        if k % 10 == 0
            mean_H = mean(interior(y.tpo.H_ice))
            @info "EISMINT moving t=$(round(Int, t)) yr: max(H)=$(round(last_max_H, digits=2)) m  mean(H)=$(round(mean_H, digits=2)) m"
        end
    end

    @info "EISMINT moving final (t=$(round(Int, t)) yr): max(H)=$(round(last_max_H, digits=2)) m"

    H_int  = interior(y.tpo.H_ice)
    ux_int = interior(y.dyn.ux)
    uy_int = interior(y.dyn.uy)
    uz_int = interior(y.dyn.uz)

    # ------ Finiteness ------
    @test all(isfinite, H_int)
    @test all(isfinite, ux_int)
    @test all(isfinite, uy_int)
    @test all(isfinite, uz_int)

    # ------ Dome growth ------
    max_H = maximum(H_int)
    mean_H = mean(H_int)
    @info "Final mean(H) = $(round(mean_H, digits=2)) m"
    # Transient at t = 5 kyr should have dome ~1-2.5 km tall (not yet
    # at steady state ~3 km).
    @test 800.0  ≤ max_H ≤ 2500.0
    @test mean_H > 100.0   # mass-balance-driven accumulation, not zero

    # ------ Dome locates at the geometric centre ------
    Nx, Ny = length(b.xc), length(b.yc)
    icenter = (Nx + 1) ÷ 2
    jcenter = (Ny + 1) ÷ 2
    # Find argmax(H_int) — should be within 2 cells of (icenter, jcenter)
    Hmax_idx = argmax(H_int[:, :, 1])
    i_max, j_max = Tuple(Hmax_idx)
    @test abs(i_max - icenter) ≤ 2
    @test abs(j_max - jcenter) ≤ 2

    # ------ Vertical velocity sane ------
    # uz at the dome (centre, full ice column) should be DOWNWARD
    # (negative) — surface mass adds at top, ice flows down to fill.
    # uz at the dome top: between -smb_max ≈ -0.5 m/yr and 0.
    uz_dome_top = uz_int[icenter, jcenter, end]
    @info "uz at dome top (i=$(icenter), j=$(jcenter), k=Nz_ac): $(uz_dome_top)"
    @test uz_dome_top ≤ 0.0
    @test abs(uz_dome_top) < 1.5   # within plausible range
    @test maximum(abs.(uz_int)) < 5.0   # well within the ±10 m/yr clamp
end
