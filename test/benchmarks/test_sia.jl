## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# SIA solver convergence test against the BUELER-B Halfar analytical
# solution.
#
# Two test sets:
#
#   1. `analytical_velocity` formula correctness — pin the closed-form
#      `dH/dr` against a numerical centred-difference of
#      `bueler_test_BC!`-evaluated H on a 1D radial slice. Catches sign /
#      exponent / factor errors in the closed-form derivative.
#
#   2. SIA convergence — drive Yelmo.jl's SIA solver on the Halfar dome
#      at three resolutions (dx = 50, 20, 10 km), evaluate the
#      depth-averaged velocity error on the dome interior (margin-
#      masked, `H > 100 m`), and assert monotone error decrease with
#      grid refinement plus a loose absolute threshold at the finest
#      grid.
#
# Notes:
#
#   - The convergence test calls `dyn_step!` directly (NOT the full
#     `step!`), so the manual `interior(y.mat.ATT) .= 1e-16` (the
#     BUELER-B isothermal A constant) survives the call. Once
#     `mat_step!` is wired into the full step orchestrator (a future
#     milestone), this manual ATT will be overwritten on each step and
#     this test's setup will need to be revisited.
#   - Per-resolution L2 errors are printed so the actual numbers are
#     visible in test output (helpful for tuning the loose 20%
#     threshold later).

using Test
using Yelmo
using Yelmo.YelmoModelPar: ydyn_params, ymat_params, YelmoModelParameters
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

# Margin-mask threshold (m) for the L2 error metric. Halfar's margin
# (H -> 0, |grad H| -> infinity) is locally singular for SIA, so any
# domain-wide L2 norm is dominated by it. Standard practice in
# Bueler 2005 is to mask out a band where H is below a fixed
# threshold; 100 m is well above the margin-band thickness at
# `t = 1000 yr` with `H0 = 3600 m`.
const H_THRESH = 100.0

# L2 relative error in the interior of an XFace 2D field, masked to
# faces whose two adjacent aa-cells average to H > H_thresh.
function _interior_l2_error(now::AbstractMatrix, ref::AbstractMatrix,
                            H_aa::AbstractMatrix, H_thresh::Real;
                            trim::Int = 2)
    Nxf, Ny = size(now)
    diff_sq = 0.0
    ref_sq  = 0.0
    n_pts   = 0
    @inbounds for j in 1+trim:Ny-trim, i in 1+trim:Nxf-trim
        # XFace face (i, j) sees aa-cells (i-1, j) and (i, j); be
        # defensive about array bounds at the leading / trailing face.
        i_west = clamp(i - 1, 1, size(H_aa, 1))
        i_east = clamp(i,     1, size(H_aa, 1))
        H_face = 0.5 * (H_aa[i_west, j] + H_aa[i_east, j])
        if H_face > H_thresh
            diff_sq += (now[i, j] - ref[i, j])^2
            ref_sq  += ref[i, j]^2
            n_pts   += 1
        end
    end
    return n_pts > 0 ? sqrt(diff_sq / max(ref_sq, eps())) : NaN
end

# ======================================================================
# Test 1 — closed-form `dH/dr` matches numerical centred-difference of
# `bueler_test_BC!`-evaluated H to numerical-differentiation tolerance.
# ======================================================================

@testset "BuelerBenchmark: analytical_velocity vs numerical dH/dr" begin
    b = BuelerBenchmark(:B; dx_km = 10.0)
    t = 1000.0

    # Sample a 1D radial slice between r = 1 km and r = 700 km — well
    # inside the dome margin (which sits near the box edge for
    # R0 = 750 km at t = 1000 yr). Avoids r = 0 singularity and the
    # margin band. `length = 2000` keeps the centred-difference
    # truncation error (O(dr^2) * d^3 H / d r^3) well below the 1e-3
    # comparison tolerance for the interior portion of the dome.
    rs = collect(range(1e3, 700e3; length = 2000))
    H_arr = zeros(length(rs))
    for (i, r) in enumerate(rs)
        H_pt  = zeros(1, 1)
        mb_pt = zeros(1, 1)
        bueler_test_BC!(H_pt, mb_pt, [r], [0.0], t;
                        R0      = b.R0_km,
                        H0      = b.H0,
                        lambda  = b.lambda,
                        n       = b.n,
                        A       = b.A,
                        rho_ice = b.rho_ice,
                        g       = b.g)
        H_arr[i] = H_pt[1, 1]
    end

    # Numerical centred-difference of H on the radial slice.
    dr = rs[2] - rs[1]
    dHdr_num = zeros(length(rs))
    for i in 2:length(rs)-1
        dHdr_num[i] = (H_arr[i+1] - H_arr[i-1]) / (2 * dr)
    end

    # Closed-form dH/dr at the same points (private helper from
    # `bueler.jl` — accessed via the `YelmoBenchmarks` module).
    dHdr_closed = zeros(length(rs))
    for (i, r) in enumerate(rs)
        dHdr_closed[i] = YelmoBenchmarks._halfar_dHdr_closed(b, r, t)
    end

    # Interior agreement to numerical-differentiation tolerance. Skip
    # the first/last 5 points (centred-difference edges) and any point
    # where the dome is thin (near the margin, where the closed-form
    # |dH/dr| diverges and a centred difference of a clipped H is no
    # longer a good approximation).
    n_checked = 0
    for i in 5:length(rs)-5
        if H_arr[i] > 100.0
            rel_err = abs(dHdr_closed[i] - dHdr_num[i]) /
                      max(abs(dHdr_num[i]), 1e-12)
            @test rel_err < 1e-3
            n_checked += 1
        end
    end
    @test n_checked > 500  # enough interior points to be meaningful
end

# ======================================================================
# Test 2 — SIA solver convergence on the Halfar dome.
# ======================================================================

@testset "SIA convergence — Halfar dome (BUELER-B)" begin
    errs_x = Dict{Float64, Float64}()
    errs_y = Dict{Float64, Float64}()

    for dx_km in (50.0, 20.0, 10.0)
        b = BuelerBenchmark(:B; dx_km = dx_km, R0_km = 750.0, H0 = 3600.0)

        # Override solver to "sia" and pin n_glen = 3 to match the
        # BUELER-B convention.
        p = YelmoModelParameters("sia_conv_$(Int(round(dx_km)))km";
                                 ydyn = ydyn_params(solver = "sia"),
                                 ymat = ymat_params(n_glen = 3.0))

        # In-memory YelmoModel from the analytical state at t = 1000 yr.
        y = YelmoModel(b, 1000.0; p = p)

        # ATT — uniform 1e-16 Pa^-3 yr^-1 (BUELER-B isothermal). Survives
        # the `dyn_step!` call because `mat_step!` is not yet wired into
        # the dynamics step (see file header).
        fill!(interior(y.mat.ATT), 1e-16)

        # Refresh tpo diagnostics so dzsdx / dzsdy / H_ice_dyn /
        # f_ice_dyn / mask_frnt are populated before `dyn_step!`. The
        # in-memory constructor leaves these at default-allocated
        # values (matches the file-based loader's behaviour).
        Yelmo.update_diagnostics!(y)

        # One dyn step. `dt` is irrelevant for the SIA velocity solve
        # (only `duxydt = (uxy_bar - uxy_prev) / dt` uses it).
        Yelmo.dyn_step!(y, 1.0)

        # Analytical reference at the same time.
        ux_ref, uy_ref = analytical_velocity(b, 1000.0)

        # Margin mask via H_ice on aa-cells.
        H_ice_aa = interior(y.tpo.H_ice)[:, :, 1]
        ux_now   = interior(y.dyn.ux_bar)[:, :, 1]
        uy_now   = interior(y.dyn.uy_bar)[:, :, 1]

        err_x = _interior_l2_error(ux_now, ux_ref, H_ice_aa, H_THRESH)
        err_y = _interior_l2_error(uy_now, uy_ref, H_ice_aa, H_THRESH)
        @test isfinite(err_x)
        @test isfinite(err_y)

        errs_x[dx_km] = err_x
        errs_y[dx_km] = err_y

        # Print so the actual numbers show up in the test output.
        println("  BUELER-B SIA: dx=$(dx_km) km  err_x=$(round(err_x; digits=4))  err_y=$(round(err_y; digits=4))")
    end

    # Monotone convergence in both directions.
    @test errs_x[20.0] < errs_x[50.0]
    @test errs_x[10.0] < errs_x[20.0]
    @test errs_y[20.0] < errs_y[50.0]
    @test errs_y[10.0] < errs_y[20.0]

    # Loose absolute threshold at the finest grid. Locked-in user
    # decision Q10: deliberately loose for now; do NOT tighten on
    # first-pass empirical results.
    @test errs_x[10.0] < 0.20
    @test errs_y[10.0] < 0.20
end
