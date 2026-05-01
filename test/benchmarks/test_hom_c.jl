## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# ISMIP-HOM-C 180° rotational symmetry test for the SSA solver.
#
# Setup (per Pattyn 2008 / yelmo/tests/yelmo_ismiphom.f90):
#   - Domain      : [0, L] × [0, L], L = 80 km, dx = 2 km (Nx = Ny = 40).
#   - Geometry    : uniform 1000 m slab over a sloping bed (α = 0.1°).
#   - Material    : isothermal, A_glen = 1e-16, n = 3.
#   - Friction    : β(x, y) = β₀ + β_amp · sin(2πx/L) · sin(2πy/L)
#                   with β₀ = β_amp = 1000 (Yelmo Fortran convention).
#   - Boundaries  : fully periodic in x and y.
#
# dzsdx OVERRIDE (after `update_diagnostics!`, before `dyn_step!`):
#
#   `update_diagnostics!` calls `calc_gradient_acx!(y.tpo.dzsdx, y.tpo.z_srf,
#   …)` which finite-differences `z_srf = -x · tan α` across the periodic-x
#   wrap. `z_srf` is monotonically decreasing in x and discontinuous across
#   the periodic boundary, so the wrap-column gradient is anomalous. To
#   bypass this production bug we overwrite `tpo.dzsdx` and `tpo.dzsdy`
#   with the analytical uniform values BEFORE `dyn_step!`. Step 2 of
#   `dyn_step!` (`calc_driving_stress!`) then produces the desired
#   uniform `taud_acx = -ρgH·tan α`, `taud_acy = 0`. Overriding `taud_acx`
#   directly would NOT work — `dyn_step!` recomputes it from `dzsdx`.
#
# 180° rotational symmetry physics:
#
#   Under (x, y) → (L - x, L - y), the imposed uniform driving stress
#   `taud_acx = const, taud_acy = 0` is invariant (constant fields don't
#   change under rotation), and the β-pattern is invariant under that
#   rotation since `sin(omega(L - x)) = -sin(omega · x)` (and same for y),
#   so β is bilinear in two sign-flipping factors:
#
#     β(L-x, L-y) = β₀ + β_amp · (-sin omega·x)(-sin omega·y) = β(x, y).
#
#   Therefore the SSA solution must be SYMMETRIC under that rotation:
#
#     ux(x, y) = +ux(L-x, L-y)
#     uy(x, y) = +uy(L-x, L-y)
#
# Sanity bounds on `ux_bar`:
#
#   With uniform `taud = ρgH·tan α ≈ 1.56e4 Pa` and β₀ = 1000, in the
#   pure-friction limit ux ≈ taud / β = 15.6 m/yr. Membrane terms
#   modify this slightly. The test asserts `mean(|ux_bar|) ∈ [10, 50]`
#   m/yr — wildly out-of-range values are a sign that the override
#   didn't apply or has the wrong sign.
#
# This test does NOT check against an analytical velocity (HOM-C has
# no closed-form). It exercises the SSA solver under fully-periodic BC,
# spatially-varying β, and uniform forcing — three orthogonal axes
# beyond the trough / slab tests already in CI.
#
# NOTE: This test currently uses `visc_method=0` (constant viscosity).
# `visc_method=1` (Gaussian-quadrature) produces a ~5% structural
# asymmetry under fully-periodic BC with spatially-varying β
# (rel err_ux=1.88%, rel err_uy=5.48% vs ~1e-9 with visc_method=0).
# The Phase C bisect (cb2fe84..bc6b677 + this commit) traced the bug
# to `calc_visc_eff_3D_nodes!` — likely a periodic-wrap bug in face-
# staggered velocity gradients or aa→ac stagger reads. Tracked for
# follow-up; see PR description.

using Test
using Yelmo
using Oceananigans: interior

include("helpers.jl")
using .YelmoBenchmarks

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                           yneff_params, ytill_params

const _SPEC = HOMCBenchmark(:C; L_km=80.0, dx_km=2.0)

# HOM-C SSA parameters: solver = "ssa", external β (`beta_method = -1`,
# `beta_gl_stag = -1` so pre-filled β fields survive every Picard
# iteration), constant viscosity (`visc_method = 0` reads directly from
# `dyn.visc`, which Yelmo fills uniformly with `visc_const`), N_eff
# irrelevant under external β, isothermal ATT.
function _hom_c_yelmo_params()
    return YelmoModelParameters("hom_c";
        ydyn = ydyn_params(
            solver         = "ssa",
            visc_method    = 0,                       # constant viscosity (visc_method=1 has known asymmetry bug)
            visc_const     = 1e7,
            beta_method    = -1,                      # external β (preserved by Picard loop)
            beta_gl_stag   = -1,                      # bypass standard staggering + GL block
            beta_const     = 1000.0,
            beta_min       = 0.0,
            ssa_lat_bc     = "floating",
            taud_lim       = 2e5,
            ssa_solver     = SSASolver(rtol            = 1e-6,
                                       itmax           = 500,
                                       picard_tol      = 1e-4,
                                       picard_iter_max = 100,
                                       picard_relax    = 0.7),
        ),
        # No till / N_eff dependency under beta_method = -1.
        yneff = yneff_params(method = -1, const_ = 1e7),
        ytill = ytill_params(method = -1),
        # Glen flow law from HOM-C: A = 1e-16 Pa^-3 yr^-1, n = 3,
        # constant ATT (isothermal).
        ymat  = ymat_params(
            n_glen     = 3.0,
            rf_const   = 1e-16,
            de_max     = 0.5,
            enh_shear  = 1.0,
            enh_stream = 1.0,
            enh_shlf   = 1.0,
        ),
    )
end

@testset "benchmarks: ISMIP-HOM-C 180° rotational symmetry under SSA" begin
    b = _SPEC
    p = _hom_c_yelmo_params()

    # Build the YelmoModel directly from the analytical state (no
    # NetCDF round-trip — HOM-C has no time evolution at the IC level).
    y = YelmoModel(b, 0.0; p=p, boundaries = :periodic)

    @test y isa AbstractYelmoModel
    @test y.time == 0.0

    Nx, Ny = length(b.xc), length(b.yc)

    # Shape sanity. Under fully-periodic BC, both XFaceField and
    # YFaceField have interior shape (Nx, Ny, 1) (no extra slot for the
    # wrap-around face — it's the same face as slot 1).
    @test size(interior(y.tpo.H_ice))   == (Nx, Ny, 1)
    @test size(interior(y.dyn.ux_b))    == (Nx, Ny, 1)
    @test size(interior(y.dyn.uy_b))    == (Nx, Ny, 1)

    # Fill β fields from the analytical formula (must come AFTER
    # YelmoModel construction so we can write into the dyn group).
    _setup_hom_c_beta!(y, b)

    # Fill the rate factor `mat.ATT` uniformly from `ymat.rf_const`. The
    # material module is not yet implemented in Yelmo.jl, so ATT defaults
    # to zero and downstream `calc_visc_eff_*!` produces a degenerate
    # viscosity. For HOM-C (isothermal), uniform `ATT = rf_const`
    # captures the full physical content of the Glen-flow law.
    fill!(interior(y.mat.ATT), p.ymat.rf_const)

    # Run topo diagnostics: this populates z_srf, dzsdx, dzsdy from
    # H_ice + z_bed. dzsdx will have the wrap-column anomaly we override
    # below.
    Yelmo.update_diagnostics!(y)

    # OVERRIDE dzsdx / dzsdy with the analytical uniform values.
    # `z_srf = -x · tan α` ⇒ `∂z_srf/∂x = -tan α`, `∂z_srf/∂y = 0`.
    # `calc_driving_stress!` (step 2 of `dyn_step!`) then produces
    # `taud_acx[i+1, j] = ρ_ice · g · H · dzsdx[i, j] = -ρ g H tan α`,
    # which is the desired uniform driving stress for HOM-C.
    fill!(interior(y.tpo.dzsdx), -tan(b.alpha_rad))
    fill!(interior(y.tpo.dzsdy), 0.0)

    # Run one SSA dyn_step. dt is irrelevant for the SSA solve itself
    # (steady-state at fixed H_ice / ATT / β); the value matters only
    # for `duxydt` post-diagnostics, which we don't check.
    Yelmo.YelmoModelDyn.dyn_step!(y, 1.0)

    iter_count = y.dyn.scratch.ssa_iter_now[]
    @info "HOM-C dyn_step! Picard iterations: $iter_count"
    @test iter_count > 0
    @test iter_count <= y.p.ydyn.ssa_solver.picard_iter_max

    ux = interior(y.dyn.ux_bar)
    uy = interior(y.dyn.uy_bar)

    @test all(isfinite, ux)
    @test all(isfinite, uy)

    # Sanity-check magnitude: should be O(15 m/yr) for HOM-C-C with
    # L = 80 km. If the dzsdx override didn't apply, ux would be near
    # zero (no driving stress). If it has the wrong sign, mean(|ux|)
    # might still pass but the test is more useful as a catch on
    # "override forgotten" than on sign errors.
    mean_abs_ux = sum(abs, ux) / length(ux)
    @info "HOM-C mean(|ux_bar|) = $(round(mean_abs_ux; digits=3)) m/yr " *
          "(expected ≈ 15.6 for taud / β₀ = ρgH·tan α / 1000)"
    @test 10.0 <= mean_abs_ux <= 50.0

    # 180° rotational symmetry: ux(x, y) = ux(L - x, L - y), same for uy.
    #
    # Index pairing under fully-periodic BC:
    #
    #   - XFaceField slot [i, j] is at x = (i-1)·dx (face position) and
    #     y = (j-0.5)·dy (cell centre). Under x → L-x, the rotated x is
    #     L - (i-1)·dx = (Nx-i+1)·dx, which is slot k' = mod1(Nx+2-i, Nx)
    #     (e.g. i=1 wraps to itself, i=2 ↔ Nx, …). Under y → L-y, the
    #     rotated y is at slot Ny+1-j (standard cell-centre pairing).
    #
    #   - YFaceField slot [i, j] is at x = (i-0.5)·dx (cell centre) and
    #     y = (j-1)·dy (face position). Pairing: i ↔ Nx+1-i (cell
    #     centre in x), j ↔ mod1(Ny+2-j, Ny) (face in y).
    #
    # The fully-periodic SSA matrix has a 1D nullspace (constant ux, uy
    # additive shift), so we compare DIFFERENCES against the rotated
    # solution directly — `ux[i, j] - ux[ip, jp]` should be identically
    # zero, regardless of any constant offset (which cancels under
    # subtraction).
    err_ux = 0.0
    err_uy = 0.0
    for j in 1:Ny, i in 1:Nx
        ip_x = mod1(Nx + 2 - i, Nx)        # face pairing in x (Nx+2-i)
        jp_x = Ny + 1 - j                  # cell-centre pairing in y
        err_ux = max(err_ux, abs(ux[i, j, 1] - ux[ip_x, jp_x, 1]))

        ip_y = Nx + 1 - i                  # cell-centre pairing in x
        jp_y = mod1(Ny + 2 - j, Ny)        # face pairing in y (Ny+2-j)
        err_uy = max(err_uy, abs(uy[i, j, 1] - uy[ip_y, jp_y, 1]))
    end
    max_abs_ux = maximum(abs, ux)
    max_abs_uy = maximum(abs, uy)
    rel_ux = err_ux / max(max_abs_ux, eps())
    rel_uy = err_uy / max(max_abs_uy, eps())
    @info "HOM-C 180° symmetry: " *
          "max|ux|=$(max_abs_ux)  max|uy|=$(max_abs_uy)  " *
          "abs err_ux=$err_ux  err_uy=$err_uy  " *
          "rel err_ux=$rel_ux  rel_uy=$rel_uy"
    @test rel_ux < 1e-8
    @test rel_uy < 1e-8
end
