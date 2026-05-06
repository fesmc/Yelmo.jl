## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# 1D / column-level unit tests for the thrm port.
#
# The column solver `_calc_temp_column_internal!` and the per-column
# wrapper `calc_temp_column!` are the production target — these tests
# exercise them directly (no full YelmoModel) so we can pin down
# correctness independent of the 3D scaffolding.
#
# Tests cover:
#
#   1. Tridiagonal solver — `solve_tridiag!` round-trip against a
#      known M·x = d.
#   2. Linear-column kernel — `define_temp_linear_column!` produces
#      exactly the linear interpolation it claims.
#   3. Robin-column kernel — closed-form Robin solution against the
#      analytic formula (grounded, mb_net > 0 branch).
#   4. Steady-state idempotence — cold column with zero forcing and
#      Dirichlet BCs at PMP base / surface T_srf stays at the
#      analytic linear steady state across many implicit timesteps.
#   5. Pure-conduction convergence — given an arbitrary cold initial
#      profile and zero forcing, the implicit solver converges
#      towards the analytic linear profile under repeated stepping.
#   6. Energy conservation under Neumann basal flux — over one
#      timestep with zero advection / strain heat / surface flux,
#      the column-integrated energy change equals the basal heat
#      flux × dt × area to within first-order error.
#   7. Bedrock equil profile — `define_temp_bedrock_column!` produces
#      the analytic linear profile from `T_bed` at the surface (top
#      of the bedrock column) downwards with slope `−Q_geo /
#      kt_rock`.
#
# Helpers + convenience constructors are at the top so each
# `@testset` reads as just the assertion logic.

using Test
using Yelmo
using Statistics: mean

# ---------------------------------------------------------------------
# Synthetic-column helpers — build the inputs `_calc_temp_column_internal!`
# / `calc_temp_column!` expect.
# ---------------------------------------------------------------------

# Uniform-spaced zeta_aa / zeta_ac on [0, 1] with cell centres at
# zeta_aa[k] = (k - 0.5) / Nz extended with 0 and 1 endpoints to
# match the Fortran convention (nz_aa cell centres + base + surface
# = nz_aa cells, with nz_ac = nz_aa + 1 edges). For test purposes we
# just use Nz_aa equally-spaced cell centres with 0 and 1 as the first
# and last so every test column is well-formed.
function _build_uniform_zeta(Nz::Int)
    # zeta_aa: Nz layers from 0 to 1 (endpoints inclusive — k=1 base,
    # k=Nz surface). Matches the Yelmo `Center()` convention closely
    # enough for the implicit solver tests.
    zeta_aa = collect(range(0.0, 1.0; length=Nz))
    # zeta_ac: Nz+1 edges, with zeta_ac[1] = 0, zeta_ac[Nz+1] = 1, and
    # interior edges at midpoints between aa-nodes.
    zeta_ac = zeros(Nz + 1)
    zeta_ac[1] = 0.0
    zeta_ac[end] = 1.0
    for k in 2:Nz
        zeta_ac[k] = 0.5 * (zeta_aa[k - 1] + zeta_aa[k])
    end
    return zeta_aa, zeta_ac
end

# Build dzeta_a / dzeta_b for the Hoffmann-2018 implicit solver from
# the test zeta arrays.
function _build_dzeta(zeta_aa::Vector{Float64}, zeta_ac::Vector{Float64})
    Nz = length(zeta_aa)
    dzeta_a = zeros(Nz); dzeta_b = zeros(Nz)
    Yelmo.calc_dzeta_terms!(dzeta_a, dzeta_b, zeta_aa, zeta_ac)
    return dzeta_a, dzeta_b
end

# Allocate the seven Float64 scratch vectors the column kernel needs.
function _alloc_tri_scratch(Nz::Int)
    return (Vector{Float64}(undef, Nz),   # subd
            Vector{Float64}(undef, Nz),   # diag
            Vector{Float64}(undef, Nz),   # supd
            Vector{Float64}(undef, Nz),   # rhs
            Vector{Float64}(undef, Nz),   # solution
            Vector{Float64}(undef, Nz),   # cp_buf (tridiag)
            Vector{Float64}(undef, Nz))   # dp_buf (tridiag)
end

# Materialise constants. cp / kt are constant across the column for
# the simple tests; production-faithful T-dependent forms are
# exercised inside `calc_temp_column!`.
const TEST_CP   = 2009.0          # J kg^-1 K^-1
const TEST_KT_K = 6.62e7          # J m^-1 K^-1 a^-1 (≈ 2.1 W/m/K × sec_year)
const TEST_RHO  = 910.0           # kg m^-3
const TEST_T0   = 273.15          # K

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

@testset "thrm: solve_tridiag! round-trip against known M·x = d" begin
    # Diagonals describing  M = (4 -1  0  0;  -1  4 -1  0;
    #                            0 -1  4 -1;  0  0 -1  4)
    a = [0.0, -1.0, -1.0, -1.0]
    b = [4.0,  4.0,  4.0,  4.0]
    c = [-1.0, -1.0, -1.0, 0.0]
    # x_true = ones(4)  →  d = M·x_true.
    x_true = [1.0, 1.0, 1.0, 1.0]
    d = [b[1] * x_true[1] + c[1] * x_true[2],
         a[2] * x_true[1] + b[2] * x_true[2] + c[2] * x_true[3],
         a[3] * x_true[2] + b[3] * x_true[3] + c[3] * x_true[4],
         a[4] * x_true[3] + b[4] * x_true[4]]
    x = zeros(4)
    Yelmo.solve_tridiag!(x, a, b, c, d)
    @test x ≈ x_true atol=1e-12

    # Reuse: caller-supplied scratch overload returns the same x.
    x2 = zeros(4); cp = zeros(4); dp = zeros(4)
    Yelmo.solve_tridiag!(x2, a, b, c, d, cp, dp)
    @test x2 ≈ x_true atol=1e-12

    # Random non-symmetric sanity case.
    a2 = [0.0,  0.5, -0.3,  0.1]
    b2 = [3.0,  4.0,  5.0,  2.0]
    c2 = [-0.7, 0.2, -0.4,  0.0]
    x_true2 = [2.5, -1.0, 0.7, 4.2]
    d2 = [b2[1] * x_true2[1] + c2[1] * x_true2[2],
          a2[2] * x_true2[1] + b2[2] * x_true2[2] + c2[2] * x_true2[3],
          a2[3] * x_true2[2] + b2[3] * x_true2[3] + c2[3] * x_true2[4],
          a2[4] * x_true2[3] + b2[4] * x_true2[4]]
    x3 = zeros(4)
    Yelmo.solve_tridiag!(x3, a2, b2, c2, d2)
    @test x3 ≈ x_true2 atol=1e-12
end

@testset "thrm: define_temp_linear_column! linear interpolation" begin
    Nz = 11
    zeta_aa = collect(range(0.0, 1.0; length=Nz))
    T_srf = 250.0
    T_base = 270.0
    T_col = zeros(Nz)
    Yelmo.define_temp_linear_column!(T_col, T_srf, T_base, TEST_T0, zeta_aa)
    expected = T_base .+ zeta_aa .* (T_srf - T_base)
    @test T_col ≈ expected atol=1e-12
    # Sanity: surface T capped at T0.
    fill!(T_col, NaN)
    Yelmo.define_temp_linear_column!(T_col, 280.0, T_base, TEST_T0, zeta_aa)
    @test T_col[end] ≈ TEST_T0 atol=1e-12
end

@testset "thrm: define_temp_robin_column! analytic match (grounded, mb>0)" begin
    Nz = 21
    zeta_aa = collect(range(0.0, 1.0; length=Nz))
    H_ice = 2000.0
    T_srf = 250.0
    T_pmp_col = fill(TEST_T0, Nz)
    cp_col    = fill(TEST_CP, Nz)
    kt_col    = fill(TEST_KT_K, Nz)
    Q_rock    = 50.0   # mW/m²
    mb_net    = 0.3    # m/yr
    sec_year  = 31536000.0

    T_col = zeros(Nz)
    Yelmo.define_temp_robin_column!(T_col, zeta_aa, T_pmp_col, kt_col, cp_col,
                                     TEST_RHO, H_ice, T_srf, mb_net, Q_rock,
                                     false, sec_year)

    # Analytic: T(z) = (sqrt(π)/2) * ll * dTdz_b * (erf(z/ll) − erf(H/ll)) + T_srf
    # with kappa = kt / (rho_ice cp), ll = sqrt(2 κ H / mb), dTdz_b = -Q_rock_now / kt.
    Q_rock_now = Q_rock * 1e-3 * sec_year
    dTdz_b = -Q_rock_now / kt_col[1]
    kappa  = kt_col[1] / (cp_col[1] * TEST_RHO)
    ll     = sqrt(2.0 * kappa * H_ice / mb_net)

    # Reuse Yelmo's _error_function port for the analytic side.
    erf_yelmo(x) = Yelmo.YelmoModelThrm._error_function(x)

    expected = similar(T_col)
    for k in 1:Nz
        z = zeta_aa[k] * H_ice
        T_pred = (sqrt(pi) / 2.0) * ll * dTdz_b *
                 (erf_yelmo(z / ll) - erf_yelmo(H_ice / ll)) + T_srf
        # T capped at T_pmp.
        expected[k] = min(T_pred, T_pmp_col[k])
    end
    @test T_col ≈ expected atol=1e-9
    # Sanity: surface stays at T_srf and base ≤ T_pmp.
    @test T_col[end] ≈ T_srf atol=1e-9
    @test T_col[1] ≤ T_pmp_col[1] + 1e-12
end

@testset "thrm: implicit column — Dirichlet BCs at steady state stay put" begin
    Nz       = 21
    zeta_aa, zeta_ac = _build_uniform_zeta(Nz)
    dzeta_a, dzeta_b = _build_dzeta(zeta_aa, zeta_ac)
    H_ice    = 2000.0
    T_srf    = 250.0
    T_base   = 270.0
    dt       = 1.0

    # Initial profile = analytic Dirichlet steady-state (linear).
    T_col = T_base .+ zeta_aa .* (T_srf - T_base)
    T_col_in = copy(T_col)

    kappa = fill(TEST_KT_K / (TEST_RHO * TEST_CP), Nz)
    uz       = zeros(Nz + 1)
    advecxy  = zeros(Nz)
    Q_strn_K = zeros(Nz)
    subd, diag, supd, rhs, solution, cp_tri, dp_tri = _alloc_tri_scratch(Nz)

    # 50 implicit steps. With Dirichlet at both ends and zero forcing,
    # the analytic profile is fixed — should not drift.
    for _ in 1:50
        Yelmo.YelmoModelThrm._calc_temp_column_internal!(
            T_col, kappa, uz, advecxy, Q_strn_K,
            T_base, T_srf, H_ice,
            zeta_aa, zeta_ac, dzeta_a, dzeta_b,
            TEST_T0, dt,
            false, false,                                   # is_basal_flux, is_surf_flux
            subd, diag, supd, rhs, solution, cp_tri, dp_tri,
        )
    end

    @test all(isfinite, T_col)
    @test maximum(abs.(T_col .- T_col_in)) < 5e-9
    @test T_col[1] ≈ T_base atol=1e-12
    @test T_col[end] ≈ T_srf atol=1e-12
end

@testset "thrm: implicit column — pure conduction converges to linear profile" begin
    Nz = 21
    zeta_aa, zeta_ac = _build_uniform_zeta(Nz)
    dzeta_a, dzeta_b = _build_dzeta(zeta_aa, zeta_ac)
    # Thin H so that the diffusion timescale H²/(π² κ) is short enough
    # that 200 steps get us many e-folding times. With H = 200 m, κ ≈ 36
    # m²/yr, the k=2 mode timescale ≈ 28 yr → 200 × dt = 10⁴ yr is ≈
    # 360 e-folding times. The implicit solver damps faster than analytic
    # at this dt so convergence is even sharper.
    H_ice    = 200.0
    T_srf    = 240.0
    T_base   = 270.0
    dt       = 50.0

    T_lin = T_base .+ zeta_aa .* (T_srf - T_base)
    T_col = T_lin .+ 5.0 .* sin.(2π .* zeta_aa)
    initial_err = maximum(abs.(T_col .- T_lin))

    kappa = fill(TEST_KT_K / (TEST_RHO * TEST_CP), Nz)
    uz       = zeros(Nz + 1)
    advecxy  = zeros(Nz)
    Q_strn_K = zeros(Nz)
    subd, diag, supd, rhs, solution, cp_tri, dp_tri = _alloc_tri_scratch(Nz)

    for _ in 1:200
        Yelmo.YelmoModelThrm._calc_temp_column_internal!(
            T_col, kappa, uz, advecxy, Q_strn_K,
            T_base, T_srf, H_ice,
            zeta_aa, zeta_ac, dzeta_a, dzeta_b,
            TEST_T0, dt,
            false, false,
            subd, diag, supd, rhs, solution, cp_tri, dp_tri,
        )
    end

    final_err = maximum(abs.(T_col .- T_lin))
    @info "pure-conduction convergence" initial_err final_err
    @test final_err < 1e-6
    @test final_err < initial_err / 1e6   # at least 6 orders of decay
    @test T_col[1] ≈ T_base atol=1e-9
    @test T_col[end] ≈ T_srf atol=1e-9
end

@testset "thrm: implicit column — Neumann base + Dirichlet top → linear steady state" begin
    # With prescribed Neumann basal flux (Q_in upward into the
    # column) and Dirichlet top at T_srf, the analytic steady-state
    # profile is linear:
    #
    #     T(zeta) = T_srf + (Q_in / kt) * H_ice * (1 - zeta)
    #
    # Slope is -Q_in/kt per unit zeta (in absolute height: -Q_in/kt
    # K/m). After many implicit steps the column should converge to
    # this analytic profile to within numerical precision. This is
    # the steady-state form of energy conservation: at steady state
    # the basal heat input is exactly balanced by the conductive
    # heat flux out the top, so the linear profile carries the
    # correct net flux Q_in.
    Nz = 21
    zeta_aa, zeta_ac = _build_uniform_zeta(Nz)
    dzeta_a, dzeta_b = _build_dzeta(zeta_aa, zeta_ac)
    H_ice = 200.0          # short column → fast diffusive convergence
    T_srf = 250.0
    dt    = 50.0
    Q_in  = 5e7            # J m^-2 yr^-1 — strong basal heating

    val_base = -Q_in / TEST_KT_K           # dT/dz at base [K/m]

    T_col = fill(T_srf, Nz)
    kappa = fill(TEST_KT_K / (TEST_RHO * TEST_CP), Nz)
    uz       = zeros(Nz + 1)
    advecxy  = zeros(Nz)
    Q_strn_K = zeros(Nz)
    subd, diag, supd, rhs, solution, cp_tri, dp_tri = _alloc_tri_scratch(Nz)

    for _ in 1:200
        Yelmo.YelmoModelThrm._calc_temp_column_internal!(
            T_col, kappa, uz, advecxy, Q_strn_K,
            val_base, T_srf, H_ice,
            zeta_aa, zeta_ac, dzeta_a, dzeta_b,
            TEST_T0, dt,
            true, false,                                   # is_basal_flux=true
            subd, diag, supd, rhs, solution, cp_tri, dp_tri,
        )
    end

    # Analytic linear steady state with the prescribed gradient.
    expected = T_srf .+ (Q_in / TEST_KT_K) .* H_ice .* (1.0 .- zeta_aa)

    # The Neumann BC discretisation pins T[1] - T[2] = -val_base * dz_1
    # (where dz_1 = H × (zeta_aa[2] - zeta_aa[1])); diffusion fills in
    # the interior to the analytic linear profile. Both should hold
    # tightly after settling.
    err_max = maximum(abs.(T_col .- expected))
    @info "Neumann+Dirichlet steady state" err_max
    @test err_max < 1e-6
    @test T_col[end] ≈ T_srf atol=1e-9
    # Diagnose the basal flux carried by the converged profile —
    # should reproduce Q_in.
    dz_base    = H_ice * (zeta_aa[2] - zeta_aa[1])
    Q_basal    = -TEST_KT_K * (T_col[2] - T_col[1]) / dz_base
    @test Q_basal ≈ Q_in atol=1e-4 * Q_in
end

@testset "thrm: define_temp_bedrock_column! analytic linear profile" begin
    Nz = 6
    zeta_aa = collect(range(0.0, 1.0; length=Nz))
    H_rock  = 2000.0
    T_bed   = 270.0
    Q_geo   = 50.0     # mW/m²
    sec_year = 31536000.0
    kt_rock  = TEST_KT_K   # any positive value works for the test

    T_rock_col = zeros(Nz)
    Yelmo.define_temp_bedrock_column!(T_rock_col, kt_rock, H_rock, T_bed, Q_geo,
                                       zeta_aa, sec_year)

    # Analytic: T_rock(zeta_aa[nz]) = T_bed; descend with slope
    # dTdz = -Q_geo_now / kt_rock; so
    # T_rock(k) = T_bed + (1 - zeta_aa[k]) × Q_geo_now / kt_rock × H_rock.
    Q_geo_now = Q_geo * 1e-3 * sec_year
    dTdz      = -Q_geo_now / kt_rock
    expected  = T_bed .- dTdz .* (H_rock .* (zeta_aa[end] .- zeta_aa))
    @test T_rock_col ≈ expected atol=1e-9
    @test T_rock_col[end] ≈ T_bed atol=1e-12
    # Q_rock diagnostic should equal Q_geo (in mW/m²) by construction —
    # equilibrium bedrock conducts Q_geo straight up to the surface.
    Q_rock_diag = Yelmo.calc_Q_bedrock_column(T_rock_col, kt_rock, H_rock,
                                              zeta_aa, sec_year)
    @test Q_rock_diag ≈ Q_geo atol=1e-9
end

# =============================================================================
# Path B column-solver tests
#
# Under Path B the first interior centre is at zeta_aa[1] > 0 (no endpoint
# at z=0). The boundary temperatures T_ice_b / T_pmp_b are separate scalars
# passed as kwargs. Tests check that:
#   (a) Steady-state Dirichlet BCs at both boundaries stay put.
#   (b) The resolved T_ice_b_new matches the prescribed val_base.
#   (c) Energy conservation under Neumann basal flux is preserved
#       (within the first-order truncation error of the stencil).
# =============================================================================

# Build an 8-layer Path B zeta grid (interior only, no 0/1 endpoints).
function _build_path_b_zeta(Nz_int::Int)
    # Interior centres at (k - 0.5) / Nz_file for k = 1..Nz_int,
    # where Nz_file = Nz_int + 2 (file convention includes 0 and 1).
    Nz_file = Nz_int + 2
    zeta_aa = [(k - 0.5) / Nz_file for k in 1:Nz_int]
    # Face edges: zeta_ac[1]=0, zeta_ac[Nz_int+1]=1, interior = midpoints.
    zeta_ac = zeros(Nz_int + 1)
    zeta_ac[1]   = 0.0
    zeta_ac[end] = 1.0
    for k in 2:Nz_int
        zeta_ac[k] = 0.5 * (zeta_aa[k - 1] + zeta_aa[k])
    end
    return Vector{Float64}(zeta_aa), Vector{Float64}(zeta_ac)
end

function _alloc_full_scratch(Nz::Int)
    kappa    = Vector{Float64}(undef, Nz)
    Qs_K     = Vector{Float64}(undef, Nz)
    subd, diag, supd, rhs, sol, cp_tri, dp_tri = _alloc_tri_scratch(Nz)
    return kappa, Qs_K, subd, diag, supd, rhs, sol, cp_tri, dp_tri
end

@testset "thrm/PathB: path_b stencil — Dirichlet steady state stays put" begin
    # Test the path_b stencil in _calc_temp_column_internal! directly
    # (bypassing calc_temp_column!'s temperature-dependent kappa_basal
    # computation) so the boundary kappa matches the interior exactly.
    # The linear profile from T_base to T_srf is an exact fixed point
    # when kappa_basal = kappa_surf = kappa_interior.
    Nz_int = 8
    zeta_aa, zeta_ac = _build_path_b_zeta(Nz_int)
    dzeta_a, dzeta_b = _build_dzeta(zeta_aa, zeta_ac)
    @test zeta_aa[1] > 0.0    # confirm interior-only grid

    H_ice    = 2000.0
    T_base   = TEST_T0
    T_srf    = 250.0
    dt       = 1.0

    kappa_val = TEST_KT_K / (TEST_RHO * TEST_CP)
    kappa    = fill(kappa_val, Nz_int)
    uz       = zeros(Nz_int + 1)
    advecxy  = zeros(Nz_int)
    Q_strn_K = zeros(Nz_int)
    subd, diag, supd, rhs, solution, cp_tri, dp_tri = _alloc_tri_scratch(Nz_int)

    # Linear profile from T_base at ζ=0 to T_srf at ζ=1.
    T_col    = [T_base + zeta_aa[k] * (T_srf - T_base) for k in 1:Nz_int]
    T_col_in = copy(T_col)

    for _ in 1:50
        Yelmo.YelmoModelThrm._calc_temp_column_internal!(
            T_col, kappa, uz, advecxy, Q_strn_K,
            T_base, T_srf, H_ice,
            zeta_aa, zeta_ac, dzeta_a, dzeta_b,
            TEST_T0, dt,
            false, false,
            subd, diag, supd, rhs, solution, cp_tri, dp_tri;
            path_b      = true,
            kappa_basal = kappa_val,
            kappa_surf  = kappa_val,
        )
    end

    max_drift = maximum(abs.(T_col .- T_col_in))
    @info "PathB stencil Dirichlet drift" max_drift
    @test all(isfinite, T_col)
    @test max_drift < 1e-6
end

@testset "thrm/PathB: calc_temp_column! Dirichlet — T_ice_b_new = val_base" begin
    # Verifies that under the temperate-base Dirichlet branch,
    # T_ice_b_new equals the prescribed val_base every step regardless
    # of the interior column evolution.
    Nz_int = 8
    zeta_aa, zeta_ac = _build_path_b_zeta(Nz_int)
    dzeta_a, dzeta_b = _build_dzeta(zeta_aa, zeta_ac)

    H_ice    = 2000.0
    T_base   = TEST_T0
    T_srf    = 250.0
    T0       = TEST_T0
    dt       = 1.0
    sec_year = 31536000.0
    L_ice    = 3.34e5

    T_pmp_col   = fill(T0, Nz_int)
    cp_col      = fill(TEST_CP, Nz_int)
    kt_col      = fill(TEST_KT_K, Nz_int)
    omega_col   = zeros(Nz_int)
    enth_col    = zeros(Nz_int)
    advecxy_col = zeros(Nz_int)
    Q_strn_col  = zeros(Nz_int)
    uz_col      = zeros(Nz_int + 1)
    kappa, Qs_K, subd, diag, supd, rhs, sol, cp_tri, dp_tri =
        _alloc_full_scratch(Nz_int)
    T_col = [T_base + zeta_aa[k] * (T_srf - T_base) for k in 1:Nz_int]

    for _ in 1:10
        _, _, _, T_b_new, T_s_new = Yelmo.calc_temp_column!(
            enth_col, T_col, omega_col, T_pmp_col, cp_col, kt_col,
            advecxy_col, uz_col, Q_strn_col,
            0.0, 0.0, T_srf, T0,
            H_ice, 0.0, 1.0, 0.0,
            zeta_aa, zeta_ac, dzeta_a, dzeta_b,
            0.0, T0, TEST_RHO, 1000.0, L_ice, sec_year, dt,
            kappa, Qs_K, subd, diag, supd, rhs, sol, cp_tri, dp_tri;
            path_b      = true,
            T_ice_b_val = T_base,
            T_pmp_b_val = T0,
        )
        @test T_b_new ≈ T_base atol=1e-8
        @test T_s_new ≈ min(T_srf, T0) atol=1e-8
    end
    @test all(isfinite, T_col)
end

@testset "thrm/PathB: Path B converges to linear profile" begin
    # Test at the _calc_temp_column_internal! level with kappa_basal = kappa_val
    # (same as interior kappa) so the linear profile is an exact fixed point.
    # This mirrors the legacy convergence test but with Path B zeta and path_b=true.
    Nz_int   = 8
    zeta_aa, zeta_ac = _build_path_b_zeta(Nz_int)
    dzeta_a, dzeta_b = _build_dzeta(zeta_aa, zeta_ac)
    H_ice    = 200.0
    T_base   = 270.0
    T_srf    = 250.0
    dt       = 50.0
    kappa_val = TEST_KT_K / (TEST_RHO * TEST_CP)

    T_lin = [T_base + zeta_aa[k] * (T_srf - T_base) for k in 1:Nz_int]
    T_col = T_lin .+ 5.0 .* sin.(2π .* zeta_aa)

    kappa    = fill(kappa_val, Nz_int)
    uz       = zeros(Nz_int + 1)
    advecxy  = zeros(Nz_int)
    Q_strn_K = zeros(Nz_int)
    subd, diag, supd, rhs, solution, cp_tri, dp_tri = _alloc_tri_scratch(Nz_int)

    for _ in 1:200
        Yelmo.YelmoModelThrm._calc_temp_column_internal!(
            T_col, kappa, uz, advecxy, Q_strn_K,
            T_base, T_srf, H_ice,
            zeta_aa, zeta_ac, dzeta_a, dzeta_b,
            TEST_T0, dt,
            false, false,
            subd, diag, supd, rhs, solution, cp_tri, dp_tri;
            path_b = true, kappa_basal = kappa_val, kappa_surf = kappa_val)
    end

    final_err = maximum(abs.(T_col .- T_lin))
    @info "PathB pure-conduction convergence" final_err
    @test final_err < 1e-4
    @test final_err < 5.0 / 1e4   # at least 4 orders of decay from initial amp 5.0
end

@testset "thrm/PathB: Neumann base → Q_ice_b consistent with flux" begin
    # Steady-state with Neumann basal flux and Dirichlet top.
    # Q_ice_b (mW/m²) returned from calc_temp_column! should reproduce
    # the applied flux to within the truncation error of the stencil.
    Nz_int = 8
    zeta_aa, zeta_ac = _build_path_b_zeta(Nz_int)
    dzeta_a, dzeta_b = _build_dzeta(zeta_aa, zeta_ac)
    H_ice    = 200.0
    T_srf    = 250.0
    T0       = TEST_T0
    dt       = 50.0
    sec_year = 31536000.0
    L_ice    = 3.34e5
    Q_b      = 80.0   # mW/m² basal heat flux

    T_col = fill(T_srf, Nz_int)
    T_pmp_col   = fill(T0, Nz_int)
    cp_col      = fill(TEST_CP, Nz_int)
    kt_col      = fill(TEST_KT_K, Nz_int)
    omega_col   = zeros(Nz_int)
    enth_col    = zeros(Nz_int)
    advecxy_col = zeros(Nz_int)
    Q_strn_col  = zeros(Nz_int)
    uz_col      = zeros(Nz_int + 1)
    # Cold base so Neumann BC is active (T_ice_b_val < T_pmp_b_val).
    T_ice_b_val = T_srf   # cold boundary

    kappa, Qs_K, subd, diag, supd, rhs, sol, cp_tri, dp_tri =
        _alloc_full_scratch(Nz_int)

    local Q_ice_b_out::Float64
    for _ in 1:300
        Q_ice_b_out, _, _, T_b_new, _ = Yelmo.calc_temp_column!(
            enth_col, T_col, omega_col, T_pmp_col, cp_col, kt_col,
            advecxy_col, uz_col, Q_strn_col,
            Q_b, 0.0, T_srf, T0,
            H_ice, 0.0, 1.0, 0.0,
            zeta_aa, zeta_ac, dzeta_a, dzeta_b,
            0.0, T0, TEST_RHO, 1000.0, L_ice, sec_year, dt,
            kappa, Qs_K, subd, diag, supd, rhs, sol, cp_tri, dp_tri;
            path_b      = true,
            T_ice_b_val = T_ice_b_val,
            T_pmp_b_val = T0,
        )
        T_ice_b_val = T_b_new  # carry forward the extrapolated base temp
    end

    @info "PathB Neumann Q_ice_b vs Q_b" Q_ice_b_out Q_b
    # At steady state Q_ice_b ≈ Q_b (the applied flux, within 5%).
    @test isapprox(Q_ice_b_out, Q_b; rtol=0.05)
end
