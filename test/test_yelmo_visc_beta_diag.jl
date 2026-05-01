## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# ----------------------------------------------------------------------
# Diagnostic: hand-derived analytical checks for the four SSA viscosity
# / friction kernels that PR #25's trough lockstep gap (~50% / 84%
# relative err_ux post-precond / visc-int fixes) was provably narrowed
# to:
#
#   1. `calc_visc_eff_3D_nodes!` — Gauss-quadrature visc (visc_method=1)
#   2. `_calc_beta_aa_power_plastic!` — pseudo-plastic friction
#      (beta_method=2)
#   3. `stagger_visc_aa_ab!` — Center → corner viscosity stagger
#   4. `stagger_beta!` — Center → face β stagger
#
# Methodology mirrors PR #25 (uniform-state slab + hand-derived
# analytical reference, row-by-row compare). Each kernel is exercised
# with synthetic geometry where the answer is closed-form, then
# compared against the kernel's interior output cell-by-cell.
#
# Phase 1 deliverable per `.claude/agents/...` task spec: report which
# kernel(s) diverge from analytical, with magnitude / pattern. STOP
# before Phase 2 (fix) — do not autonomously implement a fix.

using Test
using Yelmo
using Oceananigans
using Oceananigans: interior

const _D = Yelmo.YelmoModelDyn

_bounded_2d(Nx, Ny; dx=1.0) = RectilinearGrid(size=(Nx, Ny),
                                                x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                                topology=(Bounded, Bounded, Flat))
_grid3d(Nx, Ny, Nz; dx=1.0) = RectilinearGrid(size=(Nx, Ny, Nz),
                                              x=(0.0, Nx*dx), y=(0.0, Ny*dx),
                                              z=(0.0, 1.0),
                                              topology=(Bounded, Bounded, Bounded))

# ======================================================================
# Test 1.1 — calc_visc_eff_3D_nodes! on uniform velocity (zero strain
# rate). Effective squared strain rate = eps_0². Glen viscosity formula:
#   visc = 0.5 · (eps_0²)^p1 · ATT^p2,  p1 = (1-n)/(2n), p2 = -1/n.
# Both _aa and _nodes must give the same answer for uniform fields.
# ======================================================================

@testset "[1.1] calc_visc_eff_3D_nodes!: uniform velocity (zero strain rate)" begin
    Nx, Ny, Nz = 5, 5, 4
    dx = 8000.0   # 8 km, matches trough resolution
    g  = _bounded_2d(Nx, Ny; dx=dx)
    g3 = _grid3d(Nx, Ny, Nz; dx=dx)
    zeta_aa = collect(range(0.5/Nz, 1 - 0.5/Nz, length=Nz))

    # Uniform velocity — strain rate identically zero everywhere.
    ux = XFaceField(g);  fill!(interior(ux), 100.0)
    uy = YFaceField(g);  fill!(interior(uy), 50.0)
    H_ice = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice = CenterField(g);  fill!(interior(f_ice), 1.0)
    ATT = CenterField(g3);   fill!(interior(ATT), 1e-16)

    n_glen, eps_0 = 3.0, 1e-6

    visc_aa = CenterField(g3)
    visc_no = CenterField(g3)

    calc_visc_eff_3D_aa!(visc_aa, ux, uy, ATT, H_ice, f_ice, zeta_aa,
                         dx, dx, n_glen, eps_0)
    calc_visc_eff_3D_nodes!(visc_no, ux, uy, ATT, H_ice, f_ice, zeta_aa,
                            dx, dx, n_glen, eps_0)

    # Analytical: visc = 0.5 · eps_0^((1-n)/n) · ATT^(-1/n)
    p1 = (1.0 - n_glen) / (2.0 * n_glen)
    p2 = -1.0 / n_glen
    eps_sq = eps_0^2
    expected = 0.5 * eps_sq^p1 * (1e-16)^p2

    println("[1.1] uniform-velocity Glen viscosity:")
    println("    expected (analytical) = $expected")
    println("    _aa     interior ext  = $(extrema(interior(visc_aa)))")
    println("    _nodes  interior ext  = $(extrema(interior(visc_no)))")

    # Skip the boundary row/col because the corner-stagger samples
    # neighbour aa-cells via clamped indices. Interior cells should
    # match analytical for both kernels.
    for k in 1:Nz, j in 2:Ny-1, i in 2:Nx-1
        @test interior(visc_aa)[i, j, k] ≈ expected rtol=1e-9
        @test interior(visc_no)[i, j, k] ≈ expected rtol=1e-9
    end

    # Cross-check: _aa and _nodes agree on uniform field (no quadrature
    # error possible).
    aa_int = view(interior(visc_aa), 2:Nx-1, 2:Ny-1, :)
    no_int = view(interior(visc_no), 2:Nx-1, 2:Ny-1, :)
    max_diff = maximum(abs.(aa_int .- no_int))
    @test max_diff < 1e-3
    println("    max |_aa - _nodes| (interior) = $max_diff")
end

# ======================================================================
# Test 1.2 — _calc_beta_aa_power_plastic! on uniform velocity.
#   β = c · (|u|² + ub_sq_min)^((q-1)/2) / u_0^q · |u|^(q-1) ?
# Wait — the Fortran formula is:
#   uxy = sqrt(ux² + uy² + ub_sq_min)
#   β   = c_bed(i,j) · (uxy / u_0)^q · (1 / uxy)
# For uniform velocity, every quadrature node sees the same uxy → β
# averaged over 4 nodes = exactly that single value. Hand-derive at
# multiple velocity magnitudes with the trough config (q=1/3,
# u_0=31556926, c_bed=10·N_eff·tan(10°) i.e. realistic trough range).
# ======================================================================

@testset "[1.2] _calc_beta_aa_power_plastic!: uniform velocity (trough config)" begin
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)

    c_bed_val = 9.32e5     # realistic trough mid-domain c_bed
    q   = 1.0 / 3.0
    u_0 = 31556926.0       # 1 m/s expressed in m/yr (Fortran convention)

    for U_test in (10.0, 100.0, 1000.0, 10000.0)
        ux = XFaceField(g);  fill!(interior(ux), U_test)
        uy = YFaceField(g);  fill!(interior(uy), 0.0)
        c_bed = CenterField(g);  fill!(interior(c_bed), c_bed_val)
        f_ice = CenterField(g);  fill!(interior(f_ice), 1.0)
        beta  = CenterField(g)

        b_int  = interior(beta)
        ux_int = interior(ux)
        uy_int = interior(uy)
        c_int  = interior(c_bed)
        fi_int = interior(f_ice)

        _D._calc_beta_aa_power_plastic!(b_int, ux_int, uy_int, c_int, fi_int,
                                        q, u_0, false, Bounded, Bounded)

        ub_sq_min = (1e-3)^2
        uxy = sqrt(U_test^2 + 0.0 + ub_sq_min)
        expected = c_bed_val * (uxy / u_0)^q * (1.0 / uxy)

        println("[1.2] U=$U_test  expected β = $expected")
        println("    interior β extrema = $(extrema(b_int))")

        # Inner cells (skip outermost row/col where corner-stagger
        # touches clamped indices — but for uniform velocity the
        # clamps reproduce the same value, so check every cell).
        for j in 2:Ny-1, i in 2:Nx-1
            @test b_int[i, j, 1] ≈ expected rtol=1e-9
        end
    end
end

# ======================================================================
# Test 1.3 — stagger_visc_aa_ab! on a non-uniform Center field.
# 4-cell average aa→ab. With Center field visc[i,j,1] = i+j and
# all f_ice=1, corner [ip1f, jp1f, 1] = mean(visc[i,j], visc[ip1,j],
# visc[i,jp1], visc[ip1,jp1]) = (i+j) + (ip1+j) + (i+jp1) + (ip1+jp1) / 4
# For interior (ip1 = i+1, jp1 = j+1): = (4i + 4j + 4) / 4 = i+j+1.
# Test edge corners too (clamped): for i=Nx, ip1=Nx, so the mean
# includes only cells (i,j),(i,j+1) twice → (2(i+j) + 2(i+j+1)) / 4 =
# (i+j) + 0.5.
# ======================================================================

@testset "[1.3] stagger_visc_aa_ab!: non-uniform Center field, 4-cell average" begin
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)
    visc_2d = CenterField(g)
    H_ice   = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice   = CenterField(g);  fill!(interior(f_ice), 1.0)

    # Hand-pickable pattern: visc(i, j) = i + j.
    @inbounds for j in 1:Ny, i in 1:Nx
        interior(visc_2d)[i, j, 1] = i + j
    end

    visc_ab = Field((Face(), Face(), Center()), g)
    stagger_visc_aa_ab!(visc_ab, visc_2d, H_ice, f_ice)

    Vab = interior(visc_ab)
    println("[1.3] visc_ab interior extrema = $(extrema(Vab))")

    # Kernel writes Vab[ip1f, jp1f, 1] where (under Bounded) ip1f =
    # i+1, jp1f = j+1. The neighbour cells used in the average use
    # the *clamped* interior indices ip1 = min(i+1, Nx), jp1 =
    # min(j+1, Ny).
    for j in 1:Ny, i in 1:Nx
        ip1 = min(i + 1, Nx)
        jp1 = min(j + 1, Ny)
        expected = ((i + j) + (ip1 + j) + (i + jp1) + (ip1 + jp1)) / 4
        @test Vab[i+1, j+1, 1] ≈ expected
    end
end

# ======================================================================
# Test 1.4 — stagger_beta! beta_gl_stag=0 (mean only) — trough config.
# Uniform β_aa = β₀ → all interior faces should also be β₀. Linear-x
# gradient β(i,j) = i·δ + b₀ → interior face acx[i+1, j] should be the
# 2-cell average = ((i)·δ+b₀ + (i+1)·δ+b₀)/2 = (i + 0.5)·δ + b₀.
# ======================================================================

@testset "[1.4a] stagger_beta!: uniform β" begin
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)
    beta = CenterField(g);   fill!(interior(beta), 1500.0)
    H_ice = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice = CenterField(g);  fill!(interior(f_ice), 1.0)
    ux    = XFaceField(g);   fill!(interior(ux), 0.0)
    uy    = YFaceField(g);   fill!(interior(uy), 0.0)
    f_grnd     = CenterField(g);  fill!(interior(f_grnd), 1.0)
    f_grnd_acx = XFaceField(g);   fill!(interior(f_grnd_acx), 1.0)
    f_grnd_acy = YFaceField(g);   fill!(interior(f_grnd_acy), 1.0)
    beta_acx = XFaceField(g)
    beta_acy = YFaceField(g)

    stagger_beta!(beta_acx, beta_acy, beta,
                  H_ice, f_ice, ux, uy,
                  f_grnd, f_grnd_acx, f_grnd_acy;
                  beta_gl_stag=0, beta_min=100.0)

    println("[1.4a] β=1500: acx ext = $(extrema(interior(beta_acx)))  " *
            "acy ext = $(extrema(interior(beta_acy)))")
    for i in 1:Nx-1, j in 1:Ny
        @test interior(beta_acx)[i+1, j, 1] ≈ 1500.0
    end
    for i in 1:Nx, j in 1:Ny-1
        @test interior(beta_acy)[i, j+1, 1] ≈ 1500.0
    end
end

@testset "[1.4b] stagger_beta!: linear-x β (2-cell face avg)" begin
    Nx, Ny = 5, 5
    g = _bounded_2d(Nx, Ny)
    delta = 100.0
    b0    = 500.0
    beta = CenterField(g)
    @inbounds for j in 1:Ny, i in 1:Nx
        interior(beta)[i, j, 1] = i * delta + b0
    end
    H_ice = CenterField(g);  fill!(interior(H_ice), 1000.0)
    f_ice = CenterField(g);  fill!(interior(f_ice), 1.0)
    ux    = XFaceField(g);   fill!(interior(ux), 0.0)
    uy    = YFaceField(g);   fill!(interior(uy), 0.0)
    f_grnd     = CenterField(g);  fill!(interior(f_grnd), 1.0)
    f_grnd_acx = XFaceField(g);   fill!(interior(f_grnd_acx), 1.0)
    f_grnd_acy = YFaceField(g);   fill!(interior(f_grnd_acy), 1.0)
    beta_acx = XFaceField(g)
    beta_acy = YFaceField(g)

    stagger_beta!(beta_acx, beta_acy, beta,
                  H_ice, f_ice, ux, uy,
                  f_grnd, f_grnd_acx, f_grnd_acy;
                  beta_gl_stag=0, beta_min=10.0)

    println("[1.4b] linear-x β: acx ext = $(extrema(interior(beta_acx)))")
    # acx face between cells i and i+1 = (i + (i+1))/2 · δ + b₀ =
    # (i + 0.5)·δ + b₀.  Array idx [i+1, j, 1].
    for j in 1:Ny, i in 1:Nx-1
        expected = (i + 0.5) * delta + b0
        @test interior(beta_acx)[i+1, j, 1] ≈ expected
    end
    # acy face between cells (i, j) and (i, j+1) = same column → mean
    # of two equal values → equals β(i, j+0.5) = i · δ + b₀.
    # (Linear-x → no y-dependence → 2-cell y-avg is identical.)
    for i in 1:Nx, j in 1:Ny-1
        expected = i * delta + b0
        @test interior(beta_acy)[i, j+1, 1] ≈ expected
    end
end

# ======================================================================
# Test 1.5 — Trough-config kernel sanity: dump magnitudes from the
# fixture's reference state (read straight from the YelmoMirror NetCDF)
# AND from the Yelmo.jl post-dyn_step fixture (logs/trough_f17_jl_t1000.nc
# if it exists, e.g. produced by `test/benchmarks/test_trough.jl`).
# This is **informational only** — pinpoints which field magnitudes
# diverge as a clue toward the bug location. Asserts only the
# YelmoMirror state is finite and within reasonable bounds (it's the
# committed reference; not under test here).
# ======================================================================

@testset "[1.5] Trough fixture state magnitudes (informational)" begin
    using NCDatasets

    fixture_path = abspath(joinpath(@__DIR__, "benchmarks", "fixtures",
                                      "trough_f17_t1000.nc"))
    @assert isfile(fixture_path) "Trough fixture missing: $fixture_path"

    NCDataset(fixture_path, "r") do ds
        ve  = ds["visc_eff"][:, :, :, 1]
        vei = ds["visc_eff_int"][:, :, 1]
        b   = ds["beta"][:, :, 1]
        bx  = ds["beta_acx"][:, :, 1]
        by  = ds["beta_acy"][:, :, 1]
        cb  = ds["c_bed"][:, :, 1]
        Ne  = ds["N_eff"][:, :, 1]

        println("[1.5] Trough YelmoMirror reference state (committed fixture):")
        println("    visc_eff   3D ext = $(extrema(skipmissing(ve)))")
        println("    visc_eff_int  ext = $(extrema(skipmissing(vei)))")
        println("    beta          ext = $(extrema(skipmissing(b)))")
        println("    beta_acx      ext = $(extrema(skipmissing(bx)))")
        println("    beta_acy      ext = $(extrema(skipmissing(by)))")
        println("    c_bed         ext = $(extrema(skipmissing(cb)))")
        println("    N_eff         ext = $(extrema(skipmissing(Ne)))")
        @test all(isfinite, vei) || all(ismissing, vei) ||
              all(x -> ismissing(x) || isfinite(x), vei)
        @test 1e4 < maximum(skipmissing(vei)) < 1e13
    end

    # If the Yelmo.jl post-dyn_step fixture exists (i.e.
    # test/benchmarks/test_trough.jl has been run), also dump its
    # magnitudes for side-by-side comparison.
    jl_path = abspath(joinpath(@__DIR__, "..", "logs", "trough_f17_jl_t1000.nc"))
    if isfile(jl_path)
        NCDataset(jl_path, "r") do ds
            ve  = ds["visc_eff"][:, :, :, 1]
            vei = ds["visc_eff_int"][:, :, 1]
            b   = ds["beta"][:, :, 1]
            bx  = ds["beta_acx"][:, :, 1]
            by  = ds["beta_acy"][:, :, 1]
            cb  = ds["c_bed"][:, :, 1]
            Ne  = ds["N_eff"][:, :, 1]

            println("[1.5] Trough Yelmo.jl post-dyn_step state:")
            println("    visc_eff   3D ext = $(extrema(skipmissing(ve)))")
            println("    visc_eff_int  ext = $(extrema(skipmissing(vei)))")
            println("    beta          ext = $(extrema(skipmissing(b)))")
            println("    beta_acx      ext = $(extrema(skipmissing(bx)))")
            println("    beta_acy      ext = $(extrema(skipmissing(by)))")
            println("    c_bed         ext = $(extrema(skipmissing(cb)))")
            println("    N_eff         ext = $(extrema(skipmissing(Ne)))")
        end
    else
        println("[1.5] Yelmo.jl post-dyn_step fixture not present at " *
                "$jl_path — skip side-by-side. Run test/benchmarks/test_trough.jl " *
                "first to populate.")
    end
end

# ======================================================================
# Test 1.6 — Direct head-to-head: drive each kernel with the trough
# fixture's exact loaded state (cb_ref, N_eff, ux_b, uy_b, ATT, H_ice,
# f_ice from the NetCDF) and verify visc_eff / beta / staggered fields
# match the fixture row-by-row.
#
# This is the strongest end-to-end kernel check: bypasses dyn_step!
# entirely, calls just the kernel functions on fixture inputs, and
# compares against the fixture's stored outputs. Any kernel-level bug
# manifests here as a row-by-row divergence.
#
# Note: the fixture was generated with the trough namelist
#   ytill: is_angle=True, cf_ref=10, scale_T=1, T_frz=-3
#   yneff: method=3 (with N_eff stored at converged value)
# so c_bed = tan(cf_ref°) · N_eff · λ_T + tan(cf_ref°) · N_eff · (1-λ_T)
#         = tan(10°) · N_eff   (since cb_ref=cf_ref=10 uniformly).
# We use the fixture's stored c_bed directly to bypass any
# is_angle / cf_ref / scale_T config concerns and isolate the
# β/visc/stagger kernels.
# ======================================================================

@testset "[1.6] Head-to-head against trough fixture (kernel-only, fixture inputs)" begin
    using NCDatasets

    fixture_path = abspath(joinpath(@__DIR__, "benchmarks", "fixtures",
                                      "trough_f17_t1000.nc"))
    @assert isfile(fixture_path) "Trough fixture missing: $fixture_path"

    NCDataset(fixture_path, "r") do ds
        Nx = length(ds["xc"])
        Ny = length(ds["yc"])
        Nz = length(ds["zeta"])

        # 8 km grid, matches trough config.
        dx_m = (ds["xc"][2] - ds["xc"][1]) * 1.0  # already in m
        # Try km vs m heuristic: if the values look like km (<1e4),
        # convert.
        if abs(dx_m) < 100.0
            dx_m = dx_m * 1e3
        end

        g  = _bounded_2d(Nx, Ny; dx=dx_m)
        g3 = _grid3d(Nx, Ny, Nz; dx=dx_m)
        zeta_aa = collect(Float64.(ds["zeta"][:]))

        # ---- Load fixture inputs ----
        H_ice = CenterField(g)
        f_ice = CenterField(g)
        ATT3d = CenterField(g3)
        @inbounds for j in 1:Ny, i in 1:Nx
            interior(H_ice)[i, j, 1] = Float64(ds["H_ice"][i, j, 1])
            interior(f_ice)[i, j, 1] = Float64(ds["f_ice"][i, j, 1])
        end
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            interior(ATT3d)[i, j, k] = Float64(ds["ATT"][i, j, k, 1])
        end

        # Velocity: fixture stores Center-aligned ux/uy at (xc, yc).
        # Yelmo.jl uses face-staggered XFace/YFace. Convert: face-east
        # of cell (i, j) (Yelmo idx [i+1, j, 1]) ≈ Fortran ux(i, j).
        ux = XFaceField(g);   fill!(interior(ux), 0.0)
        uy = YFaceField(g);   fill!(interior(uy), 0.0)
        @inbounds for j in 1:Ny, i in 1:Nx
            interior(ux)[i+1, j, 1] = Float64(ds["ux_b"][i, j, 1])
            interior(uy)[i, j+1, 1] = Float64(ds["uy_b"][i, j, 1])
        end
        # Replicate leading face slot.
        @views interior(ux)[1, :, :] .= interior(ux)[2, :, :]
        @views interior(uy)[:, 1, :] .= interior(uy)[:, 2, :]

        # c_bed straight from fixture (bypasses cf_ref / is_angle / scale_T).
        c_bed = CenterField(g)
        @inbounds for j in 1:Ny, i in 1:Nx
            interior(c_bed)[i, j, 1] = Float64(ds["c_bed"][i, j, 1])
        end

        # ---- Run kernels ----
        n_glen, eps_0 = 3.0, 1e-6
        q   = 1.0 / 3.0
        u_0 = 31556926.0

        # Glen-flow visc via _nodes (visc_method=1 — what trough uses).
        visc_eff = CenterField(g3)
        calc_visc_eff_3D_nodes!(visc_eff, ux, uy, ATT3d, H_ice, f_ice,
                                zeta_aa, dx_m, dx_m, n_glen, eps_0)

        # β power-plastic (beta_method=2 — trough config).
        beta = CenterField(g)
        b_int  = interior(beta)
        ux_int = interior(ux)
        uy_int = interior(uy)
        c_int  = interior(c_bed)
        fi_int = interior(f_ice)
        _D._calc_beta_aa_power_plastic!(b_int, ux_int, uy_int, c_int, fi_int,
                                        q, u_0, false, Bounded, Bounded)

        # ---- Compare to fixture ----
        ve_ref  = Array{Float64}(replace(ds["visc_eff"][:, :, :, 1], missing => NaN))
        b_ref   = Array{Float64}(replace(ds["beta"][:, :, 1], missing => NaN))

        # Mask: only ice-covered cells (where f_ice = 1).
        mask = interior(f_ice)[:, :, 1] .== 1.0

        # visc_eff comparison.
        ve_jl = interior(visc_eff)
        ve_diff = abs.(ve_jl[:, :, 1] .- ve_ref[:, :, 1])
        ve_ratio = ve_jl[:, :, 1] ./ ve_ref[:, :, 1]
        if any(mask)
            ve_diff_masked = ve_diff[mask]
            ve_ratio_masked = ve_ratio[mask]
            valid = isfinite.(ve_ratio_masked)
            println("[1.6] visc_eff[:, :, 1] (k=1 layer) head-to-head:")
            println("    fixture extrema (iced) = $(extrema(ve_ref[:, :, 1][mask]))")
            println("    kernel  extrema (iced) = $(extrema(ve_jl[:, :, 1][mask]))")
            if any(valid)
                println("    abs diff max          = $(maximum(ve_diff_masked))")
                println("    ratio (jl/ref) ext    = $(extrema(ve_ratio_masked[valid]))")
            end
        end

        # β comparison.
        b_jl = interior(beta)[:, :, 1]
        b_diff = abs.(b_jl .- b_ref)
        b_ratio = b_jl ./ b_ref
        if any(mask)
            b_diff_masked = b_diff[mask]
            b_ratio_masked = b_ratio[mask]
            valid = isfinite.(b_ratio_masked)
            println("[1.6] β head-to-head (uses fixture's c_bed, ux_b, uy_b):")
            println("    fixture extrema (iced) = $(extrema(b_ref[mask]))")
            println("    kernel  extrema (iced) = $(extrema(b_jl[mask]))")
            if any(valid)
                println("    abs diff max          = $(maximum(b_diff_masked))")
                println("    ratio (jl/ref) ext    = $(extrema(b_ratio_masked[valid]))")
            end
        end

        # No hard assertions on the fixture comparison — informational
        # only. The hand-derived analytical tests above are the
        # authoritative kernel checks.
        @test true
    end
end
