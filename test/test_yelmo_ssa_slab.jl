## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Regression: SSA on the SLAB-S06 setup (yelmo_slab.f90) with
# imposed-constant viscosity and imposed-constant beta.
#
# Goal: isolate WHERE the trough SSA divergence vs YelmoMirror lives.
# By forcing `visc_method = 0` (constant `visc_const`) and
# `beta_method = 0` (constant `beta_const`), the matrix-assembly +
# linear solve are the *only* dynamic components of the SSA Picard
# iteration. The viscosity / beta recompute paths are bypassed.
#
# Status: with the SSA preconditioner fix (commit landing alongside
# this test update — `precond = :jacobi` default in `SSASolver`), the
# slab Picard iteration converges to the LU reference solution to ~5
# decimal places. The "analytical plug-flow" value u_b = rho*g*H*alpha
# / beta = 8.927 m/yr is the *idealised* infinite-plane solution; the
# `boundaries = :bounded` configuration imposes Dirichlet u = 0 on all
# four edges (consistent with how the trough fixture is loaded), which
# produces a finite y-boundary drag — interior `ux_bar` peaks at the
# domain mid-line at ~8.920 m/yr (verified against LU in
# `test_yelmo_ssa_solver_bisect.jl`). The hard assertions below match
# the LU reference to a tight tolerance.
#
# Setup (from `par/yelmo_SLAB-S06.nml` and `tests/yelmo_slab.f90`):
#   - Grid: 51 x 41 cells, dx = 2 km (100 km x 80 km extent).
#   - Geometry: uniform H_ice = 1000 m, z_bed = -alpha * x_metres
#                (linear downward slope in +x), z_srf = z_bed + H.
#   - alpha = 1e-3 (so dz_s/dx = -alpha).
#   - visc_const = 1e7 [Pa yr], beta_const = 1e3 [Pa yr / m].
#   - All grounded, no calving front, ssa_lat_bc = "none".
#   - solver = "ssa", n_glen = 3.0.
#
# Pure-SSA analytical (no SIA shear contribution):
#   u_b = rho_ice * g * H * alpha / beta
#       = 910 * 9.81 * 1000 * 1e-3 / 1e3 ~= 8.927 m/yr   (uniform interior).
#
# This file is intentionally NOT under test/benchmarks/ -- it is a
# debugging probe, not a fixture-backed regression. It writes the
# Yelmo.jl post-`dyn_step!` state to `<worktree>/logs/` for the user
# to inspect in ncview / NCDatasets.
#
# Loose assertions only: finiteness, Picard iteration count, and the
# x-only flow direction (uy_bar << ux_bar). The analytical match is
# *characterised* (printed numbers) -- NOT hard-asserted, since the
# trough divergence is being investigated.

using Test
using Yelmo
using Oceananigans: interior
using Oceananigans.Grids: Bounded
using NCDatasets
using Statistics: mean
using SparseArrays: SparseMatrixCSC, sparse, nnz as sparse_nnz

using Yelmo.YelmoModelPar: YelmoModelParameters, ydyn_params, ymat_params,
                            yneff_params, ytill_params

# ----------------------------------------------------------------------
# Synthetic restart fixture for the SLAB-S06 setup.
# ----------------------------------------------------------------------
# Mirrors `_write_ssa_slab_fixture!` in `test_yelmo_ssa.jl` but with
# the larger domain (51 x 41) and the SLAB-S06 slope direction.
function _write_slab_s06_fixture!(path::AbstractString;
                                  Nx::Int, Ny::Int, dx::Float64,
                                  H_const::Float64, alpha::Float64,
                                  Nz::Int)
    xc_m = collect(range(0.5*dx, (Nx - 0.5)*dx; length=Nx))
    yc_m = collect(range(0.5*dx, (Ny - 0.5)*dx; length=Ny))
    zeta_ac = collect(range(0.0, 1.0; length=Nz + 1))
    zeta_rock_ac = collect(range(0.0, 1.0; length=5))

    NCDataset(path, "c") do ds
        defDim(ds, "xc",           Nx)
        defDim(ds, "yc",           Ny)
        defDim(ds, "zeta",         Nz)
        defDim(ds, "zeta_ac",      Nz + 1)
        defDim(ds, "zeta_rock",    length(zeta_rock_ac) - 1)
        defDim(ds, "zeta_rock_ac", length(zeta_rock_ac))

        xv = defVar(ds, "xc", Float64, ("xc",))
        xv[:] = xc_m ./ 1e3
        xv.attrib["units"] = "km"
        yv = defVar(ds, "yc", Float64, ("yc",))
        yv[:] = yc_m ./ 1e3
        yv.attrib["units"] = "km"

        zc = defVar(ds, "zeta", Float64, ("zeta",))
        zc[:] = 0.5 .* (zeta_ac[1:end-1] .+ zeta_ac[2:end])
        zc.attrib["units"] = "1"
        zac = defVar(ds, "zeta_ac", Float64, ("zeta_ac",))
        zac[:] = zeta_ac
        zac.attrib["units"] = "1"
        zrc = defVar(ds, "zeta_rock", Float64, ("zeta_rock",))
        zrc[:] = 0.5 .* (zeta_rock_ac[1:end-1] .+ zeta_rock_ac[2:end])
        zrc.attrib["units"] = "1"
        zrac = defVar(ds, "zeta_rock_ac", Float64, ("zeta_rock_ac",))
        zrac[:] = zeta_rock_ac
        zrac.attrib["units"] = "1"

        # SLAB-S06: uniform H, linear slope -alpha in +x.
        H     = fill(H_const, Nx, Ny)
        z_bed = zeros(Nx, Ny)
        f_ice = ones(Nx, Ny)
        # All-grounded: z_sl far below z_bed.
        z_sl  = fill(-1e6, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            z_bed[i, j] = -alpha * xc_m[i]
        end

        for (name, arr) in (("H_ice", H), ("z_bed", z_bed),
                            ("f_ice", f_ice), ("z_sl", z_sl))
            v = defVar(ds, name, Float64, ("xc", "yc"))
            v[:, :] = arr
        end
    end
    return path
end

@testset "diagnostic: SSA SLAB-S06 with imposed visc/beta" begin
    # ----- SLAB-S06 parameters (yelmo_SLAB-S06.nml) -----
    Nx, Ny     = 51, 41
    dx         = 2_000.0          # [m]   2 km grid
    H_const    = 1000.0           # [m]
    alpha      = 1e-3             # [-]   dimensionless slope magnitude
    visc_const = 1e7              # [Pa yr]
    beta_const = 1e3              # [Pa yr / m]
    n_glen     = 3.0
    Nz         = 4

    # Default Yelmo constants (rho_ice=910, g=9.81).
    rho_ice = 910.0
    g_acc   = 9.81

    # Pure-SSA analytical plug flow: u_b = rho g H alpha / beta.
    # (The Fortran formula `(1 + beta*F2)/beta` adds a SIA contribution
    # `beta*F2 * u_b` where F2 = H/(3*eta); for solver="ssa" the SIA
    # branch is not invoked, so the SIA contribution is absent.)
    u_b_analytical = rho_ice * g_acc * H_const * alpha / beta_const

    @info "SLAB-S06 setup" Nx Ny dx_km=dx/1e3 H_const alpha visc_const beta_const u_b_analytical

    # ----- Build YelmoModel from synthetic restart -----
    tdir = mktempdir(; prefix="ssa_slab_s06_")
    path = joinpath(tdir, "ssa_slab_s06_restart.nc")
    _write_slab_s06_fixture!(path; Nx=Nx, Ny=Ny, dx=dx,
                             H_const=H_const, alpha=alpha, Nz=Nz)

    p = YelmoModelParameters("ssa_slab_diag";
        ydyn = ydyn_params(
            solver         = "ssa",
            visc_method    = 0,                # constant viscosity = visc_const
            visc_const     = visc_const,
            beta_method    = 0,                # constant beta = beta_const
            beta_const     = beta_const,
            beta_gl_scale  = 0,                # no GL scaling
            beta_min       = 0.0,
            ssa_lat_bc     = "none",           # try "none" first per prompt
            ssa_solver     = SSASolver(rtol            = 1e-8,
                                       itmax           = 500,
                                       picard_tol      = 1e-6,
                                       picard_iter_max = 50,
                                       picard_relax    = 0.7),
        ),
        yneff = yneff_params(method = -1, const_ = 1e7),
        ytill = ytill_params(method = -1),
        ymat  = ymat_params(n_glen = n_glen),
    )

    y = YelmoModel(path, 0.0;
                   rundir     = tdir,
                   alias      = "ssa_slab_s06",
                   p          = p,
                   boundaries = :bounded,    # SLAB-S06 (Bounded, Bounded)
                   strict     = false)

    # External fields used by visc_method=1/2 / cb_ref / N_eff are
    # filled but should not influence the result with imposed
    # visc/beta. Harmless.
    fill!(interior(y.mat.ATT), 1e-16)
    fill!(interior(y.dyn.cb_ref), 1.0)
    fill!(interior(y.dyn.N_eff), 1e7)

    # Initial guess: zero -- stress-test the solver.
    fill!(interior(y.dyn.ux_b),   0.0)
    fill!(interior(y.dyn.uy_b),   0.0)
    fill!(interior(y.dyn.ux_bar), 0.0)
    fill!(interior(y.dyn.uy_bar), 0.0)

    Yelmo.update_diagnostics!(y)

    # ----- Run a single dyn_step! -----
    Yelmo.YelmoModelDyn.dyn_step!(y, 1.0)

    iter_count = y.dyn.scratch.ssa_iter_now[]
    picard_iter_max = y.p.ydyn.ssa_solver.picard_iter_max
    @info "SSA SLAB-S06 Picard iterations" iter_count picard_iter_max

    # ----- Smoke checks (loose, diagnostic) -----
    Ux_bar = interior(y.dyn.ux_bar)
    Uy_bar = interior(y.dyn.uy_bar)

    @test all(isfinite, Ux_bar)
    @test all(isfinite, Uy_bar)
    @test iter_count > 0
    @test iter_count < picard_iter_max

    # ----- Verify imposed visc / beta were applied -----
    visc_eff = interior(y.dyn.visc_eff)
    beta_aa  = interior(y.dyn.beta)
    @info "Imposed-coefficient check" visc_eff_min=minimum(visc_eff) visc_eff_max=maximum(visc_eff) beta_min=minimum(beta_aa) beta_max=maximum(beta_aa)
    # visc_method=0 should fill with visc_const; beta_method=0 with beta_const.
    @test maximum(abs.(visc_eff .- visc_const)) < 1e-6 * visc_const
    @test maximum(abs.(beta_aa  .- beta_const))  < 1e-6 * beta_const

    # ----- Diagnostic: per-cell observations -----
    # Trim a 2-cell border (Dirichlet-affected) to define the
    # "interior" used for the analytical comparison.
    nb = 2
    Ux_int = Ux_bar[(1 + nb):(end - nb), (1 + nb):(end - nb), 1]
    Uy_int = Uy_bar[(1 + nb):(end - nb), (1 + nb):(end - nb), 1]

    ux_min  = minimum(Ux_int)
    ux_max  = maximum(Ux_int)
    ux_mean = mean(Ux_int)
    uy_max_abs = maximum(abs.(Uy_int))
    err_abs = maximum(abs.(Ux_int .- u_b_analytical))
    err_rel = err_abs / abs(u_b_analytical)

    @info "Interior ux_bar (border=$nb trimmed)" ux_min ux_max ux_mean u_b_analytical err_abs err_rel
    @info "Interior |uy_bar| max" uy_max_abs ratio_uy_to_ux=uy_max_abs / max(ux_max, eps())

    # Per-row x-profile (middle y-row) and per-column y-profile (middle x-col).
    j_mid = (Ny ÷ 2) + 1
    # ux_bar is XFaceField -> first dim is Nx+1 (or Nx for periodic-x).
    # Print across the full face stencil.
    println("\n=== x-profile of ux_bar at j=$j_mid (middle y-row) ===")
    println(Ux_bar[:, j_mid, 1])

    i_mid = (Nx ÷ 2) + 1
    println("\n=== y-profile of ux_bar at i=$i_mid (middle x-col) ===")
    println(Ux_bar[i_mid, :, 1])

    println("\n=== y-profile of uy_bar at i=$i_mid (middle x-col) ===")
    println(Uy_bar[i_mid, :, 1])

    println("\n=== x-profile of uy_bar at j=$j_mid (middle y-row) ===")
    println(Uy_bar[:, j_mid, 1])

    # ----- Hard-assert: cross-coupling sanity -----
    # The slope is in x only; uy_bar should be << ux_bar in magnitude.
    # If this fails, the bug is in the matrix cross-coupling.
    @test uy_max_abs < 0.1 * max(ux_max, eps())

    # ----- Hard-assert: interior solution matches LU reference -----
    # Reference: BiCGStab + Jacobi (default precond) reproduces the
    # direct sparse-LU solution `A \ b` to ~5e-9 relative residual on
    # this 4-side-Dirichlet 51 x 41 problem (see
    # `test_yelmo_ssa_solver_bisect.jl` Tests 1 & 3). The interior
    # `ux_bar` peaks at the domain mid-line at +8.9204 m/yr, slightly
    # below the idealised plug-flow value of 8.927 m/yr because the
    # y-boundaries impose lateral drag. We assert against the LU peak
    # (`8.920414`) to 4 significant figures — tight enough to catch a
    # solver regression but loose enough to be insensitive to
    # rtol-level Krylov tail noise.
    u_lu_peak = 8.920414
    @test isapprox(ux_max, u_lu_peak; atol = 1e-3)
    @test ux_min  > 0.0                      # no upstream reversal
    @test ux_mean > 0.5 * u_lu_peak          # bulk +x flow direction

    # NOTE: the *idealised* plug-flow analytical (`u_b_analytical =
    # 8.927 m/yr`) is the infinite-plane solution and is NOT
    # attainable on this all-Dirichlet 4-edge problem. The interior
    # falls short by a few percent depending on boundary location;
    # only the central peak approaches the analytical limit.

    # ----- Write Yelmo.jl post-dyn_step! state to logs/ for inspection -----
    logs_dir = abspath(joinpath(@__DIR__, "..", "logs"))
    mkpath(logs_dir)
    jl_out_path = joinpath(logs_dir, "ssa_slab_jl_dyn_step.nc")
    isfile(jl_out_path) && rm(jl_out_path)
    out = init_output(y, jl_out_path;
                      selection = OutputSelection(groups = [:tpo, :dyn, :mat, :bnd]))
    write_output!(out, y)
    close(out)
    @info "Yelmo.jl post-dyn_step! state written" path=jl_out_path

    # ----- Dump assembled SSA matrix (last Picard iteration) -----
    # Snapshot the COO triplets + RHS that `_assemble_ssa_matrix!` last
    # wrote, plus the CSC view that `calc_velocity_ssa!` actually solves.
    # Diagnostic only — feeds the analytical-stencil comparison below.
    asm_path = joinpath(logs_dir, "ssa_slab_assembly.nc")
    Yelmo.YelmoModelDyn.dump_ssa_assembly(y; path = asm_path)
    @info "SSA assembly dumped" path=asm_path

    # ===== Analytical reference stencil + row-by-row compare =====
    #
    # DIAGNOSTIC SCOPE — handles ONLY the uniform-coefficient slab:
    #   - constant H, eta, beta -> N = eta*H, Nab = N
    #   - dx = dy = h
    #   - all ice-covered, all grounded, no calving front
    #   - boundary palette = (:no_slip, :no_slip, :no_slip, :no_slip)
    #     (matches `boundaries=:bounded` -> all 4 edges Dirichlet u=0)
    #
    # For the uniform interior cell, the SSA stencil under Yelmo's
    # discretization (`_assemble_ssa_matrix!` inner-SSA branch) reduces
    # to (N := visc_eff_int = eta*H, beta = beta_const):
    #
    #   ux row (cell i, j):
    #     diag(ux(i,j)):   -10 N/h^2 - beta
    #     ux(i+/-1, j):    +4  N/h^2
    #     ux(i, j+/-1):    +1  N/h^2
    #     uy(i, j):        -3  N/h^2     [= -2 N/h^2 (Naa) - 1 N/h^2 (Nab)]
    #     uy(i+1, j):      +3  N/h^2
    #     uy(i+1, j-1):    -3  N/h^2
    #     uy(i, j-1):      +3  N/h^2
    #     b(ux(i,j)):      taud_acx (= rho*g*H*dzs/dx; copied verbatim)
    #
    #   uy row (cell i, j):  symmetric (swap x<->y, swap u<->v)
    #     diag(uy(i,j)):   -10 N/h^2 - beta
    #     uy(i, j+/-1):    +4  N/h^2
    #     uy(i+/-1, j):    +1  N/h^2
    #     ux(i, j):        -3  N/h^2
    #     ux(i, j+1):      +3  N/h^2
    #     ux(i-1, j+1):    -3  N/h^2
    #     ux(i-1, j):      +3  N/h^2
    #
    # Boundary cells (i=1, i=Nx, j=1, j=Ny under :bounded):
    #     Identity row: A[r, r] = 1.0, b[r] = 0.0.
    #
    # NOT production-ready — Yelmo's `_assemble_ssa_matrix!` handles
    # mask 0 / -1 / 3, lateral BCs, GL refinement, etc. that this
    # simplified reference omits.

    function _slab_ref_matrix(; Nx::Int, Ny::Int, h::Float64,
                              N::Float64, beta::Float64,
                              taud_x::Function, taud_y::Function)
        nrows = 2 * Nx * Ny
        I = Int[]; J = Int[]; V = Float64[]
        b = zeros(Float64, nrows)

        ij2n   = (i, j) -> (j - 1) * Nx + i
        row_ux = (i, j) -> 2 * ij2n(i, j) - 1
        row_uy = (i, j) -> 2 * ij2n(i, j)

        push_!(r, c, v) = (push!(I, r); push!(J, c); push!(V, v))

        invh2 = 1.0 / (h * h)
        diag_inner = -10.0 * N * invh2 - beta
        coef_4N = 4.0 * N * invh2
        coef_N  = 1.0 * N * invh2
        coef_3N = 3.0 * N * invh2

        for j in 1:Ny, i in 1:Nx
            r_ux = row_ux(i, j)
            r_uy = row_uy(i, j)

            # ----- ux row -----
            if i == 1 || i == Nx || j == 1 || j == Ny
                # Boundary: no-slip identity row.
                push_!(r_ux, r_ux, 1.0)
                b[r_ux] = 0.0
            else
                # Inner SSA stencil.
                push_!(r_ux, row_ux(i,   j),   diag_inner)
                push_!(r_ux, row_ux(i+1, j),   coef_4N)
                push_!(r_ux, row_ux(i-1, j),   coef_4N)
                push_!(r_ux, row_ux(i,   j+1), coef_N)
                push_!(r_ux, row_ux(i,   j-1), coef_N)
                push_!(r_ux, row_uy(i,   j),   -coef_3N)
                push_!(r_ux, row_uy(i+1, j),   +coef_3N)
                push_!(r_ux, row_uy(i+1, j-1), -coef_3N)
                push_!(r_ux, row_uy(i,   j-1), +coef_3N)
                b[r_ux] = taud_x(i, j)
            end

            # ----- uy row -----
            # Yelmo's branch order: j==1 / j==Ny boundary checks fire
            # BEFORE i==1 / i==Nx for the uy row.
            if j == 1 || j == Ny || i == 1 || i == Nx
                push_!(r_uy, r_uy, 1.0)
                b[r_uy] = 0.0
            else
                push_!(r_uy, row_uy(i,   j),   diag_inner)
                push_!(r_uy, row_uy(i,   j+1), coef_4N)
                push_!(r_uy, row_uy(i,   j-1), coef_4N)
                push_!(r_uy, row_uy(i+1, j),   coef_N)
                push_!(r_uy, row_uy(i-1, j),   coef_N)
                push_!(r_uy, row_ux(i,   j),   -coef_3N)
                push_!(r_uy, row_ux(i,   j+1), +coef_3N)
                push_!(r_uy, row_ux(i-1, j+1), -coef_3N)
                push_!(r_uy, row_ux(i-1, j),   +coef_3N)
                b[r_uy] = taud_y(i, j)
            end
        end

        A = sparse(I, J, V, nrows, nrows)
        return A, b
    end

    # Reference parameters: N = visc_const * H_const (depth-integrated);
    # beta = beta_const; taud copied straight from Yelmo's interior
    # array (it bakes in the linear slope + the H_mid).
    N_ref    = visc_const * H_const
    beta_ref = beta_const

    # Pull Yelmo's actual taud_acx / taud_acy on faces — easier than
    # rederiving the H_mid + dzsdx logic.
    Tdx = interior(y.dyn.taud_acx)
    Tdy = interior(y.dyn.taud_acy)

    A_ref, b_ref = _slab_ref_matrix(;
        Nx = Nx, Ny = Ny, h = dx,
        N = N_ref, beta = beta_ref,
        taud_x = (i, j) -> Tdx[i + 1, j, 1],   # XFace stagger: cell i -> slot i+1
        taud_y = (i, j) -> Tdy[i, j + 1, 1],   # YFace stagger: cell j -> slot j+1
    )

    # ----- Read Yelmo's dumped matrix back -----
    NCDataset(asm_path) do ds
        nrows_jl  = ds.dim["nrows"]
        nnz_coo   = ds.attrib["nnz_coo"]
        nnz_csc   = ds.attrib["nnz_csc"]
        I_coo     = ds["I"][:]
        J_coo     = ds["J"][:]
        V_coo     = ds["V"][:]
        b_jl      = ds["b"][:]
        colptr_jl = ds["csc_colptr"][:]
        rowval_jl = ds["csc_rowval"][:]
        nzval_jl  = ds["csc_nzval"][:]

        @info "Assembly stats" nrows_jl nnz_coo nnz_csc dup_summed=(nnz_coo - nnz_csc)

        # Build Julia SparseMatrixCSC directly from CSC arrays.
        A_jl = SparseMatrixCSC(nrows_jl, nrows_jl,
                               Vector{Int}(colptr_jl),
                               Vector{Int}(rowval_jl),
                               Vector{Float64}(nzval_jl))

        # ----- Dense diff (small matrix: 4202 x 4202 for 51x41) -----
        A_jl_d  = Matrix(A_jl)
        A_ref_d = Matrix(A_ref)
        A_diff  = A_jl_d .- A_ref_d
        b_diff  = b_jl .- b_ref

        max_A_err = maximum(abs.(A_diff))
        max_b_err = maximum(abs.(b_diff))
        @info "Matrix diff (Yelmo - reference)" max_A_err max_b_err

        # Per-row max-abs diff and worst-row ranking.
        row_max_err = vec(maximum(abs.(A_diff); dims = 2))
        worst_rows  = sortperm(row_max_err; rev = true)[1:min(20, length(row_max_err))]

        println("\n=== Top 20 rows by max |Δ| (Yelmo - reference) ===")
        for r in worst_rows
            row_max_err[r] == 0 && continue
            Δrow = A_diff[r, :]
            nz   = findall(!iszero, Δrow)
            # Decode r -> (eq_kind, i, j).
            cell  = (r + 1) ÷ 2
            i_r   = ((cell - 1) % Nx) + 1
            j_r   = ((cell - 1) ÷ Nx) + 1
            kind  = isodd(r) ? "ux" : "uy"
            on_b  = (i_r == 1 || i_r == Nx || j_r == 1 || j_r == Ny)
            tag   = on_b ? "BNDY" : "INNER"
            println("  row $r [$tag $kind(i=$i_r, j=$j_r)]: max|Δ|=$(row_max_err[r])")
            println("    nz cols: $nz")
            println("    Δ vals:  $(Δrow[nz])")
        end

        # RHS diff.
        b_max_idx = argmax(abs.(b_diff))
        cell_b    = (b_max_idx + 1) ÷ 2
        ib        = ((cell_b - 1) % Nx) + 1
        jb        = ((cell_b - 1) ÷ Nx) + 1
        kind_b    = isodd(b_max_idx) ? "ux" : "uy"
        on_b_b    = (ib == 1 || ib == Nx || jb == 1 || jb == Ny)
        println("\n=== RHS diff (Yelmo - reference) ===")
        println("  max |Δb| = $max_b_err at index $b_max_idx ($kind_b at i=$ib, j=$jb, $(on_b_b ? "BNDY" : "INNER"))")
        println("  b_jl  = $(b_jl[b_max_idx])")
        println("  b_ref = $(b_ref[b_max_idx])")

        # Spot-check a single representative interior row.
        i_s, j_s = (Nx ÷ 2) + 1, (Ny ÷ 2) + 1
        r_s = 2 * ((j_s - 1) * Nx + i_s) - 1   # ux row
        println("\n=== Spot-check interior row (ux at i=$i_s, j=$j_s, row=$r_s) ===")
        nz_jl  = findall(!iszero, A_jl_d[r_s, :])
        nz_ref = findall(!iszero, A_ref_d[r_s, :])
        println("  Yelmo nz cols ($(length(nz_jl))): $nz_jl")
        println("  Yelmo vals:    $(A_jl_d[r_s, nz_jl])")
        println("  Ref   nz cols ($(length(nz_ref))): $nz_ref")
        println("  Ref   vals:    $(A_ref_d[r_s, nz_ref])")
        println("  b_jl  = $(b_jl[r_s])")
        println("  b_ref = $(b_ref[r_s])")
    end
end
