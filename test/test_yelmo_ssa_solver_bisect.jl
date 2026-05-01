## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Diagnostic-only: bisect why BiCGStab fails to converge on the slab
# SSA problem despite the assembled matrix matching the hand-derived
# analytical reference to ~1.45e-11 and the RHS matching to FP.
#
# Isolates whether the bug lives in:
#   (1) the system itself (matrix or RHS layout vs unpacking convention)
#   (2) the AMG preconditioner (smoothed-aggregation mis-handling)
#   (3) the BiCGStab call (Krylov.jl API misuse)
#
# Loads `logs/ssa_slab_assembly.nc` (dumped by test_yelmo_ssa_slab.jl,
# commit 94b2aed) and runs a sequence of solvers, reporting interior
# ux/uy statistics for each. NO production source modifications.
#
# This script is NOT a Test — it prints diagnostics. Exit code 0/1 is
# not meaningful; the goal is the printed output captured in
# `logs/bisect.log`.

using NCDatasets
using SparseArrays: SparseMatrixCSC, sparse, nnz as sparse_nnz
using LinearAlgebra
using Statistics: mean
using Printf
using Krylov
using AlgebraicMultigrid

const LOGS_DIR = abspath(joinpath(@__DIR__, "..", "logs"))
const ASM_PATH = joinpath(LOGS_DIR, "ssa_slab_assembly.nc")

println("=" ^ 70)
println("SSA solver bisection diagnostic")
println("=" ^ 70)
println("Loading: ", ASM_PATH)

# ------------------------------------------------------------------
# Load assembled matrix and RHS from the diagnostic dump.
# ------------------------------------------------------------------
ds = NCDataset(ASM_PATH)
nrows = ds.dim["nrows"]
Nx = ds.attrib["Nx"]
Ny = ds.attrib["Ny"]
A = SparseMatrixCSC{Float64,Int}(nrows, nrows,
    Vector{Int}(ds["csc_colptr"][:]),
    Vector{Int}(ds["csc_rowval"][:]),
    Vector{Float64}(ds["csc_nzval"][:]))
b = Vector{Float64}(ds["b"][:])
close(ds)

@printf("nrows = %d (Nx=%d, Ny=%d, expected 2*Nx*Ny=%d)\n",
        nrows, Nx, Ny, 2*Nx*Ny)
@printf("nnz(A) = %d\n", sparse_nnz(A))
@printf("‖b‖ = %.6e   ‖b‖_inf = %.6e\n", norm(b), maximum(abs.(b)))

# Quick conditioning sanity (cheap on 4202x4202).
A_dense = Matrix(A)
σ = svdvals(A_dense)
@printf("κ_2(A) ≈ %.3e (σ_max=%.3e, σ_min=%.3e)\n",
        σ[1] / σ[end], σ[1], σ[end])

# ------------------------------------------------------------------
# Helpers — analytical interior expectation.
# ------------------------------------------------------------------
const U_ANALYTICAL = 8.927  # m/yr (rho_ice * g * H * alpha / beta)

# Row-index convention used by assembler + unpacker (see _row_ux/_row_uy in
# src/dyn/velocity_ssa.jl, lines 338-339):
#   _ij2n(i,j,Nx)   = (j-1)*Nx + i
#   _row_ux(i,j,Nx) = 2 * _ij2n(i,j,Nx) - 1   (odd row)
#   _row_uy(i,j,Nx) = 2 * _ij2n(i,j,Nx)        (even row)
@inline ij2n(i, j) = (j - 1) * Nx + i
@inline row_ux_(i, j) = 2 * ij2n(i, j) - 1
@inline row_uy_(i, j) = 2 * ij2n(i, j)

function unpack_2d(x::AbstractVector)
    ux_2d = Matrix{Float64}(undef, Nx, Ny)
    uy_2d = Matrix{Float64}(undef, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        ux_2d[i, j] = x[row_ux_(i, j)]
        uy_2d[i, j] = x[row_uy_(i, j)]
    end
    return ux_2d, uy_2d
end

function report_solution(label::AbstractString, x::AbstractVector)
    ux_2d, uy_2d = unpack_2d(x)
    res_rel = norm(A * x - b) / norm(b)
    nb = 2
    ux_int = ux_2d[(1+nb):(end-nb), (1+nb):(end-nb)]
    uy_int = uy_2d[(1+nb):(end-nb), (1+nb):(end-nb)]
    @printf("  ‖Ax - b‖ / ‖b‖ = %.6e\n", res_rel)
    @printf("  ux full:     min=%.6e  max=%.6e  mean=%.6e\n",
            minimum(ux_2d), maximum(ux_2d), mean(ux_2d))
    @printf("  uy full:     min=%.6e  max=%.6e  mean=%.6e\n",
            minimum(uy_2d), maximum(uy_2d), mean(uy_2d))
    @printf("  ux interior: min=%.6e  max=%.6e  mean=%.6e (analytical ≈ %.6f)\n",
            minimum(ux_int), maximum(ux_int), mean(ux_int), U_ANALYTICAL)
    @printf("  uy interior: min=%.6e  max=%.6e  mean=%.6e\n",
            minimum(uy_int), maximum(uy_int), mean(uy_int))
    return ux_2d, uy_2d
end

# ------------------------------------------------------------------
# Test 1 — Direct sparse LU (A \ b).
# ------------------------------------------------------------------
println()
println("=" ^ 70)
println("Test 1 — Direct sparse LU (A \\ b)")
println("=" ^ 70)
x_lu = A \ b
ux_lu, uy_lu = report_solution("LU", x_lu)

# ------------------------------------------------------------------
# Test 2 — BiCGStab with no preconditioner.
# ------------------------------------------------------------------
println()
println("=" ^ 70)
println("Test 2 — BiCGStab, no preconditioner")
println("=" ^ 70)
ws2 = BicgstabWorkspace(nrows, nrows, Vector{Float64})
bicgstab!(ws2, A, b; rtol=1e-8, itmax=2000, history=true)
@printf("  solved=%s  niter=%d\n", ws2.stats.solved, ws2.stats.niter)
@printf("  residuals[end] = %s\n",
        isempty(ws2.stats.residuals) ? "<empty>" :
        @sprintf("%.6e", ws2.stats.residuals[end]))
report_solution("BiCGStab", copy(ws2.x))

# ------------------------------------------------------------------
# Test 3 — BiCGStab with Jacobi preconditioner (inverse diagonal).
# ------------------------------------------------------------------
println()
println("=" ^ 70)
println("Test 3 — BiCGStab + Jacobi preconditioner")
println("=" ^ 70)
diag_inv = 1.0 ./ diag(A)
@printf("  diag(A) min=%.3e max=%.3e (any zero? %s)\n",
        minimum(diag(A)), maximum(diag(A)), any(iszero, diag(A)) ? "YES" : "no")
M_jac = LinearAlgebra.Diagonal(diag_inv)

ws3 = BicgstabWorkspace(nrows, nrows, Vector{Float64})
# Jacobi: M is the preconditioner matrix that we multiply by (M ≈ A^-1),
# so use M=Diagonal(1/diag(A)) and ldiv=false (default mul!).
bicgstab!(ws3, A, b; M=M_jac, ldiv=false,
          rtol=1e-8, itmax=2000, history=true)
@printf("  solved=%s  niter=%d\n", ws3.stats.solved, ws3.stats.niter)
@printf("  residuals[end] = %s\n",
        isempty(ws3.stats.residuals) ? "<empty>" :
        @sprintf("%.6e", ws3.stats.residuals[end]))
report_solution("BiCGStab+Jacobi", copy(ws3.x))

# ------------------------------------------------------------------
# Test 4 — BiCGStab + AMG (production replication).
# ------------------------------------------------------------------
println()
println("=" ^ 70)
println("Test 4 — BiCGStab + AMG smoothed_aggregation (production config)")
println("=" ^ 70)
ml = smoothed_aggregation(A;
        presmoother  = AlgebraicMultigrid.GaussSeidel(),
        postsmoother = AlgebraicMultigrid.GaussSeidel())
M_amg = aspreconditioner(ml)
@printf("  AMG levels=%d\n", length(ml.levels))

ws4 = BicgstabWorkspace(nrows, nrows, Vector{Float64})
bicgstab!(ws4, A, b; M=M_amg, ldiv=true,
          rtol=1e-8, itmax=2000, history=true)
@printf("  solved=%s  niter=%d\n", ws4.stats.solved, ws4.stats.niter)
@printf("  residuals[end] = %s\n",
        isempty(ws4.stats.residuals) ? "<empty>" :
        @sprintf("%.6e", ws4.stats.residuals[end]))
report_solution("BiCGStab+AMG", copy(ws4.x))

# ------------------------------------------------------------------
# Test 5 — GMRES no preconditioner (sanity).
# ------------------------------------------------------------------
println()
println("=" ^ 70)
println("Test 5 — GMRES (restart=30), no preconditioner")
println("=" ^ 70)
ws5 = GmresWorkspace(nrows, nrows, Vector{Float64}; memory=30)
gmres!(ws5, A, b; rtol=1e-8, itmax=2000, history=true)
@printf("  solved=%s  niter=%d\n", ws5.stats.solved, ws5.stats.niter)
@printf("  residuals[end] = %s\n",
        isempty(ws5.stats.residuals) ? "<empty>" :
        @sprintf("%.6e", ws5.stats.residuals[end]))
report_solution("GMRES", copy(ws5.x))

# ------------------------------------------------------------------
# Test 6 — Inspect the production _solve_ssa_linear! call.
# ------------------------------------------------------------------
println()
println("=" ^ 70)
println("Test 6 — Production _solve_ssa_linear! API inspection")
println("=" ^ 70)
println("""
Production call (src/dyn/velocity_ssa.jl, lines 992-997):

    bicgstab!(workspace, A, b;
              M = M, ldiv = true,
              rtol = ssa.rtol, itmax = ssa.itmax,
              history = false)

Krylov.jl 0.10.6 API for bicgstab! is:
    bicgstab!(workspace, A, b; c, M=I, N=I, ldiv=false,
              atol=√eps(T), rtol=√eps(T), itmax=0,
              timemax=Inf, verbose=0, history=false, ...)

API check:
- positional args (workspace, A, b)        : OK
- M=preconditioner                         : OK
- ldiv=true (AMG overloads ldiv!)          : OK
- rtol kwarg                               : OK
- itmax kwarg                              : OK
- atol kwarg                               : NOT PASSED — defaults to sqrt(eps) ≈ 1.49e-8
  -> ‖b‖ in slab problem is large (Pa·yr scale of taud_acx ≈ 8.93e3 → ‖b‖ ~ O(1e5)),
     so atol of 1.49e-8 is irrelevant; rtol governs.
- history=false                            : means stats.residuals stays empty,
                                              but does NOT affect convergence.

No obvious API misuse: arg order, kwargs, and ldiv flag all match the docs.
""")

# ------------------------------------------------------------------
# Test 7 — Unpacking convention cross-check.
# ------------------------------------------------------------------
println()
println("=" ^ 70)
println("Test 7 — Row-index convention check (assembler vs unpacker)")
println("=" ^ 70)

# Assembler convention (src/dyn/velocity_ssa.jl:338-339):
#   _row_ux(i,j,Nx) = 2 * _ij2n(i,j,Nx) - 1   = ODD rows
#   _row_uy(i,j,Nx) = 2 * _ij2n(i,j,Nx)        = EVEN rows
# Unpacker convention (src/dyn/velocity_ssa.jl:1454-1460):
#   row_ux = _row_ux(i,j,Nx)  ; Ux[ip1f, j, 1] = x[row_ux]
#   row_uy = _row_uy(i,j,Nx)  ; Uy[i, jp1f, 1] = x[row_uy]
# => Same convention on both sides. Now verify against the LU result:
#    if odd rows are ux (slab pushes in +x), they should be ≈ +8.927 in interior;
#    even rows are uy, should be ≈ 0.

odd_idx = 1:2:nrows
even_idx = 2:2:nrows
@printf("x_lu odd  rows (alleged ux): min=%.6e  max=%.6e  mean=%.6e\n",
        minimum(x_lu[odd_idx]),  maximum(x_lu[odd_idx]),  mean(x_lu[odd_idx]))
@printf("x_lu even rows (alleged uy): min=%.6e  max=%.6e  mean=%.6e\n",
        minimum(x_lu[even_idx]), maximum(x_lu[even_idx]), mean(x_lu[even_idx]))

println()
println("Spot-check: middle interior cell.")
i_s = (Nx ÷ 2) + 1
j_s = (Ny ÷ 2) + 1
r_ux_s = row_ux_(i_s, j_s)
r_uy_s = row_uy_(i_s, j_s)
@printf("  (i=%d, j=%d) row_ux=%d (odd? %s), x_lu[ux]=%.6e\n",
        i_s, j_s, r_ux_s, isodd(r_ux_s), x_lu[r_ux_s])
@printf("  (i=%d, j=%d) row_uy=%d (even? %s), x_lu[uy]=%.6e\n",
        i_s, j_s, r_uy_s, iseven(r_uy_s), x_lu[r_uy_s])
@printf("  b[ux]=%.6e (taud_acx; should be > 0 for +x slope)\n", b[r_ux_s])
@printf("  b[uy]=%.6e (taud_acy; should be ≈ 0)\n", b[r_uy_s])

println()
println("=" ^ 70)
println("Bisection diagnostic complete.")
println("=" ^ 70)
