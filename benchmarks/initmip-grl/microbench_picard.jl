# benchmarks/initmip-grl/microbench_picard.jl
#
# Standalone microbenchmark for picard_relax_visc! kernel variants.
# Compares the existing branched scalar loop against branchless+@simd
# and @turbo (LoopVectorization). Runs on a synthetic 106x181x10 array
# matching GRL-16KM viscosity dimensions.
#
# Usage:
#   julia --project=. microbench_picard.jl

using BenchmarkTools
using LoopVectorization
using Statistics

const NX, NY, NZ = 106, 181, 10
const FRAC_ICE   = 0.85   # rough fraction inside the Greenland mask — most cells are iced

function make_test_data(; seed::Int = 42)
    # Deterministic data: 55% positive viscosity values, rest are 0 (ice-free).
    rng_state = seed
    V    = zeros(Float64, NX, NY, NZ)
    Vnm1 = zeros(Float64, NX, NY, NZ)
    @inbounds for i in eachindex(V)
        rng_state = (rng_state * 1103515245 + 12345) & ((1 << 31) - 1)
        if (rng_state / (1 << 31)) < FRAC_ICE
            # Viscosity values span ~5 orders of magnitude (1e5 - 1e10 Pa·yr)
            r = (rng_state ÷ 1000) % 1_000_000 / 1_000_000.0
            V[i]    = 1e5 * exp(11 * r)         # 1e5 - ~5e9
            Vnm1[i] = V[i] * (0.5 + r)          # similar order of magnitude
        end
    end
    return V, Vnm1
end

# ---------- Variant 0: original (branched scalar) ----------
function picard_v0!(V, Vnm1, rel::Float64)
    one_minus_rel = 1.0 - rel
    @inbounds for i in eachindex(V)
        v_now  = V[i]
        v_prev = Vnm1[i]
        if v_now > 0.0 && v_prev > 0.0
            V[i] = exp(one_minus_rel * log(v_prev) + rel * log(v_now))
        end
    end
    return V
end

# ---------- Variant A: branchless + @simd ivdep ----------
function picard_va!(V, Vnm1, rel::Float64)
    one_minus_rel = 1.0 - rel
    @inbounds @simd ivdep for i in eachindex(V)
        v_now  = V[i]
        v_prev = Vnm1[i]
        # both_pos true → apply relaxation, else keep V[i] unchanged
        both_pos = (v_now > 0.0) & (v_prev > 0.0)
        # Floor masked-out values to 1.0 so log/exp don't NaN
        a = ifelse(both_pos, v_prev, 1.0)
        b = ifelse(both_pos, v_now,  1.0)
        v_new = exp(one_minus_rel * log(a) + rel * log(b))
        V[i] = ifelse(both_pos, v_new, v_now)
    end
    return V
end

# ---------- Variant B: LoopVectorization @turbo ----------
function picard_vb!(V, Vnm1, rel::Float64)
    one_minus_rel = 1.0 - rel
    @turbo for i in eachindex(V)
        v_now  = V[i]
        v_prev = Vnm1[i]
        both_pos = (v_now > 0.0) & (v_prev > 0.0)
        a = ifelse(both_pos, v_prev, 1.0)
        b = ifelse(both_pos, v_now,  1.0)
        v_new = exp(one_minus_rel * log(a) + rel * log(b))
        V[i] = ifelse(both_pos, v_new, v_now)
    end
    return V
end

# ---------- Correctness check: A and B must match V0 within FP tolerance ----------
function check_correctness(V0, Va, Vb)
    # Compare in log-space to avoid scale issues; positive cells only
    mask = (V0 .> 0)
    function relerr(X, Y)
        diff = abs.(log.(X[mask]) .- log.(Y[mask]))
        (mean = mean(diff), max = maximum(diff))
    end
    a = relerr(V0, Va)
    b = relerr(V0, Vb)
    println("Correctness (log-space rel err on ice cells):")
    println("  variant A vs V0:  mean = $(a.mean), max = $(a.max)")
    println("  variant B vs V0:  mean = $(b.mean), max = $(b.max)")
    # Both A and B should be exact (same arithmetic, branchless reorder only)
    # but FP reordering may introduce ulps. Tol ~1e-12 in log-space.
    ok = a.max < 1e-10 && b.max < 1e-10
    println("  → ", ok ? "PASS" : "FAIL")
    return ok
end

function main()
    println("─── picard_relax_visc! microbench ───")
    println("Array size: $NX × $NY × $NZ = $(NX*NY*NZ) cells")
    println("Active (ice) cells: ~$(round(Int, FRAC_ICE * NX * NY * NZ)) (~$(round(Int, FRAC_ICE*100))%)")
    println()

    V_ref, Vnm1 = make_test_data()
    rel = 0.7

    # Reference output
    V0 = copy(V_ref); picard_v0!(V0, Vnm1, rel)
    Va = copy(V_ref); picard_va!(Va, Vnm1, rel)
    Vb = copy(V_ref); picard_vb!(Vb, Vnm1, rel)

    ok = check_correctness(V0, Va, Vb)
    ok || error("Correctness check failed — aborting microbench")
    println()

    # Warmup
    for _ in 1:3
        Vw = copy(V_ref); picard_v0!(Vw, Vnm1, rel)
        Vw = copy(V_ref); picard_va!(Vw, Vnm1, rel)
        Vw = copy(V_ref); picard_vb!(Vw, Vnm1, rel)
    end

    println("─── benchmarks ───")
    println("V0 (original branched):")
    b0 = @benchmark picard_v0!($(copy(V_ref)), $Vnm1, $rel)
    show(stdout, MIME"text/plain"(), b0); println(); println()

    println("VA (branchless + @simd ivdep):")
    ba = @benchmark picard_va!($(copy(V_ref)), $Vnm1, $rel)
    show(stdout, MIME"text/plain"(), ba); println(); println()

    println("VB (@turbo):")
    bb = @benchmark picard_vb!($(copy(V_ref)), $Vnm1, $rel)
    show(stdout, MIME"text/plain"(), bb); println(); println()

    # Summary
    t0 = median(b0).time
    ta = median(ba).time
    tb = median(bb).time
    println("─── speedup vs V0 (median) ───")
    println("  VA: ", round(t0 / ta, digits=2), "×")
    println("  VB: ", round(t0 / tb, digits=2), "×")
    return nothing
end

main()
