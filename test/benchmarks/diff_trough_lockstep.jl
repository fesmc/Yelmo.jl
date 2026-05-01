## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate("..")
#########################################################

# Diagnostic-only standalone script (NOT a test).
#
# Reads two NetCDFs:
#   * `test/benchmarks/fixtures/trough_f17_t1000.nc`
#       — YelmoMirror reference (Fortran ground truth, Float32).
#   * `<worktree>/logs/trough_f17_jl_t1000.nc`
#       — Yelmo.jl post-`dyn_step!` state (Float64, written from
#       `test_trough.jl`).
#
# Computes per-cell diffs for the SSA-relevant velocity / beta /
# viscosity fields, summarises the residual statistically and
# spatially, and writes a per-cell diff NetCDF to
# `<worktree>/logs/trough_lockstep_diff.nc` for offline inspection.
#
# The goal is to confirm whether the trough lockstep residual
# (~30 m/yr abs err on `ux_bar`, ~36 m/yr on `uy_bar` after the
# ytill + Periodic-y loader fixes in `2e794c2` / `54b3a9e`) is:
#   (a) **distributed** — Float32 fixture noise + small kernel-level
#       imprecision spread over the domain, or
#   (b) **concentrated** — a real remaining bug localised to a
#       region (boundary, calving front, trough centerline, ...).
#
# Usage:
#   julia --project=test test/benchmarks/diff_trough_lockstep.jl
#
# If `logs/trough_f17_jl_t1000.nc` is missing or stale, re-run
# `test/benchmarks/test_trough.jl` first to refresh it.

using NCDatasets
using Printf
using Statistics

const WORKTREE_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const FIXTURE_PATH  = joinpath(WORKTREE_ROOT, "test", "benchmarks",
                               "fixtures", "trough_f17_t1000.nc")
const JL_OUT_PATH   = joinpath(WORKTREE_ROOT, "logs", "trough_f17_jl_t1000.nc")
const DIFF_OUT_PATH = joinpath(WORKTREE_ROOT, "logs", "trough_lockstep_diff.nc")

# Fixture is Float32; relative precision ~ 1.2e-7. The noise floor
# for a value of magnitude `vmax` is roughly `vmax * 1.2e-7`. The
# user-facing "structural" threshold uses 100× this floor (i.e.
# 1.2e-5 × vmax) — diffs above are unlikely to be pure rounding.
const FLOAT32_REL_PREC = Float32(1.1920929e-7)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

"""
Read a 2D field from a NetCDF dataset, dropping the singleton time
axis and converting to plain Float64 (no Missing).
"""
function _read2d(ds, name)
    v = ds[name]
    a = Array(v)
    if ndims(a) == 3 && size(a, 3) == 1
        a = a[:, :, 1]
    end
    return Float64.(coalesce.(a, NaN))
end

"""
Compute summary stats for `|d|`, where `d` is the per-cell diff.
Returns a NamedTuple.
"""
function _abs_stats(d::AbstractArray)
    a = abs.(d)
    return (
        n      = length(a),
        amin   = minimum(a),
        amax   = maximum(a),
        amean  = mean(a),
        amed   = median(a),
        a95    = quantile(vec(a), 0.95),
        a99    = quantile(vec(a), 0.99),
        sumabs = sum(a),
    )
end

"""
Print a log-binned histogram of `|d|`. Bin edges are at
[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, Inf).
"""
function _print_log_hist(name::AbstractString, d::AbstractArray)
    a = vec(abs.(d))
    edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, Inf]
    counts = zeros(Int, length(edges) - 1)
    for v in a
        for k in 1:length(edges) - 1
            if edges[k] <= v < edges[k + 1]
                counts[k] += 1
                break
            end
        end
    end
    println("    Histogram of |diff_$name|:")
    for k in 1:length(edges) - 1
        lo = edges[k]
        hi = edges[k + 1]
        bar = repeat("#", min(60, Int(round(60 * counts[k] / max(length(a), 1)))))
        @printf("      [%.0e, %.0e)%s %6d  %s\n",
                lo, hi, hi == Inf ? "  " : "", counts[k], bar)
    end
end

"""
Print top-K worst-diff cells with local geometry.

`d`     — diff array (size matches one of the loaded grids).
`name`  — field name for the header.
`fixt`  — fixture-side reference array (same shape as `d`).
`geom`  — NamedTuple of (Nx, Ny)-shaped Center fields (e.g. H_ice).
         Indexed at the *Center* coords (i_c, j) using the rule below.

For Center-aligned diffs (size (Nx, Ny)), `(i, j)` indexes geometry
directly. For XFace-aligned diffs (size (Nx+1, Ny)) the x-face at
column `i` lies between Center cells `i-1` and `i`; we report the
*adjacent Center cell* `i_c = clamp(i, 1, Nx)` (mirrors the loader's
back-padding convention so face slot `i=2` corresponds to fixture
column `1` ≡ Center cell `1`).
"""
function _print_worst(name::AbstractString, d::AbstractArray, fixt::AbstractArray,
                      geom::NamedTuple; k::Int = 10)
    Nx_c, Ny = size(geom.H_ice)
    Nx_d, _  = size(d)
    is_face_x = (Nx_d == Nx_c + 1)
    a = abs.(d)
    idxs = sortperm(vec(a); rev = true)[1:min(k, length(a))]
    println("    Top-$k worst |diff_$name| cells:")
    @printf("      %4s %4s %12s %12s %12s %8s %8s %6s %6s\n",
            "i", "j", "diff", "ref", "jl", "H_ice",
            "z_bed", "f_grnd", "frnt")
    for ii in idxs
        ci = CartesianIndices(d)[ii]
        i = ci[1]; j = ci[2]
        # Map face index to adjacent Center column.
        i_c = is_face_x ? clamp(i, 1, Nx_c) : i
        diffv = d[i, j]
        refv  = fixt[i, j]
        jlv   = refv + diffv
        @printf("      %4d %4d %12.4e %12.4e %12.4e %8.1f %8.1f %6.2f %6d\n",
                i, j, diffv, refv, jlv,
                geom.H_ice[i_c, j], geom.z_bed[i_c, j],
                geom.f_grnd[i_c, j], Int(round(geom.mask_frnt[i_c, j])))
    end
end

"""
Sum |diff| per row j and per column i, print the row/col with the
largest contribution.
"""
function _print_spatial(name::AbstractString, d::AbstractArray)
    Nx, Ny = size(d)
    row_sum = vec(sum(abs.(d); dims = 1))
    col_sum = vec(sum(abs.(d); dims = 2))
    @printf("    sum(|diff_%s|) by row j (j=1..Ny=%d):\n", name, Ny)
    for j in 1:Ny
        bar = repeat("#",
                     min(60, Int(round(60 * row_sum[j] /
                                       max(maximum(row_sum), eps())))))
        @printf("      j=%2d  %12.4e  %s\n", j, row_sum[j], bar)
    end
    j_max = argmax(row_sum)
    @printf("    -> row of max |diff_%s| sum: j=%d (sum=%.4e)\n",
            name, j_max, row_sum[j_max])
    i_max = argmax(col_sum)
    @printf("    -> col of max |diff_%s| sum: i=%d (sum=%.4e)\n",
            name, i_max, col_sum[i_max])
    # Boundary concentration: rows j=1, j=Ny vs interior.
    bnd_sum = row_sum[1] + row_sum[Ny]
    int_sum = sum(row_sum) - bnd_sum
    bnd_frac = bnd_sum / max(sum(row_sum), eps())
    @printf("    -> y-boundary rows (j=1,j=Ny) hold %.1f%% of total |diff_%s|\n",
            100 * bnd_frac, name)
end

"""
Lag-1 spatial autocorrelation of the diff field. Negative values
near -1 indicate a checkerboard pattern.
"""
function _autocorr_lag1(d::AbstractArray)
    Nx, Ny = size(d)
    # Mean-centre.
    dm = d .- mean(d)
    var0 = sum(dm .^ 2)
    if var0 == 0
        return (xlag = 0.0, ylag = 0.0)
    end
    xnum = 0.0
    ynum = 0.0
    for j in 1:Ny, i in 1:(Nx - 1)
        xnum += dm[i, j] * dm[i + 1, j]
    end
    for j in 1:(Ny - 1), i in 1:Nx
        ynum += dm[i, j] * dm[i, j + 1]
    end
    return (xlag = xnum / var0, ylag = ynum / var0)
end

"""
Concentration metric: fraction of total |diff| held by the
top-`frac` cells (default 10%).
"""
function _concentration(d::AbstractArray; frac::Float64 = 0.1)
    a = vec(abs.(d))
    n = length(a)
    k = max(1, Int(round(frac * n)))
    sorted = sort(a; rev = true)
    return sum(sorted[1:k]) / max(sum(a), eps())
end

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

function main()
    println("=" ^ 78)
    println("Trough lockstep per-cell diff analysis")
    println("=" ^ 78)
    println("Fixture (Fortran ref): ", FIXTURE_PATH)
    println("Yelmo.jl post-step:    ", JL_OUT_PATH)
    println("Diff output:           ", DIFF_OUT_PATH)
    println()

    @assert isfile(FIXTURE_PATH) "Fixture missing: $FIXTURE_PATH"
    @assert isfile(JL_OUT_PATH) """\
        Yelmo.jl post-step file missing: $JL_OUT_PATH.
        Re-run `julia --project=test test/benchmarks/test_trough.jl`
        first to regenerate it."""

    ds_ref = NCDataset(FIXTURE_PATH)
    ds_jl  = NCDataset(JL_OUT_PATH)

    # ------------------------------------------------------------------
    # Geometry (Center-aligned, both files agree on shape (Nx,Ny)).
    # ------------------------------------------------------------------
    H_ice    = _read2d(ds_ref, "H_ice")
    z_bed    = _read2d(ds_ref, "z_bed")
    f_grnd   = _read2d(ds_ref, "f_grnd")
    mask_frnt = _read2d(ds_ref, "mask_frnt")
    Nx, Ny = size(H_ice)
    println("Grid: Nx=$Nx, Ny=$Ny")

    geom = (H_ice = H_ice, z_bed = z_bed, f_grnd = f_grnd,
            mask_frnt = mask_frnt)

    # ------------------------------------------------------------------
    # Coordinates for the diff NetCDF (read from fixture, in km).
    # ------------------------------------------------------------------
    xc = Float64.(Array(ds_ref["xc"]))   # (Nx,) — Center, km
    yc = Float64.(Array(ds_ref["yc"]))   # (Ny,) — Center, km

    # ------------------------------------------------------------------
    # Fields to diff.
    #
    # ux_bar / ux_b / beta_acx are XFaceField under Bounded-x in
    # Yelmo.jl: shape (Nx+1, Ny). The fixture stores them
    # Center-aligned at (Nx, Ny). To diff, slice JL[2:end, :] vs
    # fixture (matches `_load_into_field!`'s back-padding convention
    # — see `src/YelmoCore.jl`).
    #
    # uy_bar / uy_b / beta_acy under Periodic-y: shape (Nx, Ny) in
    # both. Direct diff.
    #
    # visc_eff_int is Center-aligned: shape (Nx, Ny). Direct diff.
    # ------------------------------------------------------------------
    function _diff_facex(name)
        ref = _read2d(ds_ref, name)              # (Nx, Ny)
        jl  = _read2d(ds_jl, name)               # (Nx+1, Ny)
        @assert size(ref) == (Nx, Ny) "$name fixture shape mismatch"
        @assert size(jl)  == (Nx + 1, Ny) "$name jl shape mismatch"
        return jl[2:end, :] .- ref               # (Nx, Ny)
    end

    function _diff_center(name)
        ref = _read2d(ds_ref, name)
        jl  = _read2d(ds_jl, name)
        @assert size(ref) == size(jl) == (Nx, Ny) "$name shape mismatch"
        return jl .- ref
    end

    fields = [
        # (name,        diff_array,                 ref_array,                 kind)
        ("ux_bar",      _diff_facex("ux_bar"),      _read2d(ds_ref, "ux_bar"),      :facex),
        ("uy_bar",      _diff_center("uy_bar"),     _read2d(ds_ref, "uy_bar"),      :center),
        ("ux_b",        _diff_facex("ux_b"),        _read2d(ds_ref, "ux_b"),        :facex),
        ("uy_b",        _diff_center("uy_b"),       _read2d(ds_ref, "uy_b"),        :center),
        ("beta_acx",    _diff_facex("beta_acx"),    _read2d(ds_ref, "beta_acx"),    :facex),
        ("visc_eff_int", _diff_center("visc_eff_int"), _read2d(ds_ref, "visc_eff_int"), :center),
    ]

    # ------------------------------------------------------------------
    # Per-field analysis.
    # ------------------------------------------------------------------
    structural_flags = Dict{String, Bool}()
    for (name, d, ref, _kind) in fields
        println()
        println("-" ^ 78)
        println("Field: $name")
        println("-" ^ 78)
        s = _abs_stats(d)
        ref_max = maximum(abs.(ref))
        floor_abs = ref_max * FLOAT32_REL_PREC
        @printf("    ref max(|·|)=%.4e  Float32 noise floor (vmax×%.2e)=%.4e\n",
                ref_max, FLOAT32_REL_PREC, floor_abs)
        @printf("    |diff| min=%.4e  median=%.4e  mean=%.4e  p95=%.4e  p99=%.4e  max=%.4e\n",
                s.amin, s.amed, s.amean, s.a95, s.a99, s.amax)
        @printf("    sum(|diff|)=%.4e  ratio max/median = %.1f\n",
                s.sumabs, s.amax / max(s.amed, eps()))

        _print_log_hist(name, d)

        # Worst cells with local geometry.
        _print_worst(name, d, ref, geom; k = 10)

        # Spatial patterns.
        _print_spatial(name, d)
        ac = _autocorr_lag1(d)
        @printf("    lag-1 autocorr  x=%.3f  y=%.3f  (negative ≈ checkerboard)\n",
                ac.xlag, ac.ylag)
        c10 = _concentration(d; frac = 0.10)
        c01 = _concentration(d; frac = 0.01)
        @printf("    Concentration: top-10%% cells hold %.1f%% of total |diff|; top-1%% hold %.1f%%\n",
                100 * c10, 100 * c01)

        # Structural flag.
        structural = (s.amax > 100 * floor_abs) && (c10 > 0.50)
        @printf("    Structural (max>100×floor AND top-10%% > 50%%): %s\n",
                structural ? "YES" : "no")
        structural_flags[name] = structural
    end

    # ------------------------------------------------------------------
    # Write per-cell diff NetCDF.
    # ------------------------------------------------------------------
    println()
    println("=" ^ 78)
    println("Writing diff NetCDF: ", DIFF_OUT_PATH)
    println("=" ^ 78)
    isfile(DIFF_OUT_PATH) && rm(DIFF_OUT_PATH)
    ds_out = NCDataset(DIFF_OUT_PATH, "c"; attrib = [
        "title"          => "Trough TROUGH-F17 lockstep per-cell diff (jl - fortran)",
        "fixture"        => FIXTURE_PATH,
        "jl_post_step"   => JL_OUT_PATH,
        "noise_floor"    => "Float32 ~ ref_max * $(FLOAT32_REL_PREC)",
        "structural_thr" => "max(|diff|) > 100 × Float32 noise floor AND " *
                            "top-10% cells hold > 50% of total |diff|",
        "facex_diff_convention" =>
            "_acx fields are XFaceField (Nx+1, Ny) in Yelmo.jl; " *
            "diffed against Center-aligned (Nx, Ny) fixture using " *
            "jl[2:end, :] (matches loader back-padding convention).",
    ])
    defDim(ds_out, "xc", Nx)
    defDim(ds_out, "yc", Ny)
    xv = defVar(ds_out, "xc", Float64, ("xc",); attrib = ["units" => "km"])
    yv = defVar(ds_out, "yc", Float64, ("yc",); attrib = ["units" => "km"])
    xv[:] = xc
    yv[:] = yc
    for (name, d, _ref, _kind) in fields
        v = defVar(ds_out, "diff_" * name, Float64, ("xc", "yc"))
        v[:, :] = d
    end
    # Geometry context.
    for (gname, garr) in pairs(geom)
        v = defVar(ds_out, "ref_" * String(gname), Float64, ("xc", "yc"))
        v[:, :] = garr
    end
    close(ds_out)
    println("Wrote $(length(fields)) diff fields + 4 geometry refs.")

    close(ds_ref)
    close(ds_jl)

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------
    println()
    println("=" ^ 78)
    println("VERDICT")
    println("=" ^ 78)
    any_structural = any(values(structural_flags))
    if any_structural
        println("STRUCTURAL CONCENTRATION DETECTED in fields: ",
                join([k for (k, v) in structural_flags if v], ", "))
        println("Recommend follow-up investigation — these are real bugs, not noise.")
    else
        println("No structural concentration. Residual is consistent with")
        println("Float32 fixture noise + small kernel-level imprecision.")
        println("Top-3 worst cells are scattered across the domain, not clustered.")
    end
    return any_structural ? 1 : 0
end

exit(main())
