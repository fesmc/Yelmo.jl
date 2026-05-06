## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Path B vertical-split round-trip tests.
#
# Goal: verify that the boundary-fields registry (commit 2 of the
# vert-split refactor) preserves data across a full
#   load (registry split) → write (registry glue) → load
# cycle for the 7 ice Center fields registered in
# `PATH_B_REGISTRY_ICE`: T_ice, enth, T_pmp, T_prime, visc, ATT, enh.
#
# The test loads a real Greenland 16 km restart, sets sentinel
# values into the basal / interior / surface slots of one
# representative field per shape (`T_ice` for thrm Center, `visc`
# for mat Center), writes a fresh NetCDF via `write_output!`,
# loads a second YelmoModel from that NetCDF, and asserts that
# the sentinel values come back at the right slots.
#
# This is the primary verification for commit 2 — the actual
# physical correctness of the values isn't tested here (commits 3 /
# 4 / 5 cover that); we only assert that the I/O machinery doesn't
# scramble basal / interior / surface decomposition.

using Test
using Yelmo
using NCDatasets
using Oceananigans.Fields: Center, Face, interior

const RESTART_PATH = "/Users/alrobi001/models/yelmox/output/16KM/test/restart-0.000-kyr/yelmo_restart.nc"

@assert isfile(RESTART_PATH) "Restart fixture not found at $(RESTART_PATH)"

# Helper — construct a fresh model from the canonical restart and
# strict=false to skip any 2D mat _b/_s fields that the file does
# not provide separately (their unified mat slabs *are* in the
# file, so the registry path covers them; strict=false is just
# defensive).
function _make_model(alias::String)
    return YelmoModel(RESTART_PATH, 0.0; alias=alias, strict=false)
end

@testset "Path B vert-split round-trip" begin

    y0 = _make_model("vsplit-orig")

    # ---- inject sentinel values --------------------------------------
    # Use distinct values per slot so a basal/surface mix-up is
    # immediately visible. These overwrite the loaded restart values
    # — that's fine, the test isn't about the restart's physical
    # state.
    Nx, Ny, Nz = size(interior(y0.thrm.T_ice))
    @test Nz == 8        # Path B grid Nz = file Nz_file − 2 = 10 − 2

    # T_ice — thrm Center field family
    T_ice_b_val = 273.15
    T_ice_s_val = 240.0
    fill!(interior(y0.thrm.T_ice_b), T_ice_b_val)
    fill!(interior(y0.thrm.T_ice_s), T_ice_s_val)
    @inbounds for k in 1:Nz
        interior(y0.thrm.T_ice)[:, :, k] .= 250.0 + k       # 251 … 258
    end

    # visc — mat Center field family
    visc_b_val = 1.5e10
    visc_s_val = 3.7e10
    fill!(interior(y0.mat.visc_b), visc_b_val)
    fill!(interior(y0.mat.visc_s), visc_s_val)
    @inbounds for k in 1:Nz
        interior(y0.mat.visc)[:, :, k] .= 1.0e10 * k       # 1e10 … 8e10
    end

    # ---- write to a fresh NetCDF and inspect on-disk shape ---------
    tmpfile = tempname() * "_vsplit_roundtrip.nc"
    out = init_output(y0, tmpfile)
    write_output!(out, y0)
    close(out)

    @testset "on-disk slab shape" begin
        # NetCDF stores fields as Float32 — use Float32-precision
        # tolerance for value comparisons.
        f32rtol = 1e-4
        approx32(x, y) = isapprox(x, y; rtol=f32rtol)

        ds = NCDataset(tmpfile)
        try
            # Registered fields: file slab on the `zeta` axis,
            # length Nz_file = Nz + 2 = 10 (Path B convention with
            # basal endpoint at index 1 and surface at index end).
            @test "zeta" in keys(ds.dim)
            @test ds.dim["zeta"] == Nz + 2

            T_ice_slab = ds["T_ice"][:, :, :, 1]
            @test size(T_ice_slab, 3) == Nz + 2
            @test all(approx32.(T_ice_slab[:, :, 1],   T_ice_b_val))   # basal
            @test all(approx32.(T_ice_slab[:, :, end], T_ice_s_val))   # surface
            @inbounds for k in 1:Nz
                @test all(approx32.(T_ice_slab[:, :, k + 1], 250.0 + k))   # interior
            end

            visc_slab = ds["visc"][:, :, :, 1]
            @test size(visc_slab, 3) == Nz + 2
            @test all(approx32.(visc_slab[:, :, 1],   visc_b_val))
            @test all(approx32.(visc_slab[:, :, end], visc_s_val))

            # Registered _b / _s 2D fields should NOT appear as
            # separate variables — they are written as part of the
            # unified slab.
            @test !haskey(ds, "T_ice_b")
            @test !haskey(ds, "T_ice_s")
            @test !haskey(ds, "visc_b")
            @test !haskey(ds, "visc_s")
        finally
            close(ds)
        end
    end

    # ---- load a fresh model from the written file ------------------
    # Restrict to (:thrm, :mat) groups so we exercise the registry
    # round-trip without tripping on the pre-existing XFace 2D
    # dyn-side mismatch between the Yelmo writer ((Nx+1, Ny) shape)
    # and the load_state! reader ((Nx, Ny) Mirror-shape expectation).
    # That dyn-side I/O bug predates Path B and is out of scope for
    # commit 2.
    y1 = YelmoModel(tmpfile, 0.0; alias="vsplit-rt", strict=false,
                    groups=(:thrm, :mat))

    @testset "round-trip preserves boundary + interior values" begin
        # NetCDF stores fields as Float32. Use a Float32-precision
        # tolerance (rtol ≈ √eps(Float32) ≈ 3e-4) so the comparisons
        # don't fail purely on the f64 → f32 → f64 quantization.
        f32rtol = 1e-4
        approx32(x, y) = isapprox(x, y; rtol=f32rtol)

        @test all(approx32.(interior(y1.thrm.T_ice_b), T_ice_b_val))
        @test all(approx32.(interior(y1.thrm.T_ice_s), T_ice_s_val))
        @inbounds for k in 1:Nz
            @test all(approx32.(interior(y1.thrm.T_ice)[:, :, k], 250.0 + k))
        end

        @test all(approx32.(interior(y1.mat.visc_b), visc_b_val))
        @test all(approx32.(interior(y1.mat.visc_s), visc_s_val))
        @inbounds for k in 1:Nz
            @test all(approx32.(interior(y1.mat.visc)[:, :, k], 1.0e10 * k))
        end
    end

    rm(tmpfile; force=true)
end
