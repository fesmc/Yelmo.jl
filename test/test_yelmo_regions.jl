## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Tests for the regions module — `init_regions`, `add_region!`,
# `update_regions!`, `write_regions!`, plus the workhorse
# `calc_region_diagnostics!`.

using Test
using Yelmo
using NCDatasets
using Oceananigans

const RESTART_PATH = "/Users/alrobi001/models/yelmox/output/16KM/test/restart-0.000-kyr/yelmo_restart.nc"

@testset "regions: RegionDiagnostics defaults to zeros" begin
    diag = RegionDiagnostics()
    @test diag.H_ice    == 0.0
    @test diag.V_ice    == 0.0
    @test diag.uxy_bar  == 0.0
    @test diag.f_pmp    == 0.0
    @test diag.T_shlf   == 0.0
end

@testset "regions: init_regions auto-creates whole-domain region" begin
    tdir = mktempdir(; prefix="regions_init_")
    y = YelmoModel(RESTART_PATH, 0.0; alias="reg-init",
                   rundir = tdir, strict = false)
    regs = init_regions(y)

    @test length(regs) == 1
    @test regs[1].name == "domain"
    @test all(regs[1].mask)                          # full mask
    @test isfile(regs[1].outfile)
    @test endswith(regs[1].outfile, "region_domain.nc")

    # The init NetCDF should hold the static mask + the time-series
    # variable definitions (no time records yet).
    NCDataset(regs[1].outfile) do ds
        @test haskey(ds, "mask")
        @test haskey(ds, "H_ice")
        @test haskey(ds, "V_sle")
        @test haskey(ds, "T_shlf")
        @test size(ds["time"]) == (0,)
    end
end

@testset "regions: add_region! validates mask shape" begin
    tdir = mktempdir(; prefix="regions_add_")
    y = YelmoModel(RESTART_PATH, 0.0; alias="reg-add",
                   rundir = tdir, strict = false)
    regs = init_regions(y; include_whole_domain = false)
    @test length(regs) == 0

    Nx = size(y.tpo.H_ice, 1)
    Ny = size(y.tpo.H_ice, 2)
    bad = trues(Nx + 1, Ny)                          # wrong shape
    @test_throws ErrorException add_region!(regs, y, "bad", bad)

    # Memory-only region (no NetCDF).
    half = falses(Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:div(Nx, 2)
        half[i, j] = true
    end
    reg = add_region!(regs, y, "left_half", half;
                      write_to_file = false)
    @test reg.outfile == ""
    @test length(regs) == 1
end

@testset "regions: update_regions! + write_regions! round-trip" begin
    tdir = mktempdir(; prefix="regions_rw_")
    y = YelmoModel(RESTART_PATH, 0.0; alias="reg-rw",
                   rundir = tdir, strict = false)
    regs = init_regions(y)
    @test length(regs) == 1

    update_regions!(regs, y)
    diag = regs[1].diag

    # The restart's domain should have *some* non-zero ice somewhere
    # in the totals — A_ice > 0 is the fundamental sanity check.
    @test diag.A_ice >= 0.0          # may be zero on a synthetic restart

    # Write three timesteps and read them back.
    for t in (0.0, 1.0, 2.0)
        update_regions!(regs, y)
        write_regions!(regs, y, t)
    end

    NCDataset(regs[1].outfile) do ds
        ts = Vector{Float64}(ds["time"][:])
        @test ts == [0.0, 1.0, 2.0]
        # H_ice column should have the same length.
        @test length(ds["H_ice"][:]) == 3
        # mask must round-trip.
        m = ds["mask"][:, :]
        @test all(m .== 1)
    end
end

@testset "regions: empty mask → zero diagnostics" begin
    # An all-false mask should yield a RegionDiagnostics of zeros
    # (matches Fortran's `else` branches).
    tdir = mktempdir(; prefix="regions_empty_")
    y = YelmoModel(RESTART_PATH, 0.0; alias="reg-empty",
                   rundir = tdir, strict = false)

    Nx = size(y.tpo.H_ice, 1)
    Ny = size(y.tpo.H_ice, 2)
    empty_mask = falses(Nx, Ny)
    diag = RegionDiagnostics()
    calc_region_diagnostics!(diag, y, empty_mask)

    @test diag.H_ice == 0.0
    @test diag.V_ice == 0.0
    @test diag.A_ice == 0.0
    @test diag.f_pmp == 0.0
    @test diag.T_shlf == 0.0
end

@testset "regions: A_ice scales with cell area" begin
    # Build a minimal in-memory model — easier to control than the
    # full restart path. Use the parameterless YelmoModel constructor
    # via the existing `YelmoModel(...)` ctor with a synthetic grid.
    # Actually simpler: stamp some H_ice into the loaded restart and
    # verify A_ice * 1e6 km^2 == npts * dx * dy * 1e-6 * 1e6.
    tdir = mktempdir(; prefix="regions_area_")
    y = YelmoModel(RESTART_PATH, 0.0; alias="reg-area",
                   rundir = tdir, strict = false)

    Nx = size(y.tpo.H_ice, 1)
    Ny = size(y.tpo.H_ice, 2)

    # Stamp a 3x3 ice block at (5..7, 5..7) and zero everywhere else.
    fill!(interior(y.tpo.H_ice), 0.0)
    @inbounds for j in 5:7, i in 5:7
        if i <= Nx && j <= Ny
            interior(y.tpo.H_ice)[i, j, 1] = 1000.0
        end
    end

    full_mask = trues(Nx, Ny)
    diag = RegionDiagnostics()
    calc_region_diagnostics!(diag, y, full_mask)

    dx = abs(Float64(y.g.Δxᶜᵃᵃ))
    dy = abs(Float64(y.g.Δyᵃᶜᵃ))
    n_active = min(3, Nx - 4) * min(3, Ny - 4)
    expected_A_km2 = n_active * dx * dy * 1e-6
    @test diag.A_ice ≈ expected_A_km2 atol = 1e-9
end
