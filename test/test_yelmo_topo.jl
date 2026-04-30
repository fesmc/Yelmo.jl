## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

# Milestone 2b integration test for `tpo`. Test groups:
#   1. Analytical advection — Gaussian return-to-origin and
#      uniform-stationary checks against `advect_tracer!` directly
#      (no full YelmoModel).
#   2. Square-pulse advection — informational only; reports
#      first-order-upwind diffusion characteristics. Always passes.
#   3. mask_ice post-step pass — verifies the per-cell behavior of
#      `topo_step!`'s mask handling. SMB is zeroed so the new 2b mass
#      balance stages don't perturb the FIXED-mask check.
#   4. Real-restart 5-step smoke — volume conservation when SMB is
#      zeroed (mb_resid may still trim margins, so the tolerance is
#      loose; the only strict invariant is `dHidt = dHidt_dyn + mb_net`).
#   5. SMB conservation test (new for 2b) — slab geometry, prescribed
#      melt rate; checks that `apply_tendency!` realises the requested
#      thinning per step and that `mb_net = smb + mb_resid` and
#      `dHidt = dHidt_dyn + mb_net` to within numerical tolerance.

using Test
using Yelmo
using Oceananigans
using Oceananigans: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!

const RESTART_PATH = "/Users/alrobi001/models/yelmox/output/16KM/test/restart-0.000-kyr/yelmo_restart.nc"

# ------------------------------------------------------------------
# Analytical advection — kernel-level
# ------------------------------------------------------------------

@testset "tpo: analytical advection" begin

    @testset "Gaussian returns to origin (periodic)" begin
        Nx = 50
        Lx = 100.0
        g = RectilinearGrid(size=(Nx, Nx),
                            x=(0.0, Lx), y=(0.0, Lx),
                            topology=(Periodic, Periodic, Flat))
        H = CenterField(g)
        u = XFaceField(g)
        v = YFaceField(g)

        xc = xnodes(g, Center())
        yc = ynodes(g, Center())
        @inbounds for j in 1:Nx, i in 1:Nx
            interior(H)[i, j, 1] =
                exp(-((xc[i] - 50.0)^2 + (yc[j] - 50.0)^2) / 100.0)
        end

        u_speed = 1.0  # x-only velocity
        fill!(interior(u), u_speed)
        fill!(interior(v), 0.0)

        H_init = copy(interior(H))

        # One full revolution: T = Lx / |u|
        T = Lx / u_speed
        advect_tracer!(H, u, v, T; cfl_safety=0.5)

        # First-order upwind on a 50×50 grid over one period diffuses
        # the Gaussian noticeably. Acceptance: peak should still be
        # within 60% of original (i.e. shape persists, even if dampened),
        # *and* total volume conserved to machine precision.
        peak_ratio = maximum(interior(H)) / maximum(H_init)
        vol_diff_rel = abs(sum(interior(H)) - sum(H_init)) /
                       max(sum(H_init), eps())

        @test peak_ratio  > 0.4    # peak persists
        @test peak_ratio  < 1.05   # but doesn't grow (no instability)
        @test vol_diff_rel < 1e-10 # volume conservation
    end

    @testset "Stationary state stays stationary" begin
        Nx = 30
        g = RectilinearGrid(size=(Nx, Nx),
                            x=(0.0, 30.0), y=(0.0, 30.0),
                            topology=(Periodic, Periodic, Flat))
        H = CenterField(g)
        u = XFaceField(g)
        v = YFaceField(g)

        H_value = 5.0
        fill!(interior(H), H_value)
        fill!(interior(u), 1.0)  # nonzero so kernel actually runs
        fill!(interior(v), 0.5)

        advect_tracer!(H, u, v, 10.0; cfl_safety=0.5)

        max_dev = maximum(abs.(interior(H) .- H_value))
        @test max_dev < 1e-12  # bit-stable for uniform field
    end

end

# ------------------------------------------------------------------
# Square-pulse advection — informational only
# ------------------------------------------------------------------

@testset "tpo: square pulse (informational)" begin
    Nx = 50
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 100.0), y=(0.0, 100.0),
                        topology=(Periodic, Periodic, Flat))
    H = CenterField(g)
    u = XFaceField(g)
    v = YFaceField(g)

    # Square pulse along x at the center
    @inbounds for j in 1:Nx, i in 1:Nx
        interior(H)[i, j, 1] = (20 ≤ i ≤ 30) ? 1.0 : 0.0
    end
    fill!(interior(u), 1.0)
    fill!(interior(v), 0.0)

    H_init = copy(interior(H))
    advect_tracer!(H, u, v, 100.0; cfl_safety=0.5)

    peak_ratio = maximum(interior(H)) / maximum(H_init)
    width_at_half_max = count(>=(0.5), view(interior(H), :, Nx ÷ 2, 1))
    init_width = count(>=(0.5), view(H_init, :, Nx ÷ 2, 1))

    @info "Square pulse after one revolution" peak_ratio width_at_half_max init_width
    @test true  # informational; just records that the run completed
end

# ------------------------------------------------------------------
# mask_ice post-step pass — unit test
# ------------------------------------------------------------------

@testset "tpo: mask_ice post-step pass" begin
    @assert isfile(RESTART_PATH) "Restart fixture not found at $(RESTART_PATH)"

    y = YelmoModel(RESTART_PATH, 0.0;
                   rundir = mktempdir(; prefix="tpo_mask_test_"),
                   alias  = "tpo-mask-test",
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    # Zero SMB so the 2b mass-balance stages don't perturb the
    # mask-pass invariants we're checking. (In 2a the mass-balance
    # stages didn't exist; the test was written against that.)
    fill!(interior(y.bnd.smb_ref), 0.0)

    H_ice    = interior(y.tpo.H_ice)
    mask_ice = interior(y.bnd.mask_ice)
    Nx, Ny   = size(H_ice, 1), size(H_ice, 2)

    # Set up a known checkerboard region with all three mask values
    # and a known H_ice value, then take one step and verify per-cell
    # behavior. We work on a 5x5 patch in the deep interior so that
    # domain-edge BCs and the new 2b `resid_tendency!` margin/island
    # cleanup don't perturb the test cells. We:
    #   - set the surrounding 5x5 patch to a thick uniform 100 m of
    #     ice so the inner 3x3 test cells are not at a margin;
    #   - flip `ice_allowed = 1` over the patch so `resid_tendency!`
    #     does not zero the patch via the boundary-mask rule
    #     (the chosen patch sits over open ocean in the restart);
    #   - lift `z_bed` above sea level over the patch so the cells
    #     are unambiguously grounded (`f_grnd = 1`), keeping the
    #     `H_min_grnd = 5` margin-thinning rule trivially satisfied.
    patch_i = 9:13
    patch_j = 9:13
    for j in patch_j, i in patch_i
        H_ice[i, j, 1] = 100.0
        interior(y.bnd.ice_allowed)[i, j, 1] = 1.0
        interior(y.bnd.z_bed)[i, j, 1] = 100.0
        interior(y.bnd.z_sl)[i, j, 1]  = 0.0
    end

    test_cells = [(i, j) for i in 10:12, j in 10:12]
    expected = Dict{Tuple{Int,Int}, Tuple{Int, Float64}}()  # (i,j) → (mask, expected_H)

    pre_value = 100.0
    for (idx, (i, j)) in enumerate(test_cells)
        mask_val = (idx - 1) % 3  # cycle through 0, 1, 2
        H_ice[i, j, 1]    = pre_value
        mask_ice[i, j, 1] = Float64(mask_val)
        expected_H = mask_val == MASK_ICE_NONE  ? 0.0       :
                     mask_val == MASK_ICE_FIXED ? pre_value :
                                                  pre_value  # dynamic — H stays roughly,
                                                             # advection of a mostly-uniform
                                                             # patch barely changes it
        expected[(i, j)] = (mask_val, expected_H)
    end

    step!(y, 1.0)

    # Cells with MASK_ICE_NONE must be exactly zero.
    # Cells with MASK_ICE_FIXED must equal pre_value (mask pass restores
    #   H_prev, SMB=0 means `apply_tendency!` is a no-op for SMB; the
    #   cell is interior to the ice sheet so `resid_tendency!` does not
    #   trim it).
    # Cells with MASK_ICE_DYNAMIC must be ≥ 0 and finite.
    for ((i, j), (mask_val, expected_H)) in expected
        if mask_val == MASK_ICE_NONE
            @test H_ice[i, j, 1] == 0.0
        elseif mask_val == MASK_ICE_FIXED
            @test H_ice[i, j, 1] == pre_value
        else  # MASK_ICE_DYNAMIC
            @test H_ice[i, j, 1] >= 0.0
            @test isfinite(H_ice[i, j, 1])
        end
    end
end

# ------------------------------------------------------------------
# Real-restart smoke test: 5 steps, mass conservation accounting
# ------------------------------------------------------------------

@testset "tpo: real-restart 5-step smoke test" begin
    @assert isfile(RESTART_PATH)

    y = YelmoModel(RESTART_PATH, 0.0;
                   rundir = mktempdir(; prefix="tpo_smoke_"),
                   alias  = "tpo-smoke",
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    # Zero SMB. With SMB = 0 the only mass losses come from
    # `mb_resid` margin cleanup; the rest of the volume should be
    # advected without source/sink.
    fill!(interior(y.bnd.smb_ref), 0.0)

    V0     = sum(interior(y.tpo.H_ice))
    H0_max = maximum(interior(y.tpo.H_ice))

    for _ in 1:5
        step!(y, 1.0)
    end

    V5     = sum(interior(y.tpo.H_ice))
    H5_max = maximum(interior(y.tpo.H_ice))

    @test y.time == 5.0
    @test isfinite(H5_max)
    @test H5_max < 1.05 * H0_max         # no upwind blow-up

    # mb_resid may zero thin margins / islands — the constraint here
    # is one-sided (volume can only decrease) and small in relative
    # terms. 1% slack covers margin cleanup on a 16km Greenland grid.
    @test V5 <= V0
    @test (V0 - V5) / V0 < 0.01

    # Mass-conservation accounting on the *last* step, per cell.
    # `dHidt` should equal `dHidt_dyn + mb_net` to within roundoff.
    H_now     = interior(y.tpo.H_ice)
    dHidt     = interior(y.tpo.dHidt)
    dHidt_dyn = interior(y.tpo.dHidt_dyn)
    mb_net    = interior(y.tpo.mb_net)
    smb       = interior(y.tpo.smb)
    bmb       = interior(y.tpo.bmb)
    mb_resid  = interior(y.tpo.mb_resid)

    fmb       = interior(y.tpo.fmb)
    dmb       = interior(y.tpo.dmb)
    mb_relax  = interior(y.tpo.mb_relax)

    err_total = maximum(abs.(dHidt .- (dHidt_dyn .+ mb_net)))
    err_net   = maximum(abs.(mb_net .- (smb .+ bmb .+ fmb .+ dmb .+
                                        mb_relax .+ mb_resid)))
    @test err_total < 1e-9
    @test err_net   < 1e-12
    # DMB and relaxation are no-ops by default (dmb_method = 0, topo_rel = 0).
    @test all(dmb      .== 0.0)
    @test all(mb_relax .== 0.0)

    # `f_grnd` is now subgrid (CISM bilinear scheme) — admits values in [0,1].
    f_grnd = interior(y.tpo.f_grnd)
    @test all(0.0 .<= f_grnd .<= 1.0)
    @test sort(unique(interior(y.tpo.f_ice)))  ⊆ [0.0, 1.0]   # binary stub
    # f_ice is exactly H_ice > 0
    f_ice = interior(y.tpo.f_ice)
    @test all((f_ice .> 0) .== (H_now .> 0))
end

# ------------------------------------------------------------------
# SMB conservation test (new for 2b)
# ------------------------------------------------------------------

@testset "tpo: SMB conservation (slab + prescribed melt)" begin
    @assert isfile(RESTART_PATH)

    y = YelmoModel(RESTART_PATH, 0.0;
                   rundir = mktempdir(; prefix="tpo_smb_test_"),
                   alias  = "tpo-smb-test",
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    # Reset to a uniform 1000 m slab so we don't have to reason about
    # the restart geometry. Keep grounded everywhere (`z_bed > z_sl`
    # works for any positive bed; the restart bed is grounded almost
    # everywhere on the Greenland fixture, so this is a low-noise
    # path).
    H_ice = interior(y.tpo.H_ice)
    fill!(H_ice, 1000.0)

    # Zero velocities: isolate SMB; advection becomes a no-op on a
    # uniform field anyway, but this also bypasses the CFL kernel.
    fill!(interior(y.dyn.ux_bar), 0.0)
    fill!(interior(y.dyn.uy_bar), 0.0)

    # Ensure all cells are dynamic and ice is allowed everywhere.
    fill!(interior(y.bnd.mask_ice),    Float64(MASK_ICE_DYNAMIC))
    fill!(interior(y.bnd.ice_allowed), 1.0)

    # Lift the bed well above sea level so f_grnd = 1 everywhere
    # (avoids any grounded/floating ambiguity for `mbal_tendency!`).
    fill!(interior(y.bnd.z_bed), 100.0)
    fill!(interior(y.bnd.z_sl),    0.0)

    # Zero BMB / FMB inputs so the only mass change comes from SMB.
    fill!(interior(y.thrm.bmb_grnd), 0.0)
    fill!(interior(y.bnd.bmb_shlf),  0.0)
    fill!(interior(y.bnd.fmb_shlf),  0.0)

    # Prescribe 1 m/yr melt over the whole domain.
    smb_rate = -1.0
    fill!(interior(y.bnd.smb_ref), smb_rate)

    # Take a snapshot before stepping.
    Nx, Ny  = size(H_ice, 1), size(H_ice, 2)
    H_init  = copy(H_ice)

    n_steps = 3
    dt      = 1.0
    for _ in 1:n_steps
        step!(y, dt)
    end

    # Interior cells (avoid the 1-cell border which `resid_tendency!`
    # may treat as a margin against the Dirichlet 0-halo neighbors).
    interior_view = view(H_ice,  2:Nx-1, 2:Ny-1, 1)
    initial_view  = view(H_init, 2:Nx-1, 2:Ny-1, 1)

    expected = initial_view .+ n_steps * dt * smb_rate
    err = maximum(abs.(interior_view .- expected))
    @test err < 1e-9   # SMB is realised exactly cell-by-cell

    # Per-cell mass-balance accounting on the final step.
    dHidt     = interior(y.tpo.dHidt)
    dHidt_dyn = interior(y.tpo.dHidt_dyn)
    mb_net    = interior(y.tpo.mb_net)
    smb       = interior(y.tpo.smb)
    bmb       = interior(y.tpo.bmb)
    mb_resid  = interior(y.tpo.mb_resid)

    fmb       = interior(y.tpo.fmb)
    dmb       = interior(y.tpo.dmb)
    mb_relax  = interior(y.tpo.mb_relax)

    err_total = maximum(abs.(view(dHidt, 2:Nx-1, 2:Ny-1, 1) .-
                             (view(dHidt_dyn, 2:Nx-1, 2:Ny-1, 1) .+
                              view(mb_net,    2:Nx-1, 2:Ny-1, 1))))
    err_net   = maximum(abs.(mb_net .- (smb .+ bmb .+ fmb .+ dmb .+
                                        mb_relax .+ mb_resid)))
    @test err_total < 1e-6
    @test err_net   < 1e-12

    # Interior dHidt should equal smb_rate (no advection, no resid, no bmb, no fmb).
    err_dh_interior = maximum(abs.(view(dHidt, 2:Nx-1, 2:Ny-1, 1) .- smb_rate))
    @test err_dh_interior < 1e-9

    # f_ice should be binary 0/1 and exactly mirror H_ice > 0.
    f_ice = interior(y.tpo.f_ice)
    @test sort(unique(f_ice)) ⊆ [0.0, 1.0]
    @test all((f_ice .> 0) .== (H_ice .> 0))
end

# ------------------------------------------------------------------
# BMB conservation test (Phase 4)
# ------------------------------------------------------------------

@testset "tpo: BMB conservation (slab + uniform basal melt)" begin
    @assert isfile(RESTART_PATH)

    y = YelmoModel(RESTART_PATH, 0.0;
                   rundir = mktempdir(; prefix="tpo_bmb_test_"),
                   alias  = "tpo-bmb-test",
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    H_ice = interior(y.tpo.H_ice)
    fill!(H_ice, 1000.0)

    fill!(interior(y.dyn.ux_bar), 0.0)
    fill!(interior(y.dyn.uy_bar), 0.0)

    fill!(interior(y.bnd.mask_ice),    Float64(MASK_ICE_DYNAMIC))
    fill!(interior(y.bnd.ice_allowed), 1.0)

    # Bed well above sea level → grounded everywhere so the `pmp`
    # default reduces to `bmb = bmb_grnd` (f_grnd = 1).
    fill!(interior(y.bnd.z_bed), 100.0)
    fill!(interior(y.bnd.z_sl),    0.0)

    # Zero SMB / FMB inputs; bmb_grnd is the only mass sink. bmb_shlf is
    # irrelevant (nothing floats) but zero it for clarity.
    fill!(interior(y.bnd.smb_ref),   0.0)
    fill!(interior(y.bnd.bmb_shlf),  0.0)
    fill!(interior(y.bnd.fmb_shlf),  0.0)

    bmb_rate = -1.5
    fill!(interior(y.thrm.bmb_grnd), bmb_rate)

    Nx, Ny  = size(H_ice, 1), size(H_ice, 2)
    H_init  = copy(H_ice)

    n_steps = 3
    dt      = 1.0
    for _ in 1:n_steps
        step!(y, dt)
    end

    interior_view = view(H_ice,  2:Nx-1, 2:Ny-1, 1)
    initial_view  = view(H_init, 2:Nx-1, 2:Ny-1, 1)

    expected = initial_view .+ n_steps * dt * bmb_rate
    err = maximum(abs.(interior_view .- expected))
    @test err < 1e-9

    dHidt     = interior(y.tpo.dHidt)
    dHidt_dyn = interior(y.tpo.dHidt_dyn)
    mb_net    = interior(y.tpo.mb_net)
    smb       = interior(y.tpo.smb)
    bmb       = interior(y.tpo.bmb)
    mb_resid  = interior(y.tpo.mb_resid)

    fmb       = interior(y.tpo.fmb)
    dmb       = interior(y.tpo.dmb)
    mb_relax  = interior(y.tpo.mb_relax)

    err_total = maximum(abs.(view(dHidt, 2:Nx-1, 2:Ny-1, 1) .-
                             (view(dHidt_dyn, 2:Nx-1, 2:Ny-1, 1) .+
                              view(mb_net,    2:Nx-1, 2:Ny-1, 1))))
    err_net   = maximum(abs.(mb_net .- (smb .+ bmb .+ fmb .+ dmb .+
                                        mb_relax .+ mb_resid)))
    @test err_total < 1e-6
    @test err_net   < 1e-12

    # bmb_ref/bmb interior should match the prescribed grounded rate.
    bmb_ref = interior(y.tpo.bmb_ref)
    @test maximum(abs.(view(bmb_ref, 2:Nx-1, 2:Ny-1, 1) .- bmb_rate)) < 1e-12
    @test maximum(abs.(view(bmb,     2:Nx-1, 2:Ny-1, 1) .- bmb_rate)) < 1e-12

    err_dh_interior = maximum(abs.(view(dHidt, 2:Nx-1, 2:Ny-1, 1) .- bmb_rate))
    @test err_dh_interior < 1e-9
end

# ------------------------------------------------------------------
# BMB combiner — kernel-level (calc_bmb_total!)
# ------------------------------------------------------------------

@testset "tpo: calc_bmb_total!" begin
    Nx = 6
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 60e3), y=(0.0, 60e3),
                        topology=(Bounded, Bounded, Flat))
    bmb      = CenterField(g)
    bmb_grnd = CenterField(g)
    bmb_shlf = CenterField(g)
    H_ice    = CenterField(g)
    H_grnd   = CenterField(g)
    f_grnd   = CenterField(g)

    fill!(interior(bmb_grnd), -1.0)
    fill!(interior(bmb_shlf), -10.0)
    fill!(interior(H_ice),  1000.0)
    fill!(interior(H_grnd),  500.0)
    fill!(interior(f_grnd),    1.0)

    # Fully grounded: every method picks bmb_grnd.
    for method in ("fcmp", "fmp", "pmp", "nmp")
        calc_bmb_total!(bmb, bmb_grnd, bmb_shlf, H_ice, H_grnd, f_grnd, method)
        @test all(interior(bmb) .≈ -1.0)
    end

    # Fully floating (H_grnd ≤ 0, f_grnd == 0): every method picks bmb_shlf.
    fill!(interior(H_grnd), -500.0)
    fill!(interior(f_grnd),    0.0)
    for method in ("fcmp", "fmp", "pmp", "nmp")
        calc_bmb_total!(bmb, bmb_grnd, bmb_shlf, H_ice, H_grnd, f_grnd, method)
        @test all(interior(bmb) .≈ -10.0)
    end

    # Partial f_grnd = 0.5: pmp blends 50/50.
    fill!(interior(H_grnd), -100.0)
    fill!(interior(f_grnd),    0.5)
    calc_bmb_total!(bmb, bmb_grnd, bmb_shlf, H_ice, H_grnd, f_grnd, "pmp")
    @test all(interior(bmb) .≈ 0.5 * -1.0 + 0.5 * -10.0)

    # nmp leaves bmb_grnd at f_grnd > 0; only fully-floating cells get bmb_shlf.
    calc_bmb_total!(bmb, bmb_grnd, bmb_shlf, H_ice, H_grnd, f_grnd, "nmp")
    @test all(interior(bmb) .≈ -1.0)

    # Bare grounded land (H_grnd > 0, H_ice == 0) is forced to 0.
    fill!(interior(H_grnd),  500.0)
    fill!(interior(f_grnd),    1.0)
    fill!(interior(H_ice),    0.0)
    calc_bmb_total!(bmb, bmb_grnd, bmb_shlf, H_ice, H_grnd, f_grnd, "pmp")
    @test all(interior(bmb) .== 0.0)

    # Unsupported / unknown methods error.
    @test_throws ErrorException calc_bmb_total!(bmb, bmb_grnd, bmb_shlf,
        H_ice, H_grnd, f_grnd, "pmpt")
    @test_throws ErrorException calc_bmb_total!(bmb, bmb_grnd, bmb_shlf,
        H_ice, H_grnd, f_grnd, "bogus")
end

# ------------------------------------------------------------------
# Frontal mass balance — kernel-level (calc_fmb_total!)
# ------------------------------------------------------------------

@testset "tpo: calc_fmb_total!" begin
    Nx = 6
    dx = 16e3
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, Nx*dx), y=(0.0, Nx*dx),
                        topology=(Bounded, Bounded, Flat))
    fmb       = CenterField(g)
    fmb_shlf  = CenterField(g)
    bmb_shlf  = CenterField(g)
    H_ice     = CenterField(g)
    H_grnd    = CenterField(g)
    f_ice     = CenterField(g)

    rho_ice, rho_sw = 910.0, 1028.0

    # 1. Method 0 = pass-through: fmb = fmb_shlf, regardless of state.
    fill!(interior(fmb_shlf), -2.0)
    fill!(interior(bmb_shlf), -3.0)
    fill!(interior(H_ice),    1000.0)
    fill!(interior(H_grnd),   500.0)
    fill!(interior(f_ice),    1.0)
    calc_fmb_total!(fmb, fmb_shlf, bmb_shlf, H_ice, H_grnd, f_ice,
                    0, 1.0, rho_ice, rho_sw, dx)
    @test all(interior(fmb) .≈ -2.0)

    # 2. Method 2, no marine-ice cells (H_grnd ≥ H_ice everywhere) →
    #    fmb = 0 even where fmb_shlf is nonzero.
    fill!(interior(H_grnd),   1000.0)
    calc_fmb_total!(fmb, fmb_shlf, bmb_shlf, H_ice, H_grnd, f_ice,
                    2, 1.0, rho_ice, rho_sw, dx)
    @test all(interior(fmb) .== 0.0)

    # 3. Method 2 with floating ice and a single ice-free neighbour.
    #    Only the cell adjacent to the gap has nonzero fmb. Submerged
    #    front depth dz = H_eff * rho_ice/rho_sw → area_flt = 1*dz*dx,
    #    fmb = fmb_shlf * (area_flt / dx^2) = fmb_shlf * dz / dx.
    fill!(interior(H_ice),  1000.0)
    fill!(interior(H_grnd), -100.0)   # floating everywhere
    fill!(interior(f_ice),    1.0)
    fill!(interior(fmb_shlf), -5.0)
    interior(H_ice)[3, 3, 1] = 0.0    # carve a single ice-free cell
    interior(f_ice)[3, 3, 1] = 0.0
    calc_fmb_total!(fmb, fmb_shlf, bmb_shlf, H_ice, H_grnd, f_ice,
                    2, 1.0, rho_ice, rho_sw, dx)
    F = interior(fmb)
    H_eff = 1000.0
    dz = H_eff * rho_ice / rho_sw
    expected_neighbour = -5.0 * (dz * dx) / (dx * dx)
    # The four orthogonal neighbours of (3,3) each see exactly one
    # ice-free neighbour (the gap).
    @test F[2, 3, 1] ≈ expected_neighbour
    @test F[4, 3, 1] ≈ expected_neighbour
    @test F[3, 2, 1] ≈ expected_neighbour
    @test F[3, 4, 1] ≈ expected_neighbour
    # The gap cell itself has H_ice=0 → not marine-ice → fmb=0.
    @test F[3, 3, 1] == 0.0
    # A cell two away from the gap on the interior has all ice neighbours
    # (and is not on the domain boundary), so fmb = 0.
    @test F[2, 2, 1] == 0.0

    # 4. Method 1: bmb_eff comes from the ice-free neighbours' bmb_shlf.
    #    With uniform bmb_shlf = -3, neighbour mean is -3, so
    #    fmb = -3 * (dz/dx) * fmb_scale.
    fmb_scale = 0.5
    calc_fmb_total!(fmb, fmb_shlf, bmb_shlf, H_ice, H_grnd, f_ice,
                    1, fmb_scale, rho_ice, rho_sw, dx)
    expected_method1 = -3.0 * dz / dx * fmb_scale
    F = interior(fmb)
    @test F[2, 3, 1] ≈ expected_method1
    @test F[4, 3, 1] ≈ expected_method1

    # 5. Grounded marine ice (H_grnd in (0, H_ice)): dz uses the
    #    submerged-only depth max((H_eff - H_grnd)*rho_ice/rho_sw, 0).
    fill!(interior(H_ice),  1000.0)
    fill!(interior(H_grnd),  200.0)   # grounded but partially submerged
    fill!(interior(f_ice),    1.0)
    fill!(interior(fmb_shlf), -5.0)
    interior(H_ice)[3, 3, 1] = 0.0
    interior(f_ice)[3, 3, 1] = 0.0
    calc_fmb_total!(fmb, fmb_shlf, bmb_shlf, H_ice, H_grnd, f_ice,
                    2, 1.0, rho_ice, rho_sw, dx)
    F = interior(fmb)
    dz_grnd = max((1000.0 - 200.0) * rho_ice / rho_sw, 0.0)
    expected_grnd = -5.0 * dz_grnd / dx
    @test F[2, 3, 1] ≈ expected_grnd

    # 6. Unknown method errors.
    @test_throws ErrorException calc_fmb_total!(fmb, fmb_shlf, bmb_shlf,
        H_ice, H_grnd, f_ice, 99, 1.0, rho_ice, rho_sw, dx)
end

# ------------------------------------------------------------------
# Subgrid discharge mass balance — kernel-level (calc_mb_discharge!)
# ------------------------------------------------------------------

@testset "tpo: calc_mb_discharge!" begin
    Nx = 4
    dx = 16e3
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, Nx*dx), y=(0.0, Nx*dx),
                        topology=(Bounded, Bounded, Flat))
    dmb         = CenterField(g)
    H_ice       = CenterField(g)
    z_srf       = CenterField(g)
    z_bed_sd    = CenterField(g)
    dist_grline = CenterField(g)
    dist_margin = CenterField(g)
    f_ice       = CenterField(g)

    # Prefill `dmb` with a sentinel; method 0 should overwrite to 0.
    fill!(interior(dmb), 99.0)
    fill!(interior(H_ice), 1000.0)
    fill!(interior(f_ice),    1.0)

    calc_mb_discharge!(dmb, H_ice, z_srf, z_bed_sd,
                       dist_grline, dist_margin, f_ice,
                       0,                # method
                       dx, 60.0, 100.0, 300.0, 3.0, 1.0)
    @test all(interior(dmb) .== 0.0)

    # Any non-zero method errors with the deferred-implementation note.
    @test_throws ErrorException calc_mb_discharge!(dmb, H_ice, z_srf, z_bed_sd,
        dist_grline, dist_margin, f_ice,
        1, dx, 60.0, 100.0, 300.0, 3.0, 1.0)
end

# ------------------------------------------------------------------
# Relaxation helpers — kernel-level (set_tau_relax! + calc_G_relaxation!)
# ------------------------------------------------------------------

@testset "tpo: set_tau_relax!" begin
    Nx = 6
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 60e3), y=(0.0, 60e3),
                        topology=(Bounded, Bounded, Flat))
    tau_relax = CenterField(g)
    H_ice     = CenterField(g)
    f_grnd    = CenterField(g)
    mask_grz  = CenterField(g)
    H_ref     = CenterField(g)

    fill!(interior(H_ice), 1000.0)
    fill!(interior(H_ref), 1000.0)
    tau = 10.0

    # topo_rel = 0 → no relaxation anywhere (sentinel = -1).
    fill!(interior(f_grnd), 1.0)
    set_tau_relax!(tau_relax, H_ice, f_grnd, mask_grz, H_ref, 0, tau)
    @test all(interior(tau_relax) .== -1.0)

    # topo_rel = 1 → tau where f_grnd == 0 (floating) OR H_ref == 0.
    fill!(interior(f_grnd),  1.0)   # all grounded
    fill!(interior(H_ref),   1000.0)
    interior(f_grnd)[3, 3, 1] = 0.0   # one floating cell
    interior(H_ref)[5, 5, 1]  = 0.0   # one ice-free-reference cell
    set_tau_relax!(tau_relax, H_ice, f_grnd, mask_grz, H_ref, 1, tau)
    T = interior(tau_relax)
    @test T[3, 3, 1] == tau
    @test T[5, 5, 1] == tau
    @test T[2, 2, 1] == -1.0   # grounded with H_ref > 0

    # topo_rel = 2 → tau on floating + on grounded cells with a
    # floating orthogonal neighbour.
    fill!(interior(f_grnd), 1.0)
    fill!(interior(H_ref),  1000.0)
    interior(f_grnd)[3, 3, 1] = 0.0   # one floating cell
    set_tau_relax!(tau_relax, H_ice, f_grnd, mask_grz, H_ref, 2, tau)
    T = interior(tau_relax)
    @test T[3, 3, 1] == tau     # floating itself
    @test T[2, 3, 1] == tau     # grounded, has floating neighbour to E
    @test T[4, 3, 1] == tau     # grounded, has floating neighbour to W
    @test T[3, 2, 1] == tau     # grounded, has floating neighbour to N
    @test T[3, 4, 1] == tau     # grounded, has floating neighbour to S
    @test T[2, 2, 1] == -1.0    # diagonal neighbour: not on the GZ
    @test T[1, 1, 1] == -1.0    # corner, no floating neighbour

    # topo_rel = 3 → tau everywhere.
    set_tau_relax!(tau_relax, H_ice, f_grnd, mask_grz, H_ref, 3, tau)
    @test all(interior(tau_relax) .== tau)

    # topo_rel = 4 → tau on grounding-zone cells (mask_grz ∈ {0, 1}).
    fill!(interior(mask_grz), 2.0)             # default: out of zone
    interior(mask_grz)[2, 2, 1] = 0.0          # GL cell
    interior(mask_grz)[3, 3, 1] = 1.0          # grounded in zone
    interior(mask_grz)[4, 4, 1] = -1.0         # floating in zone — NOT relaxed
    interior(mask_grz)[5, 5, 1] = -2.0         # floating out of zone
    set_tau_relax!(tau_relax, H_ice, f_grnd, mask_grz, H_ref, 4, tau)
    T = interior(tau_relax)
    @test T[2, 2, 1] == tau       # mask_grz = 0
    @test T[3, 3, 1] == tau       # mask_grz = 1
    @test T[4, 4, 1] == -1.0      # mask_grz = -1 (floating in zone, not relaxed)
    @test T[5, 5, 1] == -1.0      # mask_grz = -2 (floating out of zone)
    @test T[1, 1, 1] == -1.0      # mask_grz = 2 (grounded out of zone)
end

@testset "tpo: calc_G_relaxation!" begin
    Nx = 4
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 40e3), y=(0.0, 40e3),
                        topology=(Bounded, Bounded, Flat))
    dHdt      = CenterField(g)
    H_ice     = CenterField(g)
    H_ref     = CenterField(g)
    tau_relax = CenterField(g)

    fill!(interior(H_ice),  500.0)
    fill!(interior(H_ref), 1000.0)

    # tau > 0: dHdt = (H_ref - H_ice) / tau.
    fill!(interior(tau_relax), 50.0)
    fill!(interior(dHdt), 99.0)   # sentinel
    calc_G_relaxation!(dHdt, H_ice, H_ref, tau_relax, 1.0)
    @test all(interior(dHdt) .≈ (1000.0 - 500.0) / 50.0)

    # tau == 0 with dt > 0: impose H_ref in a single step → dHdt = (H_ref-H_ice)/dt.
    fill!(interior(tau_relax), 0.0)
    calc_G_relaxation!(dHdt, H_ice, H_ref, tau_relax, 2.5)
    @test all(interior(dHdt) .≈ (1000.0 - 500.0) / 2.5)

    # tau == 0 with dt == 0: fall-back denominator is 1.0.
    calc_G_relaxation!(dHdt, H_ice, H_ref, tau_relax, 0.0)
    @test all(interior(dHdt) .≈ 1000.0 - 500.0)

    # tau < 0: no relaxation (dHdt = 0).
    fill!(interior(tau_relax), -1.0)
    fill!(interior(dHdt), 99.0)
    calc_G_relaxation!(dHdt, H_ice, H_ref, tau_relax, 1.0)
    @test all(interior(dHdt) .== 0.0)

    # Mixed mask: one cell on, one off.
    fill!(interior(tau_relax), -1.0)
    interior(tau_relax)[2, 2, 1] = 100.0
    calc_G_relaxation!(dHdt, H_ice, H_ref, tau_relax, 1.0)
    G = interior(dHdt)
    @test G[2, 2, 1] ≈ (1000.0 - 500.0) / 100.0
    @test G[1, 1, 1] == 0.0
    @test G[3, 3, 1] == 0.0
end

# ------------------------------------------------------------------
# Relaxation integration test (Phase 8)
# ------------------------------------------------------------------

@testset "tpo: relaxation conservation (slab + H_ref)" begin
    @assert isfile(RESTART_PATH)

    # Use YelmoModelPar.ytopo_params explicitly — Yelmo re-exports the
    # mirror module's `ytopo_params` at top level (not the model's), so
    # build the model-flavoured params struct via its own helper.
    p_ytopo = Yelmo.YelmoModelPar.ytopo_params(
        topo_rel       = 3,         # relax all cells
        topo_rel_tau   = 5.0,       # 5-yr timescale
        topo_rel_field = "H_ref",
    )
    p = YelmoModelParameters("tpo-relax-test"; ytopo = p_ytopo)

    y = YelmoModel(RESTART_PATH, 0.0;
                   p      = p,
                   rundir = mktempdir(; prefix="tpo_relax_test_"),
                   alias  = "tpo-relax-test",
                   groups = (:bnd, :dyn, :mat, :thrm, :tpo),
                   strict = false)

    # Slab geometry; everything zeroed except the relaxation target.
    H_ice = interior(y.tpo.H_ice)
    fill!(H_ice, 500.0)
    fill!(interior(y.dyn.ux_bar), 0.0)
    fill!(interior(y.dyn.uy_bar), 0.0)
    fill!(interior(y.bnd.mask_ice),   Float64(MASK_ICE_DYNAMIC))
    fill!(interior(y.bnd.ice_allowed), 1.0)
    fill!(interior(y.bnd.z_bed), 100.0)
    fill!(interior(y.bnd.z_sl),    0.0)
    fill!(interior(y.bnd.smb_ref),   0.0)
    fill!(interior(y.bnd.bmb_shlf),  0.0)
    fill!(interior(y.bnd.fmb_shlf),  0.0)
    fill!(interior(y.thrm.bmb_grnd), 0.0)

    # Reference thickness 1000 m → ice should grow toward it.
    fill!(interior(y.bnd.H_ice_ref), 1000.0)

    Nx, Ny = size(H_ice, 1), size(H_ice, 2)
    H_init = copy(H_ice)

    step!(y, 1.0)

    # After one step of 1 yr with tau = 5 yr toward H_ref = 1000 m:
    # dHdt = (1000 - 500) / 5 = 100 m/yr → ΔH = 100 m → H_new = 600 m.
    interior_view = view(H_ice, 2:Nx-1, 2:Ny-1, 1)
    @test all(abs.(interior_view .- 600.0) .< 1e-9)

    # mb_relax should be exactly 100 m/yr in the interior.
    @test all(abs.(view(interior(y.tpo.mb_relax), 2:Nx-1, 2:Ny-1, 1) .- 100.0)
              .< 1e-9)

    # mb_net accounting still balances (smb=bmb=fmb=dmb=mb_resid=0).
    smb      = interior(y.tpo.smb)
    bmb      = interior(y.tpo.bmb)
    fmb      = interior(y.tpo.fmb)
    dmb      = interior(y.tpo.dmb)
    mb_relax = interior(y.tpo.mb_relax)
    mb_resid = interior(y.tpo.mb_resid)
    mb_net   = interior(y.tpo.mb_net)
    @test maximum(abs.(mb_net .- (smb .+ bmb .+ fmb .+ dmb .+
                                  mb_relax .+ mb_resid))) < 1e-12
end

# ------------------------------------------------------------------
# Grounding helpers — kernel-level
# ------------------------------------------------------------------

@testset "tpo: calc_H_grnd!" begin
    Nx = 8
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 100e3), y=(0.0, 100e3),
                        topology=(Bounded, Bounded, Flat))
    H_ice  = CenterField(g)
    z_bed  = CenterField(g)
    z_sl   = CenterField(g)
    H_grnd = CenterField(g)

    rho_ice, rho_sw = 910.0, 1028.0

    # 1. Bed above sea level: H_grnd = H_ice (no flotation contribution).
    fill!(interior(H_ice), 500.0)
    fill!(interior(z_bed), 100.0)
    fill!(interior(z_sl),    0.0)
    calc_H_grnd!(H_grnd, H_ice, z_bed, z_sl, rho_ice, rho_sw)
    @test all(interior(H_grnd) .≈ 500.0)

    # 2. Thin shelf in deep water: H_grnd = H_ice - depth * rho_sw/rho_ice < 0.
    fill!(interior(H_ice),  100.0)
    fill!(interior(z_bed), -500.0)
    fill!(interior(z_sl),     0.0)
    calc_H_grnd!(H_grnd, H_ice, z_bed, z_sl, rho_ice, rho_sw)
    expected = 100.0 - 500.0 * rho_sw / rho_ice
    @test all(interior(H_grnd) .≈ expected)
    @test all(interior(H_grnd) .< 0.0)

    # 3. At flotation: H_ice * rho_ice = depth * rho_sw → H_grnd = 0.
    depth = 200.0
    H_flot = depth * rho_sw / rho_ice
    fill!(interior(H_ice),  H_flot)
    fill!(interior(z_bed), -depth)
    fill!(interior(z_sl),    0.0)
    calc_H_grnd!(H_grnd, H_ice, z_bed, z_sl, rho_ice, rho_sw)
    @test maximum(abs.(interior(H_grnd))) < 1e-12
end

@testset "tpo: determine_grounded_fractions!" begin
    Nx = 12
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 100e3), y=(0.0, 100e3),
                        topology=(Bounded, Bounded, Flat))
    H_grnd = CenterField(g)
    f_grnd = CenterField(g)

    # 1. Uniformly grounded → f_grnd = 1 everywhere. Boundary corner
    #    stencils mix the in-bounds value with out-of-domain zeros, but
    #    the in-bounds magnitude (|f_flt|=100) keeps every corner mean
    #    well below the negative snap threshold, so all four quadrants
    #    classify as fully grounded even on the domain edge.
    fill!(interior(H_grnd), 100.0)
    determine_grounded_fractions!(f_grnd, H_grnd)
    @test all(interior(f_grnd) .== 1.0)

    # 2. Uniformly floating → f_grnd = 0 everywhere. Out-of-domain
    #    zeros are non-negative, so "all corners ≥ 0" holds and every
    #    quadrant returns 0.
    fill!(interior(H_grnd), -100.0)
    determine_grounded_fractions!(f_grnd, H_grnd)
    @test all(interior(f_grnd) .== 0.0)

    # 3. East-West split (W grounded, E floating). Cells well inside
    #    each side keep f_grnd ∈ {0,1}; cells straddling the boundary
    #    take fractional values. Every cell must end up in [0,1].
    half = Nx ÷ 2
    Hg = interior(H_grnd)
    @inbounds for j in 1:Nx, i in 1:Nx
        Hg[i, j, 1] = i <= half ? 100.0 : -100.0
    end
    determine_grounded_fractions!(f_grnd, H_grnd)
    fg = interior(f_grnd)
    @test all(0.0 .<= fg .<= 1.0)
    # Two columns into the grounded side → fully grounded.
    @test all(view(fg, 1:half-1, :, 1) .== 1.0)
    # Two columns into the floating side → fully floating.
    @test all(view(fg, half+2:Nx, :, 1) .== 0.0)

    # 4. Single floating cell embedded in a grounded sheet → that cell
    #    has f_grnd < 1, and its 8-cell neighbourhood is partially
    #    affected via the corner stencils. Cells two away are pristine.
    fill!(interior(H_grnd), 100.0)
    Hg[6, 6, 1] = -100.0
    determine_grounded_fractions!(f_grnd, H_grnd)
    @test fg[6, 6, 1] < 1.0
    @test fg[6, 6, 1] >= 0.0
    @test fg[3, 3, 1] == 1.0   # far from the perturbation
    @test fg[9, 9, 1] == 1.0
end

# ------------------------------------------------------------------
# Analytical benchmarks for determine_grounded_fractions!
# ------------------------------------------------------------------

# Tier 1 helper: Sutherland-Hodgman clip of the unit square against the
# half-plane {α·u + β·v + γ ≥ 0}, then shoelace area. Returns the area
# in [0, 1] of the grounded sub-region.
function _grounded_area_linear_unit_cell(α::Float64, β::Float64, γ::Float64)
    @inline f(p) = α * p[1] + β * p[2] + γ
    poly = NTuple{2,Float64}[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    out  = NTuple{2,Float64}[]
    n = length(poly)
    for k in 1:n
        p1 = poly[k]
        p2 = poly[mod1(k + 1, n)]
        f1 = f(p1)
        f2 = f(p2)
        in1 = f1 >= 0.0
        in2 = f2 >= 0.0
        if in1 && in2
            push!(out, p2)
        elseif in1 && !in2
            t = f1 / (f1 - f2)
            push!(out, (p1[1] + t*(p2[1] - p1[1]),
                        p1[2] + t*(p2[2] - p1[2])))
        elseif !in1 && in2
            t = f1 / (f1 - f2)
            push!(out, (p1[1] + t*(p2[1] - p1[1]),
                        p1[2] + t*(p2[2] - p1[2])))
            push!(out, p2)
        end
    end
    isempty(out) && return 0.0
    A = 0.0
    m = length(out)
    for k in 1:m
        x1, y1 = out[k]
        x2, y2 = out[mod1(k + 1, m)]
        A += x1*y2 - x2*y1
    end
    return abs(A) / 2.0
end

@testset "tpo: determine_grounded_fractions! — linear GL (analytical)" begin
    # For each cell the analytical grounded fraction of a linear
    # `H_grnd(x,y) = a·x + b·y + c` is the area of
    # `{a·x + b·y + c ≥ 0} ∩ cell` divided by cell area, computable in
    # closed form by half-plane / unit-square clipping.
    #
    # The stabilised kernel detects the linear case (bilinear cross-
    # coefficient |dd| < _LINEAR_DD_TOL) and computes the grounded
    # area by exact polygon clip — no perturbation, no snap. So the
    # algorithm should reproduce the analytical fraction to floating-
    # point precision for any linear `H_grnd`.
    #
    # Tested only on the inner (Nx-2)×(Ny-2) block: outermost cells
    # have corner stencils that read out-of-domain values as 0, so the
    # algorithm sees a different `H_grnd` near the boundary than the
    # analytical function suggests.

    Nx = 12
    dx = 1.0
    L  = Nx * dx
    g  = RectilinearGrid(size=(Nx, Nx),
                         x=(0.0, L), y=(0.0, L),
                         topology=(Bounded, Bounded, Flat))
    H_grnd = CenterField(g)
    f_grnd = CenterField(g)

    # (label, a, b, c) such that f(x,y) = a*x + b*y + c.
    cases = [
        ("vertical_GL",     1.0,  0.0, -5.5),
        ("vertical_offset", 1.0,  0.0, -6.3),
        ("horizontal",      0.0,  1.0, -5.7),
        ("diag_45",         1.0,  1.0, -10.5),
        ("anti_diag",       1.0, -1.0,  -1.3),
        ("oblique_a",       2.0,  1.0, -16.7),
        ("oblique_b",      -1.0,  3.0, -10.4),
        ("oblique_c",       0.7, -2.3,   8.3),
    ]

    tol = 1e-12

    for (label, a, b, c) in cases
        Hg = interior(H_grnd)
        @inbounds for j in 1:Nx, i in 1:Nx
            xc = (i - 0.5) * dx
            yc = (j - 0.5) * dx
            Hg[i, j, 1] = a*xc + b*yc + c
        end

        determine_grounded_fractions!(f_grnd, H_grnd)
        fg = interior(f_grnd)

        # Compare on the inner block where the 9-point corner stencil
        # is unaffected by the out-of-domain Dirichlet halo.
        max_err = 0.0
        for j in 2:Nx-1, i in 2:Nx-1
            x_W = (i - 1) * dx
            y_S = (j - 1) * dx
            α = a * dx
            β = b * dx
            γ = a*x_W + b*y_S + c
            expected = _grounded_area_linear_unit_cell(α, β, γ)
            max_err = max(max_err, abs(fg[i, j, 1] - expected))
        end

        @testset "linear: $label" begin
            @test max_err < tol
        end
    end
end

@testset "tpo: determine_grounded_fractions! — circular GL (convergence)" begin
    # Smooth nonlinear flotation field with an analytically known
    # grounded area: H_grnd(x,y) = R² − (x−x₀)² − (y−y₀)². The grounded
    # set is a disk of radius R, area π·R². The CISM scheme is not
    # exact here (bilinear is not exact for r²), but the error should
    # converge with refinement.
    #
    # The disk is centred well inside the domain so that boundary
    # cells (which read out-of-domain f_flt as 0) never see grounded
    # ice — the contamination is confined to a region with f_grnd = 0
    # anyway.

    R   = 4.0
    x0  = 5.0
    y0  = 5.0
    L   = 10.0
    A_exact = π * R^2

    spacings = Float64[]
    rel_errs = Float64[]
    for Nx in (20, 40, 80, 160)
        dx = L / Nx
        g  = RectilinearGrid(size=(Nx, Nx),
                             x=(0.0, L), y=(0.0, L),
                             topology=(Bounded, Bounded, Flat))
        H_grnd = CenterField(g)
        f_grnd = CenterField(g)

        Hg = interior(H_grnd)
        @inbounds for j in 1:Nx, i in 1:Nx
            x = (i - 0.5) * dx
            y = (j - 0.5) * dx
            Hg[i, j, 1] = R^2 - (x - x0)^2 - (y - y0)^2
        end
        determine_grounded_fractions!(f_grnd, H_grnd)

        A_est   = sum(interior(f_grnd)) * dx * dx
        rel_err = abs(A_est - A_exact) / A_exact
        push!(spacings, dx)
        push!(rel_errs, rel_err)
    end

    # Convergence rates between successive resolutions:
    #   rate_k = log(err_k / err_{k+1}) / log(dx_k / dx_{k+1})
    # For dx halving each step, log(dx_k / dx_{k+1}) = log(2).
    rates = [log(rel_errs[k] / rel_errs[k+1]) /
             log(spacings[k]  / spacings[k+1])  for k in 1:length(rel_errs)-1]

    println("circular-GL convergence:")
    for k in 1:length(rel_errs)
        println("  Nx=$(round(Int, L/spacings[k]))  dx=$(spacings[k])  rel_err=$(rel_errs[k])")
    end
    for k in 1:length(rates)
        println("  rate[dx=$(spacings[k])→$(spacings[k+1])] = $(rates[k])")
    end

    # Bounds tuned to the stabilised kernel's actual behaviour: clean
    # 2nd-order convergence all the way to the finest grid (no noise
    # floor — the Fortran d=0 perturbation and corner snap have been
    # replaced by analytical limits).
    @test all(rel_errs .> 0.0)
    @test rel_errs[end] < 1e-3              # < 0.1% error at finest grid
    @test all(diff(rel_errs) .< 0.0)        # strictly monotone refinement
    @test all(1.7 .<= rates .<= 2.3)        # all rates ~ 2nd-order
end

# ------------------------------------------------------------------
# Calving (phase 7) — kernel-level + end-to-end smoke tests.
# ------------------------------------------------------------------

using Yelmo.YelmoModelPar: YelmoModelParameters, ytopo_params, ycalv_params

# Helper: tiny Bounded grid for kernel-level calving tests.
_calv_grid(nx, ny; dx=1.0) = RectilinearGrid(size=(nx, ny),
                                              x=(0.0, nx*dx), y=(0.0, ny*dx),
                                              topology=(Bounded, Bounded, Flat))

@testset "tpo: lsf_init!" begin
    g = _calv_grid(8, 3)
    H   = CenterField(g); fill!(interior(H),   0.0)
    zb  = CenterField(g); fill!(interior(zb), -100.0)
    zsl = CenterField(g); fill!(interior(zsl),  0.0)
    lsf = CenterField(g)

    # Ice in left half; cliff (above SL) at i=7.
    for i in 1:4, j in 1:3; interior(H)[i, j, 1] = 100.0; end
    for j in 1:3; interior(zb)[7, j, 1] = 50.0; end

    lsf_init!(lsf, H, zb, zsl)
    expected = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
    @test interior(lsf)[:, 1, 1] == expected
    @test interior(lsf)[:, 2, 1] == expected
end

@testset "tpo: extrapolate_ocn_ac!" begin
    g = _calv_grid(8, 2)

    # acx case.
    w  = XFaceField(g); fill!(interior(w), 0.0)
    ux = XFaceField(g); fill!(interior(ux), 0.0)
    for i in 3:6, j in 1:2
        interior(w)[i, j, 1]  = 5.0
        interior(ux)[i, j, 1] = 5.0
    end
    ux_orig = copy(interior(ux))
    extrapolate_ocn_acx!(w; reference = ux)
    @test all(interior(w)[:, 1, 1] .== 5.0)         # filled in both directions
    @test interior(ux) == ux_orig                    # reference must not be mutated

    # Empty row stays zero.
    fill!(interior(w),  0.0); fill!(interior(ux), 0.0)
    extrapolate_ocn_acx!(w; reference = ux)
    @test all(interior(w) .== 0.0)

    # acy case.
    w2  = YFaceField(g); fill!(interior(w2), 0.0)
    uy  = YFaceField(g); fill!(interior(uy), 0.0)
    for i in 1:8, j in 2:2
        interior(w2)[i, j, 1] = 7.0
        interior(uy)[i, j, 1] = 7.0
    end
    extrapolate_ocn_acy!(w2; reference = uy)
    @test all(interior(w2)[1, :, 1] .== 7.0)
end

@testset "tpo: lsf_redistance!" begin
    g = _calv_grid(20, 3)
    lsf = CenterField(g)

    # Sharp ±10 step at i=10/11 boundary.
    for i in 1:20, j in 1:3
        interior(lsf)[i, j, 1] = i <= 10 ? -10.0 : 10.0
    end

    lsf_redistance!(lsf, 1.0, 1.0; n_iter = 30)

    # Zero level set preserved between cells 10 and 11.
    @test interior(lsf)[10, 1, 1] ≈ -0.5 atol = 1e-3
    @test interior(lsf)[11, 1, 1] ≈  0.5 atol = 1e-3

    # |∇φ| = 1 near the front (cells 8..13).
    diffs = [interior(lsf)[i+1, 1, 1] - interior(lsf)[i, 1, 1] for i in 8:12]
    @test all(abs.(diffs .- 1.0) .< 1e-3)

    # Far from the front the Sussman scheme leaves residual deviation,
    # but signs are still preserved.
    @test all(interior(lsf)[1:10,  :, :] .< 0.0)
    @test all(interior(lsf)[11:20, :, :] .> 0.0)
end

@testset "tpo: lsf_update! (transport)" begin
    g = _calv_grid(20, 3)
    lsf = CenterField(g)
    u   = XFaceField(g); fill!(interior(u), 0.5)   # eastward at 0.5 m/yr
    v   = YFaceField(g); fill!(interior(v), 0.0)
    crx = XFaceField(g); fill!(interior(crx), 0.0)
    cry = YFaceField(g); fill!(interior(cry), 0.0)

    # Step front at i=10/11: ice on left (-1), ocean on right (+1).
    for i in 1:20, j in 1:3
        interior(lsf)[i, j, 1] = i <= 10 ? -1.0 : 1.0
    end

    # Advance 1 yr; with cr=0, lsf passively translates at +0.5 cells/yr.
    lsf_update!(lsf, u, v, crx, cry, 1.0; cfl_safety = 0.5)

    # The zero level set should have moved to the east. Specifically the
    # cell at i=11 (was +1) should now be near 0 (the front passed through).
    @test interior(lsf)[11, 1, 1] < 1.0
    @test interior(lsf)[11, 1, 1] >= 0.0
    @test all(interior(lsf) .>= -1.0)
    @test all(interior(lsf) .<=  1.0)
end

@testset "tpo: calc_calving_equil_ac!" begin
    g = _calv_grid(4, 3)
    u = XFaceField(g); fill!(interior(u), 100.0)
    v = YFaceField(g); fill!(interior(v),  50.0)
    crx = XFaceField(g)
    cry = YFaceField(g)

    calc_calving_equil_ac!(crx, cry, u, v)
    @test all(interior(crx) .== -100.0)
    @test all(interior(cry) .==  -50.0)
end

@testset "tpo: calc_calving_threshold_ac!" begin
    g = _calv_grid(5, 3)
    H = CenterField(g); fill!(interior(H), 0.0)
    F = CenterField(g); fill!(interior(F), 0.0)
    u = XFaceField(g); fill!(interior(u), 100.0)
    v = YFaceField(g); fill!(interior(v),   0.0)
    crx = XFaceField(g)
    cry = YFaceField(g)

    # Ice in cells i=2..3, j=2..2.
    for i in 2:3, j in 2:2
        interior(H)[i, j, 1] = 100.0
        interior(F)[i, j, 1] = 1.0
    end

    calc_calving_threshold_ac!(crx, cry, u, v, H, F, 200.0)
    # Faces at j=2 (XFaceField has nx+1 = 6 faces):
    #   i=1: ocean/ocean,  H_acx=0,   wv=2.0,  cr=-200
    #   i=2: ocean→ice,    H_acx=100, wv=1.5,  cr=-150 (ice-side stagger)
    #   i=3: ice/ice,      H_acx=100, wv=1.5,  cr=-150
    #   i=4: ice→ocean,    H_acx=100, wv=1.5,  cr=-150
    #   i=5: ocean/ocean,  H_acx=0,   wv=2.0,  cr=-200
    #   i=6: ocean/(out),  H_acx=0,   wv=2.0,  cr=-200
    expected = [-200.0, -150.0, -150.0, -150.0, -200.0, -200.0]
    @test interior(crx)[:, 2, 1] ≈ expected atol = 1e-12

    # Hc must be positive.
    @test_throws ErrorException calc_calving_threshold_ac!(crx, cry, u, v, H, F, 0.0)
end

@testset "tpo: merge_calving_rates!" begin
    g = _calv_grid(4, 3)
    u = XFaceField(g); fill!(interior(u), 100.0)
    v = YFaceField(g); fill!(interior(v),   0.0)

    cmbfx = XFaceField(g); fill!(interior(cmbfx), -10.0)
    cmbfy = YFaceField(g); fill!(interior(cmbfy),  -5.0)
    cmbgx = XFaceField(g); fill!(interior(cmbgx), 999.0)
    cmbgy = YFaceField(g); fill!(interior(cmbgy), 999.0)
    crx   = XFaceField(g)
    cry   = YFaceField(g)

    fgax = XFaceField(g)
    fgay = YFaceField(g)
    zb   = CenterField(g)
    zsl  = CenterField(g); fill!(interior(zsl), 0.0)

    # Branch 1: floating face → use cmb_flt.
    fill!(interior(fgax), 0.0); fill!(interior(fgay), 0.0)
    fill!(interior(zb), -500.0)
    merge_calving_rates!(crx, cry, cmbfx, cmbfy, cmbgx, cmbgy,
                         u, v, fgax, fgay, zb, zsl)
    @test all(interior(crx) .== -10.0)
    @test all(interior(cry) .==  -5.0)

    # Branch 2: grounded face below SL → use cmb_grnd.
    fill!(interior(fgax), 1.0); fill!(interior(fgay), 1.0)
    merge_calving_rates!(crx, cry, cmbfx, cmbfy, cmbgx, cmbgy,
                         u, v, fgax, fgay, zb, zsl)
    @test all(interior(crx) .== 999.0)

    # Branch 3: grounded face above SL → pin at -u_bar.
    fill!(interior(zb), 100.0)
    merge_calving_rates!(crx, cry, cmbfx, cmbfy, cmbgx, cmbgy,
                         u, v, fgax, fgay, zb, zsl)
    @test all(interior(crx) .== -100.0)
    @test all(interior(cry) .==    0.0)
end

@testset "tpo: calving_step! kill on synthetic shelf" begin
    # 16-km grid with a shelf in the left half. Use a huge calving-rate
    # forcing so retreat is large enough to kill cells in 1 step.
    nx, ny = 8, 4
    dx = 16.0e3
    g = _calv_grid(nx, ny; dx=dx)

    # Synthetic boundary fields.
    z_bed = CenterField(g);  fill!(interior(z_bed), -500.0)
    z_sl  = CenterField(g);  fill!(interior(z_sl),     0.0)
    smb_ref = CenterField(g); fill!(interior(smb_ref), 0.0)

    p = YelmoModelParameters("calv-kill";
        ytopo = ytopo_params(topo_fixed=true, use_bmb=false,
                             dmb_method=0, topo_rel=0),
        ycalv = ycalv_params(use_lsf=true, calv_flt_method="equil",
                             calv_grnd_method="zero", dt_lsf=0.0),
    )
    y = YelmoModel(RESTART_PATH, 0.0; alias="calv-kill", p=p, strict=false)

    # Replace state with the synthetic geometry: shelf cells i=1..4.
    # Force mask_ice = DYNAMIC everywhere so the post-advection mask
    # pass doesn't wipe our patched ice (the restart's mask_ice may
    # mark most of the synthetic grid as MASK_ICE_NONE).
    fill!(interior(y.tpo.H_ice),    0.0)
    fill!(interior(y.bnd.z_bed),   -500.0)
    fill!(interior(y.bnd.z_sl),       0.0)
    fill!(interior(y.bnd.smb_ref),    0.0)
    fill!(interior(y.bnd.mask_ice),
          Float64(Yelmo.YelmoCore.MASK_ICE_DYNAMIC))
    H_pre_full = interior(y.tpo.H_ice)
    nx_full, ny_full = size(H_pre_full, 1), size(H_pre_full, 2)
    for i in 1:min(4, nx_full), j in 1:ny_full
        H_pre_full[i, j, 1] = 500.0
    end

    # Patch the lsf so the right half is ocean (lsf=+1) including a
    # column that overlaps the existing ice (i=4) — this forces a kill.
    lsf_init!(y.tpo.lsf, y.tpo.H_ice, y.bnd.z_bed, y.bnd.z_sl)
    target_i = min(4, nx_full)
    for j in 1:ny_full
        interior(y.tpo.lsf)[target_i, j, 1] = 1.0
    end

    Yelmo.step!(y, 1.0)

    # Column `target_i` had H = 500 with lsf > 0 ⇒ kill: H → 0, cmb < 0.
    @test all(interior(y.tpo.H_ice)[target_i, :, 1] .== 0.0)
    @test all(interior(y.tpo.cmb)[target_i, :, 1] .< 0.0)
    @test interior(y.tpo.cmb)[target_i, 1, 1] ≈ -500.0 atol = 1e-9

    # Mass balance still closes.
    err_total = maximum(abs.(interior(y.tpo.dHidt) .-
                             (interior(y.tpo.dHidt_dyn) .+
                              interior(y.tpo.mb_net))))
    @test err_total < 1e-9
end

@testset "tpo: calving_step! method dispatch" begin
    # vm-m16 errors (mat not yet ported); unknown method also errors.
    p_vm = YelmoModelParameters("calv-vm";
        ytopo = ytopo_params(topo_fixed=true, use_bmb=false,
                             dmb_method=0, topo_rel=0),
        ycalv = ycalv_params(use_lsf=true, calv_flt_method="vm-m16"),
    )
    y_vm = YelmoModel(RESTART_PATH, 0.0; alias="calv-vm",
                      p=p_vm, strict=false)
    lsf_init!(y_vm.tpo.lsf, y_vm.tpo.H_ice, y_vm.bnd.z_bed, y_vm.bnd.z_sl)
    fill!(interior(y_vm.bnd.smb_ref), 0.0)
    @test_throws ErrorException Yelmo.step!(y_vm, 1.0)

    p_bogus = YelmoModelParameters("calv-bogus";
        ytopo = ytopo_params(topo_fixed=true, use_bmb=false,
                             dmb_method=0, topo_rel=0),
        ycalv = ycalv_params(use_lsf=true, calv_flt_method="bogus"),
    )
    y_b = YelmoModel(RESTART_PATH, 0.0; alias="calv-bogus",
                    p=p_bogus, strict=false)
    lsf_init!(y_b.tpo.lsf, y_b.tpo.H_ice, y_b.bnd.z_bed, y_b.bnd.z_sl)
    fill!(interior(y_b.bnd.smb_ref), 0.0)
    @test_throws ErrorException Yelmo.step!(y_b, 1.0)
end

@testset "model: YelmoConstants plumbing" begin
    # Default constants land on `y.c`.
    y = YelmoModel(RESTART_PATH, 0.0; alias="c-default", strict=false)
    @test y.c isa YelmoConstants
    @test y.c.rho_ice == 910.0
    @test y.c.rho_sw  == 1028.0
    @test y.c.g       == 9.81

    # Override via the `c=` keyword; same instance can be shared.
    custom = YelmoConstants(rho_ice=917.0, rho_sw=1027.0)
    y1 = YelmoModel(RESTART_PATH, 0.0; alias="c-shared-1",
                    c=custom, strict=false)
    y2 = YelmoModel(RESTART_PATH, 0.0; alias="c-shared-2",
                    c=custom, strict=false)
    @test y1.c === custom
    @test y2.c === custom
    @test y1.c.rho_ice == 917.0
    @test y2.c.rho_ice == 917.0

    # The masked-cell enums live in YelmoConst now; the values match
    # what YelmoCore re-exports for back-compat.
    @test MASK_ICE_NONE    == 0
    @test MASK_ICE_FIXED   == 1
    @test MASK_ICE_DYNAMIC == 2
end

@testset "model: named constants constructors" begin
    # earth_constants — identical to the bare default.
    e = earth_constants()
    @test e == YelmoConstants()

    # EISMINT — diverges from Earth in three fields; rest fall through.
    ei = eismint_constants()
    @test ei.sec_year   == 31556926.0
    @test ei.rho_ice    == 917.0
    @test ei.T_pmp_beta == 9.7e-8
    @test ei.rho_sw     == 1028.0    # falls through to default
    @test ei.g          == 9.81

    # MISMIP3D — rho_ice = 900.0.
    m = mismip3d_constants()
    @test m.sec_year    == 31556926.0
    @test m.rho_ice     == 900.0
    @test m.T_pmp_beta  == 9.7e-8

    # TROUGH — rho_ice = 918.0.
    t = trough_constants()
    @test t.sec_year    == 31556926.0
    @test t.rho_ice     == 918.0
    @test t.T_pmp_beta  == 9.7e-8

    # Kwargs override on top of the named-experiment defaults.
    custom = eismint_constants(rho_sw=1027.5, rho_ice=915.0)
    @test custom.rho_sw     == 1027.5
    @test custom.rho_ice    == 915.0           # kwargs win over the EISMINT default
    @test custom.sec_year   == 31556926.0      # untouched EISMINT field still set
    @test custom.T_pmp_beta == 9.7e-8

    # Symbol-dispatch constructor agrees with the named functions and
    # supports the Fortran-style aliases.
    @test YelmoConstants(:Earth)    == earth_constants()
    @test YelmoConstants(:EISMINT)  == eismint_constants()
    @test YelmoConstants(:EISMINT1) == eismint_constants()
    @test YelmoConstants(:EISMINT2) == eismint_constants()
    @test YelmoConstants(:MISMIP)   == mismip3d_constants()
    @test YelmoConstants(:MISMIP3D) == mismip3d_constants()
    @test YelmoConstants(:TROUGH)   == trough_constants()

    # kwargs flow through the symbol dispatch.
    sym_override = YelmoConstants(:MISMIP3D, rho_sw=1027.5)
    @test sym_override.rho_sw  == 1027.5
    @test sym_override.rho_ice == 900.0

    @test_throws ErrorException YelmoConstants(:bogus)
end

# ------------------------------------------------------------------
# Distance-to-feature kernels and bed-state masks
# ------------------------------------------------------------------

@testset "tpo: calc_distance_to_grounding_line!" begin
    # 8x8 grid with a 4-cell-wide grounded square. dx = 1 metre to make
    # distance arithmetic trivially auditable.
    Nx = 8
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 8.0), y=(0.0, 8.0),
                        topology=(Bounded, Bounded, Flat))
    f_grnd  = CenterField(g)
    dist_gl = CenterField(g)
    dx = 1.0

    fill!(interior(f_grnd), 0.0)
    @inbounds for j in 3:6, i in 3:6
        interior(f_grnd)[i, j, 1] = 1.0
    end

    calc_distance_to_grounding_line!(dist_gl, f_grnd, dx)
    D = interior(dist_gl)

    # 1. Grounding-line cells (grounded with floating direct neighbour)
    #    are the perimeter of the 4x4 block: dist = 0.
    @test D[3, 3, 1] == 0.0
    @test D[3, 6, 1] == 0.0
    @test D[6, 3, 1] == 0.0
    @test D[6, 6, 1] == 0.0
    @test D[4, 3, 1] == 0.0
    @test D[3, 4, 1] == 0.0

    # 2. Interior of the block (no floating neighbour) is grounded but
    #    not on the GL → positive distance equal to one direct step.
    @test D[4, 4, 1] ≈ 1.0
    @test D[5, 5, 1] ≈ 1.0
    @test D[4, 5, 1] ≈ 1.0
    @test D[5, 4, 1] ≈ 1.0

    # 3. Direct floating neighbour of the GL → -1 (one cell out, negated).
    @test D[2, 4, 1] ≈ -1.0
    @test D[7, 4, 1] ≈ -1.0
    @test D[4, 2, 1] ≈ -1.0
    @test D[4, 7, 1] ≈ -1.0

    # 4. Diagonal floating neighbour of a GL corner → -√2 (chamfer).
    @test D[2, 2, 1] ≈ -sqrt(2.0)
    @test D[7, 7, 1] ≈ -sqrt(2.0)

    # 5. Two cells out (direct) → ≈ -2 (chamfer routes via direct path).
    @test D[1, 4, 1] ≈ -2.0
    @test D[8, 4, 1] ≈ -2.0
end

@testset "tpo: calc_distance_to_grounding_line! — Neumann boundary" begin
    # All-grounded 4x4 grid. With the default Neumann-zero clamp BC on
    # Bounded sides, halo f_grnd = first interior = grounded — so no
    # cell sees a floating neighbour, and there is no GL anywhere.
    # Distance is +Inf everywhere (positive because all cells are
    # grounded; no sign flip in Phase 3).
    Nx = 4
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 4.0), y=(0.0, 4.0),
                        topology=(Bounded, Bounded, Flat))
    f_grnd  = CenterField(g)   # default BC = Neumann zero
    dist_gl = CenterField(g)
    fill!(interior(f_grnd), 1.0)

    calc_distance_to_grounding_line!(dist_gl, f_grnd, 1.0)
    D = interior(dist_gl)
    @test all(isinf, D)
    @test all(>(0), D)
end

@testset "tpo: calc_distance_to_grounding_line! — periodic axes" begin
    # 8x8 grid with a single grounded cell at (1,1). Test that under
    # full-periodic topology, the chamfer distance from the diametric
    # corner (5,5) wraps around the boundary (chamfer distance via
    # wrap is 4·√2 ≈ 5.66, vs. straight-through 4·√2 also — they're
    # equal in this geometry, but periodic-aware code must produce a
    # finite distance).
    Nx = 8
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 8.0), y=(0.0, 8.0),
                        topology=(Periodic, Periodic, Flat))
    f_grnd  = CenterField(g)
    dist_gl = CenterField(g)

    fill!(interior(f_grnd), 0.0)
    interior(f_grnd)[1, 1, 1] = 1.0   # single grounded cell at corner

    calc_distance_to_grounding_line!(dist_gl, f_grnd, 1.0)
    D = interior(dist_gl)

    # The single grounded cell is a GL source (all 4 direct neighbours
    # are floating, including via wrap).
    @test D[1, 1, 1] == 0.0

    # All floating cells get finite (negative) chamfer distance — no
    # cell remains -Inf, since periodic wrap connects every cell to
    # the source.
    @test all(isfinite, D)
    @test all(D[i, j, 1] <= 0.0 for j in 1:Nx, i in 1:Nx)

    # Periodic wrap: cell (1, 8) is one cell south of (1, 1) via the
    # north wrap → distance 1.
    @test D[1, 8, 1] ≈ -1.0
    @test D[8, 1, 1] ≈ -1.0
    @test D[8, 8, 1] ≈ -sqrt(2.0)   # diagonal wrap
end

@testset "tpo: calc_distance_to_ice_margin!" begin
    # Same geometry as the GL test but using f_ice as the source flag.
    Nx = 6
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 6.0), y=(0.0, 6.0),
                        topology=(Bounded, Bounded, Flat))
    f_ice = CenterField(g)
    dist_mrgn = CenterField(g)

    fill!(interior(f_ice), 0.0)
    @inbounds for j in 2:5, i in 2:5
        interior(f_ice)[i, j, 1] = 1.0
    end

    calc_distance_to_ice_margin!(dist_mrgn, f_ice, 100.0)
    D = interior(dist_mrgn)

    # Ice perimeter is the margin → dist = 0.
    @test D[2, 2, 1] == 0.0
    @test D[2, 5, 1] == 0.0
    @test D[5, 2, 1] == 0.0
    @test D[5, 5, 1] == 0.0

    # Interior of the ice block — at distance dx from the margin.
    @test D[3, 3, 1] ≈ 100.0
    @test D[4, 4, 1] ≈ 100.0

    # Outside the ice block (ice-free) → negative distances.
    @test D[1, 3, 1] ≈ -100.0
    @test D[6, 3, 1] ≈ -100.0
end

@testset "tpo: calc_grounding_line_zone!" begin
    Nx = 6
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 60e3), y=(0.0, 60e3),
                        topology=(Bounded, Bounded, Flat))
    dist_gl  = CenterField(g)
    mask_grz = CenterField(g)

    # Threshold = 5 km = 5000 m. Use a column of distances spanning the
    # bin boundaries.
    interior(dist_gl)[1, 1, 1] = 0.0          # exactly on GL
    interior(dist_gl)[2, 1, 1] = 3000.0       # grounded in zone
    interior(dist_gl)[3, 1, 1] = 5000.0       # grounded at threshold (in)
    interior(dist_gl)[4, 1, 1] = 7000.0       # grounded out of zone
    interior(dist_gl)[1, 2, 1] = -3000.0      # floating in zone
    interior(dist_gl)[2, 2, 1] = -5000.0      # floating at threshold (in)
    interior(dist_gl)[3, 2, 1] = -7000.0      # floating out of zone

    calc_grounding_line_zone!(mask_grz, dist_gl, 5000.0)
    M = interior(mask_grz)

    @test M[1, 1, 1] == 0.0
    @test M[2, 1, 1] == 1.0
    @test M[3, 1, 1] == 1.0
    @test M[4, 1, 1] == 2.0
    @test M[1, 2, 1] == -1.0
    @test M[2, 2, 1] == -1.0
    @test M[3, 2, 1] == -2.0
end

@testset "tpo: gen_mask_bed!" begin
    Nx = 4
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 4.0), y=(0.0, 4.0),
                        topology=(Bounded, Bounded, Flat))
    mask_bed = CenterField(g)
    f_ice    = CenterField(g)
    f_pmp    = CenterField(g)
    f_grnd   = CenterField(g)
    mask_grz = CenterField(g)

    fill!(interior(f_ice),    0.0)
    fill!(interior(f_pmp),    0.0)
    fill!(interior(f_grnd),   0.0)
    fill!(interior(mask_grz), 2.0)   # default: grounded out of zone
    fill!(interior(mask_bed), -99.0) # sentinel to confirm overwrite

    # Layout (one cell per bed-mask category):
    #   (1,1): ice-free ocean  (f_ice=0, f_grnd=0)
    #   (2,1): ice-free land   (f_ice=0, f_grnd=1)
    #   (3,1): partial         (f_ice=0.5)
    #   (4,1): grounding line  (mask_grz=0, regardless of other flags)
    #   (1,2): floating ice    (f_ice=1, f_grnd=0)
    #   (2,2): grounded frozen (f_ice=1, f_grnd=1, f_pmp=0)
    #   (3,2): grounded stream (f_ice=1, f_grnd=1, f_pmp=0.8)
    interior(f_grnd)[2, 1, 1] = 1.0

    interior(f_ice)[3, 1, 1]  = 0.5

    interior(mask_grz)[4, 1, 1] = 0.0

    interior(f_ice)[1, 2, 1]  = 1.0

    interior(f_ice)[2, 2, 1]  = 1.0
    interior(f_grnd)[2, 2, 1] = 1.0

    interior(f_ice)[3, 2, 1]  = 1.0
    interior(f_grnd)[3, 2, 1] = 1.0
    interior(f_pmp)[3, 2, 1]  = 0.8

    gen_mask_bed!(mask_bed, f_ice, f_pmp, f_grnd, mask_grz)
    Mb = interior(mask_bed)

    @test Mb[1, 1, 1] == Float64(MASK_BED_OCEAN)
    @test Mb[2, 1, 1] == Float64(MASK_BED_LAND)
    @test Mb[3, 1, 1] == Float64(MASK_BED_PARTIAL)
    @test Mb[4, 1, 1] == Float64(MASK_BED_GRLINE)
    @test Mb[1, 2, 1] == Float64(MASK_BED_FLOAT)
    @test Mb[2, 2, 1] == Float64(MASK_BED_FROZEN)
    @test Mb[3, 2, 1] == Float64(MASK_BED_STREAM)

    # Grounding-line dominates other states.
    interior(mask_grz)[1, 2, 1] = 0.0
    gen_mask_bed!(mask_bed, f_ice, f_pmp, f_grnd, mask_grz)
    Mb = interior(mask_bed)
    @test Mb[1, 2, 1] == Float64(MASK_BED_GRLINE)
end

@testset "tpo: MASK_BED_* enum values" begin
    @test MASK_BED_OCEAN   == 0
    @test MASK_BED_LAND    == 1
    @test MASK_BED_FROZEN  == 2
    @test MASK_BED_STREAM  == 3
    @test MASK_BED_GRLINE  == 4
    @test MASK_BED_FLOAT   == 5
    @test MASK_BED_ISLAND  == 6
    @test MASK_BED_PARTIAL == 7
end

# ------------------------------------------------------------------
# Boundary plumbing — grid topology, BC palette, corner halos
# ------------------------------------------------------------------

using Yelmo.YelmoCore: resolve_boundaries, neumann_2d_field, dirichlet_2d_field,
                       fill_corner_halos!
using Oceananigans.BoundaryConditions: fill_halo_regions!

@testset "core: resolve_boundaries" begin
    @test resolve_boundaries(:bounded)    == (Bounded,  Bounded)
    @test resolve_boundaries(:periodic)   == (Periodic, Periodic)
    @test resolve_boundaries(:periodic_x) == (Periodic, Bounded)
    @test resolve_boundaries(:periodic_y) == (Bounded,  Periodic)

    @test resolve_boundaries((:periodic, :bounded))  == (Periodic, Bounded)
    @test resolve_boundaries((:bounded,  :periodic)) == (Bounded,  Periodic)

    # Direct Oceananigans tuples pass through unchanged.
    @test resolve_boundaries((Periodic, Bounded))           == (Periodic, Bounded)
    @test resolve_boundaries((Bounded, Periodic, Flat))     == (Bounded,  Periodic)

    @test_throws ErrorException resolve_boundaries(:bogus)
    @test_throws ErrorException resolve_boundaries((:bogus, :bounded))
end

@testset "core: neumann/dirichlet field BCs" begin
    Nx = 4
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 4.0), y=(0.0, 4.0),
                        topology=(Bounded, Bounded, Flat))

    fN = neumann_2d_field(g)
    fill!(interior(fN), 5.0)
    fill_halo_regions!(fN)
    # Neumann zero / clamp: halo = first interior.
    @test fN[0,    1, 1] ≈ 5.0
    @test fN[Nx+1, 1, 1] ≈ 5.0

    fD = dirichlet_2d_field(g, 0.0)
    fill!(interior(fD), 5.0)
    fill_halo_regions!(fD)
    # Dirichlet zero face value: halo = -first_interior so the face
    # mean is 0.
    @test fD[0,    1, 1] ≈ -5.0
    @test fD[Nx+1, 1, 1] ≈ -5.0
end

@testset "core: fill_corner_halos! — bounded clamps to nearest corner" begin
    Nx = 4
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 4.0), y=(0.0, 4.0),
                        topology=(Bounded, Bounded, Flat))
    f = neumann_2d_field(g)
    int = interior(f)
    fill!(int, 0.0)
    int[1,  1,  1] = 11.0   # SW
    int[Nx, 1,  1] = 12.0   # SE
    int[1,  Nx, 1] = 13.0   # NW
    int[Nx, Nx, 1] = 14.0   # NE

    fill_halo_regions!(f)
    fill_corner_halos!(f)

    @test f[0,    0,    1] == 11.0    # SW corner clamps to (1,1)
    @test f[Nx+1, 0,    1] == 12.0    # SE clamps to (Nx,1)
    @test f[0,    Nx+1, 1] == 13.0    # NW clamps to (1,Nx)
    @test f[Nx+1, Nx+1, 1] == 14.0    # NE clamps to (Nx,Nx)
end

@testset "core: fill_corner_halos! — periodic wraps to opposite corner" begin
    Nx = 4
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 4.0), y=(0.0, 4.0),
                        topology=(Periodic, Periodic, Flat))
    f = CenterField(g)
    int = interior(f)
    fill!(int, 0.0)
    int[1,  1,  1] = 11.0   # SW interior
    int[Nx, 1,  1] = 12.0   # SE
    int[1,  Nx, 1] = 13.0   # NW
    int[Nx, Nx, 1] = 14.0   # NE

    fill_halo_regions!(f)
    fill_corner_halos!(f)

    # SW corner halo (0, 0) wraps to (Nx, Nx).
    @test f[0,    0,    1] == 14.0
    # SE corner halo (Nx+1, 0) wraps to (1, Nx).
    @test f[Nx+1, 0,    1] == 13.0
    # NW corner halo (0, Nx+1) wraps to (Nx, 1).
    @test f[0,    Nx+1, 1] == 12.0
    # NE corner halo (Nx+1, Nx+1) wraps to (1, 1).
    @test f[Nx+1, Nx+1, 1] == 11.0
end

@testset "core: fill_corner_halos! — mixed periodic-x + bounded-y" begin
    Nx = 4
    g = RectilinearGrid(size=(Nx, Nx),
                        x=(0.0, 4.0), y=(0.0, 4.0),
                        topology=(Periodic, Bounded, Flat))
    f = CenterField(g)
    int = interior(f)
    fill!(int, 0.0)
    int[1,  1,  1] = 11.0
    int[Nx, 1,  1] = 12.0
    int[1,  Nx, 1] = 13.0
    int[Nx, Nx, 1] = 14.0

    fill_halo_regions!(f)
    fill_corner_halos!(f)

    # SW corner: x wraps (Nx → 0 halo), y clamps (1 → 0 halo) →
    # source = (Nx, 1) = 12.
    @test f[0,    0,    1] == 12.0
    # SE corner: x wraps (1 → Nx+1 halo), y clamps → source (1, 1) = 11.
    @test f[Nx+1, 0,    1] == 11.0
    # NW corner: x wraps, y clamps → source (Nx, Nx) = 14.
    @test f[0,    Nx+1, 1] == 14.0
    # NE corner: x wraps, y clamps → source (1, Nx) = 13.
    @test f[Nx+1, Nx+1, 1] == 13.0
end
