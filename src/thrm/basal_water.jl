# ----------------------------------------------------------------------
# Basal water layer evolution.
#
# Direct port of Fortran `calc_basal_water_local`
# (`physics/thermodynamics.f90:1537`). Single-cell mass balance for
# the basal water layer thickness `H_w`:
#
#     dH_w/dt = bmb_w - till_rate
#
# clamped to `[0, H_w_max]`, with three boundary-conditional overrides:
#
#   - Floating / ice-free ocean (`f_grnd == 0`)        → `H_w = H_w_max`.
#   - Grounded, fully iced, with floating neighbour
#     (any of N/S/E/W has `f_grnd == 0`)               → `H_w = H_w_max`.
#   - Grounded, partial / no ice (`f_ice < 1`)         → `H_w = 0`.
#   - Otherwise (grounded, fully iced, no floating
#     neighbour)                                       → integrate.
#
# Boundary handling is Fortran-style clamped (`im1 = max(i-1, 1)`).
#
# `dHwdt` is computed as a backward difference between the value at
# entry and the value at exit, divided by `dt`. With `dt == 0`,
# `dHwdt = 0`.
# ----------------------------------------------------------------------

"""
    calc_basal_water_local!(H_w_field, dHwdt_field, f_ice_field, f_grnd_field,
                            bmb_w_field, dt, till_rate, H_w_max) -> H_w_field

Update the basal water layer `H_w` (with rate-of-change `dHwdt`) per
the local single-cell mass balance described in the file header.
`bmb_w_field` carries the basal water mass balance (m/yr); the
caller is responsible for sign/conversion (Fortran passes
`-bmb_grnd * (rho_ice / rho_w)` so positive bmb_w means melt-fed
water gain).
"""
function calc_basal_water_local!(H_w_field, dHwdt_field,
                                 f_ice_field, f_grnd_field,
                                 bmb_w_field,
                                 dt::Real, till_rate::Real, H_w_max::Real)
    Hw_d   = H_w_field.data
    dH_d   = dHwdt_field.data
    fi_d   = f_ice_field.data
    fg_d   = f_grnd_field.data
    bw_d   = bmb_w_field.data
    Nx     = H_w_field.grid.Nx
    Ny     = H_w_field.grid.Ny
    return _calc_basal_water_local_kernel!(Hw_d, dH_d, fi_d, fg_d, bw_d,
                                           Float64(dt),
                                           Float64(till_rate),
                                           Float64(H_w_max),
                                           Nx, Ny)
end

function _calc_basal_water_local_kernel!(Hw, dH, fi, fg, bw,
                                         dt::Float64,
                                         till_rate::Float64,
                                         H_w_max::Float64,
                                         Nx::Int, Ny::Int)
    # Snapshot H_w at entry (for the dHwdt backward difference). Stored
    # in `dH` directly to avoid a second 2D buffer (Fortran uses the
    # same trick: `dHwdt = H_w` at the top).
    @inbounds for j in 1:Ny, i in 1:Nx
        dH[i, j, 1] = Hw[i, j, 1]
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        im1 = max(i - 1, 1)
        ip1 = min(i + 1, Nx)
        jm1 = max(j - 1, 1)
        jp1 = min(j + 1, Ny)

        fg_ij = fg[i, j, 1]
        fi_ij = fi[i, j, 1]

        if fg_ij == 0.0
            # Floating or ice-free ocean → saturate.
            Hw[i, j, 1] = H_w_max
        elseif fg_ij > 0.0 && fi_ij == 1.0 &&
               (fg[im1, j, 1] == 0.0 || fg[ip1, j, 1] == 0.0 ||
                fg[i, jm1, 1] == 0.0 || fg[i, jp1, 1] == 0.0)
            # Grounded but with a floating neighbour → saturate.
            Hw[i, j, 1] = H_w_max
        elseif fg_ij > 0.0 && fi_ij < 1.0
            # Grounded, partial / no ice → drain.
            Hw[i, j, 1] = 0.0
        else
            # Grounded, fully iced, away from floating → integrate.
            new_Hw = Hw[i, j, 1] + dt * (bw[i, j, 1] - till_rate)
            new_Hw = max(new_Hw, 0.0)
            new_Hw = min(new_Hw, H_w_max)
            Hw[i, j, 1] = new_Hw
        end

        # Convert dH (which currently holds the entry value) into the
        # rate of change. Fortran convention: dHwdt = (H_w_init - H_w_now) / dt
        # — note the sign (positive when H_w decreased over the step).
        dH[i, j, 1] = dt != 0.0 ? (dH[i, j, 1] - Hw[i, j, 1]) / dt : 0.0
    end
    return nothing
end
