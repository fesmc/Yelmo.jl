# ----------------------------------------------------------------------
# Basal mass balance combiner.
#
# `calc_bmb_total!` fuses a grounded-melt field (`bmb_grnd`, from the
# thermodynamics solver) and a sub-shelf melt field (`bmb_shlf`, from
# the boundary forcing) into a single per-cell basal mass balance, with
# the grounding-line treatment selected by `bmb_gl_method`.
#
# Port of `physics/topography.f90:calc_bmb_total` (line 1555).
# ----------------------------------------------------------------------

using Oceananigans.Fields: interior

export calc_bmb_total!

"""
    calc_bmb_total!(bmb, bmb_grnd, bmb_shlf, H_ice, H_grnd, f_grnd,
                    bmb_gl_method) -> bmb

Combine the grounded (`bmb_grnd`) and floating (`bmb_shlf`) basal mass
balance fields into a single output `bmb`. The grounding-line treatment
is selected by `bmb_gl_method`:

  - `"fcmp"` (flotation criterion): `bmb_shlf` where `H_grnd ≤ 0`,
    else `bmb_grnd`.
  - `"fmp"` (full melt): `bmb_shlf` where `f_grnd < 1`, else `bmb_grnd`.
  - `"pmp"` (partial melt): linear blend by `f_grnd` for `f_grnd < 1`,
    else `bmb_grnd`.
  - `"nmp"` (no melt): `bmb_shlf` only where `f_grnd == 0`, else
    `bmb_grnd`.
  - `"pmpt"` (partial melt with subgrid tidal zone): not yet ported —
    raises an error.

After the switch, cells with positive `H_grnd` (grounded) and zero
`H_ice` are forced to `bmb = 0` (no melt on bare grounded land).

Notes vs the Fortran:
  - The Fortran `pmpt` branch calls `calc_bmb_gl_pmpt`, which itself
    requires `calc_subgrid_array`. Both are deferred to a follow-up.
  - The `gz_Hg0`/`gz_Hg1`/`gz_nx` parameters and the `boundaries`
    code drive only the `pmpt` branch in Fortran, so they don't appear
    in this v1 signature.
"""
function calc_bmb_total!(bmb, bmb_grnd, bmb_shlf,
                         H_ice, H_grnd, f_grnd,
                         bmb_gl_method::AbstractString)
    B   = interior(bmb)
    Bg  = interior(bmb_grnd)
    Bs  = interior(bmb_shlf)
    H   = interior(H_ice)
    Hg  = interior(H_grnd)
    Fg  = interior(f_grnd)

    if bmb_gl_method == "fcmp"
        @inbounds for j in axes(B, 2), i in axes(B, 1)
            B[i, j, 1] = Hg[i, j, 1] <= 0.0 ? Bs[i, j, 1] : Bg[i, j, 1]
        end

    elseif bmb_gl_method == "fmp"
        @inbounds for j in axes(B, 2), i in axes(B, 1)
            B[i, j, 1] = Fg[i, j, 1] < 1.0 ? Bs[i, j, 1] : Bg[i, j, 1]
        end

    elseif bmb_gl_method == "pmp"
        @inbounds for j in axes(B, 2), i in axes(B, 1)
            fg = Fg[i, j, 1]
            if fg < 1.0
                B[i, j, 1] = fg * Bg[i, j, 1] + (1.0 - fg) * Bs[i, j, 1]
            else
                B[i, j, 1] = Bg[i, j, 1]
            end
        end

    elseif bmb_gl_method == "nmp"
        @inbounds for j in axes(B, 2), i in axes(B, 1)
            B[i, j, 1] = Fg[i, j, 1] == 0.0 ? Bs[i, j, 1] : Bg[i, j, 1]
        end

    elseif bmb_gl_method == "pmpt"
        error("calc_bmb_total!: bmb_gl_method=\"pmpt\" not yet ported " *
              "(requires calc_bmb_gl_pmpt + calc_subgrid_array). " *
              "Use \"pmp\" for the closest analytical alternative.")

    else
        error("calc_bmb_total!: unknown bmb_gl_method=\"$bmb_gl_method\". " *
              "Supported: \"fcmp\", \"fmp\", \"pmp\", \"nmp\".")
    end

    # No melt on bare grounded land.
    @inbounds for j in axes(B, 2), i in axes(B, 1)
        if Hg[i, j, 1] > 0.0 && H[i, j, 1] == 0.0
            B[i, j, 1] = 0.0
        end
    end

    return bmb
end
