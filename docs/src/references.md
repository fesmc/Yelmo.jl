# References

Selected papers cited in the physics pages.

## Subgrid grounded fraction

- **Leguy, G. R., Lipscomb, W. H., & Asay-Davis, X. S. (2021).**
  *Marine ice sheet experiments with the Community Ice Sheet Model.*
  The Cryosphere, 15(7), 3229–3253.
  doi:[10.5194/tc-15-3229-2021](https://doi.org/10.5194/tc-15-3229-2021).
  Source of the CISM bilinear-interpolation scheme used in
  [`determine_grounded_fractions!`](@ref). The Yelmo Fortran port is
  via IMAU-ICE v2.0; the Julia port is in
  [`src/topo/grounded.jl`](https://github.com/fesmc/Yelmo.jl/blob/main/src/topo/grounded.jl)
  and described in detail on the
  [grounded fraction page](physics/grounded-fraction.md).

## Calving — front-velocity laws

- **Morlighem, M., Bondzio, J., Seroussi, H., Rignot, E.,
  Larour, E., Humbert, A., & Rebuffi, S. (2016).**
  *Modeling of Store Gletscher's calving dynamics, West Greenland,
  in response to ocean thermal forcing.*
  Geophysical Research Letters, 43(6), 2659–2666.
  doi:[10.1002/2016GL067695](https://doi.org/10.1002/2016GL067695).
  von-Mises calving law referenced as `vm-m16` in
  `ycalv.calv_flt_method`. See the
  [calving page](physics/calving.md). (Stub in Yelmo.jl — calling
  the law errors at step time, pending a port of the principal-stress
  diagnostic from `mat`.)

## Discharge mass balance

- **Calov, R., Beyer, S., Greve, R., Beckmann, J., Willeit, M.,
  Kleiner, T., Rückamp, M., Humbert, A., & Ganopolski, A. (2018).**
  *Simulation of the future sea level contribution of Greenland with a
  new glacial system model.*
  The Cryosphere, 12(10), 3097–3121.
  doi:[10.5194/tc-12-3097-2018](https://doi.org/10.5194/tc-12-3097-2018).
  The subgrid-discharge parameterisation referred to as "Calov+
  2018" in [`calc_mb_discharge!`](@ref). Not yet ported in Yelmo.jl
  (the current port stubs `dmb_method = 0` only — see the
  [mass balance page](physics/mass-balance.md)).

## Level-set redistancing

- **Sussman, M., Smereka, P., & Osher, S. (1994).**
  *A level set approach for computing solutions to incompressible
  two-phase flow.*
  Journal of Computational Physics, 114(1), 146–159.
  doi:[10.1006/jcph.1994.1155](https://doi.org/10.1006/jcph.1994.1155).
  Source of the Hamilton-Jacobi redistancing PDE used in
  [`lsf_redistance!`](@ref). The implementation replaces Fortran's
  periodic `±1` re-flag with a real signed-distance restoration —
  see the [calving page](physics/calving.md).

## Yelmo Fortran reference

The pure-Julia [`YelmoModel`](@ref) is an in-progress port of the
Fortran Yelmo ice-sheet model. Each ported kernel cites its Fortran
reference (file + line). The upstream tree lives at:

- [Yelmo Fortran source](https://github.com/fesmc/yelmo) (private
  repository at the time of writing — contact the maintainers for
  access).

The port is structured to make the cross-reference easy: a Julia
function `foo!` with a citation to `yelmo/src/physics/X.f90:N` is a
faithful port of the corresponding Fortran subroutine, with any
deliberate divergences flagged in the docstring or in the relevant
physics page on this site.
