# ----------------------------------------------------------------------
# Topology-dispatched index helpers for face-staggered writes / reads.
#
# Background — topology and face-field shapes:
#
#   Under `RectilinearGrid` topology `Bounded` for an axis with N cells,
#   the corresponding Face dimension has N+1 nodes (including both
#   boundary endpoints). Under `Periodic` for the same axis, the Face
#   dimension has N nodes (the `N+1`-th face is the `1`-st by wrap and
#   is not stored). So:
#
#     - `XFaceField(g)` interior is `(Nx+1, Ny, 1)` under
#       `(Bounded,  Bounded,  Flat)`,
#       but `(Nx,   Ny, 1)` under `(Periodic, Bounded, Flat)`.
#     - `YFaceField(g)` interior is `(Nx, Ny+1, 1)` under
#       `(Bounded,  Bounded,  Flat)`,
#       but `(Nx, Ny,   1)` under `(Bounded, Periodic, Flat)`.
#
# The dyn kernels in this module follow the convention that the
# x-face value associated with cell `(i, j)` (i.e. the eastern face of
# the cell) lives at array index `[i+1, j, 1]` of an `XFaceField` —
# correct under `Bounded`, where slot `i+1` ∈ 2..Nx+1 is in bounds.
# Under `Periodic` the `i+1` write at `i = Nx` is `Nx+1`, which is out
# of bounds. The same applies to y-face writes at `[i, j+1, 1]` under
# `Periodic` y.
#
# Option (a-1) — implemented here — is to keep the existing eastern-/
# northern-face write convention but route the +1 through topology-
# aware helpers that wrap modularly under `Periodic` and behave as `+1`
# under `Bounded`. This is a minimally invasive change: only the write
# index expressions need updating; reads at `[i+1, j, 1]` etc. continue
# to use the field-indexed (halo-aware) path which already wraps under
# Periodic via `fill_halo_regions!`.
#
# TODO: Option (a-2) — south-face write convention — would replace these
# helpers with uniform `[i, j, 1]` writes (writing to the *western* /
# *southern* face slot rather than the eastern / northern), which is
# the direction Oceananigans itself uses for face-aligned operators.
# That is a bigger refactor across all dyn kernels and is deferred to a
# future cleanup.
#
# The dispatch is on the `AbstractTopology` SUBTYPE (`Type{Bounded}` /
# `Type{Periodic}`) — not the singleton instance. Standard Oceananigans
# pattern: extract via `topology(grid)`, which returns a 3-tuple of
# subtypes; splat as `(Tx, Ty, Tz) = topology(grid)` and pass `Tx` /
# `Ty` to the kernel as `Type{<:AbstractTopology}` parameters.
# ----------------------------------------------------------------------

using Oceananigans.Grids: Bounded, Periodic, AbstractTopology

"""
    _ip1_modular(i, Nx, topo_x)

Return the canonical interior index for "the X-face one cell east of
cell `i`" given the x-axis topology subtype `topo_x`:

  - Under `Bounded` this is `i + 1` (writing to the eastern face slot,
    range 2..Nx+1; the leftmost slot `1` is reserved for boundary
    handling and conventionally replicated from slot `2`).
  - Under `Periodic` this is `mod1(i + 1, Nx)` (writing to the eastern
    face slot, which under periodic-x wraps to slot `1` at `i = Nx`).
"""
@inline _ip1_modular(i::Int, Nx::Int, ::Type{Bounded})  = i + 1
@inline _ip1_modular(i::Int, Nx::Int, ::Type{Periodic}) = mod1(i + 1, Nx)

"""
    _jp1_modular(j, Ny, topo_y)

Y-axis analogue of `_ip1_modular`: returns the canonical interior index
for "the Y-face one cell north of cell `j`".

  - Under `Bounded`:  `j + 1`.
  - Under `Periodic`: `mod1(j + 1, Ny)`.
"""
@inline _jp1_modular(j::Int, Ny::Int, ::Type{Bounded})  = j + 1
@inline _jp1_modular(j::Int, Ny::Int, ::Type{Periodic}) = mod1(j + 1, Ny)

# ----------------------------------------------------------------------
# Topology-aware "previous neighbour" helpers for plain-Matrix scratch
# buffers (no halo). Used by SIA `fact_ab` and any future kernel that
# allocates a local Matrix and needs to read a wrapped/clamped neighbour
# at `(i-1, j)` / `(i, j-1)`.
#
# `Bounded`  → clamp to 1.
# `Periodic` → wrap modularly.
# ----------------------------------------------------------------------
@inline _neighbor_jm1(j::Int, Ny::Int, ::Type{Bounded})  = max(j - 1, 1)
@inline _neighbor_jm1(j::Int, Ny::Int, ::Type{Periodic}) = j == 1 ? Ny : j - 1
@inline _neighbor_im1(i::Int, Nx::Int, ::Type{Bounded})  = max(i - 1, 1)
@inline _neighbor_im1(i::Int, Nx::Int, ::Type{Periodic}) = i == 1 ? Nx : i - 1

# Mirror "next neighbour" helpers. `Bounded` clamps to N; `Periodic` wraps
# modularly (i = Nx → 1, j = Ny → 1).
@inline _neighbor_ip1(i::Int, Nx::Int, ::Type{Bounded})  = min(i + 1, Nx)
@inline _neighbor_ip1(i::Int, Nx::Int, ::Type{Periodic}) = i == Nx ? 1 : i + 1
@inline _neighbor_jp1(j::Int, Ny::Int, ::Type{Bounded})  = min(j + 1, Ny)
@inline _neighbor_jp1(j::Int, Ny::Int, ::Type{Periodic}) = j == Ny ? 1 : j + 1
