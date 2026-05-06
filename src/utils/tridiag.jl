# ----------------------------------------------------------------------
# Tridiagonal-matrix solver (Thomas algorithm).
#
# Direct port of Fortran `solver_tridiagonal.f90:solve_tridiag`. Used
# by every column-wise thermodynamics solver (temp, enthalpy, bedrock,
# and eventually tracers).
#
# The kernel takes caller-supplied scratch buffers so the hot loop in
# a 3D column solver stays alloc-free; an overload that allocates
# internal scratch is provided for non-perf-critical callers (one-off
# solves, tests).
# ----------------------------------------------------------------------

"""
    solve_tridiag!(x, a, b, c, d, cp, dp) -> x

Solve a tridiagonal linear system M·x = d by the Thomas algorithm,
with caller-supplied scratch buffers `cp`, `dp` (length `length(x)`).

Diagonals (1-based, i=1..n):

  - `a[i]` — sub-diagonal (a[1] is unused)
  - `b[i]` — main diagonal
  - `c[i]` — super-diagonal (c[n] is unused)
  - `d[i]` — right-hand side
  - `x[i]` — solution (output)

Direct port of Fortran `solver_tridiagonal.f90:solve_tridiag`. The
kernel is alloc-free; pass `cp`, `dp` from a thrm column-solver scratch
struct so a 3D loop over (i,j) reuses one pair of buffers.
"""
function solve_tridiag!(x::AbstractVector{Float64},
                        a::AbstractVector{Float64},
                        b::AbstractVector{Float64},
                        c::AbstractVector{Float64},
                        d::AbstractVector{Float64},
                        cp::AbstractVector{Float64},
                        dp::AbstractVector{Float64})
    n = length(x)
    @inbounds begin
        cp[1] = c[1] / b[1]
        dp[1] = d[1] / b[1]
        for i in 2:n
            m     = b[i] - cp[i-1] * a[i]
            cp[i] = c[i] / m
            dp[i] = (d[i] - dp[i-1] * a[i]) / m
        end
        x[n] = dp[n]
        for i in (n-1):-1:1
            x[i] = dp[i] - cp[i] * x[i+1]
        end
    end
    return x
end

"""
    solve_tridiag!(x, a, b, c, d) -> x

Convenience overload that allocates internal scratch buffers. Use the
seven-arg form above on the hot path to stay alloc-free.
"""
function solve_tridiag!(x::AbstractVector{Float64},
                        a::AbstractVector{Float64},
                        b::AbstractVector{Float64},
                        c::AbstractVector{Float64},
                        d::AbstractVector{Float64})
    n  = length(x)
    cp = Vector{Float64}(undef, n)
    dp = Vector{Float64}(undef, n)
    return solve_tridiag!(x, a, b, c, d, cp, dp)
end
