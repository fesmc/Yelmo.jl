# MISMIP3D SSA regression — investigation handoff

## Status

**Three failing tests** under `test/benchmarks/`, all symptoms of the
same underlying SSA-side regression. Pre-existing on `main` since
some commit landed between 2026-05-04 and 2026-06-08; **not**
caused by the IceSheetBenchmarks extraction (#83, #84) — confirmed
reproduces on plain `main` with identical numerical values.

Failing tests:

- `test_mismip3d_stnd_implicit.jl` — `rel_err = 0.89` (expected `< 0.05`)
- `test_mismip3d_stnd_att.jl` — 8 / 16 assertions fail
- `test_mismip3d_stnd_lockstep.jl` — 3 / 6 assertions fail (against
  the Mirror fixture at `t = 500`)

## Symptom

The SSA solver produces **runaway velocities** that hit the
`ssa_vel_max = 5000 m/yr` ceiling, vs Mirror's realistic ~700 m/yr.
The velocity saturation propagates downstream:

| Metric                              | Yelmo.jl       | Mirror ref |
| ----------------------------------- | -------------- | ---------- |
| `max\|ux\|` lockstep, t = 500       | **5000** (clamped) | 779    |
| `max\|ux\|` ATT, t = 1500           | **5000** (clamped) | 692    |
| `mean(f_grnd)` lockstep             | 0.98 (over-grounded) | 0.49 (half-shelf) |
| `mean(f_grnd)` ATT, t = 1500        | 0.98           | 0.51       |
| `n_shelf` ATT, all phases           | **0**          | (> 0 expected) |
| `gl_idx` ATT, all phases            | 50 (= max, pinned to eastern boundary) | (interior, advances/retreats with A) |

Because velocities saturate, no shelf forms — the entire domain
grounds up to the eastern column. The Phase-0 / 1 / 2 GL-advance and
GL-retreat assertions in the ATT test fail because the GL has nowhere
to advance or retreat from.

The `test_mismip3d_stnd_implicit.jl` `rel_err = 0.89` is downstream
of the same saturation: at the cap, the implicit and explicit
advection schemes propagate the ill-posed dynamics differently. Fix
the SSA regression first; the implicit-vs-explicit agreement should
fall back into line.

## Baseline (passing) reference

When `test_mismip3d_stnd_implicit.jl` was first added in commit
`096c7c9` (2026-05-04), it passed with observed agreement
**`max\|ΔH\| / max(H) = 0.4%`, `mean\|ΔH\| / max(H) = 0.04%`**
(commit message). So the regression came in *after* `096c7c9`.

## Suspect commit range

From `git log 096c7c9..HEAD --no-merges -- src/dyn src/topo`
(20+ commits across SSA / dyn / topo). The likeliest culprits, in
rough order of plausibility:

| SHA       | Date       | Description                                                                   |
| --------- | ---------- | ----------------------------------------------------------------------------- |
| `1352824` | 2026-05-08 | topo/dyn: unify `f_grnd_ac{x,y}` on Oceananigans i-1/2 face convention        |
| `0808bb6` | 2026-05-06 | topo: PR6 mask_ice Heun-corrector fix + pre-existing test cleanup             |
| `433b6e5` | (later)    | ssa: split `SSASolver.method` into formulation + linear_method                |
| `ec673e7` | 2026-05-27 | Consolidate ice boundary control onto `bnd.mask_ice`                          |
| `ce0b107` | 2026-05-10 | initmip-grl benchmark: real-domain Greenland sandbox + DIVA viscosity fix (#73) |
| `d64e842` | 2026-06-03 | topo: branchless + @turbo on five hot per-cell kernels                        |

Don't trust the order — bisect.

## Plan for next session

### Step 1 — bisect on `src/`

Use a dedicated bisect worktree so the test files at `HEAD` stay
fixed while `src/` cycles through candidates. ~6 iterations × ~80 s
per test = 10–15 min wall-clock.

Skeleton:

```bash
git worktree add .claude/worktrees/mismip-bisect HEAD
cd .claude/worktrees/mismip-bisect

# Save HEAD test files (we'll keep these constant across all bisect steps).
cp test/benchmarks/test_mismip3d_stnd_implicit.jl /tmp/test_impl_HEAD.jl
cp test/benchmarks/harness.jl                       /tmp/harness_HEAD.jl
cp -r test/benchmarks/yelmo                          /tmp/yelmo_glue_HEAD

# Bisect range. 096c7c9 was passing (commit-message confirms). HEAD = bad.
git bisect start --no-checkout HEAD 096c7c9

# Run with a helper that:
#   1. checks out the candidate src/ over the worktree,
#   2. restores HEAD test files,
#   3. instantiates,
#   4. runs the impl test,
#   5. exits 0 if rel_err < 0.05, 1 otherwise, 125 if can't even build.
git bisect run bash scripts/bisect_mismip_impl.sh    # author this script
```

Sketch of `scripts/bisect_mismip_impl.sh` (template; tighten as needed):

```bash
#!/usr/bin/env bash
set -euo pipefail
SHA=$(git rev-parse BISECT_HEAD)

# Reset src/ to candidate, keep test/ at HEAD.
git checkout HEAD -- src       # baseline (the worktree's HEAD == bisect-start HEAD)
git checkout "$SHA" -- src

# Restore HEAD test files (saved out of band).
cp /tmp/test_impl_HEAD.jl test/benchmarks/test_mismip3d_stnd_implicit.jl
cp /tmp/harness_HEAD.jl   test/benchmarks/harness.jl
cp -r /tmp/yelmo_glue_HEAD/* test/benchmarks/yelmo/

cd test
rm -f Manifest.toml
julia --project=. -e 'import Pkg; Pkg.instantiate()' > /tmp/inst.log 2>&1 \
    || { echo "$SHA: build fail (skip)"; exit 125; }

LOG=/tmp/bisect_$SHA.log
julia --project=. benchmarks/test_mismip3d_stnd_implicit.jl > "$LOG" 2>&1 || true

rel=$(grep -oE 'max\|ΔH\|/max\(H\)=[0-9.e+-]+' "$LOG" | head -1 | sed 's/.*=//')
[ -z "$rel" ] && { echo "$SHA: no rel_err line (skip)"; exit 125; }
echo "$SHA: rel_err = $rel"
awk -v r="$rel" 'BEGIN{exit !(r < 0.05)}'
```

Caveats:

- Older src/ may not work with HEAD's test files (public-API drift,
  missing fields). When that happens, `exit 125` (skip) so bisect
  steps over. If too many candidates skip, narrow the bisect range
  to commits-touching-physics-only via
  `git bisect skip $(git rev-list --first-parent --grep='doc\|README\|test:' HEAD..main)`.
- The `IceSheetBenchmarks` path-dep was removed in #84. For commits
  *before* #84, the harness expects the path-dep — easiest fix is to
  temporarily restore `benchmarks/IceSheetBenchmarks/` in the bisect
  worktree (copy from a checkout of #83's merge commit). Or just
  start the bisect *after* #84 if the regression is recent enough
  — sanity-check by running the test at `e495212` (just before #84)
  and at `dc7e39c` (HEAD); if both fail equally, the regression
  pre-dates the ISB-extraction work, and the bisect range can stay
  pre-#83.

### Step 2 — once the breaking commit is identified

Read the diff. Three likely outcomes:

1. **Direct bug** — a copy-paste / stagger / sign error in the
   commit. Patch the bug, the three tests should go green together.
2. **Latent bug exposed** — the commit didn't introduce the bug but
   exposed it (e.g. tightened a tolerance, changed dispatch). Trace
   back to the actual cause.
3. **Test was always borderline** — agreement was 0.4% at the time
   of write but drifted under accumulated refactors; no single
   commit is the culprit. Investigate by comparing the SSA solution
   field-by-field against a Mirror fixture at `t = 1` (one step).

### Step 3 — also check during the bisect

Are any of these *also* broken on `main` right now? Run them via
`julia --project=test test/benchmarks/test_yelmo_ssa_solver.jl` and
the slab / HOM-C tests. If yes, that points at a shared SSA bug; if
no, it's MISMIP3D-specific (probably the Bounded-x / calving-column
boundary handling).

### Step 4 — once fixed

Update this file with the root cause + fix commit + delete it (or
keep as a postmortem under `docs/regressions/`).

## Useful starting commands

```bash
# Quick repro on main (no worktree, no instantiation overhead if
# test/Manifest.toml is already current):
cd /Users/alrobi001/models/Yelmo.jl/test
julia --project=. benchmarks/test_mismip3d_stnd_implicit.jl
# expect: rel_err = 0.8901, mean_err = 0.5395 (failing)

# All three failures in one go:
for f in test_mismip3d_stnd_{implicit,att,lockstep}.jl; do
    julia --project=. benchmarks/$f.jl 2>&1 | tee /tmp/$f.log
done
```

## Cross-reference

- Mirror fixtures: `test/benchmarks/fixtures/mismip3d_stnd_{t0,t500}.nc`,
  `mismip3d_stnd_att_t1500.nc`. Regenerate with
  `julia --project=test test/benchmarks/regenerate.jl mismip3d_stnd --overwrite`
  (requires `libyelmo_c_api.so`).
- Test-side Yelmo glue: `test/benchmarks/yelmo/mismip3d.jl`.
- ISB spec: `IceSheetBenchmarks.jl/src/mismip3d.jl` (standalone repo,
  not in-tree).
- SSA solver: `src/dyn/velocity_ssa.jl`, `src/dyn/_assemble_ssa_matrix.jl`.
- Velocity cap: `src/YelmoPar.jl:297` (`ssa_vel_max = 5000.0`).
