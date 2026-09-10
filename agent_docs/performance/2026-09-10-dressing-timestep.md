# 2026-09-10 — Dressing time step: dt 1/30 x 3 against dt 1/60 x 6

- Status: Rejected as the pretraining default. dt 1/30 is 2.13 times faster, but on the merged placements it loses elbow-stage successes. dt 1/60 stays, and dt 1/40 x 4 is being probed.
- Code under test: `0b9ccfc9`, from a detached worktree. The `--dt` flag landed later in `fb38b326`.
- Benchmark manifest: none. These are expert-ceiling probes on the dressing environment.

## Question

A policy decision lasts 0.1 s. The environment spends it as six libuipc steps of 1/60 s.
The IPC solve is 98 % of an environment step, and fused PCG is 78 % of a frame.

Would three steps of 1/30 s do the same task for less solver work? The task must stay
the same: same expert ceilings and the same grip.

The held cuff is a soft position constraint whose physical stiffness is
strength x mass / dt^2. So the dt 1/30 variant runs strength 4e4 against 1e4, the same
spring, and half the settle steps, the same 0.5 s.

## Environment

| Field | Value |
|---|---|
| GPU / driver | RTX PRO 6000 Blackwell Workstation Edition, 595.84 |
| Software | torch 2.12.0+cu130, Genesis 1.1.2, Linux 7.0.0-30 |
| Build | libuipc wheel of this branch, Release |
| Worktree state | clean detached worktree at `0b9ccfc9` |
| GPU sharing | four agents and 11-18 other processes at 100 % utilisation, so wall time is not a primary signal |

## Workload and method

`expert_dt.py` (session scratchpad `perf/`) runs the scripted seven-stage expert on live
cells. The settings are 48 anchors and 300 decisions of 0.1 s. Each run reports, per
cell, the highest forearm ratio, the highest upper-arm ratio and the largest held-cuff
error over the episode.

Solver work is counted exactly. With linear-system graph mode 0, libuipc's host stage
timers see every scope. That gives Newton iterations (`Newton Iteration`) and PCG
iterations (`Apply Preconditioner`) per decision. Graph mode 0 launches the same kernels
as training's mode 2, so the ceilings are those of training. Ceiling-only runs use
mode 2.

The variants:

| Variant | dt | Steps per decision | Cuff strength | Settle steps |
|---|---|---|---|---|
| A (default) | 1/60 | 6 | 1e4 | 30 |
| B | 1/30 | 3 | 4e4 | 15 |

## Correctness and safety

- No simulation error in any run.
- Every snapshot hold error after the settle was 0.8-1.2 mm.
- The comparison is per cell: identical bodies, drapes and expert.

## Results

Solver work on tshirt_26, bodies 0-3, exact counts over 300 decisions:

| Metric | A | B | B / A |
|---|---:|---:|---:|
| Newton iterations per decision | 18.2 | 9.3 | 0.51 |
| PCG iterations per decision | 1051 | 457 | 0.43 |
| PCG iterations per Newton iteration | 57.7 | 49.1 | 0.85 |

The same counts per 50-decision window:

| Decisions | A Newton | A PCG | B Newton | B PCG |
|---|---:|---:|---:|---:|
| 0-49 | 15.0 | 614 | 8.0 | 213 |
| 50-99 | 19.3 | 1050 | 11.3 | 592 |
| 100-149 | 22.9 | 1199 | 14.5 | 603 |
| 150-199 | 20.4 | 1163 | 13.3 | 801 |
| 200-249 | 22.3 | 1708 | 5.0 | 313 |
| 250-299 | 9.3 | 572 | 3.6 | 221 |

Expert ceilings, same cells:

| Cells | A forearm | A upper >= 0.7 | B forearm | B upper >= 0.7 | Hold error median, A / B |
|---|---|---|---|---|---|
| tshirt_26 bodies 0-3 | 4/4 | 3/4 | 4/4 | 3/4 | 14 / 18 mm |
| tshirt_26 bodies 4-7 | 4/4 | 2/4 | 4/4 | 3/4 | 24 / 30 mm |
| tshirt_68 bodies 0-7 | 5/8 | 3/8 | 8/8 | 6/8 | 37 / 42 mm |
| All 16 cells | 13/16 | 8/16 | 16/16 | 12/16 | |

In detail:
- A's failures reproduce the dressing-correctness record exactly: tshirt_68 bodies 1, 4 and 7 never reach the forearm.
- B reaches the forearm on all three, and the upper arm on all three.
- On tshirt_26 body 6, A hooks at the elbow and B dresses.
- No cell dresses under A and fails under B.
- On cells that dress, B's hold error is larger: 6.7-11 mm against 3.2-9.5 mm on tshirt_26 bodies 0, 2 and 3.

Mechanism probes on tshirt_68 bodies 1, 4 and 7, the cells A fails:

| Variant | Forearm | Upper >= 0.7 |
|---|---|---|
| A | 0/3 | 0/3 |
| A, `contact/eps_velocity` 0.02 (B's friction transition displacement) | 0/3 at decision 100, then stopped | 0/3 |
| A, 2 s settle (120 steps) | 1/3 | 1/3 |
| B | 3/3 | 3/3 |
| B, `contact/eps_velocity` 0.005 (A's friction transition displacement) | 3/3 | 1/3 |
| B, 2 s settle (60 steps) | 3/3 | 1/3; one cell lost the grip late, 315 mm |

### Production wall clock, 16 cells

The production-mode pair used graph mode 2 and no timers. Each variant ran one world of
tshirt_26 and tshirt_68 on bodies 0-7, 16 cells. Both built first, then started their
first decision at the same second, so they shared one contention window until
dt 1/30 finished.

| Window | A s per decision | B s per decision | A / B |
|---|---:|---:|---:|
| Decisions 0-137, both running | 5.20 | 2.45 | 2.13 |
| Whole episode; A ran its last 162 decisions alone | 3.82 | 2.44 | 1.56, a lower bound |

The ceilings in that shared world:

| Variant | Forearm reached | Upper arm >= 0.7 | Hold error median |
|---|---|---|---|
| A | 14/16 | 9/16 | 31 mm |
| B | 16/16 | 8/16 | 35 mm |

### The swing hypothesis, tested

The garment agent re-rolled tshirt_68 to its baked hang (commit `30441b28`). That cuts its
settle displacement to 0.258 m. It also brings bodies 1, 4 and 7 onto the forearm at
dt 1/60, giving forearm 8/8 and upper arm >= 0.7 on 5/8 over bodies 0-7.

So the start-state swing accounts for most of B's ceiling advantage on tshirt_68. The
speed gain does not depend on it.

### Re-validation on the merged placements (`06b49176`)

The merged placements put the gown and tshirt_68 in their baked hang, and tshirt_4 and
tshirt_392 sleeve outward. Each garment ran on bodies 0-7 in its own world, with dt 1/60
twins on the same commit.

| Garment | dt 1/60 forearm | dt 1/60 upper >= 0.7 | dt 1/30 forearm | dt 1/30 upper >= 0.7 |
|---|---|---|---|---|
| tshirt_26 | 8/8 | 5/8 | 8/8 | 5/8 |
| tshirt_68, baked hang | 8/8 | 5/8 | 8/8 | 3/8 |
| hospital_gown, baked hang | 7/8 | 5/8 | 8/8 | 2/8 |
| tshirt_392, sleeve outward | 8/8 in the garment agent's run | 0/8 | 8/8 | 0/8 |

- The dt 1/60 twins reproduce the garment agents' figures exactly: tshirt_68 on bodies 0, 2, 3, 5 and 7, the gown on 1, 2, 4, 5 and 7, and tshirt_4 at 8/8 and 3/8.
- Under dt 1/30, tshirt_68 bodies 3 and 5 and gown bodies 1, 2 and 7 hook at the elbow at an upper-arm ratio of 0.21-0.30.

The no-move rule's granularity is ruled out. The rule is checked once per physics step, so
at dt 1/30 each check covers twice the tool motion. A prototype that checks the tool path in
1/60 s increments whatever the step left both results unchanged: tshirt_68 at 3/8 and the
gown at 2/8. The patch is archived in the session scratchpad as
`perf/no_move_granularity.patch`.

## Interpretation

Measured directly:
- B halves the Newton iterations per decision and cuts PCG iterations 2.3 times, over a whole episode.
- The two differ in Newton and PCG counts, so this is an algorithmic-path change, not a throughput gain on the same path.
- Reaching the forearm is robustly better under B. Over three variants each on the cells A fails, B reaches it 9/9 and A 1/9.

Ruled out:
- IPC's friction smoothing does not explain it. Swapping `eps_velocity` so that each variant gets the other's transition displacement leaves the forearm result unchanged in both directions.
- A longer settle rescues one of the three cells under A.

Inferred, not tested:
- A's failing cells lose the grip early: 40-49 mm by decision 100, against 12-21 mm under B.
- The gown record in the dressing-correctness record found that the live placement's socket roll can start a garment far from its hang, still swinging when the expert starts.
- Implicit Euler at the larger step damps that swing sooner, which would explain both the forearm gain and the partial effect of a longer settle.

The elbow stage is sensitive to initial conditions under both time steps. Across B's
variants, upper >= 0.7 on these cells ranges from 1/3 to 3/3.

Wall time is indicative only. The paired ceiling runs gave 1.98 against 5.01 s per
decision for B and A on four cells, but they overlapped only partly, under different
contention.

## Decision

dt 1/30 is rejected as the pretraining default, and the trainer and `pretrain_wang` stay at
dt 1/60.

The baked hang removed the start-state swing that had flattered dt 1/30. What remains of
its effect is at the elbow, where it costs 5 of 16 successes on tshirt_68 and the gown.
The mechanism is not established. Candidates are implicit Euler's larger numerical damping
and friction at the larger step.

`--dt` stays available, and `pretrain_wang --dt 1/30` still fills in the matched cuff
strength and settle, for runs that trade elbow fidelity for the 2.1 times speed.

dt 1/40 x 4 (cuff strength 2.25e4, settle 20) is being probed on tshirt_68 and the gown.

## Reproduction and artifacts

From the session scratchpad `perf/`, with `PYTHONPATH=<worktree>/python`. Results are
written to `expert_dt_<tag>.json` and summarised by `compare_dt.py <tags>`.

```bash
python expert_dt.py tshirt_26 --bodies 0,1,2,3 --tag dt60                      # A, graph 0, counts
python expert_dt.py tshirt_26 --bodies 0,1,2,3 --dt 0.03333333333333333 --repeat 3 \
    --settle-steps 15 --strength 4e4 --tag dt30s4                              # B, graph 0, counts
python expert_dt.py tshirt_68 --bodies 0,1,2,3,4,5,6,7 --graph 2 --tag dt60_t68
python expert_dt.py tshirt_68 --bodies 0,1,2,3,4,5,6,7 --dt 0.03333333333333333 --repeat 3 \
    --settle-steps 15 --strength 4e4 --graph 2 --tag dt30s4_t68
python expert_dt.py tshirt_68 --bodies 1,4,7 --graph 2 --eps-velocity 0.02 --tag dt60e2_t68
python expert_dt.py tshirt_68 --bodies 1,4,7 --graph 2 --settle-steps 120 --tag dt60s120_t68
T0=$(( $(date +%s) + 420 ))   # production pair: both start their first decision together
python expert_dt.py tshirt_26,tshirt_68 --bodies 0,1,2,3,4,5,6,7 --graph 2 --start-at $T0 --tag wall_dt60_n16 &
python expert_dt.py tshirt_26,tshirt_68 --bodies 0,1,2,3,4,5,6,7 --dt 0.03333333333333333 --repeat 3 \
    --settle-steps 15 --strength 4e4 --graph 2 --start-at $T0 --tag wall_dt30_n16 &
```
