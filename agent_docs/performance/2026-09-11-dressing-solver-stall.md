# The region-13 teacher stall and the policy's slow tail

Status: bounded by a Newton cap and a decision watchdog. The slow tail under the learned policy was the
hold spring, now bounded by a 6 cm tether on every held vertex. Relaunch 3 is running.

## What happened

The first Wang teacher ran on region 13: 24 environments, `wang_static_arm`, and six 1/60 s steps
per decision, on `638e63f6`. It started at 01:45 on 2026-09-11 and was stopped by hand at 05:03.
The table gives environment seconds per vector step, over 20-step windows of `train_log.csv`:

| Transitions | Episode | Environment s per vector step |
|---|---:|---:|
| 960 to 6,720 | 1 | 3.6 to 6.7 |
| 7,200 to 10,560 | 2 | 4.6 to 7.4 |
| 11,040 to 12,480 | 2 | 6.7 to 14.6 |
| 12,960 | 2 | 21.1 |
| 13,440 | 2 | 33.1 |
| 13,920 | 2 | 130.2 |

- After the 13,920-transition row the log stayed silent for 58 minutes.
- The run made 13,920 transitions in 7,657 s. That is 1.8 per second, against the 8.9 per second
  the expert benchmark measured.
- The world holds all 24 cells in one libuipc scene, so one hard cell sets the step time of every
  cell.

## What bounds a step

- **Newton.** `newton/max_iter` is 1024 per frame, the libuipc default. The environment did not
  set it.
- **PCG.** The fused PCG stops at twice the degrees of freedom, so its cap never binds.
- **Other settings.** The FEM preconditioner is MAS, with a relative PCG tolerance of 1e-2. Contact
  uses kappa 1e7 and a d_hat of 1 mm.

## Probes

The probes run in graph mode 0, where the host timers count iterations exactly. They use the
teacher's own settings, built through `pretrain_wang.prepare` and `train_sac.dressing_config`.

**Expert replay.** tshirt_68 on eight region-13 bodies:
- Decisions 0 to 30: 6 to 31 Newton iterations per decision and 6 to 42 PCG iterations per Newton
  iteration, with the hold error at 7 mm or less.
- `middle` stage, decisions 81 to 97: Newton per decision rises from 14 to 176 and PCG per Newton
  from 40 to 104. The decision time grows from 5 s to 93 s, while the hold error stays at 5 to 6 mm.

**Persistent push.** Each axis holds a random action of magnitude 0.7 to 1.0 for 30 decisions, on
eight shirts. From the first decision it needs 35 to 63 Newton iterations per decision and 57 to 96
PCG per Newton. The hold error stays at 17 mm or less.

**Uniform random actions.** Eight mixed cells, two of them gowns. From the first decision it needs
21 to 29 Newton iterations per decision and 90 to 120 PCG per Newton.

In these probes the hold error stays at millimetres while the cost climbs, so an unbounded hold force
is not what slows these steps. The cost is the linear solve. The learned policy is another matter:
see [The learned policy's slow tail is the hold](#the-learned-policys-slow-tail-is-the-hold). PCG iterations per Newton step rise with contact,
as in the expert's `middle` stage, and with large actions, from the first decision. Against the
expert's six PCG iterations per Newton iteration at decision 0, large actions make every step about
fifteen times as expensive from the start. The teacher's first episode, at 3.6 to 6.7 s per vector
step, is therefore the learning policy's baseline, well below the expert benchmark.

## Neither contact parameter nor step size is the lever

Each run pushes the same eight shirts with the same seeded sequence of held actions, for 30
decisions in graph mode 0. The baseline was stopped after 21 decisions.

| Run | Newton per decision | PCG per decision | PCG per Newton | Line search per Newton |
|---|---:|---:|---:|---:|
| Baseline: d_hat 1 mm, kappa 1e7, full step | 46.9 | 3,383 | 72.2 | |
| d_hat 3 mm | 47.4 | 3,695 | 78.0 | 1.00 |
| kappa 1e6 | 45.3 | 3,477 | 76.7 | 1.00 |
| Half the translation per decision | 35.4 | 3,330 | 94.0 | 1.00 |

- **d_hat 3 mm.** Over the first 14 decisions it cut PCG per decision by 26 per cent. By decision 27 it
  was back at the baseline.
- **kappa 1e6.** It changed nothing.
- **Half the translation per decision.** It took 25 per cent fewer Newton iterations, but each needed
  more PCG iterations, so the PCG work per decision did not move.
- **Line search.** It accepted on the first trial in every Newton iteration of every run, so neither
  the barrier nor CCD limits the step.

The cost under large actions is the conditioning of the linear system, and none of these three
settings changes it by the factor of two that would name a lever.

## The expert does not stall in production mode, with or without a Newton cap

The expert replay ran on the eight stalled tshirt_68 cells, in graph mode 2 (the production mode),
for 300 decisions. Each run used a different `newton/max_iter`:

| Newton cap | Minutes | Mean s per decision, by 50-decision window | Slowest decision | Upper arm >= 0.7 |
|---|---:|---|---:|---:|
| 1024 (library default) | 12.8 | 1.5, 2.6, 4.0, 3.2, 2.7, 1.4 | 11.4 s | 4 of 8 |
| 128 | 12.5 | 1.6, 2.6, 3.9, 3.4, 2.4, 1.2 | 12.8 s | 2 of 8 |
| 32 | 13.0 | 1.3, 2.5, 4.0, 4.0, 2.9, 0.9 | 13.7 s | 4 of 8 |

- **Graph mode 0 inflated the wall time.** Around decision 97, where graph mode 0 took 93 s, graph
  mode 2 needs at most 11 s for the same iterations. Graph mode 0 pays a host round trip on every PCG
  iteration.
- **The cap hardly binds.** The windows barely differ, so these states rarely need more than 32 Newton
  iterations in a step.
- **Outcomes are chaotic per cell.** The cap-32 run passes the same four cells as the uncapped run.
  The cap-128 run loses cells 0 and 4, at 0.99 without the cap. One changed step changes where a cell
  ends, so eight single samples cannot rank the caps.
- **The stall came from somewhere else.** The expert's states do not reproduce the stalled teacher's
  130 s vector step. That came from states the learning policy reached, in a world of 24 cells.

## The five-garment expert check with the cap

The five-garment check ran on bodies 0 to 7 with the committed placements. It read the same drape
bakes as the 21-of-40 run at `638e63f6`, and the watchdog was off:

| Garment | Upper arm >= 0.7 at `638e63f6` | With the cap of 128 |
|---|---:|---:|
| tshirt_26 | 5 of 8 | 6 of 8 |
| tshirt_68 | 5 of 8 | 5 of 8 |
| tshirt_4 | 5 of 8 | 5 of 8 |
| tshirt_392 | 0 of 8 | 0 of 8 |
| hospital_gown | 6 of 8 | 6 of 8 |
| Total | 21 of 40 | 22 of 40 |

- **Forearm.** It is reached on 39 of 40 cells in both runs.
- **Per-body ratios.** Every per-body ratio agrees within 0.03, except tshirt_26 on body 3, which now
  passes (0.25 to 1.00).
- **Result.** The cap costs the expert nothing on these cells.

## Fix

Three settings in `DressingConfig`:
- **`newton_max_iterations`.** 128 per simulation step, against the library's 1024. A step that needs
  more ends unconverged but free of penetration, because every line-search step is CCD-filtered.
  The five-garment expert check does not move.
- **A decision watchdog.** The budget is `decision_time_factor` (8) times the median of the last 64
  decisions, and never below `decision_time_floor_s` (30 s). Until eight decisions are timed, it is
  the floor times the factor. A decision past the budget raises a simulator error, and
  `pretrain_wang` handles it like any other: it drops the step's transitions and rebuilds the world.
  A stall like the one above now costs about a minute.
- **`anchor_tether_m`.** Off by default. The hold did not cause the slowdown, so the tether stays
  opt-in.

A trip deletes the step it trips on, and the slow steps are the contact-rich ones the policy most
needs to learn from. So the trip rate is watched in the run: `pretrain_wang` prints
`[wang] simulator error` on each trip and counts the trips in `sim_errors`.

## Stress gate

The gate ran 24 environments with the persistent push on region-13 cells of all five garments. It
used graph mode 2, the cap and the watchdog.
- **Trips.** The watchdog did not trip in the first 102 decisions.
- **Decision time.** The mean was 28.8 s and then 20.2 s over the first two 50-decision windows,
  while five expert checks shared the GPU. Once they finished it was about 9 to 10 s.
- **Cost.** A held push near saturation is the most expensive action pattern measured here.

## Reproduce

The probes are in the session scratchpad, under `perf/`:
- **`stall_probe.py`.** It builds the teacher's settings through `pretrain_wang.prepare` and takes
  `--policy expert|random|persistent`, `--graph`, `--newton-max-iter`, `--d-hat`,
  `--contact-resistance` and `--max-translation-scale`. It appends one JSON line per decision.
- **`ab_compare.py` and `cap_compare.py`.** They summarise those records.
- **`expert_dt.py --no-watchdog`.** It runs the five-garment expert check.

## Relaunch 1: the watchdog trips in every policy episode

The chain restarted at 06:44 from `953b0cff`, with the cap and the watchdog. The first world built in
62 s from the seeded bake cache.

| Transitions | Episode | Environment s per vector step |
|---|---:|---:|
| 0 to 7,200 | 1 (random actions) | 4.1 to 5.3 |
| 7,200 to 9,600 | 2 | 5.0 |
| 9,600 to 12,384 | 2 | 7.2 to 12.1, then a trip at episode step 215 (budget 74 s) |
| 14,400 to 15,336 | 3 | 5.2 to 13.9, then a trip at episode step 124 (budget 51 s) |

- **The second trip is partly confounded.** It came as two probe processes started on the same GPU.
  Its episode was already climbing along the same curve as the second episode's.
- **A known limit of the watchdog.** The budget follows the median of the last 64 decisions, so it
  cannot adapt to contention that starts in the middle of an episode. Probes running beside a teacher
  therefore cause spurious trips.
- **The loop skipped its schedule on a trip.** Evaluation and checkpoints ran only when an episode
  reached its horizon, and none did once the policy acted. The run would never have evaluated or saved.
  `300d743c` runs the schedule on the trip path too.
- **The run was stopped at 08:30,** at 16,272 transitions, and kept as
  `wang_teacher_r13_s1_relaunch1_tripped_20260911`.

Once the policy acts, the cost of a step is the price of contact at 24 cells, about 2 transitions per
second. Every episode then climbs until one decision passes eight times the median. The watchdog
bounds each stall, but it also cuts off the end of the episode, the phase that pulls the sleeve up the
upper arm.

## The PCG tolerance is not the lever either

The expert replay ran on the eight tshirt_68 cells in graph mode 2, with both arms at once on an
otherwise free GPU:

| PCG tolerance | Minutes | Mean s per decision, decisions 50 to 200 | Upper arm >= 0.7 |
|---|---:|---:|---:|
| 1e-2 (default) | 10.7 | 2.65 | 5 of 8 |
| 5e-2 | 12.1 | 3.01 | 5 of 8 |

The looser tolerance is 14 per cent slower, so the default stays at 1e-2.

## Relaunch 2

The chain restarted at 08:43 from `300d743c`. It carries the cap, the watchdog, the schedule on the
trip path, and `--checkpoint-every 10000`, so that a learned actor exists about an hour in. That
checkpoint drives the next diagnostic: `stall_probe.py --policy checkpoint` on mixed region-13 cells.
At every decision it records each slot's forearm and upper-arm ratios, collision and threading.

## Evaluation under the watchdog

Relaunch 2's second evaluation, at 14,400 transitions, reported a NaN held-out upper-arm ratio. All 25
episodes carried a simulator error.
- **Why the budget was too low.** The evaluation world is built once and reused, so its watchdog set round
  two's budget from round one's decisions. The untrained actor never touched a garment, so those
  decisions were cheap and the budget sat at the 30 s floor.
- **Why the whole round was lost.** The first contact-heavy decision passed that budget. A trip ends every
  slot's episode at once, and without metrics.
- **The fix, in `8c3923a8`.** The environment clears its decision history on every reset, so each
  episode and each evaluation round is budgeted on its own decisions. An episode cut by a simulator
  error is scored at its last completed decision, and still does not count as a success.
- **`best.pt` was never at risk.** `checkpoint_score` already scores a non-finite value as the floor.

The same episode shows the watchdog's other side. Its budget follows a rising median, so the last 2,400
transitions of episode 2 ran at 36 s per vector step, 53 s at worst, without a trip, and the episode
reached its horizon. The tail is slow, not stalled.

## The learned policy's slow tail is the hold

Relaunch 2's checkpoint at 14,400 transitions drove `stall_probe.py --policy checkpoint`, the actor
sampling as in training, on eight region-13 cells:
- hospital_gown 14002 and 14014;
- tshirt_26 14003 and 14013;
- tshirt_68 14005 and 14012;
- tshirt_4 14008;
- tshirt_392 14010.

Graph mode 2, no watchdog, the GPU shared with the running teacher. The hold error is the largest
distance between a held cuff vertex and its commanded position.

| Decisions | Mean s per decision | Worst s | Largest hold error, mm |
|---|---:|---:|---:|
| 0 to 49 | 3.8 | 12 | 32 |
| 50 to 99 | 3.5 | 13 | 38 |
| 100 to 124 | 4.3 | 8 | 89 |
| 125 to 149 | 5.4 | 23 | 131 |
| 150 to 182 | 25.1 | 128 | 179 |

- **The slow cells are the dragged ones.** tshirt_4 14008 passed 50 mm at decision 113 and reached
  179 mm with the forearm at 0.84; tshirt_26 14003 reached 163 mm.
- **The earlier probes never reached this regime.** The expert's hold peaked at 59 mm (the gown, body 1)
  and the persistent push stayed at 17 mm or less.
- **The picker differs from the reference.** Wang's PyFlex picker is kinematic: the picked particles
  follow it exactly and cannot lag. The soft hold here can, and its spring force grows with the lag.

The run stopped at decision 182.

## A tether on the patch centre is not enough

`anchor_tether_m` as first written (`095e72e2`) dropped a translation that took the commanded tool more
than the tether from the held patch's centre. The run below used 5 cm on the same cells and checkpoint.
The actor samples, so the two runs do not share a trajectory: compare bounds, not decisions.

| Decisions | Mean s per decision | Worst s | Largest hold error, mm |
|---|---:|---:|---:|
| 0 to 49 | 3.0 | 5 | 4 |
| 50 to 99 | 5.8 | 58 | 72 |
| 100 to 124 | 7.7 | 47 | 97 |
| 125 to 149 | 8.4 | 25 | 113 |
| 150 to 182 | 12.8 | 46 | 112 |

- **The hold is a lever.** The tail halved, 25.1 to 12.8 s per decision, and the worst decision fell
  from 128 to 58 s.
- **The tether did not bound it.** The hold still reached 112 mm. Five cells passed 60 mm, against two
  without the tether, and 8 decisions took over 30 s, against 7.

## A tether on every held vertex

Two more runs used the same cells and checkpoint on a free GPU:
- **Centre tether, 5 cm.** The gate above, now logging each slot's centre gap and the distance the
  grasp rotation has moved the commanded vertices.
- **Vertex tether, 6 cm.** A move, rotation included, is dropped when it would leave any held vertex
  farther than the tether from its target, unless it brings the farthest one closer.

| Decisions | Centre 5 cm: mean s | worst s | hold max, mm | Vertex 6 cm: mean s | worst s | hold max, mm |
|---|---:|---:|---:|---:|---:|---:|
| 0 to 49 | 2.3 | 6 | 14 | 2.5 | 6 | 14 |
| 50 to 99 | 3.7 | 17 | 102 | 3.9 | 17 | 46 |
| 100 to 124 | 4.6 | 13 | 119 | 6.6 | 40 | 60 |
| 125 to 149 | 12.8 | 46 | 123 | 3.9 | 14 | 60 |
| 150 to 182 | 16.1 | 103 | 125 | 5.8 | 18 | 60 |
| Total, minutes | 21.1 | | | 12.9 | | |

- **Why the centre tether leaked.** The hold passed 60 mm at 190 slot-decisions. At those, the centre
  gap had a median of 45 mm, capped at the tether's 50, and the rotation had moved the commanded
  vertices by a median of 35 mm. The grasp radius is 56 to 71 mm and the policy may turn 5 degrees per
  decision. The farthest vertex trails by about the sum of the two, and the centre gate checks only
  the first.
- **The vertex tether bounds the hold.** It stopped at 60 mm. The tail stays flat: decisions 150 to 182
  average 1.5 times decisions 0 to 125, against 4.8 times with the centre tether.
- **The tether binds often.** At 60 of the 183 decisions some slot sat at the tether.
- **Two slow decisions remain.** Decisions 100 and 101 took 32 and 40 s with every hold under 47 mm,
  below the tether. They are contact, not the hold.
- **The value.** 0.06 m sits just above the scripted expert's 59 mm ceiling, and the blowup runs from
  90 to 180 mm. `anchor_tether_m` now defaults to 0.06. The rule differs from the no-move collision,
  which drops only the translation.

## The five-garment expert check with the vertex tether

The same 40 cells as the check with the cap, bodies 0 to 7 of each garment, with `anchor_tether_m`
at 0.06:

| Garment | Upper arm >= 0.7, cap only | With the tether | Bodies that changed |
|---|---:|---:|---|
| hospital_gown | 6 of 8 | 6 of 8 | 2 lost, 6 gained |
| tshirt_26 | 6 of 8 | 6 of 8 | none |
| tshirt_392 | 0 of 8 | 0 of 8 | none |
| tshirt_4 | 5 of 8 | 5 of 8 | none |
| tshirt_68 | 5 of 8 | 5 of 8 | none |
| **Total** | **22 of 40** | **22 of 40** | |

- **The gown's two changes are not the tether.** Bodies 2 and 6 held within 14 and 12 mm, far inside
  it. The GPU solve is not bitwise repeatable, so such flips also separate the runs with and without
  the cap.
- **The tether binds where the hold ran away.** tshirt_392 body 6 reached 131 mm with the cap only and
  59 mm with the tether.

## Relaunch 3

The chain restarted fresh at 12:53 from `d19489c8`, with the vertex tether on by default and relaunch 2's
flags. Against relaunch 2 at the same steps:

| Step | Transitions | Relaunch 2 elapsed s | Relaunch 3 elapsed s | Env s per step, last 20: relaunch 2 | Relaunch 3 |
|---:|---:|---:|---:|---:|---:|
| 300 | 7,200 | 1,914 | 2,026 | 5.6 | 7.2 |
| 500 | 12,000 | 3,780 | 3,763 | 13.6 | 9.6 |
| 540 | 12,960 | 4,878 | 4,112 | 25.1 | 6.2 |
| 580 | 13,920 | 6,452 | 4,539 | 49.6 | 10.9 |

- **Episode 1 runs at the same speed in both.** From late in episode 2, where relaunch 2 climbed to 50 s
  per step, relaunch 3 stays under 11 s.
- **No simulator error through 14,400 transitions.** Relaunch 2's first came at step 704.
- **The first evaluation under the policy.** At 14,400 transitions the held-out upper-arm ratio is 0.054
  and the forearm ratio 0.48, with a mean return of 24.1 and no simulator error, in 1,074 s. Relaunch 2's
  round at the same point lost all 25 episodes to the watchdog and reported NaN.

