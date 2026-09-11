# The region-13 teacher stall: the linear solve, not the hold

Status: bounded by a Newton cap and a decision watchdog. The relaunched teacher is the remaining check.

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

The hold error stays at millimetres while the cost climbs, so an unbounded hold force is not what
slows these steps. The cost is the linear solve. PCG iterations per Newton step rise with contact,
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
