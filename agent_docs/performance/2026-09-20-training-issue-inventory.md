# Everything measured to be wrong with dressing training — 2026-09-20

Review status: **historical inventory; several causal claims corrected** by the
[evidence review](2026-09-20-training-issue-review.md). In particular, continuity
does not imply a flat reward; the privileged run changes physical control timing;
top-two action gaps are not policy advantages; and 15 peak-covered cells are not
a reachable upper bound. Read the item-by-item disposition before using the
claims below to select or reject an algorithm. Original wording is retained for
provenance.

Status: inventory, not new measurement. Every entry names the artifact or record it
comes from. Entries marked **[unverified]** are read from a record and not re-measured
in this pass.

## A. The objective

**A1. The objective is flat at both events the task is about.** Wang's task term is
minus the fingertip-to-opening distance while the opening is off the arm, so it rises to
exactly zero as the opening arrives, and forearm progress is measured back from the
finger, so threading starts at zero. At the elbow, forearm progress ends at `forearm_len`
and the upper-arm branch starts at `forearm_len + 5 * 0`. Both junctions are continuous
and have no slope. (`dressing_reward.py::wang_progress`,
`2026-09-20-upper-bound-and-ral-calibration.md`)

**A2. One plateau costs more than a workstation budget.** A state actor with a
privileged critic, on the 25 cells it is evaluated on with nothing held out, spent about
175,000 transitions crossing the fingertip plateau and the remaining 95,000 covering
42 % of the forearm without reaching the elbow. Peak upper-arm coverage was exactly
0.0000 at all 22 evaluations of 270,000 transitions.

**A3. The reward has no grasp term; the success rule has a hard 2 cm grasp gate.** The
coupling is indirect - a slipped grip stops the cloth following the tool, so progress
stops - but nothing in the objective names the constraint the score enforces.

**A4. Nothing rewards stopping.** Four of the teacher's cells reach 0.997-1.000 coverage
and end at 0.000. Freezing the command at the crossing converts three of four into
successes, with final coverage equal to peak to the digit.
(`2026-09-19-stop-when-covered.md`)

## B. The success criterion and the metric

**B1. Three different success rules exist in the code and disagree.** The environment's
own `valid_grasp_success` checks the grasp of the *final decision only*
(`dressing_env.py`, `tracking_max` is reset at the top of each `step`). The evaluators
used by every study from 2026-09-17 on use whole-episode max tracking with coverage
sustained over the last twelve decisions. The FQL tables use final coverage with
whole-episode grasp. Cross-study numbers are not comparable. **[unverified]**

**B2. Two coverage geometries.** `upperarm_extension_m` is an opt-in 5 cm shoulder ray.
The teacher's 9 of 25 uses it; the 6 of 25 does not; every `abl_*` and `wang_teacher_*`
evaluation predates it. **[unverified]**

**B3. Most training evaluation logs have no grasp column at all.** `success_rate` in
them is coverage-only, so the historical "best 7 of 25" is not a coverage-and-grasp
number. **[unverified]**

**B4. The coverage metric is coarse at the rim.** It ray-casts against a six-vertex
opening polygon triangulated into four triangles; 4-9 % of a typical episode's decisions
report no progress while the opening still encircles the arm. Verified in this pass:
these are *contiguous marginal states*, not spikes - isolated one-decision zeros are
0.024 % over 96,960 decisions, and the six captures that raised the concern report zero
each. The metric should not be read at a single decision; it is not corrupting the
signal at random. (`2026-09-19-intervention-and-agent-baseline.md`)

## C. The environment as a source of learning signal

**C1. The advantage is below the noise at many states.** At states a policy visits, the
gap between the best intervention and the next best is 1.31 pooled repeat standard
deviations on `tshirt_26`, 1.24 on `tshirt_392`, and below one at 42 % of states. On the
cloth-drag benchmark, where the same code base's SAC goes from -4 to +13 return, the
same statistic is 193.6 with 8 %. (`2026-09-19-decision-signal-to-noise.md`)

**C2. An update driven by that is a random walk.** The signature appears in every long
run: the 270,216-transition SAC peaked at 3 of 25 at 174,264 and ended at 0 of 25.
**[unverified]**

**C3. Nothing resolves at single-decision granularity.** Displacing one command by its
full magnitude moves the best score by 1.45 pooled standard deviations. Only a
four-decision segment resolves, at 5.29.
(`2026-09-20-upper-bound-and-ral-calibration.md`)

**C4. Identical commands from an exactly restored state never reproduce.** Restore error
is 0.00e+00 m, yet separation reaches millimetres within ten decisions at most states.
It is the solver's own run-to-run non-determinism, amplified by contact.
(`2026-09-18-predictability-horizon.md`)

**C5. The dressing environment runs looser solver tolerances than the benchmark that
learns.** Newton velocity 0.1 against 0.001, conjugate-gradient 1e-2 against the
library's 1e-3. Tightening to 0.01 and 1e-4 drops the replay separation at twenty
decisions by 139 times and the coverage spread by 583 times, for about 2.5 times the
wall time and no code change. Recorded and not pursued, by direction.
(`2026-09-19-tolerance-sets-the-noise-floor.md`)

**C6. Contact force is unusable.** Replaying identical commands, the steadiest channel
is a contact count at 24 % relative spread; summed and peak normal forces are 50-126 %
and friction about 190 %. It cannot be a policy input, a reward term or a safety
threshold. (`2026-09-19-force-reproducibility.md`)

## D. The data and the prior

**D1. The behaviour prior extrapolates exactly where a recovery matters.** Reversing the
flow at learner-visited states puts the needed action at noise percentile 1.000 with
reconstruction error 0.282, against 0.027 and 0.019 after labelling the policy's own
states. (`2026-09-19-flow-prior-support.md`)

**D2. No dressing replay on disk carries privileged state.** All eight compatible
replays, 945,864 transitions, have `priv_dim` 0. **[unverified]**

**D3. Imitation is capped by the teacher, with one exception.** Expert BC scores 2 of 8
and both successes are cells the teacher passes; withheld bodies are 0 of 4 at 0.00000
coverage. The exception is FQL finishing `tshirt_68/14049` - a withheld cell where the
teacher peaks at 0.997 and ends at 0.000 - at coverage 1.000 with a valid grasp, one
seed, two rounds, every episode tripping `early_turn`.

## E. Exploration and decision structure

**E1. The value of choosing per state is small.** 0.046 on `tshirt_26` and 0.000 on
`tshirt_392` against the best fixed macro; ranking agreement between independent repeats
is 0.47 against 1.00 on the control task, where chance over seven macros is 0.14.

**E2. The neighbourhood of a good action is a plateau.**
(`2026-09-18-recovery-decisions.md`)

**E3. Training moves the policy away from the event.** Random-action evaluation threads
the opening in 52 % of episodes at 12,500 transitions; the policy then spends 160,000
transitions before threading returns to 1.00, and the reward it climbs in between is the
approach term's ceiling.

## F. Protocol and infrastructure

**F1. The privileged state was invisible to every actor.** `PRIVILEGED_LAYOUT` is
documented as critic-only by design, `StateActor` existed in `sac.py` but `--actor` did
not accept `state`, and the only asymmetric-critic dressing run ever started died at
13,920 transitions with no checkpoint. Every negative result before this session is
confounded with a 5,383-float point cloud. Fixed in this session.

**F2. `--body-seeds` silently holds ten of twenty-five slots out of replay** unless
`--heldout-bodies 0` is passed. A run can train on 60 % of what it appears to.

**F3. The garment curriculum admits one garment per 100 vector steps by default**, so
the first 500 steps feed replay from a fraction of the slots. Not a bug, but the
transition count and the simulated count diverge and that has been misread.

**F4. Evaluation is a quarter of the wall clock.** The longest run spent 24,079 s of
94,161 s in evaluation. **[unverified]**

**F5. A killed shell does not kill its child.** Two training processes ran against one
output directory in this session, interleaving `eval_log.csv` and the checkpoints. The
directory is kept as `state_upper_bound_s1_CONTAMINATED`.

## G. The benchmark itself

**G1. The reachable ceiling is 15 of 25 and the teacher scores 6.** Four cells are
covered and lost; ten never reach threshold and nine of those have a worst tracking
error of 1.84 cm or more. (`2026-09-19-what-actually-fails.md`)

**G2. One garment fails on every body.** All five `tshirt_392` cells peak between 0.23
and 0.50 with tracking 2.87-3.17 cm. It is a different problem from the rest of the
grid.

**G3. Two garments were never solved once** in the 270,216-transition run:
`hospital_gown` and `tshirt_68` read 0.0 at every evaluation. **[unverified]**

## H. Where the numbers stand

| controller | score | rule |
|---|---|---|
| scripted teacher, with the stopping rule | 9 of 25 | coverage and grasp |
| scripted teacher, as written | 6 of 25 | coverage and grasp |
| best SAC ever on the full grid | 7 of 25, not reproducible within its own run | **coverage only** |
| longest SAC run, 270,216 transitions, 26.2 h | peak 3 of 25, final 0 of 25 | coverage only |
| privileged upper bound, 270,000 transitions | **0 of 25** at all 22 evaluations | coverage and grasp |
| FQL, 4 cells | 3 of 4 episodes on one withheld cell the teacher fails | coverage and grasp |

No learned controller has ever matched a 733-second script on this benchmark.
