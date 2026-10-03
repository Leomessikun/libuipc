# Observed-motion teacher and matched students — 2026-10-02

## Stopped after the owner's review — 2026-10-03

The owner authorized stopping launched dressing jobs if further work lacked
sufficient research value. M4 remained a conventional privileged-teacher
distillation baseline and did not implement a new post-training algorithm.
Its remaining automatic collection/training/evaluation budget is cancelled;
this supersedes the continuation approvals recorded below.

At inspection, one of eight initialization episodes had finished and two were
running. The queue (PID 1058515), reporter (1486967), newly started body-14048
pass worker (2243778) and its inference server were stopped. The other worker
(1132541) had completed all 51 expensive planning calls and was allowed at most
ten minutes to finish saving; it completed and exited within that bound.
Five stale shell monitors (1023005, 4041806, 4050841, 4070488, 4078518) were
also terminated after confirming that their target jobs were no longer live.

| Retained initialization episode | Final success | Decisions | First success | Planning calls | Elapsed |
|---|---|---|---|---|---|
| body14047 pass | Yes | 450 | 148 | 51 (49 valid labels) | 10,385 s |
| body14047 lift | Yes | 450 | 174 | 51 (51 valid labels) | 9,615 s |

These are training-data episodes, not an independent generalization test.
Both complete trajectory archives passed ZIP CRC checks; hashes are in the
[stop record](2026-10-03-dressing-job-stop.json). The interrupted body14048
pass run retains seven complete candidate-query records and its initial state,
but no complete trajectory. It is not counted as a policy failure.

Five initialization episodes were cancelled before launch. Student training,
DAgger and all 72 held-out evaluations had not started; no student model or
held-out result exists. The pipeline status and report now show the owner stop
and cancelled evaluation episodes. Pre-stop snapshots and the exact process
manifest remain in `output/uipc_manip/m4_privileged_20261002/stop_20261003/`.
All selected processes were verified exited. Existing data, checkpoints,
source worktrees and unrelated workloads remain available. No new experiment
was launched. Respect `STOP_REQUESTED.json` before any future launch.

## Scope and evidence

The owner authorized a corrected small validation, followed by dynamic teacher
data, history/current-only student controls, DAgger, and held-out evaluation.
This is a bounded learning experiment, not an established algorithm contribution.

The completed M2 pilot on body 14046 and two s1 GRAB clips reports current-pose
planning 0/6, exact-state causal planning 6/6, true-future planning 5/6. These are
repeats of two tasks, not six independent tasks. The GICP proxy's 1/6 came from
an older evaluation path. See [M2 record](2026-10-01-garment-and-motion.md).

Static r2 failing to improve does not establish a universal one-round ceiling.
A weak frozen teacher is a plausible local limitation, not proof that a student
cannot outperform its teacher. The existing flow experiments do not rule out
all generative-policy architectures.

## Corrected validation

Implementation: `scripts/wang_transfer/motion_lookahead_probe.py`, with
`--candidate-set observed_common --planning-window 0 5 --save-trajectory`.

- Every belief uses the same 12-action generator: nominal, half, stop, no
  rotation, six translation perturbations, and two **observed-cloud** follow
  actions. At a matched state the candidates are exactly identical. Subsequent
  closed-loop states may differ. Legacy behavior remains explicitly selectable.
- Both observed planning and GICP correction use visible segmented arm clouds
  and measured tool position. Tool translation is removed before registration.
  There are no vertex correspondences, ground-truth human velocities, motion
  identifiers or future samples in that estimate. Invalid estimates become zero.
- The teacher retains privileged current cloth/body geometry for simulation.
  The observed forecast applies the estimated rigid displacement field to that
  geometry at constant velocity. This is an approximation for articulated
  motion, not a learned motion predictor. Speed is bounded at 0.5 m/s; GICP RMS
  must be at most 2 cm. It may fail where exact per-vertex velocity succeeds.
- The planning window is fixed elapsed time 0–5 s, not computed from a motion
  onset or clip duration. It is a bounded pilot budget, not a deployment trigger.
- Score, H=4, action hold, physics, completion controller and 450-step budget
  stay fixed. Report **final_success**: all interior sleeve sections wrap, armhole
  fraction >=0.7 for the final 21 states, and grasp valid throughout. Force is
  still an uncalibrated teacher score proxy and is never a student input.
- Store initial state arrays, code/checkpoint/motion hashes, candidate outcomes,
  observation history, proposal/teacher/executed actions, query and intervention
  masks. Failed episodes are retained; invalid-physics episodes are not training
  data. Observations align as `obs[t] -> action[t] -> obs[t+1]`.

Earlier inspection under `output/uipc_manip/m3_observed_20261002/`:
pass/current completed with grasp failure at decision 13 (13 queries, no action
changes); pass/GICP completed with grasp failure at decision 161 (maximum
armhole fraction about 0.551). Pass/observed and lift/observed remain active;
lift/current and lift/GICP are pending. There is no completed observed-motion
teacher or trained-student result yet. The old 6/6 result must not be relabeled
as point-cloud or student performance. The separate
[contact-intervention proposal](2026-10-02-executable-contact-interventions.md)
does not alter this experiment or its scaling gate.

## Training protocol

The first proposed student is frozen r1 point features/base actor plus a small
zero-initialized action adapter, using four causal frames and measured tool
displacements. The same-size control replaces every older frame with the current
frame. Identical labels, optimizer updates, minibatch seeds, parameter counts,
and inference controls are required. This tests whether history helps on dynamic
data; it does not establish that any benefit is specifically future prediction.
`dynamic_student.py` provides the CPU model/bridge;
`train_dynamic_student.py` caches features and trains the adapter. Encoder FPS
is seeded per observation identically during extraction and deployment. A zero
adapter checkpoint provides a deployment-equivalent r1 control to isolate this
change from learned improvements. All six action components are supervised in
the simulator frame; no force or simulator human-motion state is an input.

Keep subject s1 for training/development. Reserve whole GRAB source sequences
and subjects, plus different recipient body IDs, for final evaluation. Hold out
an entire training-domain body for validation; never randomly split frames.
Start with teacher-executed rollouts and call them **teacher initialization**.
DAgger requires querying on states visited by the student; preserve the label
even when a different action is executed. Include both students' visited states
in a shared aggregate dataset for the next matched training round.

Scaling gate: the corrected teacher must first succeed on both pilot clips and
have at least one paired win over each of current-pose planning and GICP. This
is a practical continuation gate, not a significance test. If it fails, retain
the failure and diagnose perception/action/forecast error before larger runs.

## Bounded executable pipeline

`run_dynamic_pipeline.py` implements these stages in order:

1. Six corrected validation runs (two clips, three conditions), including the
   already running observed/pass job. Check actual initial state arrays across
   conditions within 10 micrometres, then apply the teacher gate above.
2. If it passes: eight teacher-initialization episodes. s1 pass/lift on bodies
   14047, 14048 and 14050 for fitting; body 14052 for validation. Onsets 0.8/1.2 s.
   All eight motion assets have already been prepared on CPU; this is asset
   preparation, not eight collected dressing trajectories.
3. Cache frozen features on CPU. Train history/current controls with identical
   data, 1,000 Adam updates, batch 128, and three seeds. No test-set selection.
4. Four student-roll-in DAgger episodes: both seed-0 students on body 14047/pass
   and 14048/lift, onset 1.4 s, teacher-execution probability **zero**. Query the
   frozen-r1-proposal observed teacher on visited states. Keep failures and
   aggregate both students' labels into the same dataset for both models.
5. Refit both adapters from r1 with the shared aggregate, the same budget and
   three seeds. This is one bounded DAgger round, not an endless improvement loop.
6. Held-out evaluation: bodies 14054/14055 crossed with s2/mug_pass_1 and
   s3/phone_call_1, using original r1, GICP proxy, fixed-encoder zero-adapter r1,
   and both student variants (three seeds). These are four task cells; seeds do
   not multiply the number of independent motions.

The queue writes `pipeline_status.json`, stage job manifests, per-job commands,
logs and output markers. Start/resume with:

```bash
/home/ge47gax/kun/genesis-world/.venv/bin/python -u \
  scripts/wang_transfer/run_dynamic_pipeline.py \
  --root output/uipc_manip/m3_observed_20261002 --max-jobs 2
```

Only one queue may own this root. It adopts matching live jobs on resume and
rejects changed commands or incomplete exited jobs. It never stops other jobs.
At most two own simulations run; each launch requires total GPU use below
70 GiB, measured with CUDA because local NVML is broken. Feature extraction and
student updates run on CPU with CUDA hidden. A failed teacher gate ends the
bounded pipeline with its table; it does not start more labeling or training.

## Checks

Nine focused CPU tests pass: robot ego-motion cancellation, invalid-registration
reset, causal episode-local history, equal-size current-only inputs, and exact
candidate equality across planning beliefs at a fixed state; binary bridge
integrity despite model banners; failed-gate stopping; and complete mocked
pipeline ordering/shared aggregation/test separation; branch-local error/reset
handling. Two-update synthetic
training smokes and both real-point-cloud policy bridges pass, including exact
history reset. These are software checks under `_software_smoke`, excluded from
research data and task results. FMVP's stdout FiLM banner initially corrupted
the unbuffered student bridge; redirecting model output to stderr fixed it and
the binary-channel regression covers it.
The real-cloud feature-extraction fixture also checks that non-query frames use
nominal r1 targets and completion holds use zero, even if an invalid candidate
action remains in the raw teacher-action array.

At the initial implementation check, native simulation was in progress and no
dynamic research student had been trained. The first nominal-action replay differed by about 6.8
micrometres; exact snapshot restoration passed. This is one local replay check,
not proof of deterministic physics.

The first observed/pass attempt hit an invalid **candidate** forecast at decision
12: human drive error 3.52 mm exceeded the unchanged 2 mm tolerance. The base
environment's unconditional error reset then attempted to settle with stale
forecast targets and crashed. The complete attempt is preserved under
`failed_attempts/pass_observed_branch_reset/`. The planning environment now
suppresses resets inside candidate branches so `step` returns `sim_error`, the
candidate is rejected, and the planner restores its own snapshot. Real executed
steps retain the original error behavior. States with no feasible candidate are
not correction labels. The failed attempt is an implementation failure, not a
completed task result. A corrected retry is active; the current-pose job remains
running. No tolerance, score or success threshold was relaxed.

## Corrected validation result and review (recorded by the supervising session, 2026-10-02 19:35)

One run per condition, `observed_common` candidates (identical 12-action set under every belief, follow
actions from the visible cloud), fixed 0-5 s window, final-state success, `output/uipc_manip/m3_observed_20261002`:

| clip | observed forecast | current pose | GICP |
|---|---|---|---|
| s1 mug pass | success (held from 138) | grasp lost at 13 | grasp lost at 161 |
| s1 mug lift | grasp lost at 142 | success | grasp lost at 181 |

The teacher gate (observed teacher succeeds on both clips) fails on mug lift. The queue then stopped with
`Initial states differ in pass: 1.4e-4 m`, so the gate table was not written by the pipeline.

Review:

- With the common candidate set, the **current-pose** planner dresses mug lift, where the legacy current-pose
  planner failed 3 of 3. The legacy 6/6 for exact-state causal planning therefore cannot be attributed to the
  motion inside the rollouts alone; the belief-dependent follow candidates (now observation-based and shared)
  and lift's run-to-run variance are live alternatives. Mug pass still separates the beliefs (current pose fails
  at 13 in every run so far). Repeats per condition are needed before any gate decision, which the pending
  `validation_repeats` option provides.
- The 10-micrometre initial-state check compares states settled in separate processes; they differ by 0.14 mm
  here, consistent with the solver's run-to-run variation, not with a placement change. Either compare against
  a measured same-command reset spread, or run all conditions of a clip in one process from one reset, as
  `probe_arm_motion.py` does, which makes the starts identical by construction.
- On the EXPO mainline: the motion env runs one body per process at roughly 3 s per decision without planning,
  so a 450-decision episode takes about 20-25 min and two to four concurrent runs on the shared GPU give about
  8-12 episodes per hour. An online editor/critic loop needing hundreds of episodes per seed is several days per
  seed before controls. Earlier IPC-label SAC updates in this project did not improve (IAQL, closed 2026-09-17).
  Order of work: (1) the offline critic-ranking diagnostic on existing candidate logs, (2) the teacher-to-student
  distillation with history vs current-only students as the deliverable that existing evidence supports, (3)
  plain EXPO from that student with a declared episode budget, (4) the consequence-metric variant only if plain
  EXPO learns at that budget and its ranking errors are shown to matter.

## M4: privileged causal teacher baseline

The observed-only training-teacher requirement is relaxed for this bounded
baseline. A training teacher may use privileged recent body targets and joints;
the deployable students still receive only point-cloud features, base-policy
outputs and measured robot state. The `causal` forecast extrapolates the last
two prescribed body states; it does **not** sample the real future GRAB motion.
It is not a perfect future oracle or a learned predictor.

Use a new root, `output/uipc_manip/m4_privileged_20261002`, preserving M3's
outputs and terminal error. The selected teacher remains subject to a common
candidate-set check: M2's legacy 6/6 is not evidence of its performance with
`observed_common`. M3's current-pose lift success also prevents attributing the
old difference entirely to forecast quality.

Protocol changes from M3:

- Two repeats per clip and condition, with the same 12 candidates, H=4,
  fixed 0–5 s window, physics and final success rule. Reuse M3's completed
  current/GICP repeat-zero results after protocol and input-hash checks.
  Four causal validation runs and four new comparator runs remain.
- Predeclared feasibility gate: at least one causal success on each clip,
  and more total successes than each comparator. This permits a bounded
  learning pilot; it is neither a significance test nor proof of anticipation.
- `--independent-resets` records cloth settling differences in
  `initial_comparisons.json` instead of calling independent process resets
  identical. Human/tool starts must still match within 10 micrometres and all
  differences must be finite. Cloth differences have no numeric acceptance
  cutoff in this mode; inspect the report before interpreting outcomes as
  comparable. Exact candidate-state restoration inside each planner is unchanged.
- On passing: eight causal teacher-initialization episodes, shared features,
  history/current-only students with three seeds each, four student-roll-in
  DAgger episodes, and one shared-data refit. Failed task episodes are retained;
  invalid physics is excluded. Teacher condition is recorded in the feature
  and checkpoint manifests, never fed to the student.
- Before final evaluation, run six seconds of passive zero-action motion on
  every fixed held-out body/clip cell, including s3/phone_call_1. Require valid
  attachment, no physical error, and body tracking within 2 mm. This checks
  motion/fixture compatibility; passive grasp loss alone does not prove that
  the human motion is physically invalid. Stop for inspection without dropping
  hard test cells or using their data for fitting.
- Two evaluation repeats of each policy/cell: 72 episodes across four task
  cells, three model seeds, and the original r1, GICP and fixed-encoder r1
  controls. Repeats and seeds do not create additional independent task cells.

Start the separate queue with:

```bash
/home/ge47gax/kun/genesis-world/.venv/bin/python -u \
  scripts/wang_transfer/run_dynamic_pipeline.py \
  --root output/uipc_manip/m4_privileged_20261002 \
  --teacher-condition causal --validation-repeats 2 --evaluation-repeats 2 \
  --reference-root output/uipc_manip/m3_observed_20261002 \
  --wait-for-root output/uipc_manip/m3_observed_20261002 \
  --preflight-heldout --independent-resets --max-jobs 2
```

The dependency waits for M3's owner and recorded jobs to exit; it never signals
them. Configuration is saved and frozen per root. The queue still launches at
most two own simulations, only below 70 GiB total GPU usage. Sixteen CPU checks
pass, including causal input isolation, teacher-condition feature selection,
same-data control ordering, configuration immutability, dependency waiting,
independent-reset reporting and preflight stopping.

This is the dynamic DAgger baseline in the post-training plan, not the proposed
EXPO extension. Privileged planning plus history-based visual distillation has
direct precedent in [GenH2R](https://arxiv.org/html/2401.00929v2). A history
student win would establish useful history, not future prediction or algorithm
novelty. The EXPO learner and shared consequence-metric candidate remain
separate, unimplemented research work. No dynamic research student result is
available at this update. The October 3
[prior-art review](2026-10-02-post-training-mainline.md#research-assessment-and-bounded-continuation--2026-10-03)
downgrades the separate shared-metric candidate and specifies the bounded
continuation decision.

## M4 validation results — 2026-10-03

All planned validation jobs exited. The common-candidate causal teacher now
has four valid successes, establishing its pilot result independently of M2's
legacy candidates. The current-pose planner also succeeds on both lift repeats.

| Condition | Mug pass | Mug lift | Valid successes / valid runs |
|---|---|---|---|
| Privileged causal teacher | 2/2 | 2/2 | 4/4 |
| Current-pose planner | 0/2 (grasp lost at decision 13 twice) | 2/2 | 2/4 |
| Observed GICP correction | 0/2 (grasp lost at 161 and 13) | 0/1 valid; one invalid-physics run | 0/3, plus one invalid run |

Teacher first sustained success decisions are 149/157 for pass and 138/160
for lift. Current-pose lift succeeds from 165/180. Every success remains valid
through the 450-decision endpoint. These are repeats of two clips on body 14046,
not four independent tasks or evidence of unseen-motion generalization. The
observed advantage is on pass; there is no success-rate advantage on lift.

The queue stopped before its gate table and before collection/training because
`validate_lift_gicp_rep1` raised a human target tracking error at 16.0667 s:
2.03787 mm versus the unchanged 2 mm validity limit. Do not count that run as a
valid policy failure. The base environment resets after a simulator exception,
so that run's summary `time_s=0` and tiny `body_tracking_max_m` are post-reset
values; the retained exception is the diagnostic evidence. This is not evidence
that the exact clip is physically impossible or that the error is pure noise.
All M4 processes were confirmed exited despite stale `running` entries in the
terminal queue status. No teacher initialization or research student exists yet.

An initial-array audit found matched human/tool starts and maximum cloth
settling differences of 0.1571 mm across the planned comparisons. These are the
declared independent resets, not identical states.

Recovery is bounded to one same-command, same-seed repeat of the invalid GICP
job, without changing physics, success criteria or the teacher. Preserve its
entire original output, launch metadata and log in
`m4_privileged_20261002/failed_attempts/validate_lift_gicp_rep1_attempt0/` and
record the retry in `physics_retry.json`. Keep the invalid attempt visible in
all reports; the retry does not erase it. If the retry is physically valid,
resume the existing feasibility gate and eight-episode collection regardless
of whether GICP succeeds. If it is invalid again, retain both and stop for
diagnosis rather than retrying until a usable result appears. No other completed
validation is rerun. This recovery does not establish an algorithm contribution.

### Bounded retry completed; initialization collection started

The single unchanged GICP retry was physically valid and failed through grasp
loss at decision 166 (final fraction 0.5191, maximum human tracking error about
0.460 mm). The valid GICP aggregate is now **0/4**, with **one additional
invalid-physics attempt** retained separately. Causal remains 4/4 and current
pose 2/4. `teacher_gate.json` records `passed: true`; it is a feasibility gate,
not a statistical significance test.

All eight CPU motion preparations completed. The queue entered
`teacher_initialization`, with `init_body14047_pass` and `init_body14047_lift`
active at inspection. No dynamic student has been trained yet. The queue and
its fixed collection/training/evaluation budget are unchanged.

A read-only audit of the four causal logs also limits the interpretation:
the first changed action is at elapsed time 1.1-1.2 s, after motion starts at
1.0 s. Planning takes about 146-189 s per query on average, and each successful
episode takes 2.2-2.9 hours for 45 s of simulated time. These results motivate
distilling a slow planner, but do not demonstrate pre-onset anticipation or a
new teacher/student algorithm. The GICP comparator is our registration proxy,
not a reproduction of all components of Dressing in Motion. See the
[research assessment](2026-10-02-post-training-mainline.md#research-assessment-and-bounded-continuation--2026-10-03).

## Student evaluation authorized and reporting prepared — 2026-10-03

The owner approved completing the bounded evaluation after the novelty review.
The existing pipeline is healthy and remains in teacher initialization, with
the first pass/lift episodes active. No student checkpoint or held-out student
success rate exists yet. Evaluation starts automatically after the existing
eight initialization episodes, matched training, four DAgger roll-ins, shared
refit and passive held-out checks; no additional simulation was launched here.

`evaluation_protocol.json` records the fixed 72-run grid and shared success
criteria before any held-out policy results: bodies 14054/14055, s2 mug pass
and s3 phone call, two evaluation repeats, and nine checkpoints/controllers.
The latter are r1, GICP, zero-adapter/fixed-encoder r1, and three seeds each of
history/current-only students. These remain four task cells, not 72 independent
generalization tasks. The pilot does not test ClothesNet generalization.

`scripts/wang_transfer/report_dynamic_evaluation.py` writes
`evaluation_report.json` in the experiment root. It reports each checkpoint's
valid-success denominator, task failures, pending/running jobs and invalid
physics separately; history/current comparisons remain nested in task cells.
It checks the final evaluation manifest against the fixed grid and rejects
success records without the full valid endpoint. It does not compute a
significance test from repeated seeds, and partial results do not constitute
the final comparison.

A CPU-only companion watches the existing queue for at most 72 hours, recording
its PID/command in `evaluation_report_launch.json`. It starts no simulation,
does not alter or restart jobs, and exits on completion, queue failure, loss of
the owner process, protocol mismatch or its time limit. Its log is
`evaluation_report_watch.log`. The current report correctly shows 72 pending
evaluation episodes. Four focused CPU tests cover invalid/pending denominators,
final rather than temporary success, seed pairing, and changed protocols.

Manual report refresh, when the companion is not holding its lock:

```bash
python3 scripts/wang_transfer/report_dynamic_evaluation.py \
  --root output/uipc_manip/m4_privileged_20261002
```

The local NVML version mismatch remains, but the pipeline's CUDA memory query
works (about 74.7 GiB used at inspection). Keep its existing limit of two own
simulation jobs and launch threshold below 70 GiB; do not change drivers or
restart unrelated GPU work for reporting.
