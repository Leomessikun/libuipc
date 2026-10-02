# Observed-motion teacher and matched students — 2026-10-02

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

Latest read-only inspection under `output/uipc_manip/m3_observed_20261002/`:
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

Native corrected simulation is in progress. No dynamic research student has
been trained yet. The first nominal-action replay differed by about 6.8
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
