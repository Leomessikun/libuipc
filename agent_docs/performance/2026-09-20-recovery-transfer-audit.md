# Reuse the successful recovery before designing another search algorithm

## Evidence already available

The September 17 [recovery-teacher experiment](2026-09-17-recovery-teacher.md)
already tested a **closed-loop, full-continuation** controller. From an original
BC policy's decision-120 state on tshirt_68/14046, the outward-route teacher
achieved .98444/.97604 sustained upper-arm coverage with whole-episode valid
grasp in an independent verification world. It supplied 360 recovery labels.
The actor fitted those labels about ten times better after training, yet still
failed this target in both full-reset evaluations. Caps on its commands did not
rescue the full-reset result. This is a positive physical-control result and a
negative transfer result, not an absence of successful recovery trajectories.

The latest [action-selection report](2026-09-20-action-selection-evidence.md)
correctly rejects extrapolating single-action improvements to full dressing,
but its proposed next question was too broad: closed-loop teachers were already
tested. Their transfer into the deployable policy is the narrower unresolved
question. The [SAC-plus-recovery pilot](2026-09-17-recovery-sac-pretraining.md)
was stopped before its treatment or paired comparison; it is **not** a completed
negative training result. Historical CEM, macro, duration and gradient experiments
remain completed controls, not experiments to relaunch.

This audit concerns the existing BC teacher-to-policy transfer. It does not use
a BC failure as evidence about the cause of ordinary SAC's training failures.

## Missing comparison and fixed protocol

The earlier student started from reset and changed the approach as well as the
recovery behavior. No matched comparison made the student and verified teacher
continue from the same original-policy prefix. Test the following factorial:

- Prefix actor: original BC or the existing recovery-trained BC.
- At decision 120, restore the same snapshot for three continuations: original
  BC, recovery BC, or the unchanged previously selected outward-route teacher.
- One development cell, tshirt_68/14046; two batched copies, seeds 4197/4198;
  two repeats with cyclic slot assignment and reversed assignment on the second
  repeat. Four continuations per prefix/route, **not four independent task draws**.
- Continue through decision 300 without automatic reset. Both prefixes use the
  actors' normal commands. All continuations use the historical teacher's common
  8 mm translation and .05 rad rotation norm caps, with X rotation disabled.
- Success requires minimum coverage over the last 12 decisions >= .7 and
  maximum tracking error over **prefix and continuation** <= .02 m, with no
  simulator error. Retain the separate historical early-turn filter; do not
  conflate it with this criterion.
- Restore the checkpoint's physical contract, including the historical 6 mm arm
  erosion, .1 s control period and original solver tolerances. Do not substitute
  the warm-SAC audit's different geometry.
- No actor/critic updates, new demonstration fitting, CEM, gradient queries,
  route tuning, checkpoint selection or early termination on success.

Predeclared budget: 4,800 physical decisions including both prefixes; 1,500 s
internal deadline and 1,600 s external timeout. Hashes of checkpoints, historical
results and executing source are saved. Reuse `recovery_teacher.branch_world`
with frozen checkpoint routes and repeated assignments. A zero position-restore
error does not prove restoration of every solver cache or deterministic dynamics.

Reproduction (use a fresh output directory):

```bash
env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib \
  timeout --signal=TERM --kill-after=20s 1600s \
  /home/ge47gax/kun/genesis-world/.venv/bin/python scripts/evaluate_recovery_transfer.py \
  --out output/uipc_manip/recovery_transfer_audit_20260920
```

## Interpretation fixed before completion

If the teacher succeeds but the student fails from the original prefix, a changed
approach cannot be the sole cause of failed transfer. If the teacher also succeeds
from the student's prefix, that state remains recoverable by this controller.
Neither result alone identifies observation aliasing, network capacity, inadequate
fitting or accumulated feedback errors. If the teacher fails to reproduce, report
that failed positive control instead of diagnosing student representability.

## Research implications and prior art

The relevant distinction is between improving a controller and transferring its
improvement to the policy that will actually execute. Removing action gradients
only changes one possible improvement mechanism. It does not automatically fix
the transfer step, the observations, or the trajectory distribution.

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses imitation on
states induced by the learner. It is an established control for collecting
corrective labels on the student's own observations, not a new algorithm.
The primary abstract of [DART](https://proceedings.mlr.press/v78/laskey17a.html)
also describes injecting noise into the supervisor to approximate learner errors
and collect recovery experience. Perturbing a successful teacher to widen its
data distribution is therefore another existing baseline, not new on its own.

[End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/html/1504.00702v5)
adapts trajectory-centric teachers and the observation-based policy together.
Its state-distribution agreement is materially different from collecting a fixed
successful trajectory set and fitting it once. Its Gaussian/local-dynamics
assumptions do not automatically apply to our cloth contacts.

[Mirror Descent Guided Policy Search](https://arxiv.org/html/1607.04614), especially
Sections 3--4, explicitly treats policy fitting as an approximate projection and
adapts update size when the global policy cannot reproduce a local controller's
performance. Its guarantees have stated simplified assumptions; they are not a
guarantee for this simulator. Consequently, adding a trust region, rollout check,
teacher adaptation, or imitation loss is already covered by substantial prior
art. None is a sufficient novelty claim for this project.

The current geometric teacher uses named opening/alignment mesh vertices and
internal stage/coverage history. These are not supplied directly to the frozen
single-frame point-cloud actors. That information difference is a concrete
alternative explanation to test, not proof that their observations are
insufficient: the relevant geometry or stage may still be inferable. A proper
history/privileged ablation must preserve physical time and actuator limits;
the earlier confounded state-actor run does not settle it.

The existing proposed query-allocation operator must therefore demonstrate gains
in the **fitted policy's complete episodes per workstation hour**, not merely
better branch scores. Before investing in a new controller optimizer, establish
which transfer error remains in this already-successful recovery. Treat standard
on-policy corrective imitation and guided policy search as comparison methods.

## Completed native results

All 4,800 decisions completed in **682.80 s (11.38 minutes)**, including world
construction, prefixes, restores and artifact writes. The timer starts after
argument/config/checkpoint validation. Original-prefix world: 292.38 s;
student-prefix world: 389.68 s. The small difference from their sum is orchestration
and teardown. Checkpoint training and historical teacher collection are shared
inputs; this timer also excludes the later read-only analysis and research time.
All workers exited; no simulation failures or nonzero recorded
position-restore errors. All continuation commands obey the common caps.

| Prefix through decision 120 | Continuation | Valid sustained successes | Whole-episode valid grasp | Mean sustained coverage |
|---|---|---:|---:|---:|
| Original BC | Original BC | 0/4 | 4/4 | .249131 |
| Original BC | Recovery BC | 0/4 | 2/4 | .338994 |
| Original BC | Existing outward teacher | **4/4** | 4/4 | **.979249** |
| Recovery BC | Original BC | 0/4 | 0/4 | .000000 |
| Recovery BC | Recovery BC | 0/4 | 0/4 | .000000 |
| Recovery BC | Existing outward teacher | **4/4** | 4/4 | **.987735** |

Original prefix coverage is .18463/.18245; student prefix coverage is
.10665/.07613. Both prefixes preserve grasp: maximum tracking errors
5.30--6.41 mm across the four states, below the 20 mm criterion. Teacher
continuations reach sustained coverage .97473--.99243, with maximum tracking
errors 9.35--12.56 mm. **All 24 continuations fail the separate historical
early-turn/paper filter**, including the eight successes under this audit's
coverage/grasp criterion. Do not claim paper-protocol success or real-world
safety from these results.

The unchanged original policy also degrades from its own prefix to the student's
prefix. Approach differences matter, but they cannot be the entire explanation:
the student still fails when given the original prefix, and the teacher succeeds
from every tested student prefix. These particular student-prefix states are
therefore not physically unrecoverable under this teacher and time budget.

The supported result is a **controller-to-policy transfer failure on one cell**.
It does not distinguish insufficient fitting, missing state/history information,
or accumulating feedback errors, and it does not establish ordinary SAC's root
cause. Seeds/repeats are development checks on one garment/body, not population
success estimates or independently trained policies.

Do not restart CEM or collect another route merely because the student failed:
the useful controller already exists here. A future replacement RL update must
show that its improvement survives execution by the learned policy. The next
learning comparison should isolate policy fitting and learner-state correction,
with standard DAgger/GPS controls, before introducing another search optimizer.
This is an experimental requirement, **not a claim of a new algorithm**.

Artifacts are under `output/uipc_manip/recovery_transfer_audit_20260920/`:
`summary.json`, each prefix's `result.json`, prefix observations/commands, and all
24 observation/action/metric episode tapes. Implementation was pushed in
`af120bc6`; execution began from parent `c2ad7921` with the same collector edits,
whose exact hashes are recorded. Runtime uses the existing native Release/CUDA
12.8 build on the RTX PRO 6000 Blackwell workstation. Sampled GPU utilization
was 96%; this is not an occupancy or optimal-throughput claim.

`scripts/analyse_recovery_transfer.py` audits completed artifacts and recomputes
selected actor commands from saved observations. It also separates moving from
stopped teacher-command errors and writes `analysis.json` and `continuations.png`
without new simulation. Run with the interpreter/environment above and
`--root output/uipc_manip/recovery_transfer_audit_20260920`.

## Command fitting on old and newly verified teacher trajectories

Read-only inference evaluates the same frozen recovery actor on the historical
360 training rows and the eight newly successful teacher continuations. No
new fitting occurs. A moving row has nonzero teacher command norm (>1e-6).
Physical errors compare the teacher commands to actor commands after applying
the same continuation caps; MSE uses the five active normalized coordinates.

| Teacher observation set | Moving / total rows | Student raw active MSE, all rows | Translation RMS error on moving rows | Translation RMS error on stopped rows |
|---|---:|---:|---:|---:|
| Historical admitted training trajectories | 239/360 | .012806 | 2.286 mm | .537 mm |
| New teacher continuations from original prefix | 494/720 | .057100 | **5.216 mm** | .465 mm |
| New teacher continuations from student prefix | 383/720 | .046372 | **5.353 mm** | .839 mm |

The existing .01281 training MSE reproduces. Even at the same garment/body,
error on new teacher trajectories is much larger; the moving-command error is
substantial relative to the 8 mm command cap. Low error on stopped actions also
makes all-row MSE understate errors during motion. The old set is not dominated
by stopped rows (121/360), so this does not explain the entire tenfold fitting
improvement. Recovery training improved motion fitting as well.

This supplies direct evidence of a gap between fitting the recorded recovery
and predicting commands along freshly executed successful recoveries. It does
not distinguish conventional generalization error from hidden teacher phase or
other information unavailable in the single-frame input. These teacher-state
errors are not on-policy student-state labels, nor an intervention proving that
reducing this MSE will improve completion.

Artifact checks pass for all 24 continuations: unique route/slot/repeat coverage,
expected 180-decision lengths and true terminal flags, finite recorded outcomes,
whole-prefix grasp accounting, common command bounds, and matching last-12
coverage scores. Recomputing four selected commands per learned continuation
from the correct checkpoint on CPU gives maximum absolute normalized error
**1.15e-6**, validating route wiring against the GPU-executed tapes. Ten existing
focused cap/admission/schedule tests pass. The analysis plot shows means and
min--max bands across the four repeated executions, not confidence intervals.
