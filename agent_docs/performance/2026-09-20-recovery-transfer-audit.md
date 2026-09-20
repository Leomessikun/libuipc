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

The existing proposed query-allocation operator must therefore demonstrate gains
in the **fitted policy's complete episodes per workstation hour**, not merely
better branch scores. Before investing in a new controller optimizer, establish
which transfer error remains in this already-successful recovery. Treat standard
on-policy corrective imitation and guided policy search as comparison methods.

## Execution status

The bounded native matrix is running. Preliminary outcomes are not a final
result. `scripts/analyse_recovery_transfer.py` will audit completed artifacts,
recompute selected actor commands from saved observations, and separate teacher
moving-action errors from stopped-action errors without new simulation.
