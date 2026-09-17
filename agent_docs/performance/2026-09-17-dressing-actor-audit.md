# Dressing actor gradient and finite-action audit — 2026-09-17

## Scope and decision

The owner approved testing IPC guidance for the SAC actor at elbow failures before
another training run. Offline experiments remain paused. This stage corrects and
audits the diagnostic actor path; it does not replace the batched training signal
with an unvalidated host-factorization routine.

The reference is the preserved-state SAC continuation
`output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt`.
Its saved architecture and environment configuration are retained. The bounded
development audit uses tshirt_26 on bodies 14049 and 14046, saved-policy decisions
180 and 240, seed 1097. The approach recreates physical snapshots with the existing
policy; observation replay alone is not a restorable cloth state. Every snapshot
records coverage, forearm progress, tracking and grasp status so an arbitrary
decision is not silently labelled an elbow snag. These cells are not held-out tests.

## Corrections and remaining approximation

`ObservationCapture.torch_observation` now optionally differentiates the tool:
cloth coordinates are `centroid - tool`, static arm/goal rows shift by `-tool`,
the explicit tool row stays zero, and absolute-tool/relative-goal extras change.
`physics_gradient` adds the direct tool derivative to the six-frame cloth chain.
It retains the cloth-only chain and historical last-frame result for comparison.

An opt-in controller trace records actual translation/rotation acceptance each
substep without changing decisions. The diagnostic chain differentiates clipping
and finite Rodrigues rotations by cheap central differences of the rigid
controller under those fixed masks. It verifies the controller reconstructs the
recorded anchors/offsets. A rejected zero command now has zero sensitivity;
the historical displacement-ratio fallback incorrectly assigned `k/6` at zero.
Snapshot restore also restores each environment's NumPy observation RNG.

This is **not the complete analytic IPC derivative**. The six-frame reverse pass
still uses the projected last-iterate Hessians and inertia coupling. The backend's
lagged coupling exporter currently covers half-plane friction, not cloth–arm mesh
friction. Changing contact, visibility, voxel and controller branches also remain
outside that local analytic model. The continuation surrogate
`V(o_next) = min Q(o_next, mu(o_next))` is not the full SAC objective; no immediate
reward derivative or entropy term is silently claimed. The finite-difference
oracle executes the real environment and therefore includes its actual solver,
controller and observation responses at the tested finite scales.

## Protocol fixed before reading results

`python -m uipc_manip.physics_actor_audit` performs:

1. At each current policy action, compare corrected six-frame value guidance,
   cloth-only chain, historical last-frame guidance, and a six-frame task-proxy
   gradient. The task proxy is the existing opening-axis plus coverage distance
   in metres; its gradient is not a new reward definition for SAC.
2. Full executed finite differences on five live action dimensions at normalized
   steps .05 and .1. Report full observation, frozen discrete observation with
   moving tool, and the historical fixed-tool reconstruction separately. Clipped
   action intervals use their actual span. These slopes are diagnostic finite-scale
   secants, especially near action bounds and contact branch changes.
3. Compare unmodified SAC against six finite first-action corrections: IPC value,
   IPC task proxy, finite-difference value, finite-difference task proxy, SAC's
   learned action gradient, and random direction. Translation/rotation corrections
   each have normalized norm .5 before action clipping. All then follow the same
   saved SAC for 12 decisions. The finite-difference directions use the last
   configured epsilon (.1), not the best outcome selected retrospectively.
4. Execute every candidate and SAC twice from the same snapshot, shuffle order
   and reverse it on the second round. A candidate is accepted only when **both**
   coverage gains exceed `max(.01, 2*abs(reference_1-reference_2))`, with valid
   grasp throughout both candidate and reference continuations. Axis, return,
   controller rejections and accepted anchor motion are secondary diagnostics.
   These repetitions are not independent task samples. Position restore error
   must be at most 10 micrometres, and horizons must precede automatic reset.

Do not start policy training merely because one candidate passes. Analytic value
guidance should agree directionally with executed differences across both scales
(cosine at least .9), and improve valid-grasp coverage repeatably on more than one
snapshot, with evidence stronger than the random control. Otherwise retain the
negative or mixed result and identify which derivative/objective gate failed.

## Reproduction and status

The first four snapshots turned out to be later partial-dressing stalls, with
upper-arm coverage .47–.58, rather than the elbow crossing itself. After inspecting
the approach traces, two earlier development states were added: body 14049 at
decision 80 and body 14046 at decision 60. Each supplemental run uses two
finite-difference draws per epsilon (`--fd-repeats 2`) to distinguish numerical
variability from step-size dependence. Candidate radii, continuation horizon and
acceptance thresholds are unchanged. State selection used the approach phase,
not candidate improvement. Recreated cold-run coverage may differ from the
original approach; the actual initial measurements are retained in each result.

```bash
env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.physics_actor_audit \
  --checkpoint output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt \
  --out output/uipc_manip/dressing_actor_audit_20260917
```

The supplemental commands add `--cells tshirt_26:14049 --steps 80 --fd-repeats 2`
or `--cells tshirt_26:14046 --steps 60 --fd-repeats 2`, with output names
`dressing_actor_audit_20260917_elbow_14049` and
`dressing_actor_audit_20260917_elbow_14046`, respectively.

38 focused CPU checks pass, including tool-motion cancellation, static arm/goal
and extra derivatives, finite-rotation control with partial rejection, rejected
zero commands, clipping-aware finite differences, and grasp/confirmation gates.
Three native environment tests were deselected by the initial CPU test profile.
The actual native audit below exercised the modified controller trace and checked
its reconstructed anchors/offsets in every gradient query.

## Completed results

All three jobs finished: **1,794 simulator decisions, 404.92 seconds** in total,
including world setup and policy approaches. The first four-state audit used
1,236 decisions / 205.31 s; the earlier 14049 and 14046 cases used 289 / 74.70 s
and 269 / 124.91 s. No policy weights, critic weights or replay buffers were
trained. Artifacts are each run's `result.json` plus
`output/uipc_manip/dressing_actor_audit_20260917/summary.json`.

Cosines below compare the analytic direction with **full executed** finite
differences at normalized epsilon .05 / .1 (1 is aligned, 0 orthogonal).

| Body / decision | Initial coverage | IPC value cosine | IPC task-proxy cosine | SAC coverage after 12 decisions, two executions |
|---|---:|---|---|---|
| 14049 / 180 | .4871 | .846 / .153 | .862 / .975 | .48491 / .48488 |
| 14049 / 240 | .4700 | -.005 / .979 | .989 / .994 | .44771 / .44748 |
| 14046 / 180 | .5605 | .532 / .761 | .988 / .999 | .56576 / .56586 |
| 14046 / 240 | .5752 | .651 / .553 | .972 / .996 | .57283 / .57345 |
| 14049 / 80 | .1242 | .572 / .813 | .815 / .808 | .20211 / .19929 |
| 14046 / 60 | .0478 | -.115 / -.022 | .902 / .929 | .25473 / .18764 |

**Every proposal type passes 0/6 state gates**: IPC value, IPC task proxy,
finite-difference value, finite-difference task proxy, SAC action gradient and
random. All 84 reference/candidate continuations preserved grasp; none completed
dressing in the tested horizon. This result is failure to establish a sufficiently
large, repeatable local benefit under the preset protocol, not proof that IPC,
gradients or sequence optimization cannot help.

The four later-state tests did show small effects: mean coverage gains over SAC
were .00230 for IPC value, .00328 for IPC task, .00334 for finite-difference value,
.00282 for finite-difference task, -.00065 for SAC gradient, and .00067 for random.
No individual proposal exceeded the .01 minimum gain in both repetitions there.
On earlier 14049, IPC value gained .00638 / .00616, while finite-difference value
gained .01726 / .00705 and failed confirmation at the preset threshold.

The earlier **14046 reference is unreliable**: its repeated final coverages
differ by .06709 despite restored positions and observation RNG. Its gate therefore
required .13418 coverage gain. IPC value gains of .01412 / .06206 cannot be treated
as reliable improvement. Repeated finite-difference directions in that state also
vary: full-value repeat cosine .540 / .605 and task-proxy repeat cosine .698 / .803.
The 14049 elbow value slopes repeat much better (.972 / .988), yet differ across
epsilon (cosine .665). These are distinct limitations: numerical/execution
variability and finite-scale dependence. Matching restored positions alone does
not certify all hidden solver state or subsequent numerical repeatability. The
current test does not isolate the source of that discrepancy.

All recorded continuation restore errors satisfy the 10-micrometre bound; all
controller rejection counts lie within [0, 6]. The historical last-frame value
direction is sometimes closer to finite differences than the corrected chain,
and sometimes worse. Correcting an omitted mathematical path does not establish
that the remaining approximate model gives a better practical direction.

## Consequence for further work

Do not scale this value-gradient actor term or resume policy training on the basis
of this audit. No state clears the value-direction accuracy threshold at both
scales, and no proposal clears the task-improvement gate. Geometry-based task
derivatives are encouraging in several later states but do not yet demonstrate
useful elbow recovery or a stronger policy. Any next comparison at the early
14046 elbow must first establish repeatability of complete restored trajectories;
changing only the RL loss would leave that measurement problem unresolved.

This is not evidence that the critic alone causes failure. The learned value's
local geometry, discrete encoder/observation choices, omitted dynamics derivatives
and measured numerical variability have not been independently isolated. The
fixed single-action, 12-decision test also says nothing decisive about optimizing
coordinated multi-action recovery sequences. The production batched actor signal
remains the historical last-frame surrogate; this diagnostic is not silently
installed as a complete analytic training derivative.

## GPU status during the audit

While the owner asked about a cool GPU, `nvidia-smi` reported 98% GPU activity,
2% memory-controller activity, 6,866 MiB allocated, 48 C and 156.16 W. The audit
process used about one CPU core. The workload contains sequential IPC decisions,
small policy calls and host sparse-LU gradient calculations, not large-batch
policy optimization. These instantaneous counters do not establish occupancy,
peak-compute use or which component limits elapsed time. After the jobs finished,
their CUDA contexts disappeared; only unrelated desktop/MPS processes remained.
