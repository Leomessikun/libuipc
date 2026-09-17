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

```bash
env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.physics_actor_audit \
  --checkpoint output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt \
  --out output/uipc_manip/dressing_actor_audit_20260917
```

17 focused CPU checks pass, including tool-motion cancellation, static arm/goal
and extra derivatives, finite-rotation control with partial rejection, rejected
zero commands, clipping-aware finite differences, and grasp/confirmation gates.
Three native environment tests were deselected by the default CPU test profile.
The native four-state audit is in progress; no learning result is claimed.
