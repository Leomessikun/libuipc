# Why no expert episode passes the reference filter

Date: 2026-09-13. Closes G14 of `2026-09-10-one-policy-protocol.md`. CPU measurement on the 25
recorded held-out expert episodes of `output/uipc_manip/expert_r13_heldout_s0`; script
`output/uipc_manip/early_turn_probe.py`. No GPU job ran.

Tags: [MI] measured here, [RA] read from the source.

## The question

The expert dresses 11 of 25 held-out cells but 0 of 25 pass the reference trajectory filter
(upper-arm ratio ≥ 0.7 and no early turn). G14 suspected the criterion: ours resolves the signed
cross products in the arm's bend plane (`dressing_reward.py:221`), FMVP's Appendix A.1 in the
horizontal plane of its y-up simulator, ported literally as world XZ in
`newton_isaaclab_tasks/dressing/training/eval_metrics.py:309` [RA]. If the criterion were the
cause, "no filter-passing prior" would be an artifact.

## Method

Each cell's SMPL-X body is regenerated on the CPU from its body id — deterministic, and the
rebuilt forearm length and shoulder coordinates match the recorded privileged arm block in all 25
episodes to 2 mm [MI]. The recorded tool position, stored in the arm frame, is mapped back to the
world through the frame's transpose. The flag is then evaluated per decision under three criteria:
ours; FMVP's intent, the horizontal plane, which in this z-up world is XY; and the Newton port's
literal XZ. Our recomputation reproduces the recorded `early_turn` flag in 25 of 25 episodes [MI].

## Result

| Flag evaluated over | ours | FMVP intent (XY) | Newton literal (XZ) |
|---|---|---|---|
| all 300 decisions | 0/11 pass | 0/11 pass | 2/11 pass |
| before completion (`done` excluded) | 0/11 | 0/11 | 2/11 |
| before completion, after `approach` | 0/11 | 0/11 | 2/11 |
| `elbow_hook` and `last` stages only | 8/11 | 11/11 | 8/11 |

Passes are counted among the 11 dressed episodes (final upper-arm ratio ≥ 0.7). Our criterion
fires on 867 decisions in the `middle` stage, 354 in `done`, 201 in `approach`, 188 in `last`, and
once in `elbow_hook` [MI].

**The criterion is not the cause.** Under FMVP's own intended plane every dressed episode is
flagged, and the two XZ passes come from a formula applied in a vertical plane of our world, not
from a cleaner trajectory.

**The cause is one stage of the scripted expert.** In all 11 dressed episodes the flag first fires
in `middle` (decisions 13–68; 12–48 flagged decisions each) or already in `approach`, and it is
raised while the gripper pulls the sleeve along the forearm into the last quarter of the
finger–elbow segment on the arm's inner side. The elbow hook itself and the final pull are clean:
restricted to `elbow_hook` and `last`, 8 of 11 pass under our criterion and 11 of 11 under FMVP's.
The 354 `done`-stage hits are the gripper resting on the inner side after completion, which our
fixed 300-decision episodes keep scoring and a terminating episode would not; they do not change
the count because every dressed episode is already flagged earlier.

## What follows

- G14 is closed: keep our bend-plane criterion, which is FMVP's test made independent of world
  handedness, and keep `paper_filter_rate` out of checkpoint selection as decided.
- "No filter-passing prior" (`2026-09-13-prior-and-residual-rl.md`) stands, but the defect is now
  located: the `middle` stage's path, and on some bodies the `approach`, run inside the elbow. The
  cheapest route to a filter-passing prior is therefore not a search over all sixteen expert
  parameters but an outward offset of the `middle`-stage target path within the elbow region, with
  `approach` arriving from the outer side. That is a change to `dressing_heuristic.py` and a rerun
  of `expert_baseline` on the same 25 cells; whether success survives the detour is the
  measurement, not assumed here.
- The probe generalises: any recorded run with `--save-observations` can be re-scored under both
  criteria without the simulator.
