# Body-dependent checkpoint rollout failures

The 500-episode job has accepted trajectories on eight of its 16 bodies. Body
14046 is especially reliable, but it is not the only working body. This audit
investigates why some bodies have no accepted episodes without changing the
checkpoint or the running production collector.

## A late stopping condition on body 14053

Among the first 21 production attempts on body 14053, 18 contain a state where:

- the actual cuff and all three mesh sections wrap the arm;
- the proximal sleeve section is at least 90% of the way from elbow to shoulder;
- the held cloth patch has remained within the existing 20 mm tracking limit.

In all 18 cases, the legacy upper-arm score is below the 0.70 stopping threshold
at the first such state. These are transient geometric observations, not 18
successful held episodes. The collector continues pulling because it waits for
the legacy score as well as the sleeve-wrapping check.

Concrete saved example:
`output/uipc_manip/fmvp_dataset_500_20260924/batch_00008_body_14053/body_14053_seed_20260932/baseline.npz`.
At state 241, the proximal sleeve section is at upper-arm fraction 0.909, all
four sections wrap, the legacy score is 0.541, and gripper load is 83.6 N.
The grasp has stayed valid, with maximum tracking error about 19.4 mm.
The first invalid grasp occurs later, at state 263. The original episode
eventually reaches the force cutoff without a valid held completion.

The legacy score in `dressing_reward.wang_progress` intersects an arm-axis ray
with the sparse armhole polygon and takes the minimum valid upper-arm progress.
Its response to a tilted or folded armhole differs from the actual sleeve
sections' positions. It is not a direct measure of where the proximal sleeve
section lies. On body 14046 the two checks usually overlap before tracking is
lost; on 14053 the old stop can be late.

## Limits of that explanation

Many attempts on other bodies stall before getting the sleeve past the elbow.
Across the production attempts inspected, bodies 14050, 14056 and 14060 have no
near-shoulder, fully wrapped state with grasp valid throughout. A new stopping
condition alone cannot rescue those attempts. Pose-dependent pulling direction,
initial placement and contact mechanics still need controlled comparisons;
this audit does not establish which of those causes each early stall.

Nor does shirt size alone explain the failures. Closed mesh sections at the
upper-arm midpoint give circumferences of approximately 0.324 m for body 14046,
0.318 m for 14045, and 0.320 m for 14053. The undeformed shirt cuff circumference
is approximately 0.609 m. Other failed bodies are thicker, but the successful
body is not uniformly thinner than the failed ones. These measurements do not
prove that the garment fits every posture or can pass every contact bottleneck.

Derived reports are saved beside the dataset as `body_fit_diagnostic.json`,
`body_geometry_diagnostic.json`, and `endpoint_diagnostic.json`. The
`upper_radius_p90` field in the geometry report is a rough point projection
contaminated by other body regions; use the closed-section circumferences in
the fit report instead.

## Isolated stopping experiment

`collect_better_rollouts.py --stop-proximal-upper 0.9` replaces the legacy
endpoint threshold only for an explicitly requested diagnostic run. It still
requires all four sleeve sections to wrap throughout a 20-decision (two-second)
hold, the proximal section to remain at or above 0.9, a valid grasp throughout,
and no runtime or load-cutoff failure. Production defaults are unchanged.

The diagnostic at
`output/uipc_manip/fmvp_body14053_physical_stop090_20260924` reuses body 14053,
seed 20260932, placement, three speed variants, material and full-body collision
settings from production batch 8. Comparing the saved environment configuration
files shows no differences. The run metadata changes only the diagnostic stop,
its stated acceptance rule, collector hash and output path.

IPC trajectories can diverge across repeated executions, so this is not an
exact state-restoration counterfactual. Any positive result supports the new
stopping hypothesis but does not establish a multi-body success rate. These
diagnostic episodes use a different endpoint definition and are kept outside
the official 500-episode manifest.

### Result

All three variants retain the wrapped sleeve near the shoulder throughout the
two-second hold. Only quarter speed also passes the unchanged grasp and
deformation checks:

| Variant | Stop state | Minimum proximal fraction in hold | Maximum tracking error | Peak gripper load | Edge p99 / max | Passes all diagnostic checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 240 | 0.909 | 20.95 mm | 161.0 N | 2.384 / 3.381 | no |
| half | 370 | 0.933 | 22.00 mm | 128.3 N | 2.399 / 3.611 | no |
| quarter | 569 | 0.903 | 16.69 mm | 74.1 N | 2.060 / 3.091 | yes |

The independent recomputation is saved in `independent_checks.json` in the
diagnostic run. Baseline and half first exceed the grasp tracking limit before
the stop (states 229 and 338 respectively), so stopping alone does not repair
them. Quarter speed has 589 transitions including the hold and never exceeds
the tracking limit. Its final legacy score is only 0.617. Thus a complete,
held, grasp-valid near-shoulder sleeve state is achievable on this previously
zero-yield body with the unchanged checkpoint and IPC. This is one successful
diagnostic episode, not evidence that all failed bodies are fixed, and not a
pass under the original legacy-score definition.

The next collection improvement to validate is anatomical stopping combined
with reduced pulling speed. Keep original and revised endpoint definitions
explicit in dataset metadata; do not silently merge this run into the existing
manifest. The background 500-episode job remains active with its original rule.

## Why the FMVP paper is not a like-for-like success-rate comparison

Checked against [FMVP Sections 5 and 6, Table 1](https://arxiv.org/html/2509.12741v1).
The PyBullet experiments fine-tune on 204 target-domain trajectories, then
evaluate four sizes of simplified articulated humans, 14 motions and three
garments. Table 1 reports mean upper-arm dressed ratios of 0.63, 0.71, 0.62 and
0.61 (average 0.64). These are continuous coverage measurements, not binary
rates of complete, held, grasp-valid, deformation-filtered episodes.

The current IPC collection is a further transfer of `fmvp_sim.pt`; there has
been no IPC network fine-tuning in this collection route. Its force input is
explicitly zero (`policy_force_input` in run metadata and `WangPolicyClient.act`
called without force). Thus force conditioning cannot respond to measured
IPC resistance. The observation route is `wang_static_arm`, which retains a
pre-captured arm cloud. FMVP Section 5.1 describes observing the unoccluded part
of the arm at each step, but the released PyBullet implementation actually
hides the cloth while rendering its arm cloud (`assistive_gym/dressing.py`,
lines 320–323). Therefore bare-arm visibility itself is not a verified mismatch
to this released simulation checkpoint. Camera geometry and sampling still
differ. Material, grasp constraints, human
geometry (full SMPL-X here), initialization and executed action scale also
differ. These are verified setup differences, not individually proven causes
of each failed body. The local official FMVP README additionally states that
its release is sim-to-sim only and PyBullet garment parameters are not fully
optimized.

Multi-body generalization in the paper therefore remains compatible with
poorer performance in this altered IPC pipeline. The stopping experiment
identifies a concrete collection issue. Determining the remaining early-stall
causes requires matched observation/action/force and contact comparisons; it
would be premature to blame either the checkpoint or IPC alone.
