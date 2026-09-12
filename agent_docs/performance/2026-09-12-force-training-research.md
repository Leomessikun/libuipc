# Force-aware dressing training and the elbow bottleneck

> **Status update, 2026-09-13:** This is a historical audit/proposal against `4bfe88c2`, not the current implementation status. Subsequent commit `024c5716` closed the proposed per-decision force-training direction after the reliability gate failed; see the [closure](2026-09-12-research-direction.md) and [measurements](2026-09-12-contact-force-calibration.md). No force-training result is established here. The dense action-per-point critic was subsequently implemented in `ef1b3c81`, with optional residual trunks in `01bf913e`. Any future experiment must use those changes as an explicit baseline and revalidate its labels. Literature findings remain reference material; implementation recommendations below are conditional.

Force can enter standard learning objectives mathematically, but the current dressing labels have not passed the reliability gate required by this proposal. The immediate
engineering objective is reliable elbow passage with less snagging. A separate research objective
is to establish whether spatial contact information adds useful information beyond strong visual
and aggregate-force baselines. Reimplementing force-conditioned policies or teleoperation plus
diffusion is not a new contribution.

This report reflects source and local-artifact review on 2026-09-12 at base `4bfe88c2`. No new
policy training or force-aware success result is claimed. The companion
[integration audit](2026-09-12-force-training-integration.md) gives code locations and replay/reset
requirements; the [methods review](2026-09-12-force-training-methods.md) covers six neighboring
learning/control approaches. The [novelty audit](2026-09-12-force-training-novelty.md) identifies direct competitors, and the [intervention specification](2026-09-12-elbow-experiment-design.md) bounds the first pilot. A Chinese reader report is delivered separately outside the English-only
repository at `/home/ge47gax/kun/force-training-research/force-training-zh.md`.

## Evidence about the elbow

The [archived raw-evidence summary](2026-09-12-elbow-evidence.json) contains hashes of input files
and all 27 evaluation rows. A zero result with simulator errors is not an error-free policy test.
The actual teacher uses gamma 0.995 and reward scale 0.5, not the 0.25 seen in mocked tests.

| Existing artifact | Result | Interpretation |
|---|---|---|
| Expert, region 13, 25 held-out cells | 11 final successes; 15 ever reach upper ratio 0.7 | Some trajectories lose progress after passing the elbow |
| Same expert | 8 end in elbow_hook; all 5 tshirt_392 cases among them | Specific garment/path bottleneck, not proof of physical impossibility |
| Same expert | 24 early-turn flags; zero paper-filter yield | Blind filtered BC has no accepted demonstrations in this dataset |
| Teacher, 125016 admitted transitions | Mean final upper 0.282891; 1/25 success/filter; no sim errors | Best error-free mean in this run |
| Teacher, 174264 transitions | 3/25 success; zero paper-filter yield; no sim errors | Some learned elbow passage, but not robust filtered completion |
| Teacher, 265584 transitions | Zero mean; 25 sim errors | Invalid as a pure policy-capability measurement |

Source-code mechanisms are more specific than “RL needs more training.” Near-body no-move can
reject translation, and the grasp tether can reject both translation and rotation. The expert
already has a lifting correction for the elbow. Absolute forearm-progress reward can remain positive
while a garment stalls; selected ray intersections can change abruptly during rotation. The actor
has no contact-force/history input. These mechanisms motivate matched diagnostics; their relative
causal contributions to the failed learned trajectories remain unmeasured.

Early-turn flags are geometric proxies, not independently measured human collisions. Likewise,
expert stage `done`, maximum progress, final progress and paper-filter success must not be merged.
The expert manifest also serializes a watchdog-enabled base config even though world construction
disables it; this provenance inconsistency must be fixed before treating serialized config as the
exact executed configuration.

## What prior work already covers

| Prior work | Established method | Consequence for this project |
|---|---|---|
| Clegg et al., RA-L2020 [1] | Contact-force penalties and a low-to-high penalty curriculum | Force-cost curriculum is an implementation baseline, not novelty |
| Deep Haptic MPC, 2018 [2] | Action-conditioned future distributed body-force maps and elbow/catch avoidance | “Predict spatial force to recover at the elbow” is already occupied broadly |
| FCVP, RA-L2024 [3] | Visual policy with a learned force-dynamics constraint | A force-based action filter alone is insufficient differentiation |
| FMVP, CoRL2025 [4] | Force-modulated visual features and offline-RL fine-tuning | Adding force to PointNet++ or using FiLM is already demonstrated |
| TacSL/AACD, 2024 [5] | Privileged low-dimensional learning and pretrained-critic transfer to high-dimensional actors | A teacher/critic architecture is a reusable learning tool |
| FACTR, 2025 [6] | Visual corruption curriculum for force-attending imitation | Naive force concatenation can be ignored; this is different from penalty curriculum |
| Dressing in Motion, September2026 preprint [7] | Static demonstrations, diffusion policy and visual arm-motion adaptation | Demonstration-based dressing and visual recovery are current baselines |

Deep Haptic MPC is particularly relevant: its whole-arm study reports success1.25%,97.5%,98.75%
for prediction horizons0.01,0.05,0.2s, with80trials per condition. Its objective includes forward
progress, and longer-horizon actions rotate around the elbow. That is evidence for action-dependent
contact consequences, not a performance forecast or an instruction to copy its time constants.

Neighboring TACTIC uses distributed robot-arm tactile sensing and predictive control; DPTG uses
tactile feasibility guidance for diffusion actions. Those methods further rule out claiming that
spatial contact, contact-dependent recovery, or adding force guidance is itself new [8,9].
Hybrid Control Strategies also explicitly studies force-triggered snag recovery in dressing [11]. Neither
a lack of keyword search matches nor use of a different simulator establishes publication novelty.

## An immediately useful training experiment

Keep the deployment actor observation unchanged initially. Use force in the reward, as an optional
privileged critic input, and later as an auxiliary action-conditioned prediction target. These uses
do not require differentiating through IPC or having a deployed sensor for every simulated body
vertex. A real-time force-input actor is a separate experiment with a sensor/estimator contract.

Run independent ablations from the same verified warm start and data curriculum:

- A: existing task reward and non-force privileged geometry baseline.
- B: A plus force summaries in the critic only.
- C: A plus a small fixed normalized force cost.
- D: both additions.
- E: D plus history/action-conditioned contact prediction, only if earlier evidence justifies it.

For the first cost experiment, a fixed objective and fresh, compatible replay are simpler than a
changing penalty. If lambda changes, preserve decomposed task rewards/costs and relabel at sampling;
old scalar-reward buffers cannot be reweighted correctly. Apply the existing reward multiplier once
to the combined objective. Do not describe exact resume as actor-only warm start.

For physical substeps j in a decision of duration DeltaT, a candidate dimensionless cost is
`mean_cost = sum_j(dt_j * max(L_j/F_ref - 1, 0)^2) / DeltaT`; report the peak separately.
`F_ref` is a frozen training-data scale, not a clinical threshold. Selected `L` definitions must
remain identical across comparisons. Body net force can cancel; nodal norm sums and peaks are
mesh-dependent. Begin with verified body/region model-load summaries, and reserve shear/pressure
claims for their own validation. Numerical failures or absent exporters must never become zero costs.

A high force penalty can teach contact avoidance. Clegg's random-initialized strong-penalty policy
failed completely, whereas curriculum refinement reduced forces [1]. Preserve task progress and
completion, measure success-conditioned force statistics, and retain failure/rollback trajectories
for value/prediction learning rather than imitating their actions indiscriminately.

## A research hypothesis worth testing before a major training run

A narrower, falsifiable target is whether the **location and direction of distributed garment–body
loads identify different effective recovery actions when aggregate force observations are similar**.
This is a proposed information-value experiment, not an established first contribution. IPC supplies
an instrument for controlled data collection; its name, force units and collision handling are not
the paper's result.

Use the same physical states, candidate actions, horizon and objective for: non-force state,
strong aggregate statistics and body net wrench, correctly localized force maps, and location-shuffled
maps that preserve the force vectors/totals. Include a privileged non-force geometry/control-state
baseline so oracle force is not credited for unrelated hidden state. Predict future progress and load
jointly, not only which action minimizes force. A body net wrench is not a measured wrist wrench;
claiming comparison with real F/T requires a separate gripper reaction model/measurement.

First restore and branch complete physical/controller states. A positions-only archive is inadequate:
velocities, internal state, anchors, offsets, targets, episode/history state and random generators must
match. Current snapshots restore whole worlds through disk; begin with one environment, establish
same-action repeat variability and restore invariants, then branch. Compare policy-selected action
regret against the best action in the identical finite candidate set, not an imaginary optimal action.

If spatial information gives no stable advantage over net wrench and equal-capacity nonspatial
baselines, stop the spatial-force contribution claim. If its advantage disappears under timestep,
solver tolerance or modest mesh changes, investigate discretization before training against it. If an
oracle force policy improves but deployable observations cannot recover the information, report that
observability gap and test history/sensing; do not deploy the oracle actor with zero-filled inputs.

Potential publishable outcomes require evidence beyond the baseline: a reproducible characterization
of aggregate-force ambiguity during dressing, a validated method exploiting the information to
recover, and controlled gains over relevant modern policies on held-out garments/body poses. Physical
claims additionally require independent real force/contact validation. No such result exists yet.

## Decision gates and engineering order

1. Re-evaluate a previously successful checkpoint with valid full episodes. Separate invalid evaluation
   from policy degradation; freeze resolved physics and scoring.
2. Trace expert and actor around the elbow: requested action, accepted target, actual motion,
   no-move/tether reasons, sleeve frame, reward selection and contact summaries.
3. Validate force labels and their action rankings against stricter numerical controls. The particle
   regression established cached zero-friction failure; it did not validate dynamic cloth forces.
4. Establish complete-state branch reproducibility, then run spatial-information probes. These may
   falsify the proposed direction before substantial RL expenditure.
5. Train A–D on verified reachable training cells, then expand the curriculum. Keep hard cells in
   full held-out evaluation; do not select them away based on outcomes.
6. Compare at least three training seeds for any final claim, with both matched admitted transitions
   and wall cost. Report conditional elbow passage, final/peak progress, strict filter, force tails,
   rejection rates, sim errors, and label coverage. No fixed throughput estimate is warranted yet.

IsaacIPC's rendering/physical mesh separation can improve student observation generation, while its
pressure validation motivates better label testing. It is a later integration option, not a proven
way to solve the elbow or accelerate this training pipeline [10].

## Sources

1. Clegg et al. *Learning to Collaborate from Simulation for Robot-Assisted Dressing*. RA-L2020,
   SectionsIII-D,V-C. [Original source](https://arxiv.org/html/1909.06682v2)
2. Erickson et al. *Deep Haptic Model Predictive Control for Robot-Assisted Dressing*. 2018,
   SectionsIII–V,TableI. [Original source](https://arxiv.org/html/1709.09735v2)
3. Sun et al. *Force-Constrained Visual Policy: Safe Robot-Assisted Dressing via Multi-Modal Sensing*.
   RA-L2024. [Original source](https://par.nsf.gov/servlets/purl/10573301)
4. Hao et al. *Force-Modulated Visual Policy for Robot-Assisted Dressing with Arm Motions*.
   CoRL2025,Sections5–6. [Original source](https://arxiv.org/html/2509.12741v1)
5. Akinola et al. *TacSL: A Library for Visuotactile Sensor Simulation and Learning*. 2024.
   [Original source](https://arxiv.org/html/2408.06506v2)
6. Liu et al. *FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning*. 2025.
   [Original source](https://arxiv.org/html/2502.17432v2)
7. Sun et al. *Dressing in Motion: A Human Motion-Aware Diffusion Policy for Robot-Assisted Dressing*.
   Preprint posted September4,2026. [Original source](https://arxiv.org/html/2609.04759v1)
8. Madan et al. *TACTIC: Tactile and Vision Conditioned Contact-Centric Control for Whole-Arm
   Manipulation*. RSS2026. [Original source](https://emprise.cs.cornell.edu/tactic/)
9. Liu et al. *DPTG: diffusion policy with tactile feasibility guidance*. Frontiers in Robotics and
   AI,June9,2026. [Original source](https://doi.org/10.3389/frobt.2026.1851102)
10. Liang and Han. *IsaacIPC: Coupling High-Fidelity Simulation and Realistic Rendering for
    Contact-Rich Robotic Systems*. Technical report2026. [Original source](https://arxiv.org/html/2605.24339v1)

11. Rafiq et al. *Hybrid Control Strategies for Safe and Adaptive Robot-Assisted Dressing*.
    Preprint2025. [Original source](https://arxiv.org/html/2505.07710v1)

Further primary papers on asymmetric learning, force estimation, auxiliary prediction, cumulative
cost constraints and admittance are linked with deployment requirements in the methods appendix.
Local code claims are anchored in the integration appendix; measured force results are in the
[preceding audit](2026-09-12-force-learning-audit.md). The excerpts above are bounded primary-source
summaries; experimental designs and recommendations are this project's proposals.
