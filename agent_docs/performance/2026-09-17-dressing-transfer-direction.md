# Research pivot: transferring sleeve motion across contact dynamics

Status: research recommendation, not an implemented method or a positive training
result. The owner has abandoned IPC/SAC correction and requested a new direction
after the ordinary SAC audit. Literature checked on 2026-09-17. No training,
native rollout, solver modification or new experiment was launched for this note.

## Recommendation and empirical reason

Investigate **learning desired sleeve motion from existing trajectories, with a
controller that adapts its commands to the target physics**. The research target
is reducing the target-domain experience needed for robust dressing. A new generic
RL update should no longer be the prerequisite for a useful contribution.

The [ordinary SAC audit](2026-09-17-normal-sac-rollout-audit.md) establishes two
failure patterns: partial dressing followed by a plateau, and displacement of the
opening away from the upper arm. In one captured rollout, lateral opening offset
grows from 25.5 to 226.9 mm between decisions 120 and 180. Another fails with valid
grasp throughout. These are executed-motion failures; they do not identify the
dominant learning cause, prove a friction snag, or prove that IPC is unsuitable.

The audited SAC baseline never consumed the expert trajectory folder. Its source
run has 408 complete episodes across 225 planned configurations and scarce logged
success. Network updates account for only about 12.6% of its recorded cumulative
environment/evaluation/update/rebuild time. That supports reusing successful
experience and reducing repeated simulation, rather than expecting faster neural
updates alone to fix training. It does not prove that imitation will generalize.

The relevant physical distinction is between a desired cloth outcome and the
command that produces it. Sleeve geometry relative to the arm describes the
task. The response to a gripper command also depends on grasp compliance, material
properties, friction and the current deformation. Source-domain actions may
therefore be poor target-domain labels even when both domains can dress the arm.
Whether useful sleeve-motion sequences transfer is the hypothesis to test, not
an assumed physical invariance. Different contact dynamics can require different
paths as well as different commands.

## Alternatives and the closest prior art

| Direction | Relevant published work | Decision for this project |
|---|---|---|
| Direct visual action-sequence imitation | [Dressing in Motion](https://arxiv.org/html/2609.04759v1) learns a diffusion policy and adapts trajectories using observed arm motion. Its real and simulated policies use separate datasets and training, explicitly without sim-to-real transfer. | Strong practical baseline. Replacing SAC with diffusion or adding arm-relative features is not a new contribution. |
| Learned opening dynamics and predictive control | [Garment Diffusion Models for Robot-Assisted Dressing](https://www.imperial.ac.uk/personal-robotics/publications/?id=1420028&noscript=noscript&respub-t4-action=citation.html) predicts future opening geometry from partial clouds and actions, using it in model-based RL/MPC. | Very close prior art for a garment world model. A new simulator backend would not establish novelty; model error and planning cost remain. |
| Desired cloth configurations and learned execution | [TAX3D](https://arxiv.org/html/2410.19247v2) learns deformable relative placement through dense displacement. Its simulated evaluation predicts gripper targets at the initial state and applies a PD controller. [ArticuBot](https://arxiv.org/html/2503.03045v2) learns hierarchical end-effector goals and a goal-conditioned execution policy. | Best fit for existing-data reuse. Investigate continuous sleeve threading and transfer across contact dynamics; neither goal prediction nor hierarchy itself is new. |
| Change how the garment is held | [Bimanual Robot-Assisted Dressing](https://arxiv.org/html/2508.12274v1) addresses tight garments with two arms and body-relative demonstration encoding. | Relevant if current grasp mechanics limit reachable outcomes. Existing successful native trajectories prevent assuming that a second arm is necessary. Hardware and data requirements make this a separate project. |

Additional novelty boundaries matter. [ImaginationPolicy](https://arxiv.org/html/2509.20841v1)
already represents manipulation using moving oriented keypoints, including local
patches of deformable objects. [Latent Space Alignment](https://arxiv.org/html/2406.01968v1)
already transfers manipulation policies using a shared latent space and
domain-specific encoders/decoders. [Delehelle et al.](https://arxiv.org/html/2601.21713v1)
already separate geometric policy learning from perception and demonstrate
cross-modality transfer for cloth flattening. Generic keypoints, modularity,
latent alignment or privileged-to-visual learning are insufficient novelty claims.

The Garment Diffusion comparison is based on the authors' institution-hosted
abstract; its full text was not accessible in this review. Do not infer omitted
implementation details or assign its headline success figure to a particular
real/simulation protocol. The other linked arXiv methods above were inspected in
full-text HTML. Different papers' success rates are not directly comparable to
our coverage-and-grasp metric. This search narrows the claim; it is not an
exhaustive novelty certification.

## Concrete candidate

Train a policy to predict a short sequence of desired sleeve configurations
relative to the arm, then execute it with feedback. For example, a target could
describe how the opening should change position and orientation as it passes the
elbow. Pulling the gripper farther is useful only if the observed sleeve responds
appropriately. A target sequence must permit backing up and changing alignment;
imposing monotonically increasing progress would exclude possible recoveries.

Use a compact goal sequence initially: opening center, oriented normal and shape
summary in an arm frame. The existing
[`dressing_privileged.py`](../../python/uipc_manip/dressing_privileged.py) already
defines these quantities. This coordinate system is existing infrastructure,
not a proposed invention. Full contour information may later be necessary:
center/normal/radius cannot determine whether a folded sleeve encloses an arm.
The existing ray/winding labels are geometric proxies, not guarantees of physical
threading or complete cloth state.

The minimal factorization is

$$
g_t \sim H_\theta(\cdot\mid h_t), \qquad
a_t \sim C_{\psi,d}(\cdot\mid h_t,g_t).
$$

Here $h_t$ is recent observations and executed commands, $g_t$ is a short future
sleeve-geometry sequence, $a_t$ is the existing gripper command, and $d$ identifies
the execution domain. $H_\theta$ learns useful geometry sequences from successful
trajectories. $C_{\psi,d}$ learns how commands produce those outcomes in its own
domain. The controller executes one decision, observes the cloth response, and
updates its plan. Selecting a target never certifies that it is reachable.

Training uses recorded future geometry as the goal label and recorded actions
as the executor label. Failures can supply short-term action/outcome pairs when
their state contract is valid; they should not automatically label desirable
long-term goals. Avoid averaging incompatible motion routes; compare against an
action-sequence baseline with comparable temporal and distributional capacity.
This is initially supervised sequence learning, not offline value learning or a
claim to have solved offline-to-online RL.

The intended transfer experiment shares geometry-level training examples while
keeping incompatible action labels in their respective domains. Goal generation
must remain conditioned on current geometry; copying a fixed source trajectory
is not the proposal. A source-trained goal that the target cannot track is a
failure of the proposed factorization, not a reason to quietly add a runtime IPC
optimizer. Likewise, good fitting under recorded goals does not establish that
the executor handles its own predicted goals during rollout.

## Existing-data feasibility and real deployment

The [pretraining audit](2026-09-17-expert-policy-pretraining.md) found substantial
existing collections. Its small native BC experiment covered one training body;
its negative generalization result is not a test of the full trajectory corpus.
It is also not an explanation of SAC's failures.

- Native expert recordings contain actions and 35D pre-action geometry. The
  cached reconstruction contains observations as well: 3,600 transitions from
  six source trajectories replayed twice. Replays are not independent strategies.
  These data can label short future center/normal/shape goals now. Full opening
  contours were not stored in the original expert tapes.
- A read-only schema inspection in this research pass confirmed that
  `newton/artifacts/dressing_physics_v5_demos_tshirt26_h7_n36/replay_buffer.pt`
  contains 12,857 transitions, 36 episode records, 2,005D observations and 6D
  actions. It uses x-ray observations and a kinematic cuff grasp. There is no
  explicit privileged-geometry or full-mesh array in that buffer. Its force
  arrays are marked contact proxies. Source labels and dynamics must not be
  treated as native IPC or measured real-robot quantities.
- Locate corresponding geometry-rich recordings before attempting source-goal
  pooling. If only the inspected buffer exists for a sequence, reconstructing
  geometry requires an explicit conversion or replay, whose cost and changed
  outcomes must be recorded. Padding 2,005D observations to 5,383D is not a valid
  conversion. Different timestamps, units, ring orientation, arm frames,
  grasp/control contracts and success definitions also need explicit alignment.

No new trajectory collection is justified merely by choosing this direction.
First establish which existing records contain the needed labels. Training on
cached sequences can use GPU batches without IPC calls per update. The previous
3,000-update BC loop took 86.8 s, but observation reconstruction alone cost
318.63 s; neither figure predicts this candidate's total training time.

Deployment should run camera observations and robot proprioception through the
learned policy to the current command interface. Exact simulator geometry is a
training label or an explicitly marked diagnostic input, never hidden runtime
information. Severe occlusion can make sleeve state uncertain; a short history
does not automatically resolve it. Real perception and target-controller
calibration remain necessary experimental questions. This is an autonomous
sensor-to-command pipeline with structured intermediate supervision, not a
claim that a jointly end-to-end-trained real policy already exists.

The [force-modulated dressing paper](https://proceedings.mlr.press/v305/hao25b.html)
provides a relevant deployment comparison: it fine-tunes a simulated visual
policy with limited real vision/force data. Our unsuccessful force-input trials
do not refute that result. The proposed first experiment does not require adding
force inputs, and it makes no zero-shot real-world guarantee.

## One bounded experiment and decision rule

First test the control target on compatible native data. Compare an
action-sequence imitation policy with the sleeve-goal/executor policy using the
same trajectories, observation history, controller, learning budget and complete
evaluation protocol. Give both comparable model capacity; SAC is an external
reference, not the only baseline. Otherwise a gain could simply come from using
demonstrations, memory or more parameters.

Use two training seeds and a preregistered matrix of eight garment/body cells,
with two full 300-decision evaluation rounds per seed and method: 64 episodes,
19,200 native decisions maximum. Include the documented failure configurations
as development cases, and reserve additional bodies before fitting; group all
replays and source versions of a body into one split. This is a feasibility
screen, not a definitive generalization estimate. Freeze the cell list, checkpoint
selection, available data and numeric time budget before running it. Do not
extend training because the first outcome is disappointing.

Primary evidence is sustained completion with whole-episode valid grasp. Report
the historical paper filter separately, plus per-cell/seed results, geometry
traces, stalls, loss of progress and total data-preparation/training/evaluation
time. Training loss or peak coverage cannot decide whether to continue. Saved
best and final SAC checkpoints can be evaluated on that same matrix if an SAC
comparison is reported; the 127k plain checkpoint is not the strongest known SAC
architecture by default.

Continue to the transfer study only if the goal policy improves full-episode
success on withheld configurations across both seeds, without worse grasp
validity or a materially larger total budget. Mixed results are inconclusive;
they do not authorize a new long run. If accurately predicted goals still fail
to produce useful cloth motion, the proposed controller is not yet a solution.

If that screen passes, keep the target executor and its data budget fixed and
compare the same goal learner trained on native-only versus native-plus-source
geometry sequences. This isolates whether otherwise incompatible trajectories
provide transferable information. Before claiming algorithmic novelty, compare
against established hierarchy/latent-transfer and action-adapter baselines,
then demonstrate reduced real-robot adaptation cost. A sim-to-sim gain alone
does not establish sim-to-real transfer. Further threading tasks, such as cloth
on a bent fixture or a bag handle on a hook, would test the same mechanism;
unrelated locomotion benchmarks would not.

The prospective contribution is therefore specific: **which representations of
deformable motion can be reused across contact dynamics, and how much target
experience that reuse saves while preserving successful threading**. Merely
combining the modules above is established engineering. No current result
establishes this contribution; the proposal earns further work only through the
stated comparison.

## Repository state

Based on `7cfa6608` on `research/ipc-adjoint-q-learning`. This stage changes
research documentation only. Read-only source/data inspection and literature
review completed; no policy learning or native evaluation. GPU process inspection
shows desktop applications and the CUDA MPS service, not a training worker.
Pre-existing untracked build, output and worktree files remain untouched.
