# A concrete post-training candidate: transfer interaction goals, verify new repairs

Date: 2026-10-03. Status: **prototype implemented; first bounded physical
probe found no repairs; earlier-intervention diagnostic running**.

The owner subsequently approved implementation and requested parallel launches.
Section 12 records the actual implementation, a failed initial objective and
the revised pilot. Earlier design/budget paragraphs describe the proposal at
the time of the literature review.

The owner asks for a creative, concrete way to post-train the dressing policy.
The recommendation in this note is a candidate for method development, not a
claim that a new algorithm has already been established. All M4 jobs remain
stopped. This work used primary literature, source inspection and a small CPU
analysis of existing trajectories; it launched no simulation or training.

## 1. Recommendation

Develop **post-training through garment-relative outcome edits**:

1. Extract short garment–body interaction changes from existing rollouts.
2. Learn a controller that can attempt those changes from a new observation.
3. At a policy failure, search over short sequences of interaction goals.
4. Realize each proposed sequence with the actual target garment in IPC.
5. Use verified improvements to update the pretrained policy.

A concrete example is temporarily moving a sleeve away from the wrist,
changing its opening orientation, then advancing it. This is an illustrative
behavior to discover, not a diagnosis of current failures or a hand-written
expert trajectory.

The proposed technical focus is **transferring local interaction goals across
garments while re-solving their realization under the target dynamics**.
Copying a successful robot trajectory is generally insufficient when the
garment shape, deformation or body motion changes. Whether this representation
actually makes useful repairs easier to discover is the central hypothesis.

This is a more specific research question than increasing the number of labels:
can existing imperfect experience provide transferable *repair objectives* that
help a policy acquire behavior absent from its successful full-task rollouts?

The method is task structured. Its potential contribution is an efficient
repair synthesis and post-training procedure, not a new generic RL objective.
It should not be advertised as established algorithmic novelty yet.

## 2. What the repository actually establishes

- The current motion planner's `evaluate` function repeats the **same action**
  for H decisions. H=4 does not search an arbitrary four-action sequence.
  Its proposals are the nominal action, scaled/zero/rotation-disabled variants,
  six coordinate perturbations and two arm-follow variants.
- Its objective combines terminal arm progress with force and action penalties.
  This can improve local decisions, but it does not directly measure whether
  a temporary retreat enables subsequent completion.
- Existing v5 archives contain observations, executed and policy actions,
  cloth positions, sleeve measurements and failure information. Failed
  episodes are already retained; there is no need to recollect them merely
  to learn local dynamics or goal-conditioned behavior.
- Archived positions and observations are not complete restorable IPC worlds.
  Search needs a live snapshot, or a fully validated reconstruction/replay.
- The stopped M4 pipeline left two completed successful initialization
  trajectories. There is no trained dynamic student or held-out student result.
- We have not established that an appropriately implemented EXPO learner fails
  on r1. EXPO is a required comparison, not a method already disproved here.
- Current grasp validity means held-vertex target tracking within the configured
  tolerance. It is not evidence of physical finger slip or a human comfort limit.

### New CPU-only descriptive audit

A fixed sample of eight baseline archives from each of five v5 garment
directories gives 40 episodes, of which 23 have the collector's success label.
Before their recorded first-success state, 22 of those 23 have a decrease
greater than 0.01 in the saved proximal upper-arm fraction; the median maximum
drawdown is 0.1424. All 17 failed episodes also have such a decrease.

The [audit records](2026-10-03-existing-progress-audit.json) include exact paths,
selection seed, metrics and limitations. These are dimensionless descriptor
changes, not metres. Success labels were read, not independently re-certified.

**Interpretation:** successful recorded progress is often non-monotonic.
**Not established:** a retreat caused success, that the descriptor is free of
geometric discontinuities, that this is the planner's exact score, or that
longer sequence edits outperform local edits. This evidence motivates a
sequence-level hypothesis; it does not prove it.

## 3. Primary-source map and strongest objections

| Work | Established mechanism | Implication for this candidate |
|---|---|---|
| [EXPO](https://arxiv.org/html/2507.07986v3), [EXPO-FT](https://arxiv.org/html/2605.25477v2) | Learned action edits, value-based selection and supervised updates to an expressive base policy | The outer edit-and-absorb loop is prior art. Compare identical action horizons, data and budgets. |
| [HIQL](https://arxiv.org/abs/2307.11949), [HILP](https://seohong.me/projects/hilp/) | Subgoal policy extraction and reusable behaviors from offline experience | Goal conditioning, hierarchical policies and mining imperfect data are not novel. |
| [RecoveryChaining](https://arxiv.org/html/2410.13979v1) | Learn recovery and when to hand control to existing task controllers, using continuation outcomes | A verified handoff or success-basin objective alone is not novel. |
| [Skill-Space Shooting](https://arxiv.org/html/2609.38178v1) | Try learned skill corrections, verify repairs and train the task policy on accepted segments | Autonomous repair-and-retrain is directly covered. Its seeded skill demonstrations make automatic transfer from existing rollout segments a relevant distinction to investigate. |
| [D-Cubed](https://applied-ai-lab.github.io/D-cubed/) | Learned skill latents and diffusion-based trajectory optimization for deformable manipulation | Low-dimensional sequence search and diffusion are not new contributions. |
| [IRP](https://irp.cs.columbia.edu/) | Learn how action changes alter deformable-object trajectories | Learning action effects or residual dynamics is established. |
| [DIDP](https://arxiv.org/abs/2505.17434) | Reduced-order inverse dynamics and physics-informed adaptation for dynamic deformable manipulation | An inverse model plus a physical state representation is insufficient novelty. |
| [DeformGen](https://arxiv.org/html/2606.25939v1) | Physically generate initial deformations, warp source trajectories and validate executions | State augmentation and deformable trajectory transfer already exist. The candidate must improve failure-specific, multi-step interaction repairs under target dynamics. |
| [Garment Diffusion Models](https://spiral.imperial.ac.uk/entities/publication/a0176f43-5eb7-4530-93d8-35c0118514df) | Predict garment-opening dynamics from partial point clouds and actions; use iterative model-based control | Garment dynamics, partial observations, model-based dressing and novel-garment transfer are direct prior art. |
| [Contact Mode Guided Planning](https://arxiv.org/abs/2105.14431) | Search contact transitions and continuous motions | Contact sequences or physical mode search alone are established. |
| [Topology-coordinate dressing RL](https://doi.org/10.1080/01691864.2013.777012) | Use garment–body topological relationships for dressing control | Sleeve topology is not itself a new representation claim. |
| [Hi-WM](https://hi-wm.github.io/), [WISE](https://arxiv.org/html/2609.03681v1) | Corrective world-model branches and subsequent policy improvement | Replacing a world model with IPC does not establish novelty. |
| [FAR](https://arxiv.org/html/2607.01111v1) | Failure-aware adaptation, retry exploration and continual improvement | Learning from failures or successful retries is also established. |
| [Full FMVP](https://arxiv.org/html/2509.12741v1), [Dressing in Motion](https://arxiv.org/abs/2609.04759) | Adaptive dressing with moving arms using distinct sensing/training recipes | Dynamic dressing is not a new task; `fmvp_sim` is not the full FMVP system. |

Search also examined time-reversal self-supervision, reverse curricula,
trajectory stitching, SMC policy improvement and morphology continuation.
They provide possible tools, not independent novelty claims for this project.

The most serious reviewer objection is:
**“This is goal-conditioned hierarchical control, with garment features and
a simulation verifier.”** Merely implementing the five steps above does not
defeat that objection. The work needs a specific, useful transport operator
and evidence that it transfers repairs better than generic latent goals,
trajectory warping and ordinary action-space post-training.

## 4. Candidate selection

| Observation -> mechanism hypothesis | Candidate intervention | Falsifiable prediction | Decision |
|---|---|---|---|
| Local labels are weak -> useful correction requires a sequence | Longer unconstrained action search | Equal-cost sequence search finds more successful repairs | Essential simple control, not sufficient novelty |
| Successful segments recur across garments -> their interaction goals transfer better than their actions | Garment-relative outcome edits with target-dynamics realization | Transfer improves repair yield on a garment whose successful episodes were excluded | Recommended candidate |
| Goal information is sufficient -> ordinary latent subgoals already solve transfer | HIQL-style learned goal abstraction | It matches the structured representation at equal cost | Would remove the proposed representation advantage |
| Failure states need known recovery skills | Skill library + verification | Existing skills repair them without new synthesis | Strong prior-art baseline; do not rename it |
| Reverse motion produces new successes | Undressing/reverse curriculum | Forward executions succeed after physically valid reversal | Not selected: reversal and curricula are established; friction and goal initialization add unresolved costs |
| Rare successful paths are the bottleneck | SMC/tree search and policy absorption | Better success per total IPC step | Not selected as novelty: SPO/TRT-SMC and branching post-training already cover the generic mechanism |
| Partial observation prevents useful correction | Belief estimation or probing | Same histories need different actions, and information gathering resolves this | Possible, but not diagnosed in our data |
| Physics discretization dominates cost | Multi-fidelity policy improvement | Coarse queries preserve action rankings and reduce total cost | Possible later; no validated cheap model or bias control currently exists |

## 5. Proposed algorithm in implementable terms

### 5.1 State and information contract

Let s_t contain cloth positions/velocities, body state, robot/controller state
and all persistent simulator state. The actor sees only causal history

h_t = (o_(t-L+1:t), a_(t-L+1:t-1), robot proprioception).

Garment identity/mesh g and full simulated geometry can supervise training
targets. The deployed actor must not require full meshes, hidden body joints
or future GRAB frames. Human motion is an environment input, not an action
the optimizer may alter to make dressing easier.

### 5.2 The transportable object is a path of interaction constraints

Construct an initial garment-relative descriptor from ordered material sleeve
sections and the finger–elbow–shoulder curve:

- normalized position along that curve;
- perpendicular ring-centre offset and ring normal;
- opening shape/clearance statistics;
- whether interior sleeve sections surround the arm;
- distances from held material to the opening and section boundaries.

Existing `SleeveSections` and `ring_on_arm` supply part of this geometry.
The descriptor is **not claimed to be Markov sufficient**. Full point-cloud
history and target-garment shape remain inputs to the realization model.

Do not match raw vertex indices across meshes. Sleeve sections must be assigned
a consistent order from cuff to armhole, with shape/scale retained. A jacket
whose valid sleeve correspondence cannot be established is unsupported,
not silently assigned a shirt template.

For a source segment, extract a short path of target regions C_1,...,C_J
in these coordinates. A region is a tolerance set, not an exact target mesh.
For example, the path can first decrease arm-axis progress while increasing
clearance, then rotate the opening, then cross the wrist region.

**Proposed transport operator:** align the source's current sleeve–arm relation
with the target's current relation; transfer *changes and event order* in
normalized section coordinates; instantiate target clearances using the target
opening/body scale. Preserve the target's own cloth dynamics throughout.

This operator needs implementation and validation. It does not establish
physical reachability. No cloth vertices are teleported to satisfy a goal.

### 5.3 Learn how to attempt a goal from existing experience

Use windows from successful and failed episodes, stopping at simulation errors
or invalid grasp. Relabel a window by its actually achieved interaction path:

D_goal = {(h_t, C_(t:t+H), a_t)}.

Train a small goal-conditioned residual controller initialized around r1:

a_t = clip(pi_base(h_t) + e_psi(h_t, C_(t:t+H))).

Its supervised initialization loss is

L_goal = E[ ||pi_base(h_t) + e_psi(h_t,C) - a_t||^2 ].

This is ordinary hindsight goal conditioning; it is not a novel loss.
A stochastic head can retain multiple valid realizations. Flow matching can
replace this supervised loss if needed, using an existing flow policy, but
changing the backbone is not the research contribution.

The implemented pilot encodes observations each decision, predicts eight-action
chunks and replans every eight decisions. It executes the distinct actions in
each chunk. Single-action hindsight imitation collapsed to the original policy
on its own deterministic data (Section 12), motivating this bounded change.
Past failed windows teach recorded local changes; they are not labelled
successful dressing or globally good actions.

### 5.4 Search and physically realize repairs

From a full live snapshot near an encountered difficulty:

1. Retrieve compatible source interaction paths and include a no-edit option.
2. Transfer their goal regions to the target garment/body.
3. Let the goal-conditioned controller attempt each path in IPC.
4. Optionally refine a small number of goal coordinates/durations; reject
   physically invalid paths.
5. Hand control back to a frozen version of the base policy and measure its
   continuation.

Use final dressing success as the authoritative outcome. A short branch that
merely advances the sleeve is an unverified proposal. During search, a learned
continuation value may prioritize candidates; promising repairs require an
actual continuation before entering the positive correction buffer.

For a candidate controller/goal path c of duration ell, the target is

Q_k(s,c) = P(final valid dressing success | c for ell decisions,
             then frozen policy pi_k),

and choose high-value candidates with a duration/query cost penalty.
This is a standard recovery/option value target, not a new Bellman equation.

Include ordinary action-sequence proposals in the same search budget.
Otherwise a comparison would conflate longer horizons with the proposed
transport mechanism. Every reset, rejected branch and suffix execution counts.

### 5.5 Update the pretrained policy

Retain all executed transitions for value/goal-model training. Put verified
beneficial repair segments into a separate actor correction buffer. Fine-tune
the existing actor with

L_actor = L_imitation(D_repair) + beta L_retention(D_old_success).

For the flow variant, substitute the standard conditional flow-matching loss.
Use learner roll-ins in subsequent rounds, adding repairs where the updated
policy actually fails. Absorption and aggregation are established machinery.

A nominal-versus-repair continuation comparison estimates benefit. A single
successful noisy rollout is insufficient to certify a probability improvement;
retain outcome counts/uncertainty and avoid declaring a deterministic oracle.
The training data must record policy versions and continuation semantics.

## 6. Why this could produce more than the current teacher

The existing teacher proposes nearby individual actions and holds each briefly.
The candidate can attempt a sequence whose early motion has low immediate
progress but changes the sleeve configuration for a later insertion. Its
proposal source includes local transitions from incomplete episodes and other
garments, rather than only successful complete target-garment trajectories.

This does **not** imply EXPO or a sufficiently long action-space planner cannot
discover such behavior. The intended benefit is finding transferable useful
repairs with fewer expensive physical trials. That is an empirical efficiency
claim, not an expressivity theorem.

A source segment need not be successful dressing to initialize the decoder.
However, source coverage still matters: if the existing data contains no useful
local transition and all goal realizations fail, the algorithm has no magic
source of expertise. Targeted exploration or additional demonstrations would
then be required.

## 7. Boundaries of the intended contribution

| Level | Current status |
|---|---|
| Application | Dress unseen garments and moving arms; both have substantial prior work |
| Engineering | Connect full IPC snapshots, sequence queries and existing datasets |
| Empirical | Test whether transferable interaction changes yield missing repairs |
| Algorithm | Candidate transport-and-realize edit operator for post-training |
| Theory | None claimed |

The publishable claim, if supported, would be:

> Transferring verified interaction-goal paths across garments supplies
> corrective experience beyond a pretrained policy's target-garment successes,
> and improves autonomous post-training efficiency over equally budgeted
> action edits, generic latent goals and trajectory warping.

A successful end-to-end demonstration alone is insufficient. If a generic
goal controller or longer action search matches it, report that result and
withdraw the transport novelty claim. No venue acceptance is implied.

## 8. Implementation sequence and the next three experiments

These are a proposed bounded development study, **not launched jobs**.
Do not resume the cancelled M4 queue.

### First: reuse and relabel existing trajectories

Implement an offline window extractor and the ordered section-coordinate
transport interface. Train the small goal controller from existing data.
Use whole garments as validation units. This tests goal realization, not yet
dressing superiority.

Compare transferred interaction paths against raw robot-action transfer and
generic latent goals. Report actual prediction/realization errors separately;
a low offline imitation loss is not a control result.

Deliverable: a reusable goal-conditioned correction proposal initialized from
data we already own. This directly builds the proposed method.

### Second: one capped repair-discovery study

The immediate feasibility probe is **one training failure state**, four
proposal methods and two physical replay seeds. Allocate at most 96 repair
decisions and one 450-decision verification suffix per method/seed, for 4,368
decisions before setup. Apply a global cap of 5,000 IPC decisions or eight
GPU-hours, whichever is reached first, including setup/replay costs. A cap-hit
unfinished continuation is censored, not a task failure. This is a proposed
development budget; no job has been launched. The larger panel below is an
expansion design, not an automatic next queue.

Use 12 valid training failure states across three garments, four states each.
Use two predeclared physical replay seeds per state. At each state, compare:

- unstructured temporally varying action edits;
- transferred source end-effector trajectories;
- generic learned subgoal proposals;
- the proposed garment-relative interaction paths.

Share the same goal decoder where applicable, terminal rule, nominal
continuation, maximum repair duration and total IPC budget. As an initial
development allocation, allow at most 192 repair decision transitions per
method/state, including failed branches, and at most two 450-decision
verification suffixes. Snapshot regeneration and replay checks are additional
measured costs and must be charged equally, not treated as free.

Nominally this caps repair/suffix decisions at
12 x 2 x 4 x (192 + 2 x 450) = 104,832, before setup.
Do not call this a one-day test: old teacher runtime shows that such a budget
can still be expensive. Begin with a single state to measure cost and enforce
an independently fixed wall-clock cap before any expansion.

Outcome: accepted repairs per total IPC second/decision, plus validity and
source-to-target transfer. Seeds on the same state are repeated measurements,
not independent garments. This study produces training data; it is not another
repeat of the r1/GICP comparison.

### Third: one post-training study from the same repaired dataset

Use three training seeds for a pilot. Compare the fine-tuned base against:
same-data ordinary supervised correction, same-budget action-space EXPO,
and the method without cross-garment goal transport.

Separate two questions: whether the new data helps, and whether the algorithm
produces that data more efficiently. Same-data BC may absorb our repairs just
as well; that would support a data-generation contribution, not a novel actor
optimizer.

Evaluate once on a frozen held-out set of whole garments and body/motion
sequences. Use paired cells, seed-specific effects and cluster bootstrap by
garment/motion source; a 12-state development pilot cannot support a broad
significance claim. Expand only after the new-method result warrants it.

## 9. Dynamic-arm extension and deployment

Start with garment transfer in a static arm setting to identify the transport
mechanism. Then express interaction targets relative to the currently observed
arm and condition the controller on history. This retains the original
garments x human-motion agenda without trying to learn both shifts at once.

GRAB supplies exogenous motion during physics execution. A causal teacher may
use privileged past/current state during data generation, but real future
frames must not determine deployable actions. Match histories and test whether
corrections can be represented from the available observations.

A dynamic comparison must include observed registration correction and the
full sensing/training context of FMVP. A RealSense camera supplies observations;
it does not replace the need for a robot, valid grasp control and a deployment
study.

## 10. Risks and stop conditions

1. Geometric descriptors alias different contact states: retain richer history;
   if transferable goals remain ambiguous, this representation is inadequate.
2. Target goal paths are unreachable with the fixed grasp/action space: do not
   hide this by moving the human or changing the grasp during verification.
3. Generic latent goals or action search match the method: no demonstrated
   algorithmic advantage for the proposed transport.
4. Accepted repairs cost more than the data efficiency they save: no efficiency
   claim; count suffix and failed-query costs.
5. Gains disappear after policy absorption: planner competence alone is not
   evidence of successful post-training.
6. Unseen-garment gains vanish or depend on target success data: the intended
   transfer claim fails.
7. Force/tracking thresholds are simulator checks, not validated human safety
   guarantees.
8. Neither this report nor a literature search proves absence of equivalent
   prior art. The strongest open novelty threat is the combination of HIQL,
   deformable trajectory transfer and skill-space repair.

## 11. Practical decision

Continue method development only around a concrete new capability and a
bounded cost question. The current proposal makes existing data useful for
learning repair realizations and explicitly targets missing target-garment
corrections. It is worth a small implementation study, with the stated novelty
and feasibility risks. It is not grounds to restart a large collection queue
or to claim that the paper contribution is already finished.

## 12. Implemented pilot and initial evidence

The owner approved this candidate and then requested more concurrent project
work. Implementation is on `research/expo-ft-dressing`:

- `scripts/wang_transfer/outcome_goals.py`: ordered material coordinates,
  goal transport, tool-relative goal inputs and a residual chunk decoder.
- `train_outcome_goals.py`: valid-prefix extraction, whole-garment split and
  matched geometry/generic/no-goal CPU training.
- `outcome_policy.py`: CPU inference with explicitly restored causal history.
- `probe_outcome_repairs.py`: full IPC snapshots, three proposals per method,
  selected replay and physical continuation under a strict budget.
- `test_outcome_goals.py`: five checks covering rigid-pose invariance,
  stationary transport, valid-prefix boundaries, unsupported correspondence
  and charging setup/partial decisions against the budget.

### Data and the rejected single-action initialization

The fixed 40-episode v5 sample supplies 11,156 valid decisions. The source
policy is r1. Failed episodes contribute only valid prefixes; first invalid
transitions and successful completion holds are excluded. Four garments
(32 episodes) train the adapter; `tshirt_4` (eight episodes) is reserved for
validation. It was seen by FMVP pretraining, so this is only an adapter split.
No ClothesNet episode trains the proposal models or enters source retrieval.

The first implementation predicted a residual for the next action. Its
zero-residual validation MSE was **1.36e-14**: the saved actions are already
the deterministic base outputs under the matched encoder/action convention.
Changing the goal changed actions by only **8.30e-8 RMS** after fitting.
This objective provides no reason to learn a correction. More identical-policy
trajectories do not resolve that degeneracy. These models are retained as a
negative diagnostic and are not used for physical experiments.

The revised proposal predicts eight distinct actions from current history and
a three-waypoint future goal path, then executes the chunk and replans. Its
targets are the recorded eight-action subsequences. The four allowed action
components remain translation xyz and world-z rotation; corrections are bounded
to +/-0.5 around the current nominal action. This is ordinary conditional
sequence learning, not a new objective. It still needs physical evidence that
changing the requested outcome produces the requested interaction.

### Fixed CPU result, one seed and 1,200 updates per model

There are 8,529 training windows and 1,950 validation windows. These overlap
within episodes; they are not independent statistical samples.

| Proposal | Held-out action-chunk MSE | MSE after shuffling goals |
|---|---:|---:|
| Repeat current r1 action | 0.017141 | n/a |
| Geometric interaction goals | 0.014780 | 0.016881 |
| Generic frozen-encoder goals | 0.013655 | 0.016885 |
| History with no goal | 0.014121 | n/a |

The geometry decoder responds to goals (action RMS change 0.05466), but its
offline prediction is worse than both generic goals and history-only. This
does **not** support an advantage for the proposed representation. The physical
probe tests goal realization/repair directly; it cannot be replaced by these
imitation metrics. [Exact CPU results](2026-10-03-outcome-pilot-cpu.json).

### Material correspondence and runtime checks

The descriptor uses cuff, exact .25 and .5 sleeve cuts, and armhole, each with
arm-normalized center, oriented normal, radius mean/deviation and winding flag.
The earlier fifth ring was removed because the existing extractor's final
interior fraction varies between .75 and .55. Cached point features were reused
with a documented coordinate selection; no encoder retraining was needed.

The first two IPC starts exposed another fallback: on `cn_tcsc_top558`, the
historical success extractor chooses .4 instead of .5 because its point-count
ratio is 62/41, slightly above 1.5. Those starts exited before any repair and
each consumed 32 physical substeps (six charged decisions). The probe now
constructs exact closed goal sections independently, verifies they remain
near the sleeve, and retains the original sections for the success rule.
Unsupported geometry is checked before creating a GPU world. This is explicit
goal correspondence, not reassignment of .4 to .5.

The archived initial anchor offsets recover the scaled, rotated original hang
with maximum error **1.80e-16 m**. The new world re-settles that original rest
geometry; archived deformed positions are never used as a new rest mesh.
The single-world replay is not claimed to reproduce a historical batched
trajectory. CPU bridge checks found exactly matching cached features and
exact history restoration. All five coordinate/boundary tests passed.

### Physical work launched

Target: training garment `cn_tcsc_top558`, body 1032, root after 20 nominal
decisions. Two separate workers use replay seeds 20261003/20261004; each
compares geometry goals, generic feature goals, source TCP-path transfer and
smooth action perturbations. These are simple proposal controls, not a full
EXPO or HIQL implementation. Within each worker, all methods restore the same
full root and controller history. Two reconstructed replays are not independent
garments, bodies or tasks.

Each method tries no edit and two proposals for up to 24 decisions, then
replays the selected proposal and permits up to 450 continuation decisions.
Search ranks valid terminal arm progress; only a completed physical dressing
continuation can populate a future positive repair buffer. This inexpensive
ranking is a pilot limitation, not an outcome-value estimator. Every rejected
branch, selected replay and initialization is charged.

Each corrected worker is capped at 2,494 decisions and 3.99 hours. Together
with the two initial setup failures this stays within **5,000 decisions and
eight summed GPU-worker hours**. Budget exhaustion censors an unfinished
continuation. No full policy update or broader queue is automatic.

Artifacts: `output/uipc_manip/outcome_pilot_20261003/`. Models used are in
`models_canonical/`; `features_canonical/manifest.json` preserves source paths,
hashes and splits. `launch_canonical.json`, per-worker logs, `budget.json`,
`branches.jsonl`, `results.json` and `status.json` record the physical runs.
The machine's system NVIDIA userspace library differed from the loaded kernel;
only these workers use an existing matching 595.84 library directory via
`LD_LIBRARY_PATH`. No system driver or unrelated process was modified.

### First physical result and one bounded diagnostic

Both root-20 workers completed. **All 24 branches failed the 0.02 m grasp
tracking criterion before completing 24 decisions**, including the eight
no-edit repeats and four edited proposals from each of four methods. Geometry
proposals failed after five decisions in all four attempts. No continuation
was eligible, no positive repair was found, and r1 was not updated. These are
development failures, not independent task-level success estimates.

The two roots were valid (tracking 7.77/4.95 mm) and unthreaded; the failure
does not prove that a late root is unrecoverable. To test intervention timing,
two additional workers start from nominal decision **5** with the same target,
models, methods and seeds. Their archives additionally retain achieved
interaction descriptors and goal-position/normal errors. This diagnostic is
adaptive and must not be pooled into a preregistered benchmark.

All previous costs are deducted. The early workers have respectively
**2,401 / 2,398** decisions left, and approximately 3.962 / 3.968 worker-hours.
The combined ceiling including the two initial setup failures and root-20
runs remains 5,000 decisions / eight worker-hours. No expansion beyond this
diagnostic is queued. Read `launch_early.json` and `ipc_seed*_early/`, plus the
[compact physical record](2026-10-03-outcome-pilot-ipc.json).
