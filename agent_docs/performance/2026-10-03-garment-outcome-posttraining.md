# A concrete post-training candidate: transfer interaction goals, verify new repairs

Date: 2026-10-03. Status: **prototype and consequence-learning continuation
completed without verified repairs; Discrete Forcing research assessment in
Section 14 proposes an unvalidated next mechanism; its approved matched pilot
is implemented and completed without verified improvement in Section 15**.

The owner subsequently approved implementation and requested parallel launches.
Section 12 records the actual implementation, a failed initial objective and
the revised pilot. Earlier design/budget paragraphs describe the proposal at
the time of the literature review.

The owner asks for a creative, concrete way to post-train the dressing policy.
The recommendation in this note is a candidate for method development, not a
claim that a new algorithm has already been established. All M4 jobs remain
stopped. The initial review used primary literature, source inspection and a
small CPU analysis of existing trajectories. Subsequent authorized simulation
and training are recorded in Sections 12-15.

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

### Completed result

Both early workers also finished. All geometry, generic-goal and TCP-transfer
edited branches exceeded the 0.02 m tracking limit before 24 decisions. Early
geometry branches lasted 18, 20, 23 and 22 decisions. Smooth action noise
produced one valid 24-decision branch in each replay, but both selected replays
plus 450-decision continuations failed to dress: the sleeve was not wrapped
around the arm. Preserving grasp alone was insufficient.

| Root decision | Geometry: valid edits / 4 | Generic goals | TCP transfer | Action noise | Verified dressing repairs |
|---|---:|---:|---:|---:|---:|
| 20 | 0 | 0 | 0 | 0 | 0 |
| 5, adaptive timing diagnostic | 0 | 0 | 0 | 2 | 0 |

Across all runs, including initialization failures, the study charged **1,655
decision equivalents rounded up per worker** (9,906 actual substeps, 1,651
unrounded equivalents), and **1,180.27 summed worker-seconds**. There were
48 search branches and two full verification branches. All workers exited
naturally. The remaining budget was not consumed by further repeats.

A descriptive coverage check used 1,056 source-window starts and 241
`tshirt_4` validation starts under the same retrieval distance. Median and
maximum validation nearest distances were 0.00634 / 0.02596; the four target
roots had nearest distances 0.19766–0.27668. Thus this target lies outside the
sample's validation range in the chosen representation. This does not prove
that coverage caused failure, that more nominal rollouts would solve it, or
that the desired repairs are physically impossible.

**Decision:** this prototype has not demonstrated a useful repair operator or
post-training contribution. Do not scale its nominal-data cloning or count its
goal sensitivity as controllability. No positive actor correction buffer was
created and no r1 weights were updated. Keep the negative single-action result,
trained chunk controls and actual branch transitions for method development.
The missing evidence is a decoder that can physically realize distinct useful
interaction goals; deterministic-policy imitation alone did not establish it.
At the end of this first stage, no GPU or training queue remained running.
The owner's follow-up and subsequent continuation are recorded below.

## 13. Follow-through: learn from distinct actions at a common root

The owner challenged the absence of project jobs after the negative report.
The initial workers had all exited; the visible GPU jobs were another user's
GR00T training and this account's unrelated FoE evaluation. This was a gap in
our follow-through, not a failed launch caused by their occupancy. The next
finite development stage was launched and has now completed.

### New data, with no fabricated success labels

`probe_outcome_repairs.py --collect-only` restores the same root for each of
16 eight-decision action probes, separately in two worlds/replay seeds.
Probes include nominal actions, bounded suppression/reversal and smooth
perturbations. Every correction stays within the decoder's +/-0.5 action
support. Actual observations, actions, tool poses, geometry and the three
observations preceding the root are saved. The causal history is restored for
each branch. The collector adds **256 valid physical transitions**, with
**278 charged decisions** including setup, at a combined 220.21 worker-seconds.
All branches completed their short horizon; none is labelled successful
dressing. These data supply action alternatives missing from deterministic
policy self-cloning.

`train_outcome_goals.py branches` caches those observations and combines them
with the existing 40-episode cache. It trains on the first collection world
and validates on the entire second world, while retaining the original
`tshirt_4` adapter-validation split. Context observations enter history but
their missing geometry is NaN and is never a supervised target. The branch
world split is a local-control diagnostic on the **same body and garment**,
not evidence of generalization to an unseen recipient or garment.

### Learning and execution change

The decoder now learns **one-step inverse control** from the actually reached
next geometry, alongside retained nominal transitions. It receives observed
history and the requested geometric outcome. In the failed first attempt,
every label at a state was exactly r1's own action; now different branches
provide distinct action/outcome pairs. Whether the finite data support useful
inverse control remains an empirical question. The same ordinary supervised
loss is used; it is not claimed as a novel objective.

Two CPU training seeds each fit geometry, generic-feature and no-goal models
for 1,200 updates. Separate counterfactual validation errors and within-domain
goal-shuffling errors are logged. A new data-boundary test verifies that root
context cannot leak into action/goal targets; all six focused tests pass.

The inference bridge supports one- and eight-action models. With
`--one-step-goals`, the physical probe requests the next point on the same
transferred source path and replans after every decision. The source goal bank
remains the original Cloth3D training garments; the target's new data teach
local realization, not successful target-garment goal trajectories.

### Finite active pipeline and cost accounting

`run_outcome_continuation.py` trains both CPU seeds, then launches two physical
probes using the new decoders. It records stage, child PID, command, log and
exit status in:

`output/uipc_manip/outcome_pilot_20261003/counterfactual_pipeline/status.json`

Top-level launch metadata is `launch_continuation.json`; log is
`continuation.log`. Collection is in `counterfactual_seed*/`, combined caches
in `features_counterfactual/`, new models in `models_inverse_*/`, and physical
results in `inverse_seed*/`. A worker error stops the finite stage and is
recorded; there is no endless retry loop or M4 resumption.

Physical costs so far are **1,933 charged decisions**. Each pending probe
gets at most 1,500 further decisions and half a GPU-worker hour. The combined
ceiling is therefore **4,933 charged decisions**, below the original 5,000
budget; actual accumulated times are also checked against eight worker-hours.
No full-r1 update is queued. A physical repair would still need verified
dressing completion before becoming a positive actor label.

### Initial continuation status

Both CPU training seeds have completed. On the local counterfactual validation
transitions, nominal-action MSE is 0.03839. Geometry models reach 0.03023 / 0.02954;
no-goal controls reach 0.03025 / 0.03135. Shuffling geometric goals gives
0.03017 / 0.02915, so the error reduction does **not** establish that the
geometry decoder learned useful goal control. Generic-goal models show a
clearer shuffling effect, but still need physical evidence. Complete metrics
are in `counterfactual_learning_summary.json` under the output root.

The supervisor and both GPU children subsequently completed normally. All
eight method/replay outcomes were negative; two TCP-transfer outcomes had no
valid repair candidate, and six selected continuations failed. Geometry
continuations exceeded the grasp-tracking threshold after 146 / 163 decisions.
Both action-noise continuations and one generic-goal continuation retained
grasp through 474 decisions but did not wrap the sleeve around the arm.
The other generic continuation failed tracking after 31 decisions.

Total cost including the earlier pilots and short collection is **4,212
charged decisions / 2,903.00 summed worker-seconds**. No r1 weights were
updated. The stage is finished, not still training or queued. See the
[compact evidence record](2026-10-03-discrete-forcing-evidence.json).

## 14. Discrete Forcing: useful inspiration and a narrower research hypothesis

The owner supplied [Discrete Forcing, arXiv:2609.39526](https://arxiv.org/pdf/2609.39526).
Its main transferable idea is to make a coarse prediction define the source
of continuous action refinement. It quantizes each normalized action scalar
into 255 bins, predicts the bins, and mixes the dequantized action with noise
before a continuous flow step. Training uses masked-token classification,
flow matching and a one-step reconstruction loss. These are action bins,
not labels for physical contact events. The paper learns from demonstrations;
it does not supply missing successful dressing corrections. Its real-world
74.9% number is task progress, not binary success. These distinctions matter
when proposing an adaptation rather than reproducing its architecture.

### What the current data add

A CPU audit of all 32 new eight-decision probe archives finds all 256
transitions grasp-valid, but **zero changes in any of the four saved section
wrapping flags**. All four flags remain zero throughout. Continuous section
positions do change. Thus the new probes expose action alternatives without
examples of changing this measured garment/body relation. This does not prove
that other contact events are absent or that all action effects are identical.
Paths, hashes and limitations are in the evidence record above.

Combined with the weak goal-shuffling effect and failed physical repairs,
this suggests a specific question: can post-training allocate its action
alternatives by their physically distinct effects, instead of spending its
budget on many variations within the same ineffective behavior? This is a
hypothesis, not a diagnosed cause of every failure.

### Candidate mechanism: learn an action partition from intervention outcomes

Keep the pretrained point-cloud policy and its ordinary replay buffer. At
selected live failure roots, compare bounded action chunks from the same full
snapshot and causal controller history. Represent the observed outcome with:

- the ordered changes in material-section wrapping;
- grasp tracking validity and available geometry-validity checks;
- continuous clearance/opening changes and the time of any transition.

These measurements are training labels. The deployed controller sees causal
point-cloud/tool/action history, not future states or a privileged mesh.
Threshold crossings need hysteresis and repeated checks; they are not perfect
topological invariants or human comfort measurements.

Learn coarse action codes using observed outcome differences. For a fixed
root, action chunks with similar physical outcomes may share a code; chunks
with reliably different events should be separated even when their command
vectors are close. A practical first implementation fits an action-conditioned
outcome classifier, groups outcome distributions into a small codebook, and
trains a history-conditioned proposal decoder for each code. This does not
assume that all states share the same absolute action prototype.

The two levels would be:

    z ~ p_phi(z | observation history)
    A ~ pi_theta(A | observation history, z)

The continuous decoder can use a code-conditioned action anchor plus noise as
its flow source, following the supplied paper. Post-training changes which
codes are proposed and how their actions are realized, using actual same-root
branch outcomes. Fine refinement must be rechecked for the intended physical
effect: proximity in action space is not a guarantee that contact behavior
is preserved.

An implementable update uses (i) outcome-prediction supervision from every
valid branch, including task failures; (ii) code classification and ordinary
conditional flow fitting on action/outcome pairs; and (iii) return-weighted
code selection and actor absorption only when verified continuation data
support an improvement. Soft weights can be formed from paired continuation
returns, w_i proportional to exp((R_i - R_base) / temperature), with equal
information and total rollout budgets across controls. This weighting and
the standard flow loss are established ingredients, not novelty claims. If
all outcomes are equivalent or uncertain, they do not create a positive repair
label merely because a classifier can separate them.

The proposed methodological focus is **an intervention-derived action
partition that preserves relevant physical outcome differences during
post-training**, with more proposals allocated to unresolved useful event
transitions. A learned classifier is fallible; final repair acceptance still
requires real simulator execution. The current 32 branches cannot establish
useful event codes because their recorded wrapping events do not vary.

### Closest prior art and boundaries of the claim

| Primary source | Existing contribution | Implication |
|---|---|---|
| [Discrete Forcing](https://arxiv.org/pdf/2609.39526) | Quantized-action source followed by continuous refinement | The two-stage architecture and source initialization are borrowed. |
| [HyDo](https://arxiv.org/abs/2411.14913) | Hybrid discrete contact selection and continuous diffusion RL | Discrete contact choices plus diffusion alone are not new. |
| [Implicit Contact Diffuser](https://arxiv.org/html/2410.16571) | Contact-relation sequences, learned reachability and MPC for deformable objects | Contact subgoals, variable horizon and relation transfer are already covered. |
| [PDP](https://arxiv.org/html/2606.00336) | Trajectory-geometry-aligned latent behavior control | A searchable behavior space or geometry-conditioned generator is not enough. |
| [CATok project](https://causalactiontokenizer.github.io/) | Ordered action tokens coupled to stages of flow reconstruction | A causal-token name or token intervention inside a decoder is not physical causal-effect learning. The inspected page lists an anonymous submission and a forthcoming PDF. |
| [SA-VLA](https://arxiv.org/abs/2606.30113) | State-conditioned action decoding | State-dependent prototypes alone are not new. |
| [Adaptive discretization](https://papers.nips.cc/paper/2020/hash/285baacbdf8fda1de94b19282acd23e2-Abstract.html) | Adaptive partitions for efficient model-based RL | Adaptive resolution is established; its name cannot carry the contribution. |

EXPO, skill-space repair, goal-conditioned control and action abstraction
remain relevant objections from Section 3. This review does **not** establish
that the complete proposed procedure is novel. Its defensible research target
is whether physically supervised code construction improves correction yield
and policy improvement per simulator query, beyond generic action codes or
existing contact-guided control.

### Smallest next method experiment

Use existing trajectories to identify varied pre-entry and near-contact
contexts, then reconstruct live roots without treating archived deformation
as rest geometry. First establish at least two reproducible achievable effects
at several roots. All branches are charged, including acquisition of examples
with different effects. Simple bounded retreat/reorientation/advance sequences
are candidate probes, not claimed expert labels or guaranteed repairs.

On one fixed data/query budget, compare a flat continuous correction learner,
an action-quantized coarse/continuous learner, and the proposed outcome-based
partition. Match encoder, parameter/update budget, proposal horizon and action
support. A shuffled-effect-label control tests whether the proposed supervision
matters. Split by root and garment, never by frames of the same branch.

Measure distinct reproducible effects per query, success of requested effects,
verified dressing repairs per total simulator decision, and then improvement
of the absorbed actor on held-out garments. Local validity, decoder MSE and
event prediction accuracy are diagnostics, not task completion. If ordinary
action codes or flat corrections do as well, there is no established advantage
for the proposed physical partition. No additional GPU jobs or new flow
training were launched during this paper assessment.

## 15. Approved effect-code implementation and finite study

The owner approved implementation after the paper review. This new study is
separate from the completed 4,212-decision goal-transfer pilot. Its combined
ceiling is **8,000 charged IPC decisions and four summed worker-hours**, with
at most two physical workers at once. Setup, failed branches, nominal
continuations and verification replays all count. M4 remains retired.

### Data and action effects

Three isolated `cn_tcsc_083` roots use bodies 1032 / 3041 for training and
body 2034 exclusively for validation. Archived event locations suggested
roll-in decisions 58 / 32 / 50 respectively; the actual new worlds are fully
reconstructed and rolled in. These roots are development choices, not a
randomly sampled benchmark. Each root has 12 proposal recipes in two replays,
up to 24 decisions each. Random proposal knots differ between replay seeds;
only nominal/suppression/reversal recipes repeat. Do not describe all pairs
as repetitions of identical open-loop commands.

All **72 branch attempts** completed, costing **1,854 charged decisions**.
In the first two roots, 38/48 branches have a raw wrapping-flag change and
36/48 retain a change under a three-state persistence rule. Only 3/48 have a
persistent change in the first eight decisions. This motivated using complete
**24-decision action chunks** in the matched learners and probes; an eight-step
action target would omit most measured changes. The supervisor was reloaded
for this horizon correction while collection continued undisturbed.

Existing canonical features from 40 Cloth3D episodes are retained. Exact-root
action alternatives are explicitly upweighted, with half of training batches
drawn from nominal data and half from valid intervention windows. The new
dataset has **2,641 windows: 2,139 training and 502 validation**, including
45 / 24 complete intervention windows respectively. Three shorter invalid
branches supply no complete 24-step generative target; their raw archives and
failed outcomes are retained. A further **1,033 windows** contain a persistent
wrapping event; overlapping windows are not independent demonstrations.

Validation holds out all `tshirt_4` adapter data and all intervention branches
of body 2034. `tshirt_4` was seen by FMVP pretraining. No `cn_tcsc_top558` data
enter these new heads, including the old 256-transition diagnostic; that
garment is a held-out development case, not a sealed research test.

### Matched learner implementation

`effect_codes.py` defines a 33-dimensional outcome vector: normalized section
center/radius changes, normal changes, persistent net/acquired/lost wrapping
flags, and validity. The first implementation fits an eight-center k-means
partition of measured outcome vectors. This is an initialization from observed
effects, **not a calibrated distribution of causal effects**, an exact topology
representation, or a new clustering algorithm. Generative fitting uses valid
prefixes, so it has no learned grasp-failure class at this stage.

`train_effect_codes.py` fits four heads on identical data, sampled batches,
updates and action support:

1. `flat`: continuous flow from Gaussian noise;
2. `action`: k-means action-chunk codes and a code-conditioned coarse anchor;
3. `effect`: codes learned from measured physical outcomes;
4. `shuffled_effect`: the same effect labels shuffled within training
   bodies/episodes, preserving marginal frequencies without future leakage.

The action-code comparison is a generic learned-quantization control, not a
full reproduction of Discrete Forcing's per-scalar token architecture. The
flow backbone is reused from `uipc_manip.flow_policy`; only its input adapter
changes. Each variant has **285,512 parameters** and receives 1,600 updates
for each of two seeds, 20261005 / 20261006. The observation context contains
four frozen FMVP point-feature/tool frames, three previous executed actions,
and the current nominal action. No outcome, simulator mesh, future frame,
body ID or episode clock enters the actor.

The coded variants mix 0.3 of a learned action anchor with 0.7 noise before
flow refinement. Losses combine flow matching, one-step reconstruction,
anchor fitting and code classification. The continuous-only variant uses
pure noise with the same network shapes. These losses are existing building
blocks; this prototype's research question concerns the physical partition.
At execution all variants use four flow evaluations and the same +/-0.5
correction limit around each contemporaneous r1 action, in the original four
action axes. No candidate is exempt from grasp/geometry checks.

### What offline training establishes

Both seeds have completed. Autonomous validation MSE is approximately
0.0392 / 0.0390 for flat, 0.0360 / 0.0355 for action codes, and
0.0375 / 0.0352 for effect codes. These are action-fitting metrics, not dressing
success. Correct versus shuffled oracle effect codes changes MSE only slightly;
no useful physical effect-control advantage is established by these numbers.

Six focused tests cover persistent events, invalid geometry, causal action
history, online/episode history parity and partition inputs; the six existing
goal/budget tests also pass. A trained-model bridge check has **zero context
error**, exact snapshot/history restoration and finite **24 x 6** action
output. Extending the binary reply size is confined to the optional effect
models; the old eight-action interface remains supported.

### Physical work and next decision

`run_effect_study.py` records collection, preparation, training and simulation
child PIDs/commands/exit codes. The current physical workers use:

- body 2034 / `cn_tcsc_083`, root 50, model seed 20261005;
- body 1032 / `cn_tcsc_top558`, root 5, model seed 20261006.

Within each root all four methods share a full snapshot and causal history,
try nominal plus two proposals, and verify the selected candidate with up to
450 nominal continuation decisions and the existing 20-decision hold rule.
A shared nominal continuation is measured first. All methods use the same
selection score and success test. Model seed and case are coupled in this
small pilot; it cannot isolate training-seed variance or estimate generalization.

Full success beyond a failed paired nominal continuation is required for a
positive repair. The active pipeline does not automatically absorb short
progress, oracle labels or invalid branches into the base actor. If a verified
improvement is found, it supplies the next absorption-stage input; if none is
found, do not claim that the code partition solved post-training.

Output root: `output/uipc_manip/effect_codes_20261003/`. Live status is
`pipeline/status.json`; `collect_body*/`, `data/manifest.json`,
`models_*/summary.json`, `bridge_check.json` and `physical_*/` preserve the
actual evidence. All collection, training and physical workers have now exited
normally; the supervisor reports `completed` and zero positive repairs.

### Completed physical results

| Development case | Nominal r1 | Flat | Action codes | Effect codes | Shuffled effect codes |
| --- | --- | --- | --- | --- | --- |
| `cn_tcsc_083`, body 2034, root 50 | Success | Nominal fallback succeeds | Nominal fallback succeeds | Nominal fallback succeeds | Nominal fallback succeeds |
| `cn_tcsc_top558`, body 1032, root 5 | Tracking invalid after 21 decisions | No valid 24-step proposal | Proposal survives 24 steps; verification fails at 40 | No valid 24-step proposal | No valid 24-step proposal |

In the first case all three candidates tie at short-horizon score 0.05 for
every method; tie-breaking selects nominal candidate 0. The four successes
are therefore **fallback successes, not successes of learned corrections**.
All eight generated candidates remain valid for 24 steps and realize the
same nearest effect class, 7. The effect model requested classes 3 and 7;
requesting class 7 adds nothing beyond the nominal branch, while class 3 is
not realized. This also exposes a selection limit: the common short score
cannot rank the valid alternatives. Their full continuations were not all
evaluated, so the result does not establish that every alternative would fail.

In the second case only one of eight generated proposals remains valid through
24 steps, from ordinary action codes. Its continuation exceeds the tracking
threshold at step 40. Effect-code proposals terminate at steps 20 and 19;
their nearest class is 4, but a class assignment to a truncated invalid branch
is **not successful effect realization**. Grasp validity here means the
documented held-vertex tracking and deformation guards, not physical finger-slip
or human-comfort validation.

The study totals **4,091 charged decisions / 2,036.45 summed worker-seconds**
(1,854 collection; 1,912 first case; 325 second case), below its separate cap.
No actor update occurred. The two cases and coupled training seeds cannot
support a population success-rate claim or establish that the whole method
family is impossible. They do establish no verified advantage for this tested
implementation. Do not enlarge the unchanged training/evaluation campaign.

The concrete limitation is that an outcome cluster labels what happened;
the present objective does not learn which alternative improves on the
nominal policy from the same state. Persistent event coverage increased,
but controllable, beneficial alternatives were not established. Any next
method revision must address that distinction explicitly; more tokens or
lower action MSE do not do so. A possible relative-outcome/preference objective
remains a hypothesis, not an implemented or novel contribution.

The tracked [evidence JSON](2026-10-03-effect-code-evidence.json) preserves
all 72 branch outcomes, matched offline results, physical search/verification
summaries, exact costs, the bridge check and 116 data/model hashes. The finite
supervisor's live-cost aggregation was also corrected to read per-worker
`budget.json` during active simulation; completed accounting already used
the final worker records and was unaffected.
