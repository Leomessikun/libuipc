# Dressing post-training: EXPO reference and physical branch supervision

## Scope correction

The owner clarified that the research remains **post-training an existing
dressing policy for moving arms and unseen garments**, using the recipe in
[Towards Universal Post-Training for Robotics](https://pd-perry.github.io/posts/post-training.html)
as a reference. The contact-friction intervention proposal is not the selected
next direction. Do not start its mechanism experiments as a prerequisite.

The concrete reference learner is EXPO-style offline-to-online policy improvement:
a pretrained base, a small learned action editor, a critic, online replay, and
supervised updates that absorb improved behavior into the base. Existing dynamic
DAgger remains a necessary baseline and source of initialization data. It is not
already an implementation of this RL loop.

## What the linked work actually supplies

[EXPO](https://arxiv.org/html/2507.07986v3) trains a small editor against Q while
training the base with an imitation objective. It selects among original and
edited actions for behavior and TD targets. The interface is not restricted to
VLAs. [EXPO-FT](https://arxiv.org/html/2605.25477v2) adds chunk-level learning and
human interventions. These are established methods to implement faithfully as
references, not new contributions here.

The current [Real-Time EXPO-FT paper](https://arxiv.org/html/2609.18207v1),
appendix VII-E, explicitly updates the base with flow-matching BC, including
LoRA and vision/action parameters. The older local note saying that its base is
entirely frozen is incorrect for this paper. Its latency handling is also not
our proposed novelty. A small bounded action edit is not a safety guarantee in
contact-rich dressing.

## Existing code and data audit

- `train_dynamic_student.py` regresses selected teacher actions with a trust
  penalty. It has no TD critic, return-maximizing editor, or online RL update.
- At inspection, observed/pass had 20 queries and 240 candidate results;
  observed/lift had 18 queries and 216 candidates. Fourteen candidate branches
  reported simulation errors. These are partial teacher logs, not 456 independent
  trajectories, verified task improvements, or completed student results.
- Candidate logs include actions, endpoint progress, attachment tracking and
  feasibility. They do not contain each branch's successor observation or a full
  reward/terminal sequence. Completed M3 trajectory archives also lack explicit
  reward and termination arrays. They are usable for imitation and short-horizon
  diagnostics, not plug-and-play Bellman replay.
- Current candidates hold the same action for H=4 decisions, while actual control
  replans after one. Their scores must not be mislabeled as one-action Q targets.
- `train_flow_policy.py` uses a four-dimensional action convention and a force
  feature; the dynamic adapter uses six actions and geometry/history. Reusing a
  checkpoint requires an explicit observation/action compatibility audit.
- Prior September results used other policies/protocols. They motivate checking
  edit support and critic ranking; they do not rule out EXPO on the current r1.

## Selected baseline implementation route

1. Start with current r1 as the pretrained base in its existing action interface.
   EXPO does not require converting it into a large diffusion model first.
   Use causal point-cloud history and measured proprioception for the editor
   and critic. Retain the existing flow checkpoint as a later expressive-policy
   comparison once its interface and geometry-only initialization are matched.
2. Initialize replay with valid existing transitions where outcomes can be
   reconstructed reliably. Store new full transitions explicitly. Teacher
   corrections may initialize the editor; all autonomous successes and failures
   train the critic. Invalid-physics branches are not task failure labels.
3. Learn a bounded stochastic editor, sample original and edited proposals, and
   select using twin critics. Tune edit scale on validation tasks; do not assume
   the successful behavior lies within an arbitrary small radius.
4. Execute, append replay, update critics and editor, and update the base with
   the reference method's supervised behavior objective. Track base-only success
   as well as base-plus-editor success to test whether improvement is absorbed.
5. First preserve one-decision execution. If adopting action chunks, use the
   same executed duration and terminal semantics in all comparisons. A chunk
   ending at H uses the corresponding discounted return and gamma-to-H backup.

Use one shared task objective and endpoint across methods: sustained sleeve
completion with valid attachment. Any dense progress reward or explicit failure
cost must be specified once and shared by every baseline. Keep calibrated task
outcomes distinct from the old uncalibrated force-based teacher score.

## Focused research extension: physically calibrated edit advantages

Research question: **Can a limited number of physical comparisons teach a critic
which local policy edits actually help, so online post-training improves more
reliably per simulator hour on new garments and moving humans?**

The hypothesized failure is inaccurate local action ranking under distribution
shift, not a proven universal failure of Q-learning. The editor can otherwise
exploit ranking errors and feed bad actions back into the base. Ordinary RL
must first establish whether this is material in the current task.

At a sampled policy state, fork the base action and one edited action under
normal physics. Use the same garment, complete initial state, human-motion
realization and absolute time. For a one-step critic, execute the differing
first actions, then use the same continuation-policy version for the remaining
H-1 decisions. Do not hold each first action for H steps unless defining a
different macro-action critic. Human future samples are simulation disturbances,
not inputs to the actor/editor/critic; evaluate held-out subjects and sequences.

Let $h$ be observable history, $a_0$ the base action, $a_1$ its edit, and
$\bar V$ a target continuation value under a fixed policy version. A paired
H-step target is

$$
\widehat\Delta_H =
\sum_{k=0}^{H-1}\gamma^k(r_k^1-r_k^0)
+\gamma^H[b_H^1\bar V(h_H^1)-b_H^0\bar V(h_H^0)].
$$

$b_H^j$ is zero after true termination and one for a nonterminal continuation;
absorbing padding is used for branches that terminate before H. Time-limit
semantics must match the finite-horizon task rather than default to bootstrapping.
The tail is estimated. Short physical rollouts do not establish full task value.
Forecast-model branches and actual-environment branches must be distinguished.

Add the supervised relative-value term

$$
\mathcal L_Q = \mathcal L_{TD}
+\lambda_b\,\mathbb E\left[
w\,\ell\left(Q(h,a_1)-Q(h,a_0)-\operatorname{sg}(\widehat\Delta_H)\right)
\right].
$$

Use a robust loss and empirical reliability weights. Measure uncertainty with
repeated continuations; use fresh validation after proposal selection. Shared
human motion does not make the IPC solver deterministic or guarantee variance
reduction. Require normal replay to anchor absolute Q; differences alone leave
state-dependent offsets undetermined.

Keep the EXPO editor and base updates unchanged in the first comparison. This
isolates the proposed critic supervision. Limit physical queries to a declared
budget. Start with fixed random queried states; uncertainty-triggered allocation
is a later ablation, not an extra untested component in the first method.

The formula is a difference of multi-step TD targets, **not new mathematics**.
Short model rollouts improving value estimates have direct precedent in
[Model-Based Value Expansion](https://arxiv.org/abs/1803.00101), and pairing
stochastic rollouts also has prior art, including
[Luck Is Not Skill](https://arxiv.org/abs/2609.24144). The potential contribution
must be a demonstrably better policy-improvement procedure under expensive,
noisy contact simulation, with enough specificity to outperform those controls.
Calling EXPO plus IPC a new algorithm is insufficient. No priority claim is made.

## Three decisive comparisons

Use the same base, history, task splits, objective, edit family and total
simulation/learning budget:

| Method | Purpose |
|---|---|
| Existing dynamic teacher DAgger | Does reward-based policy improvement beat direct corrective imitation? |
| EXPO-style post-training | Does the established recipe already solve the problem? |
| EXPO plus physically calibrated edit advantages | Does the proposed supervision add value? |

Also give EXPO the same extra branch transitions without the relative-value
loss, and compare equal-cost unpaired extra transitions. These controls separate
the proposed supervision from simply purchasing more informative data. A
model-based value-expansion control shares the same H and continuation budget.

First measure held-out action-ranking error, harmful-edit rate, and calibration
against fresh physical continuation outcomes. Then measure full success and
success per simulator/total wall-clock hour. Report three training seeds and
paired task cells; task diversity is not replaced by many repeats of two clips.
Evaluate unseen garments, unseen motions, and both together.

If ordinary EXPO ranks edits correctly and improves reliably, use that result
and reject the supposed critic bottleneck. If the extension only beats DAgger
but matches ordinary EXPO, it is not an algorithm contribution. If the relative
loss is no better than identical extra transitions, it has not earned a novelty
claim. These outcomes refine the post-training method, not the application scope.

## Immediate work and status

The next implementation unit is a shared replay/outcome interface plus a small
EXPO editor/critic learner, followed by paired-branch export with correct
successor states and temporal semantics. The existing M3 queue remains a
teacher/DAgger baseline; its final result is pending. No EXPO dressing learner
or new post-training result was produced during this literature/data audit.
The present commit corrects the plan and identifies the concrete missing data
and learner components. Resource-heavy experiments retain the existing bounded
queue discipline.
