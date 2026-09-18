# Research direction after reviewing Berkeley CS 185/285

Date: 2026-09-18. Status: literature and repository assessment; proposed method
and experiments, not an implemented algorithm or a new policy result.
See the [complete course reading record](2026-09-18-cs285-reading-review.md)
for coverage and limitations.

## Recommendation

Continue the objective of a shared, efficient dressing policy. Close the current
motion-encoder initialization recipe. Investigate one narrower question:

**Can simulator interventions teach a policy which recovery action to choose
from its available observation history, with less total training time than
ordinary RL using the same prior data?**

The proposed focus is learning decisions under partially observed deformable
contact. IPC supplies training outcomes; a learned policy acts at deployment.
This is a plausible research direction, not an established novel algorithm.
History models, privileged critics, action comparisons, simulator resets and
offline-to-online learning all have substantial prior art. A paper would need
a distinct mechanism and evidence beyond assembling those components.

The immediate deliverable should be an action-selection diagnosis on existing
ordinary-SAC failures. Do not start a larger motion model, another long unchanged
training run, or a broad algorithm sweep before that diagnosis has a signal.

## What the project evidence actually says

IPC and SAC are not competing algorithms: IPC supplies physical transitions;
SAC supplies learning updates. An IPC advantage requires comparing simulators
under a controlled learner and cost budget. An algorithm advantage requires
comparing learners under a controlled environment, data and task contract.
The existing experiments establish neither comparison comprehensively.

| Evidence | Supported conclusion | Not established |
|---|---|---|
| Ordinary SAC checkpoint 127,416: eight saved episodes, four configurations, no successful episodes; stalls and lost sleeve progress | There are physical policy failures worth diagnosing; the audit was not testing an IPC-gradient correction | The cause is necessarily contact-force input, partial observability, critic error, or inaccurate physics |
| FQL at 30,000 offline updates: 5/8 coverage-plus-grasp passes, comprising 4/4 training-body and 1/4 withheld-body episodes | Existing compatible data can support useful behavior in some evaluated cells | Robustness, online adaptation, a 40k result, or a controlled improvement over the older SAC run |
| Matched small motion pilot: random / geometry / motion initialization = 2/8 / 1/8 / 0/8 passes | No observed benefit from this actor-only initialization recipe | A general rejection of dynamics pretraining or a statistically settled method ranking |
| Motion prediction improves globally, but held-out elbow error is 5.20 mm versus stationary prediction 2.45 mm | Whole-cloth prediction error is an inadequate acceptance metric | Accurate local contact decisions, even if the global prediction score improves |
| Reward has no direct grasp-tracking term; evaluation requires tracking within 2 cm for the whole episode | Training and evaluation emphasize different outcomes | This mismatch explains every failure or should be repaired by an arbitrary reward bonus |
| Some historical early-turn flags are true at reset; a progress ray can lose intersection discontinuously | Metrics require geometric validation | Every flagged episode took a physically wrong route because of its policy |

Sources: [ordinary SAC audit](2026-09-17-normal-sac-rollout-audit.md),
[FQL implementation/results](2026-09-17-fql-pretraining.md),
[motion pilot](2026-09-18-motion-pretraining.md), and
[preceding decision review](2026-09-18-research-direction-decision.md).
These runs differ in data, training configuration and task/reward history;
0/8 versus 5/8 is not a clean SAC-versus-FQL experiment. Repeating four cells
twice does not provide eight independent garment/body configurations.

The ordinary SAC source run started with empty replay and did not consume the
separately collected expert folder. Its 125,016 source transitions include about
408 complete trajectories across a plan with 225 configurations, not 125,000
independent dressing attempts. These facts make useful experience and coverage
credible bottlenecks alongside optimization; they do not prove either is the
sole cause.

There is already substantial collected data. The historical inventory contains
945,864 transitions; the current reward-compatible FQL reconstruction contains
25 episodes, with 15 training and 10 validation episodes. Only five training
episodes supplied the motion pilot. These are different usable subsets, not a
reason to tell the owner to recollect everything. Existing observations/actions
can have value even when their old rewards cannot be relabeled from absent
geometry. Failed trajectories are potentially useful RL experience; blindly
cloning their saturated stalled commands is a different objective.

Memory and counterfactual prediction are also not new to this repository. The
[RLT audit](2026-09-13-rlt-training-audit.md) identified action conditioning,
history alignment, replay weighting and timing problems in an earlier prototype;
subsequent fixes do not establish a dressing-policy gain. The
[counterfactual-response experiment](2026-09-15-counterfactual-response.md)
tested derivative and finite-response representation supervision on a small
non-dressing diagnostic. Better response prediction did not establish useful
dressing action selection. Any next experiment must differ in its measured
decision target, not merely rename these ideas.

## First-principles interpretation

A policy needs four things to improve: a meaningful task objective, information
that supports a good decision, useful experience about consequences, and an
optimizer that can exploit that experience. More accurate simulation addresses
the fidelity of consequences. It does not automatically solve the other three.

At an elbow stall, several explanations remain consistent with our evidence:

1. The task/controller may make a useful recovery inaccessible under current
   actions, or the optimized reward may favor remaining partly dressed.
2. The available observations may omit a relevant distinction, such as how
   cloth arrived at the current visible configuration. A history might help,
   but some hidden information may remain unrecoverable from it.
3. The learner may have enough information but insufficiently useful examples
   of alternative actions and their outcomes in that region.
4. The data may already contain the relevant information, while value learning,
   policy extraction, regularization or optimization fails to use it.

An episode that keeps pushing cannot distinguish these explanations. We should
measure them before treating any one as the explanation. A local contact force
or physics derivative is not automatically the right supervision: it describes
an immediate mechanical relation, while the policy needs the consequence of a
decision followed by future feedback. Differentiability remains potentially
useful, but it is neither required for policy gradients nor a demonstrated cure
for this environment.

The most relevant course connections are:

- [Lecture 14, pages 40–51](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-14.pdf):
  formulate the decision input as an observation/action history. A current cloud
  may suffice in some tasks, but that must be tested here.
- [Lecture 16, pages 17–24](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-16.pdf):
  useful simulator interaction need not backpropagate through an entire physical
  trajectory. Short interventions are a possible source of learning targets.
- [Lecture 18, pages 24–32](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-18.pdf)
  and the [offline-to-online project](https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/offline_to_online_rl_default_final_project.pdf):
  FQL, prior-data SAC, warm starts, latent actions and chunking belong in the
  baseline set. Choosing one is not a research contribution.
- [Lecture 24, pages 23–31](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-24.pdf):
  a useful representation can preserve consequences for control without
  predicting every cloth vertex. Our test should therefore concern decisions.

These are interpretations for our project, not evidence that the proposed
mechanism will improve it.

## Prior art that constrains the novelty claim

The following are primary sources checked for this assessment. "Method read"
means the relevant main-text method/experiments were examined, not that every
appendix was reproduced. Abstract-only checks identify overlap; they are not
sufficient for a detailed comparison or a definitive novelty search.

| Work and reading depth | What is already established | Consequence for our claim |
|---|---|---|
| [Rollout Sampling Approximate Policy Iteration, 2008](https://arxiv.org/abs/0805.2027), abstract; [Rollout Allocation Strategies, 2010](https://mohammadghavamzadeh.github.io/PUBLICATIONS/icml10-rollout.pdf), full main paper | Adaptive allocation of simulations to states/actions for rollout-based policy improvement | Branching at uncertain decisions, comparing returns and spending fewer samples on easy choices are not new. |
| [Recurrent Model-Free RL Can Be a Strong Baseline, 2022](https://proceedings.mlr.press/v162/ni22a.html), abstract and author material | Carefully implemented recurrent model-free RL is competitive on many POMDPs | A history-conditioned SAC/FQL control is mandatory before claiming a new memory mechanism. |
| [Unbiased Asymmetric RL under Partial Observability, 2022](https://arxiv.org/abs/2105.11674), abstract and cross-check through later derivations | History-state critics address limitations of state-only asymmetric critics | Adding privileged state to a critic is not new; a recurrent actor's context cannot simply be ignored in its value function. |
| [Informed POMDP, 2024](https://rlj.cs.umass.edu/2024/papers/Paper105.html), abstract and method excerpts | Privileged information can train representations for execution from partial observations | "Use IPC state during pretraining but only vision at execution" is not a novel principle. |
| [Informed Asymmetric Actor-Critic, v3 June 2026](https://arxiv.org/html/2509.26000), main formulation, selection criteria and experiments | Arbitrary privileged signals in history critics; signal selection by residual information and value-prediction gain | Even choosing useful privileged features is prior art. A decision-specific mechanism must distinguish itself from these criteria. |
| [To Distill or Decide?, 2025](https://arxiv.org/html/2510.03207v1), main methods/discussion | Privileged expert quality and the ability to distill it under partial observability are distinct | A simulator oracle's best action need not be a learnable deployment target. |
| [Value Equivalence, 2020](https://arxiv.org/abs/2011.03506), abstract; [Policy Similarity Embeddings, 2021](https://agarwl.github.io/pse/), author method page | Representations can preserve control-relevant behavior instead of full state | "Predict what matters for action" and grouping behaviorally similar states are insufficient novelty statements. |
| [PIGDreamer, 2025](https://arxiv.org/html/2508.02159v1), main method/experiments; [Reinformed Dreamer, July 2026](https://arxiv.org/abs/2607.26040), abstract | Privileged guidance of latent models/policies is an active, populated area | A generic privileged world model is a costly and weakly differentiated next step. |
| [Contact-Rich Manipulation under Partial Observability, RSS 2020](https://www.roboticsproceedings.org/rss16/p023.html), proceedings overview | Belief-aware contact control and uncertainty-reducing actions | Active contact sensing is not an unexplored concept. |
| [TACTIC, RSS 2026](https://emprise.cs.cornell.edu/tactic/), project methods/results | Visuotactile history, contact-informed planning and dressing evaluation | Contact memory and learned physical planning have close application precedents; this system also uses tactile inputs we should not silently assume. |
| [Adaptive Q-Chunking, May 2026](https://arxiv.org/abs/2605.05544), abstract | Adaptive action duration with calibrated value-based selection | Switching between long free-motion chunks and short feedback near contact is already proposed. |
| [One Policy to Dress Them All, RSS 2023](https://roboticsproceedings.org/rss19/p008.html), proceedings overview and local implementation audit | Shared partial-point-cloud dressing policy using regional training/distillation | Our practical motivation is reducing training complexity/cost. Its 86% figure is mean arm-length coverage, not an 86% binary success rate. |
| [Dressing in Motion, September 2026](https://arxiv.org/abs/2609.04759), abstract and prior local review | Motion-aware adaptation of dressing trajectories from static demonstrations | History or motion conditioning alone is particularly weak novelty in dressing. |

This updates the earlier local review's narrower search conclusion. In
particular, **adaptive rollout allocation based on action-decision difficulty
has direct prior art from 2008–2010**. The earlier statement that a closely
related allocation mechanism had not been found should not be used to claim
novelty. These papers do not establish our proposed method has already been
fully solved, but they set a much higher bar than "IPC plus RL."

## The candidate research mechanism

Working description: **pretraining decision representations from simulator
interventions under partial observability**. Avoid assigning a new algorithm
name before a distinct mechanism and positive evidence exist.

The specific distinction to investigate is between privileged information
that predicts return and information that changes a decision in a way the
deployed policy can learn. For example, a hidden variable can add the same
constant to the returns of every action. Knowing it improves return prediction
without changing the best action. Conversely, two hidden cloth configurations
may require opposite recoveries. A history helps only if it distinguishes
them, or supports a useful belief over them. These examples motivate a test;
they are not findings about our captured dressing episodes.

Let `h` contain available observations and previous commands, `x` be privileged
physical state, and `pi` be a fixed continuation policy that uses only `h`.
For a fixed candidate set, the deployment value is

```text
Q_pi(h, a) = E[Q_pi(h, x, a) | h].
```

It generally holds that

```text
max_a E[Q_pi(h, x, a) | h] <= E[max_a Q_pi(h, x, a) | h].
```

The right side lets the decision maker know hidden state. Distilling its
winning action without checking observability can give conflicting or
unattainable targets. Future branch actions must also come from the same
deployment-compatible continuation policy, rather than a privileged expert
whose recoveries the student cannot reproduce. These are standard conditional
expectation/information distinctions, not a proposed new theorem. Finite-window
history is itself an approximation, not automatically a sufficient statistic.

### A minimal operational prototype

1. Keep the existing data interface, action/controller limits and FQL baseline.
   Add a correctly aligned history baseline using existing history machinery;
   do not build another transformer stack. FQL currently uses current-frame
   features, although history-capable SAC/model infrastructure already exists.
2. At a small number of actual policy failures, restore the full simulator and
   controller/history state. Test a bounded set of alternatives: the policy's
   proposed action and a few feasible nearby recovery commands. Then resume
   the same frozen policy. Record complete provenance and restore cost.
3. Repeat branches sufficiently to distinguish action effects from restore or
   numerical variability. Record finite-horizon returns and physical outcomes.
   A short return is a local diagnostic, not an unbiased estimate of full
   dressing success. A bootstrapped terminal value introduces its own bias.
4. Train a history-conditioned critic to preserve measured return differences,
   alongside its ordinary RL objective. A simple control objective is

   ```text
   Delta_G = G_L(a) - G_L(a_reference)
   L_gap = E[(Q(h,a) - Q(h,a_reference) - stopgrad(Delta_G))^2].
   ```

   First use this only as a diagnostic/prototype: horizons and target policies
   must agree before adding it to a long-horizon critic. An alternative is an
   explicitly separate finite-horizon head. Raw pairwise win probabilities are
   not interchangeable with expected-return differences.
5. The actor improves through the ordinary learned-Q objective and behavior
   regularization. The simulator supplies finite outcome labels; there is no
   derivative through IPC in this prototype. With separate actor/critic
   encoders, critic supervision does not directly train the actor encoder;
   the actor learns through its own policy loss. Do not claim otherwise.
6. Retain useful critic/history parameters and their learning objective into
   subsequent RL. Do not discard the only decision head and transfer only an
   actor encoder as in the motion pilot. If flow-prior parameters are
   pretrained, preserve those too; specify every transferred component.

This simple return-gap auxiliary objective is **a baseline operationalization,
not enough algorithmic novelty**. Rollout-based policy iteration and advantage
learning already cover much of its logic. Its value now is that it can expose
whether our simulator contains actionable supervision which current learning
fails to extract.

The possible research extension is a learning-and-querying mechanism that
separates three cases: history-recoverable decision error, irreducible ambiguity
under the sensor interface, and unreliable simulator labels. It would allocate
expensive queries and train representations according to improvement in
deployable decisions, while accounting for full simulation cost. Generic
information gain, uncertainty sampling or expected regret reduction per unit
cost are also established ideas; merely assigning one of those scores is not
a sufficient contribution. The exact estimator/update and its distinction
from rollout allocation and informed asymmetric RL remain research work.

### What the pretraining infrastructure would transfer

The current IPC trajectories are **pretraining data relative to a later
deployment/adaptation stage**; data are not intrinsically "pretrain" or
"posttrain." New interaction collected after initialization supports online
adaptation. Replaying the same fixed dataset for more updates is still offline.

For this direction the training pipeline would consume compatible recorded
transitions, plus a small separately costed intervention dataset if justified.
It would preserve actor, critic, history models, behavior prior, normalizers,
optimizer/target state and episode/reward provenance as appropriate for exact
continuation. Non-compatible old rewards must not be silently mixed in.

At execution, the actor receives the declared sensor-derived point cloud,
robot state and previous commands. Hidden cloth vertices, forces, material
parameters, branch outcomes and simulator derivatives stay in training unless
an actual deployment sensor supplies them. The current simulator observation
builder is not proof that equivalent real point clouds, segmentation and
coordinates are available. Real perception, latency, robot/controller mapping
and physical model validation remain explicit deployment work.

## A bounded decision sequence

### Gate 0: establish what failure means, using existing records

Use the ordinary-SAC geometry audit and existing teacher/FQL traces to distinguish
physical stalls, lost sleeve placement, invalid grasp and metric discontinuities.
Version any corrected outcome/reward contract and apply it to every new arm.
Report completion, retained completion, grasp validity and teacher outcome
separately. Do not make the historical early-turn flag the sole ground truth.

This is necessary measurement work, not the proposed paper contribution. A
reward/controller repair that fixes the problem should be credited as such.

### Gate 1: does an affordable, reproducible action comparison exist?

Reuse completed restore/repeatability diagnostics if available; do not duplicate
other ongoing work. Otherwise retain the preceding review's cap of **one hour
of new native simulation** for an initial feasibility gate, including restores.
Start with a few saved failure states spanning a physical stall and a loss of
sleeve progress. An illustrative pilot is up to eight states, three candidate
actions and three repeats, with a short declared continuation horizon. The cap
is authoritative; these counts are not a power calculation or a promise they
fit within it.

Check identical-action repeats before interpreting between-action differences.
The existing three repeats with divergent outcomes do not identify the cause
as chaos; restore state, control history and numerical settings must be checked.
Include a separate repeated set for evaluating selected winners to avoid
reporting a winner chosen from its own favorable noise.

Proceed only if useful alternatives can be separated reproducibly at plausible
cost. If the test cannot resolve alternatives, report whether labels were too
noisy, the horizon too short, candidates inadequate, or no improvement observed.
This blocks scaling this protocol; it does not prove all recovery impossible.

### Gate 2: is the decision information available to a deployable policy?

On the same intervention data compare small predictors with:

- current observation and candidate action;
- observation/command history and candidate action;
- history plus privileged simulator state and candidate action, as a diagnostic
  reference rather than a deployable policy.

Hold out whole episodes and configurations. Do not randomly split overlapping
windows. Measure the return lost by the chosen action relative to the best
evaluated candidate, with independent repeated outcomes, as well as gap error.
Report privileged information's advantage separately from model capacity.
Use shuffled-history or matched-window controls if an apparent history benefit
could come from configuration leakage. A learned privileged predictor is not
the true optimal oracle and failure of that predictor is not a proof of no
available information.

This small pilot can reveal a promising signal, not establish robust statistical
generalization. One unique hidden state per unique history cannot identify a
posterior over hidden states. Controlled partial-observation benchmark cases
with known ambiguity are needed for claims about irreducible uncertainty.

Decision rules:

- If a plain history baseline captures the benefit, use it; do not call it a
  new algorithm or add simulator supervision unnecessarily.
- If only the privileged predictor chooses useful alternatives, test whether
  a feasible observation/history change resolves that gap. More updates cannot
  reveal truly unavailable information. Active sensing is a separate mechanism
  with its own prior art and execution cost.
- If intervention supervision improves deployable action selection beyond the
  history baseline on held-out cases, a policy-learning experiment is justified.
- If the action scores improve but complete rollouts do not, investigate target
  horizon/distribution mismatch before scaling the representation.

### Gate 3: show an actual learning benefit

Use the same task version, prior data, configuration split and one-GPU hardware.
First compare the current-frame FQL baseline, the history baseline, and the
history baseline with the simplest validated intervention supervision. Only
after a positive signal test a distinct proposed selection/learning mechanism.
It must beat uniform intervention sampling and a classical uncertainty/regret
allocation control, not just an uninformed from-scratch policy.

Include simple imitation and prior-data SAC/RLPD as essential external baselines
in the eventual study; an entire simultaneous CQL/IQL/PPO/DSRL sweep is not the
next experiment. Match tuning effort and show multiple training seeds. Report
both learning from identical data and the complete system at equal workstation
time, including unsuccessful branches, data reconstruction and evaluation.
More favorable data or extra simulated rollouts must not masquerade as a better
learning update.

## Novelty, benchmarks and publication claim

There are three distinct possible outcomes:

| Outcome | Defensible claim |
|---|---|
| Better task contract, history baseline or prior-data use solves much of dressing | Useful system/empirical result; no invented new RL algorithm |
| A distinct decision-supervision/selection mechanism improves held-out learning at equal cost beyond strong controls | Candidate algorithmic contribution, subject to broader novelty search and replication |
| Prediction or local action ranking improves without end-to-end dressing benefit | Negative result about the attempted interface; insufficient evidence to scale or claim policy improvement |

A strong algorithm paper would establish the failure mechanism, define the new
learning rule precisely, isolate why it improves decisions, and demonstrate
better time-to-performance across multiple tasks. Dressing should be the main
task, with held-out garments/bodies/poses and calibrated material/contact
variation. Add at least two distinct contact tasks only after the dressing
mechanism works; include controlled partial-observation cases and both easy
and expensive reset settings. A standard offline benchmark can test optimizer
behavior but cannot by itself validate an IPC-specific contact advantage.

Separate ablations should test partial versus fuller observations, uniform
versus selected interventions, motion versus decision supervision, and
same-data versus same-total-time training. To claim IPC is necessary, compare
label quality and downstream policy performance against a cheaper simulator
at equal cost. To claim sim-to-real benefit, evaluate the trained policy with
the real sensing/controller interface; simulator precision alone is not that
evidence. An algorithm should not be named after IPC unless it exploits a
property that ordinary restorable simulators do not supply.

## Compute and implementation implications

The ordinary-SAC audit attributes 17,813.7 s to environment work, 10,164.1 s to
evaluation and 4,078.3 s to updates: updates are about 12.7% of those three
categories. The short continuation is likewise environment-dominated. The
motion pilot observed 99–100% GPU utilization during learner work. A cold GPU
between updates and a busy GPU during updates can both occur in this pipeline.

Therefore charge data preparation, physics, restore, rendering/observation,
learning, evaluation and rejected runs separately. The Genesis-facing wrapper
does not make native IPC cloth simulation a batched Genesis GPU workload.
Cached-data learning can be batched efficiently; branch simulation must be
profiled for memory and throughput before claiming parallel speedups. Several
processes on one GPU are not automatically faster than sequential solves.

No learner, physics run or algorithm implementation was launched by this course
review. Full online FQL continuation remains unimplemented and untested. The
next technical work is the bounded decision diagnostic above, followed by a
minimal policy experiment only if its signal supports one. The recommendation
has a reasoned basis, but no new performance result or guarantee of novelty.
