# ADR 0009 — Resolvable Advantage Learning

- Status: Proposed; requires revision after evidence and prior-art review
- Date: 2026-09-20
- Owners: deformable manipulation research
- Branch: `research/flow-latent-steering`
- Supersedes: N/A. ADR 0008 (IAQL) is closed by measurement; this does not reuse it.

## Review disposition — 2026-09-20

The [training inventory review](../performance/2026-09-20-training-issue-review.md)
supersedes the causal and novelty interpretations below. A local four-step
action-effect separation is supported; the proposed milestone-based operator and
end-to-end policy improvement have not passed a gate. The reward-flatness proof
is incorrect, and the state-actor run is not a matched perception ablation.

Confidence-gated rollout policy supervision and budget allocation have direct
prior art in [RSPI (2008)](https://arxiv.org/html/0805.2027) and
[rollout allocation (2010)](https://mohammadghavamzadeh.github.io/PUBLICATIONS/icml10-rollout.pdf).
Withdraw the claim that this core mechanism has no existing counterpart.
Potential-shaping invariance does not apply to replacing the task reward with
short-horizon milestone reachability. Top-two separation is not improvement
over the incumbent; any revised gate must address mean-estimation uncertainty,
selection bias and numerical bias separately. Preserve the original proposal
below as history, not as an accepted algorithm or novelty justification.

## The problem, stated in quantities measured here

Six reinforcement-learning and imitation routes have now failed on this dressing
benchmark, and the measurements say why, in terms that are about the task rather than
about any of the methods:

1. **The return is not measurable at the resolution of an action's effect.** At states
   a policy visits, the gap between the best intervention and the next best is 1.31
   repeat standard deviations on `tshirt_26` and 1.24 on `tshirt_392`, and falls below
   a single repeat's noise at 42 % of states. On the cloth-drag benchmark, where the
   same code base's SAC goes from -4 to +13 return, that ratio is 193.6 and the failure
   fraction is 8 % (`2026-09-19-decision-signal-to-noise.md`).
2. **The objective has no slope at the event the task is about.** Wang's task term is
   minus the fingertip-to-opening distance while the opening is off the arm, so it
   rises to exactly zero as the opening arrives, and forearm progress is measured back
   from the finger, so threading also starts at zero
   (`2026-09-20-the-reward-is-flat-at-threading.md`).
3. **With free perception the learner parks on that plateau.** A state actor reading the
   35-float privileged state, with a privileged critic, on the 25 cells it is evaluated
   on, improves its task reward sixfold and its grasp validity from 0.64 to 1.00 while
   peak upper-arm coverage stays exactly 0.0000 at every evaluation.
4. **The reachable ceiling is far above every learner.** The scripted teacher puts the
   sleeve fully on the upper arm in 15 of 25 cells, and freezing the command once
   coverage crosses the threshold converts 3 of 4 of its "covered then lost" cells into
   successes with the final coverage equal to the peak to the digit
   (`2026-09-19-stop-when-covered.md`). No learner has taken any of that.
5. **Restoration is exact.** The largest vertex error over 63 and 84 restores is
   0.00e+00 m (`2026-09-18-predictability-horizon.md`). The simulator is a *resettable*
   oracle, which almost no reinforcement-learning algorithm assumes.

Points 1 and 3 together explain the failure signature the logs keep showing — a run
that peaks at 3 of 25 and ends at 0 of 25. An update driven by return differences below
the environment's own noise is a random walk in policy space. It is not slow learning;
it is learning from noise.

## Decision

Develop **Resolvable Advantage Learning (RAL)**: a policy-improvement operator that
*measures whether it can tell two actions apart before it takes a gradient step*, and
spends a fixed simulator-query budget only where the answer is decidable.

The three commitments, each forced by one of the measurements above:

**(a) Do not estimate returns; estimate short-horizon event reachability.**
The branch score is not the episode return and not the task reward. It is

> Φ(s) = the probability that a K-decision continuation from s reaches the next
> uncompleted milestone,

estimated by restoring to s and running continuations. Milestones are state predicates,
beginning with two that the privileged state already computes — the opening encircles
the arm, and the upper-arm ratio leaves zero. Φ is dense exactly where the task reward
is flat: two openings a centimetre apart at the fingertip have the same reward and
measurably different probabilities of threading within K decisions. K is chosen inside
the measured predictability horizon, where a restored pair has separated by
sub-millimetre rather than centimetres.

**(b) Compare actions in pairs from one restored state, and only learn where the
comparison resolves.**
At a visited state, M candidate actions are evaluated by branches that share the
restored start. Repeats are allocated by fixed-budget best-arm identification
(sequential halving), and the winner is written to a supervised buffer **only if its
mean exceeds the runner-up's by more than a threshold times the pooled within-action
standard deviation measured in the same call.** Where the state does not resolve, the
algorithm spends nothing further and takes no step. This is the part no existing method
has: the update is gated by the algorithm's own, per-state, in-situ estimate of the
environment's noise.

**(c) Treat the simulator query budget as the object being optimised.**
One workstation delivers 40-50 environment decisions per second, about 4 million per
day. RAL's inner loop has an explicit cost — states x actions x repeats x K — and the
allocation across states is the design problem, not an afterthought. States are selected
where the policy's own uncertainty is high and a milestone is near, because those are
where a resolvable advantage is most likely to exist.

The policy is then fitted to the resolved winners by regression. There is no Bellman
backup over three hundred decisions, no bootstrapping through the plateau, and no
gradient taken from a difference the simulator cannot report.

## Why this is not the existing literature

Honest accounting, to be checked against the literature before any claim is published:

* **Potential-based shaping** (Ng, Harada, Russell) gives the invariance argument for
  using Φ, and learned potentials exist. What is not standard is defining the potential
  as a *short-horizon event probability measured by resetting the simulator*, chosen
  because the measured signal-to-noise of that quantity is orders of magnitude better
  than that of the return.
* **Reset-based and simulator-backtracking methods** (Go-Explore, reverse curricula,
  tree search) use restoration for *exploration* or for *planning*. RAL uses it for
  *credit assignment*, and specifically to make the comparison paired.
* **Best-arm identification** is a mature field. It has not, to my knowledge, been used
  as the policy-improvement operator of a deep-RL method on a physics simulator, with
  the stopping rule tied to a measured environment noise floor.
* **The resolvability gate has no counterpart I know of.** Every standard algorithm
  assumes its return estimate carries signal. Here it provably does not at 42 % of
  states, and an algorithm that knows this and declines to move is a different object.

The novelty claim to defend is therefore the *operator* — measure, gate, then move —
together with the empirical case that the gate is necessary because the environment's
noise floor was measured to exceed the advantage.

## The calibration experiment, which must pass first

RAL is worthless if no branch score resolves. Before implementing the learner, measure,
on states the current policy visits, the margin-to-noise ratio of candidate scores under
paired branches:

| score | why it might be better |
|---|---|
| sustained upper-arm coverage | the task metric; measured at 1.31 SD, ray-cast and coarse |
| smooth geometric progress of the opening along the arm axis | continuous, no ray cast |
| Φ, threading within K decisions | binary but dense near the event |

Pre-registered criterion: **at least one score must reach a median margin of 3 pooled
standard deviations at states where the current policy's action is not already the best,
with K no longer than 20 decisions.** Below that, the gate would almost never fire and
RAL reduces to doing nothing; that result closes this ADR and is worth recording.

## Pre-registered evaluation

Against three baselines already on disk or in flight: the scripted teacher (6 of 25,
9 of 25 with the stopping rule), the longest SAC run (270,216 transitions, 26.2 h, peak
3 of 25, final 0 of 25), and the privileged-state upper bound now running. Same 25
cells, same success rule, and — because the whole point is the workstation — **the same
number of simulator decisions, not the same number of gradient steps.**

## Consequences

* RAL cannot be run on hardware, since it requires restoration. It is a simulator-side
  training method whose product is a policy that deploys normally. Distillation to the
  point-cloud observation is a separate, later step.
* If the calibration experiment fails, the finding is that this task's advantages are
  not resolvable at any affordable query budget, which is a stronger negative result
  than any this project has recorded and is itself the deliverable.
