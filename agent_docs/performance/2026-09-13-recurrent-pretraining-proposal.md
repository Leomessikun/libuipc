# Recurrent dressing pretraining infrastructure

Proposal, not an implemented architecture or a measured training improvement. Local baseline:
`01bf913e09fc1b8017ef5bd47bd182d02835d7c2`. Work branch: `pretrain/recurrent-force-memory`.

## Decision

Make pretraining preserve interaction history before choosing a new temporal backbone. Keep the
corrected dense action-per-point critic and the current regional teacher/student protocol. Add
episode-aware replay, an explicit rollout-state interface, and sequence-based distillation. Compare
finite history and GRU before adopting a small RLT-inspired model. Force inputs remain disabled:
the [per-decision reliability gate failed](2026-09-12-research-direction.md).

Here, pretraining means our simulation teacher learning and student distillation. It does not mean
language-model next-token pretraining. The same episode interface can support offline behavior
cloning, but adding an offline RL algorithm would be a separate algorithm choice.

## What the linked work contributes

The inspected [RLT repository][1] at `1bee93a9b01c21bea0c7a50ce3f6619f24731e19` contains a
technical report, translations and illustrations, with no training implementation or measured
model results. Its causal encoder supplies global memory; the decoder carries a recurrent output
and sliding-window KV caches across tokens. The growing depth follows time, with fixed block count
per token. Exact replay rebuilds parameter-dependent state under current weights; full gradients
include all caches. Decoder replay remains sequential, and global memory grows with context.
Its 48+48-layer configuration and RL discussion do not establish a robotics speed or success gain.
These are source claims from sections 2–5, not our experimental findings.[^1]

Our proposed translation is one spatially encoded observation per control decision. A useful
history could distinguish a sleeve approaching the elbow from a similar-looking sleeve that has
failed to advance after repeated pulls. This is a partial-observability hypothesis, not evidence
that missing memory caused the current failure. Extra repeated computation on a single unchanged
frame would test a different hypothesis.

## Baseline corrections that precede the memory experiment

Commit `ef1b3c81` fixes action conditioning in the critic: candidate actions now enter every point
before spatial encoding. The old global-latent critic remains relevant only to old checkpoints or
an explicitly selected ablation. Preserve the actor gradient through the action-conditioned
critic encoder while freezing critic parameters. Commit `01bf913e` adds optional residual trunks;
plain remains the default. Neither change has yet supplied a new long-run dressing result in this
audit. See the [critic defect record](2026-09-12-critic-architecture-defect.md).

The old run's best valid mean upper-arm ratio was 0.283; its final all-zero evaluation contained
25 simulator errors. That endpoint cannot be used as a clean learning result. The scripted expert
had 11/25 final successes, but none passed the reference trajectory filter. Demonstrations are
therefore imperfect training data, not verified safe targets. See the
[archived evaluation evidence](2026-09-12-elbow-evidence.json).

Memory cannot enlarge an infeasible sleeve, undo rejected controller motion, repair force labels,
or establish physically correct reward geometry. Keep action rejection, progress retention and
simulation errors visible alongside the learning metrics.

## The data contract comes first

`FlatReplayBuffer` stores independent transitions. Vector slots interleave in storage, and horizon
transitions deliberately bootstrap with `not_done=1`. Neither adjacent indices nor `not_done` can
identify a continuous episode. Old replay snapshots without identities must remain flat-only;
do not infer their histories silently.

Extend the existing storage with optional sequence metadata and a named sequence batch:

| Field | Contract |
|---|---|
| Stream, world generation, episode ID, step | Distinguish slots and recycled worlds; prove temporal adjacency even after circular overwrite |
| Observation and terminal next observation | Preserve the actual pre-reset successor; record preprocessing/version and decision interval |
| Policy command action | Keep the action space on which SAC and its density are defined; do not replace commands with cloth displacement |
| Accepted controller target and measured motion | Separate optional channels, with timestamps and availability masks; these are not interchangeable with the policy command |
| Episode end, terminated, truncated | Episode reset is independent of Bellman bootstrap; simulator errors close sequence continuity |
| Region/garment labels and buffer index | Preserve teacher routing and temperature selection; not substitutes for episode identity |
| Valid observation and loss masks | Padding contributes neither state updates nor losses |
| Optional diagnostic labels | Store provenance and validity; missing force is not a measured zero |

A sampled window returns observations `[B,L+1,D]`, command actions and rewards `[B,L,...]`, masks,
optional privileged observations and labels. Store raw frames once per existing transition layout
and gather linked windows; do not multiply replay storage by materializing H-frame stacks per row.
The current observation/next-observation arrays alone can occupy roughly 17.2 GB at 400k rows and
5383 floats per observation, before other arrays. Removing existing duplication is optional later work.

Use whole training episodes for dataset splits. Keep region poses 45–49 held out. Existing ordered
demonstration files can support BC only when their action and observation timing is verified;
do not synthesize missing terminal observations or simulator states. Preserve failure episodes for
diagnostics; imitation losses need explicit target selection because the current expert cuts corners.

## Actor, critic and execution state

Proposed actor timing is:

`history_t = update(history_(t-1), observation_t, command_(t-1), available_feedback_t)`

`action_t ~ actor(history_t)`

The first decision uses a previous-action sentinel and mask. Commands selected during random
warmup or scripted roll-in must also enter the subsequent history. Force or future-observation
labels must not leak into an actor intended to deploy with cameras and controller observations.
Point order is not time: retain spatial encoding of each cloud before temporal aggregation.

Use a pure `step(..., state) -> (distribution, next_state)` interface with state held outside the
network. Training collection, each evaluation world and each teacher own independent states.
Reset on episode end, world rebuild, slot reassignment and resume into fresh physics. Saving a
network cache without matching physics is not an exact resume. A model-version tag makes stale
state detectable when collection weights change during an episode.
After each weight update, rebuild active rollout states from their retained raw prefixes before
selecting another action. If prefixes are bounded, use the same declared finite-context definition
in collection and replay; a version tag alone does not repair stale state.

For Q, retain the current dense encoder `E(observation_t, candidate_action_t)` and add a history
summary of preceding observation/command pairs. Scoring candidates must be side-effect-free.
At the Bellman successor, advance history with the command actually recorded in replay, then
evaluate the current policy's next candidate. Do not rewrite the observed past using a newly
sampled action. Actor, online critic and target critic reconstruct histories using their respective
weights; new target-memory modules must participate in Polyak updates. The current SAC uses an
online actor for successor sampling, so this proposal does not add an unspecified target actor.

An asymmetric critic should also receive the actor's relevant history. The existing privileged
summary is not proven to be a complete Markov state. A recurrent actor with a frame-only critic
can be an ablation, but should not be assumed an adequate history-value baseline.

## Training stages and concrete change boundaries

| Stage | Change | Gate before advancing |
|---|---|---|
| 0 | Evaluate dense/plain baseline; residual trunk remains separate | Valid held-out results, architecture provenance, no simulator-error checkpoint selection |
| 1 | Optional episode metadata, linked windows and rollout adapter | Interleaved-slot, ring-overwrite, terminal-history and save/load correctness |
| 2 | H4 then H8 ordered feature history; current-weight recomputation at collection and replay | Streaming/window parity and measurable held-out history benefit |
| 3 | GRU pilot: hidden 128, sequence batch 8, burn-in 8, learning length 8 | Compare prefix reconstruction error, memory cost, learning curves and dense action gradients |
| 4 | Small recurrent-attention alternative using the same interface | Beats simpler memory controls at a declared data and compute budget |

The numerical pilot choices are proposals, not tuned values. H4/H8 use declared finite context
and recompute features under current weights. A GRU started from zero in a mid-episode burn-in
window approximates omitted history. Replaying the full prefix under current weights reconstructs
the forward state; detaching the burn-in boundary still truncates gradients. Make both choices
explicit. A later RLT adapter requires the complete recurrent, local KV, encoder-memory and
position state; detaching only the final vector does not truncate every memory path.

Train masked SAC losses on valid learning steps and normalize by their count. Use consistent
observation preprocessing across overlapping prefixes. Begin without stochastic augmentation,
then define temporally consistent transforms separately. Behavior cloning scores the current
command before consuming it. A deterministic expert need not supply a behavior density for BC;
do not invent one or transplant language-model importance ratios into SAC.

Minimal implementation locations:

- `replay.py`: optional metadata, sequence sampler, persistence/version checks; preserve flat mode.
- `models.py` and `sac.py`: feature access, pure memory transitions, history-aware dense Q,
  masked sequence updates, all target-memory updates and explicit configuration compatibility.
- `train_sac.py` and `pretrain_wang.py`: stream identities, reset events, random-roll-in history,
  independent evaluation states and rank-checked inputs.
- `collect_rollouts.py` and `distill.py`: ordered episode schema and causal sequence BC.
- Checkpoint consumers including `expert_baseline.py`: support stateful execution or reject it
  before allocating a simulation world. Unsupported FlashSAC/recurrent-teacher combinations
  should fail explicitly until their sequence algorithms are implemented.

This is a change specification. No new CLI flag or sequence feature is claimed to exist yet.

## Experiments that can answer the elbow question

Compare dense/plain single-frame, matched-capacity single-frame, H4/H8, GRU and a small RLT-inspired
variant with the same physics, action limits, demonstrations and reward. Keep a separate
no-history ablation and a shuffled-history diagnostic. Changing capacity, force reward and
material parameters at once would not identify a memory effect.

Report final and peak upper-arm ratio, retained progress after crossing, early-turn filter,
action rejection, simulator errors and per-garment results. Report both equal environment-transition
budgets and elapsed time; use multiple seeds for any improvement claim. Select models using
training-only validation, reserving the prescribed held-out cells for final comparisons. Offline
history classification or BC loss alone does not establish closed-loop elbow passage.

Before a sustained run, verify causal step/unroll and short-sequence gradients, reset isolation,
candidate-action non-mutation, dense `dQ/da`, target-memory updates, and bootstrap from the
pre-reset terminal history. Then measure one bounded inference/update smoke and a fixed-cell
rollout. If H4/GRU do not help matched ambiguous-history cases, there is no current evidence to
escalate to a more expensive temporal model.

The [historical wall-clock budget](2026-09-12-wall-clock-budget.md) assigns 63% to simulation,
25.6% to evaluation and 10.3% to updates. These timings predate the dense critic and cannot price
the new model. Sequence learning increases spatial encoding, critic and target work; record peak
GPU memory, update throughput and decision p50/p95. At a 0.1 s control period, inference must fit
the available controller budget rather than relying on the paper's unmeasured efficiency goals.

## Sources and supporting evidence

[^1]: Yifan Zhang, *Recurrent Looped Transformer*, September 12, 2026, [English report][1], sections 2–5. Inspected pinned revision above. The robotics contracts and staged experiments in this document are our engineering proposals.

[1]: https://github.com/yifanzhang-pro/recurrent-looped-tranformer/blob/1bee93a9b01c21bea0c7a50ce3f6619f24731e19/Recurrent_Looped_Transformer.pdf

Local code was inspected at the baseline hash above. Historical force and elbow findings are
linked in place; their later supersession is explicit. No policy training, runtime implementation,
or measured gain was produced by this proposal.
