# A prior and residual RL instead of SAC from scratch

Date: 2026-09-13. Status: research record and proposal; no code changed, no GPU job ran. Prompted by
the question whether the REAL lab's recent work ([shurans.github.io](https://shurans.github.io/index.html))
means our SAC structure should change.

Tags: [MI] measured here, [RA] read from the source, [E] inferred.

## In one sentence

**Our SAC is not old, it is starved: the reference solved this task with about five million
transitions and we collect fourteen thousand an hour. The lab's 2025–2026 line spends one to six
hundred thousand online steps refining a usable prior, which is a budget we can afford — but we do
not yet have a usable prior, and building one is the first infrastructure change, before any RL.**

## The numbers that carry the argument

| | Value | Source |
|---|---|---|
| Reference teacher, region 13, SAC from scratch | 0.740 upper-arm ratio | [RA] Wang RSS 2023, via `2026-09-12-task-specification-audit.md` |
| Reference training budget | about five million transitions | [RA] `2026-09-08-uipc-manip-pretraining.md`, "Newton teacher's five-million-transition budget" |
| Our throughput, evaluation excluded | 13,881 transitions per hour; 0.220 s simulation + 0.036 s update per transition | [MI] `2026-09-12-wall-clock-budget.md` |
| Five million transitions here | about 360 hours per region, before evaluation | [E] from the two rows above |
| Our best valid SAC result, same region | 0.283; the running dense/plain ablation 0.10–0.15 at 72k–125k | [MI] `2026-09-12-elbow-evidence.json`, `output/uipc_manip/abl_*_s1/eval_log.csv` |
| Scripted expert on the 25 held-out cells | 0.542 mean upper-arm ratio, 11/25 final successes, 0/25 pass the reference trajectory filter, 8 elbow hooks | [MI] `expert_r13_heldout_s0`, `2026-09-12-elbow-experiment-design.md` |
| One 300-decision episode | about 66 s of simulation; about 46 episodes per hour | [E] from the throughput row |

## What the lab's work actually is

Read at the source on 2026-09-13. None of it is a dressing or assistive paper; the lab's cloth work
(FlingBot CoRL 2021, Cloth Funnels ICRA 2023, GarmentNets) is unfolding and canonicalisation with
pick-and-fling primitives over spatial action maps, a different action space from ours.

**DICE-RL — "From Prior to Pro", Sun & Song, ICML 2026** ([arXiv 2603.10263](https://arxiv.org/abs/2603.10263)) [RA].
A frozen flow- or diffusion-based BC policy is kept as a *stochastic proposal*; RL trains a
lightweight residual on its action chunk, conditioned on the same latent noise, with a TD3+BC-style
objective: critic value maximisation plus a penalty on residual magnitude. A *BC-loss filter*
relaxes the penalty only where the critic predicts the residual improves on the base action *and*
that prediction does not exceed a Monte-Carlo return estimate (margin ε −0.25 to −0.75), so an
optimistic critic cannot pull the policy off the demonstration support. Candidates are scored by
the critic and the best executed (value-guided selection, K = 16 latent samples per state, also used
to average the TD target). Chunk-level critic with n-step targets (3–5), ensemble of 10, RLPD-style
offline/online mixing whose offline share decays linearly (0.5–0.9 → 0.1 over 160k–640k steps),
update-to-data ratio 10–20, 4–8 parallel environments. Robomimic from 20–50 proficient-human
demonstrations in **100k–600k online environment steps**; Tool Hang from a 45 % prior to ≥ 90 %
"within roughly 2000 online episodes". Three real NIST tasks with asynchronous updates every 10
episodes. Its baselines — DPPO, DSRL, EXPO, IBRL, ResFit — are the current alternatives; the paper
reports ResFit and EXPO "collapse on the more complex long-horizon tasks … in the absence of strong
BC regularization". Its critic is an MLP on *frozen* visual features with the action concatenated,
the latent arrangement this project rejected for the point-cloud critic on 2026-09-13; the cheap
UTD depends on that frozen encoder.

**Latent Policy Barrier, NeurIPS 2025** ([2508.05941](https://arxiv.org/abs/2508.05941)) [RA]: a base
diffusion policy on expert data only, plus a latent dynamics model trained on expert *and*
suboptimal rollouts; at inference the predicted future latents are optimised to stay inside the
expert distribution. No RL, no human corrections.

**Compliant Residual DAgger, NeurIPS 2025** ([2506.16685](https://arxiv.org/abs/2506.16685)) [RA]: a
residual policy learned from gentle human delta-corrections under compliance control, with force
feedback; +64 % base-policy success on four contact-rich tasks from little correction data.

**Gated Memory Policy, CoRL 2026** ([2604.18933](https://arxiv.org/abs/2604.18933)) [RA]: "simply
extending observation histories of a visuomotor policy often leads to a significant performance
drop due to distribution shift and overfitting"; the remedies are a learned binary memory gate, a
cross-attention read over cached history tokens, and diffusion-scheduled noise on past actions.
+30.1 % on their non-Markovian MemMimic, no loss on Markovian RoboMimic.

**Multisensory Continual Learning, CoRL 2026** ([2606.30988](https://arxiv.org/abs/2606.30988)) [RA]:
adapts a pretrained *vision-only real* policy to *real* force-torque sensing with a world model and
replay. It says nothing about simulated force, and does not reopen the direction closed in
`2026-09-12-research-direction.md`, whose failure was per-decision reproducibility of simulated force.

**Older, and the lab's own precedent for what follows** [RA]: TossingBot's residual physics
(analytic controller + learned residual), and "Scaling Up and Distilling Down" (privileged
sampling-based planning in simulation generating the dataset a policy is distilled from).

Not read: DF-ExpEnse (ICML 2026) is behind an OpenReview challenge page; DICE-RL's real-robot
before/after numbers are in figures the text extraction did not carry.

## The mapping, and where it breaks

The lab's line presupposes a prior with useful coverage: DICE-RL §5.2 studies which pretrained
checkpoints are *finetunable* and finds mode coverage matters; the residual penalty contracts
toward whatever the base does. Our only base is a deterministic seven-stage scripted controller
(`dressing_heuristic.py`) that cuts corners — 0/25 filter passes, 8 elbow hooks — so residual RL
with a BC penalty on it would contract toward corner-cutting. A Gaussian BC student of that expert
(`distill.py`, NLL/MSE head, not a diffusion or flow model) inherits the same support.

Two further conflicts are design decisions, not footnotes:

- **Critic cost.** Our dense critic encodes the point cloud with the candidate action on every point
  and trains PointNet++ each update, 36 ms [MI]. UTD 10 at that cost is 0.220 + 10 × 0.036 =
  0.58 s per transition, about 6,200 per hour, so 300k online steps take about 48 h [E]. A frozen
  encoder with a small trainable head would return to DICE-RL's regime but is the latent
  arrangement measured 0.11 worse by the reference; if it is tried, it is an arm, with the dense
  critic at a lower UTD as its control.
- **Action chunks against the tether.** DICE-RL edits h-step chunks executed open-loop. Our
  controller rejects sub-steps that would carry a held vertex past the 6 cm tether
  (`2026-09-12-task-specification-audit.md`), so an open-loop chunk at 0.1 s may be rejected
  mid-chunk. Start with h = 2–4 or a per-decision residual inside the chunk, and report the
  rejection rate; it has to be measured, not assumed.

## Proposal, in order

**0. The prior gate — cheapest, decides everything.** Perturb the scripted expert with a bounded
random residual per decision, of the magnitude a BC penalty would allow (start at 0.5 and 1.0 cm
per axis, against the 8.66 mm per-axis command cap), on the 25 held-out cells, a few hundred
episodes, and record whether *any* perturbed trajectory passes the reference filter or raises the
upper-arm ratio above the expert's. If the bounded residual space around this expert contains no
better trajectory, residual RL on it is dead and step 1 is mandatory first. `expert_baseline.py`
has no such flag today; adding `--residual-scale` (uniform noise added to `scripted_actions()`
before clipping, seeded per episode) is about twenty lines. Then:

```bash
PYTHONPATH=python /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.expert_baseline \
  --region 13 --poses heldout --num-envs 25 --seed 0 --residual-scale 0.005 --save-observations
```

Not launched: the GPU holds the critic ablation.

**1. A finetunable prior, if the gate fails — and probably anyway.** The lab-consistent recipe is
privileged search in simulation: a per-cell cross-entropy search over the expert's sixteen stage
parameters (`translation_step`, `z_offset`, `elbow_overshoot`, `elbow_hook_offset`,
`shoulder_overshoot`, `finish_upperarm_ratio`, `finish_drop`, `rotation_step`,
`align_tolerance_deg`, the proximity push, the step budgets), keeping every episode with its filter
and progress labels. 25 cells × 10 samples × 3 rounds is 750 episodes, about 16 h of one GPU [E], a
one-off dataset. `collect_rollouts.py` keeps only filter-passing episodes; for offline mixing it
must keep all attempts with their labels, or the offline buffer is empty. The prior can then be
the searched controller itself (deterministic, no BC needed) or a flow/diffusion BC on the dataset;
the residual machinery below is the same for both.

**2. Residual RL on the prior, at a matched budget against `abl_dense_s1`.** What each piece maps
to in this code:

| DICE-RL piece | Here |
|---|---|
| Frozen proposal | The searched scripted controller, or a flow BC; its command enters the actor as an input |
| Residual actor on the proposal | The existing stochastic SAC head on `[obs, base_action]`, output added to the base command and clipped |
| Value-guided selection, K samples | K residual samples per decision scored by the critic; the dense critic already takes the candidate on every point |
| Chunk critic, n-step 3–5 | n-step targets over the sequence replay's linked rows; h = 2–4 at first |
| Critic ensemble 10 | 2–5 at our update cost, measured |
| BC penalty + filter | Penalty on residual magnitude, relaxed only where critic gain and the Monte-Carlo return agree |
| RLPD offline/online mixing | The offline dataset as a second `FlatReplayBuffer`; the linear decay as a trainer knob |
| UTD 10–20 | A first-class knob; cost model above |

**3. Keep, and do not confuse with this:** the cell plan, evaluation, the reference filter, sequence
replay, the running critic ablation (it is the from-scratch control arm) and the H4 memory arm
gated behind it. GMP's finding is a caution for H4 — naive history can lose to single-frame through
distribution shift — but online RL grows the dataset the history is trained on, so the overfitting
half of that argument is weaker here; if H4 loses to dense, the gate-plus-noised-commands variant
is the follow-up arm, not a reason to skip H4.

## What this does not claim

No measurement here shows a prior-plus-residual policy dresses the elbow. The claim is narrower:
the from-scratch budget is unreachable, the refinement budget is reachable, and the missing
ingredient is a prior with coverage — which is why the gate in step 0 comes before any learner
change. Sources are the lab page and the arXiv texts read on 2026-09-13; the robotics numbers of
this project are our own measurements, linked in place.
