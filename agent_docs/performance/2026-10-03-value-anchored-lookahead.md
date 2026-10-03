# Value-anchored IPC lookahead: a teacher that can beat the policy it labels (2026-10-03)

Status: **method candidate under test**. Offline evidence below; the decisive teacher check is running.

## Why the earlier teachers saturated

Every post-training route so far stopped at the same point: the labeller could not tell which action is
better than the policy's own at a given state.

- The one-step / four-step IPC lookahead scores a candidate by local sleeve progress minus excess load. On
  Cloth3D it was no better than r1 (44 vs 45 of 69 units at H = 1 and H = 4), so 14x more of its labels left
  the student at r1's level (r2 127 vs r1 130 of 201).
- Most failures are grasp loss (92 % on Cloth3D, 83 % on ClothesNet) that happens tens to hundreds of
  decisions after the cause. A four-decision progress score cannot see it, and its candidates often tie.
- Repair search with outcome goals or effect codes found no verified repair (zero of two development cases).

## The idea

Keep the exact short physics rollout, replace its local progress score with a **learned continuation value**
trained on the policy's own archived rollouts: V(s) = P(final success | privileged state s) under the logging
policy (r1). At the end of an H-decision candidate hold, V estimates the success probability of continuing with
r1, so

    a* = argmax_a V(s_H(a)),   s_H(a) = IPC rollout of candidate a for H decisions from the live snapshot,

is a one-step policy improvement step over r1 with an exact model for H decisions and a learned value beyond.
Distilling a* into the actor (DAgger) and refitting V on the new policy's rollouts gives approximate policy
iteration in which the expensive part (long-horizon outcome) is learned once from existing data and the
contact-critical part (the next H decisions) stays exact.

The value is privileged: it is used only to label in simulation. The deployed student remains the point-cloud
policy.

## Offline evidence (no new simulation)

`scripts/wang_transfer/outcome_value_audit.py`, `train_outcome_value.py`, `outcome_value.py`;
outputs in `output/uipc_manip/value_audit_20261003/`.

Data: 7,276 archived r1 rollouts of the v4/v5 multi-garment collections (3,383 accepted, 4,546 grasp-loss
failures; other failures excluded), all with per-state arrays. The 41 test bodies are excluded from training;
one fifth of the remaining bodies are held out for validation.

1. **Outcome noise is moderate.** Replicas with identical garment, body, placement offset and seed agree on
   the outcome in 82 % of 3,040 groups (539 mixed). Single-episode outcomes are noisy but far from random.
2. **Failure is predictable long before it happens.** A gradient-boosted classifier on privileged per-state
   features separates eventual success from eventual grasp loss on held-out bodies with AUC 0.99 within 10
   decisions of the failure, 0.95 at 30-60 decisions, 0.94 at 60-120 decisions and 0.80 within the first 40
   decisions of an episode. The four-decision progress score sees none of this horizon.
3. **A smooth value is needed for ranking.** An MLP on window-4 state features (tool position relative to the
   arm, tracking error, sleeve ratios/fractions and their four-decision changes, episode time, garment) reaches
   held-out AUC 0.886 with all features. Its candidate ranking in a first live test was unusable: identical
   replays differed by 0.25 in value. Action features (a shortcut: the policy's own command) and the grip-force
   estimate (stiffness times sub-millimetre anchor offsets, so micrometre noise becomes newtons) were the cause.
   Without them: AUC 0.877, and consecutive archived states change by more than 0.1 in 3 % of steps instead of
   10 %. That model (`value_noact_nogrip.npz`) is the one under test; live replays now differ by < 0.0002.

## Decisive test (running)

Teacher check, identical protocol to the progress-score teacher check: r1 with the batched lookahead (H = 4,
every 3 decisions from decision 40, load > 15 N, the same 10 candidates), candidates ranked by V with a 0.01
minimum gain over the nominal action; first 7 test bodies x 5 Cloth3D garments; paired with r1 on the same
units (`output/uipc_manip/value_teacher_check_v2_20261003`).

Success criterion for continuing: the value teacher clearly beats r1 on the paired units (the progress
teacher was 41 vs 43). If it does not, the value is not actionable at this candidate set and the next step is
to diagnose whether V ranks branches that later succeed above branches that later fail.

## Prior art and what would be new

Short-horizon planning with a learned terminal value is established: POLO (Lowrey et al., 2019), blending MPC
and value functions (Bhardwaj et al., 2020), MuZero/AlphaZero-style search, model-based value expansion. DAgger
with a planner teacher is PLATO-like. A claim must therefore be specific: **post-training a pretrained
visuomotor dressing policy with an exact deformable-contact simulator for the contact-critical horizon and a
hazard value learned from the policy's own failures for the rest**, and showing (i) the teacher beats the policy
where local scores could not, (ii) iterating (refit V on the new policy) keeps improving instead of saturating
after one round, (iii) transfer to unseen garments and moving arms. If plain POLO-style planning explains all of
it, the contribution is an empirical finding for deformable post-training, not a new algorithm.
