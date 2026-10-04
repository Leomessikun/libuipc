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

## First live results and what they changed (2026-10-04)

Protocol note: test bodies 1-14 (in sorted order) are now the development set for teacher variants; bodies
15-41 are kept for a final check and are not used for any decision.

1. Value with action and grip-force features (bodies 1-7, tshirt_26): replays of one candidate differed by 0.25
   in value, so no candidate cleared the margin. Retrained without them.
2. Value without action/grip, all states (bodies 1-7, tshirt_26): teacher 4/7, r1 4/7 on the same units. The
   teacher chose "stop" in most late plans: the archive's accepted episodes end with a verified zero-action
   hold, so the value learned that standing still near the top means success. It delayed grasp loss (e.g. body
   1032: r1 at decision 225, teacher at 506) without preventing it; late plans had no feasible candidate left.
3. Value on decision states only (`value_v3_decision.npz`, hold excluded, AUC 0.877; bodies 8-14, tshirt_26):
   teacher 3/7, r1 2/7 (1:0, the rescued unit was an r1 non-grasp failure). The four grasp losses were again
   delayed by 20-120 decisions, not prevented. Stopped after one job.

Why lookahead from decision 40 can only delay: on held-out archived r1 episodes, 49 % of eventual failures
already have V < 0.2 at decision 40 (5 % of successes), 37 % are below it for good by decision 60, and the outcome
is separable from decision 5 (AUC 0.71) and decision 20 (0.82). At decision 40, failures have the gripper
further along the arm (0.26 vs 0.19 of arm length) while the cuff has barely advanced over the hand (0.003 vs
0.007): the hand missed or only grazed the sleeve opening and the robot kept pulling. Hospital gown is
over-represented among failures. The decisive decisions are in the entry phase, before the lookahead started.

Next variant (running, `value_teacher_entry_20261004`, bodies 8-14): plan from decision 5, no minimum-load gate
(no contact yet during entry), add two retreat candidates (reverse the mean of the last six executed
translations, once and twice), same value and margin.

## Entry-phase variant and what the failures really are (2026-10-04)

Entry-phase value teacher (plan from decision 5, no load gate, retreat candidates; bodies 8-14, tshirt_26):
2/7 vs r1 2/7 (1:1). It rescued body 4041 (r1 grasp loss at 209) but lost 5040 and 6035 after reaching the top
(upper-arm 0.97 / 0.93): stop and retreat choices stretched episodes to 400-557 decisions, beyond the value's
training range, and the grasp was lost on the way. Three bodies never got the sleeve past the elbow under any
policy. Stopped after one job. Four value-teacher variants are now level with r1; local one-action lookahead,
even with a long-horizon value, does not convert these failures.

Where r1 fails on the 201 test units (all 11 evaluated policies): 37 of its 71 failures fail for every policy
(35 of them never get the sleeve past the elbow); 34 are solved by at least one policy (12 by only one).
In 800 archived grasp-loss episodes, the sleeve was partly on the forearm in 43 % (forearm ratio 0.3-0.8),
barely entered in 32 % (hospital gown 129 of them), stalled at the elbow in 6 % and lost after passing the elbow
in 19 %. Once the sleeve reaches the upper arm the episode succeeds 81 % of the time; otherwise 7 %. Start
geometry predicts reaching the upper arm with AUC 0.79 (garment alone 0.66).

## Next: post-training from failures with value advantages (running)

Bc0 and r1 imitate successes only; the 4,546 archived failures are unused although they show which actions make
the state worse. Hazard-advantage-weighted post-training (`compute_advantage_weights.py`, `finetune_fmvp_bc.py
--state-weights`): every encoded state of 1,500 successes and 1,500 grasp-loss failures (training bodies only,
every third state) gets w = clip(exp((V(s_{t+8}) - V(s_t)) / 0.1), 0.05, 20); the actor is fine-tuned from r1
for 3 epochs. Control: the same data and budget with success-only uniform weights (self-imitation). Both are
evaluated on development bodies 1-14 x 5 garments against r1 (`haw_eval_dev_20261004`). This is offline
advantage-weighted regression with an asymmetric privileged critic; the established ingredients are AWR/IQL-style
weighting, the question is whether failure data with a hazard critic improves a pretrained dressing policy where
success-only imitation does not.

### Result: advantage weighting from failures hurts (2026-10-04)

`haw_eval_dev_20261004`, development bodies 1-14, stopped after 18 of 20 jobs (one HAW tshirt_26 job lost to a
CUDA start-up fault): r1 39 of 55, success-only control 37 of 56, HAW 33 of 55; HAW vs r1 2:8 (p = 0.11), HAW vs
control 1:4. Per-step credit from differences of a correlational value (V(s_{t+8}) - V(s_t)) is not reliable
enough to reweight actions; it suppresses useful ones. Together with the four value-teacher variants, every
local correction (one-action lookahead, value ranking, reweighting, distillation) lands at or below r1 on
static Cloth3D. 37 of r1's 71 test failures fail for all eleven evaluated policies, all derived from FMVP: a
shared blind spot that local post-training around the base behaviour does not leave. A scripted-expert and an
r1-to-expert handoff run on tshirt_26 and hospital gown (bodies 1-14) are measuring whether those units are
solvable at all (`expert_headroom_20261004`, `handoff_headroom_20261004`).


## Headroom: a different strategy solves units the FMVP family cannot (2026-10-04)

Scripted seven-stage expert (`dressing_heuristic.py`, privileged arm landmarks, moves at the speed cap) and an
r1-to-expert handoff at forearm ratio, development bodies 1-14 x tshirt_26 / hospital gown
(`expert_headroom_20261004`, `handoff_headroom_20261004`), paired with r1 and flow from `policy_eval_test_20260927`:

| policy | tshirt_26 | hospital gown | all |
|---|---|---|---|
| r1 | 6/14 | 8/14 | 14/28 |
| flow | 8/14 | 9/14 | 17/28 |
| expert | 5/14 | 1/14 | 6/28 |
| handoff | 2/14 | 3/14 | 5/28 |

The expert is much weaker overall (it loses the grasp after about 135 decisions on most units), but it is not
a worse version of the same behaviour. On the 12 units both r1 and flow fail, every FMVP-family policy evaluated
so far (r1, flow, bc0, r2, flow_e2e, lookahead and value teachers, HAW; 14-18 trials per unit) succeeded 14 times in
187 trials, and on tshirt_26 3047 / 5035 / 8049 twice in 51. Expert and handoff together succeed on those three
in 4 of 6 attempts (8049: 0 of 18 before, 2 of 2 now) and on gown 5035 once. The oracle over r1, flow, expert and
handoff is 21/28 against r1's 14/28.

The placement does not explain it either: across 8,246 archived r1 rollouts, 42 garment-body pairs that failed
at the default offset never succeeded reliably at another offset (0 always-succeed, 4 mixed).

Reading: r1's failures are concentrated on units its strategy family does not solve; local per-step corrections
around r1 stay inside that family. A strategy with a different global plan (hover height, waypoints through
finger, past the elbow, hooked over it, past the shoulder, explicit cuff alignment) solves some of them.

## Method hypothesis: strategy-level post-training

The decision that determines the outcome is global and made at entry, so post-training should search and learn
at the strategy level, not per action:

1. **Strategy search in IPC.** For each training unit (garment x body), evaluate a small family of strategies
   to completion: r1, the scripted expert family (parameters: hover height, elbow/shoulder overshoot, outward
   offset around the bend, speed), and r1-to-strategy handoffs. Outcomes per (unit, strategy).
2. **Strategy selection.** Learn P(success | initial observation, strategy) from those outcomes and choose
   the strategy per unit. The initial observation is the point cloud the deployed policy already sees.
3. **Distillation.** Train a strategy-conditioned student (or r1 fine-tuned on the selected strategy's
   successes only where r1 is predicted to fail), so the deployed policy covers the union.

Success criteria, in order: (a) the strategy family's union clearly exceeds r1 on development units with
reproducible rescues; (b) selection from the initial state recovers a large part of the oracle gap on held-out
bodies; (c) the distilled student beats r1 on test bodies 15-41.

Running: expert reproducibility with a new seed (`expert_seed2_20261004`), the expert on the other three
garments (`expert_headroom_20261004`), then five family members (slow, low, high, outward, deep) on development
bodies 1-14 x tshirt_26 / hospital gown (`expert_family_20261004`, `collect_garment.py --expert-params`).

## Strategy family: parameters do not diversify; phases do (2026-10-04)

The expert's rescues reproduce exactly with a new seed (`expert_seed2_20261004`: the same six units succeed). On
the other garments the default expert is weak (tshirt_68 0/14, tshirt_4 0/14, tshirt_392 3/13, both of the latter
already solved by r1). Varying its parameters (`expert_family_20261004`: half speed, hover 8 cm, hover 16 cm)
changes almost nothing: every variant succeeds on the same units. Parameter search inside the scripted family
does not widen coverage. (`outward_offset` does not exist in the installed `dressing_heuristic`; that variant
did not run.)

What differs is the phase. The expert covers the whole forearm (forearm ratio 1.0) on 39 of 42 units by about
decision 90 and then loses the grasp on the upper arm. r1 is the opposite: 75 % of its failures are at entry
or on the forearm, and once the sleeve reaches the upper arm it succeeds 81 % of the time. The earlier handoff
ran r1 first and the expert second, the wrong way round.

### Reverse composition: expert entry, then r1

`collect_garment.py --variants entry --entry-forearm 0.95`: the scripted expert threads hand and forearm, r1
takes over once the sleeve covers 95 % of the forearm (decision 84-100). Development bodies 1-14 x 5 garments
(`entry_compose_20261004`), paired with r1 and flow from `policy_eval_test_20260927`:

| garment | entry | r1 | flow | entry-only : r1-only |
|---|---|---|---|---|
| hospital gown | 12/14 | 8/14 | 9/14 | 4 : 0 |
| tshirt_4 | 13/14 | 11/14 | 11/14 | 3 : 1 |
| tshirt_392 | 11/13 | 11/13 | 12/13 | 0 : 0 |
| tshirt_68 | 6/14 | 9/14 | 8/14 | 3 : 6 |
| tshirt_26 | 0/14 | 6/14 | 8/14 | 0 : 6 |
| all | 42/69 | 45/69 | 48/69 | 10 : 13 |

The per-unit oracle of entry and r1 is 55/69. Successful compositions are faster (gown 191-198 decisions
against 281-289 for r1). On tshirt_26 every composition reaches upper-arm 0.4-0.76 and then loses the grasp
under r1: the handoff state carries more tension than r1's own states (gripper force p90 110-124 N against
25-65 N for r1 alone on tshirt_26/68). The tool rotation at handoff is small (0-43 degrees), so it is not an
orientation mismatch.

The r1 continuation value at the handoff state separates outcomes with AUC 0.85, but only through the garment:
within a garment it saturates near 1 (tshirt_68 failures 0.99), because handoff states lie outside the r1
state distribution it was trained on. It cannot yet time the handoff.

Running: handoff thresholds 0.6 and 0.8 on tshirt_26, tshirt_68 and hospital gown (`entry60`, `entry80` in
`entry_compose_20261004`).

### Handoff timing decides it; the expert's rotation is not needed (2026-10-04)

Same protocol, handoff when the sleeve covers 60 % / 80 % of the forearm instead of 95 %
(`entry60`, `entry80`), and a translation-only expert (`entry60nr`, `--expert-no-rotation`: the expert's
rotation command is zeroed, the tool keeps r1's vertical-only rotation rule after the handoff):

| condition | tshirt_26 | tshirt_68 | gown | tshirt_4 | tshirt_392 | all | vs r1 (win : loss) |
|---|---|---|---|---|---|---|---|
| r1 | 6/14 | 9/14 | 8/14 | 11/14 | 11/13 | 45/69 | |
| flow | 8/14 | 8/14 | 9/14 | 11/14 | 12/13 | 48/69 | |
| entry, handoff 0.95 | 0 | 6 | 12 | 13 | 11 | 42/69 | 10 : 13 |
| entry, handoff 0.8 | 0 | 11 | 12 | - | - | 23/42 | 8 : 8 |
| entry, handoff 0.6 | 9 | 12 | 11 | 11 | 13 | 56/69 | 15 : 4 |
| translation-only entry, 0.6 | 11 | 13 | 14 | 13 | 13 | **64/69** | **20 : 1** (sign test p = 2e-5) |

Handing off at 60 % of the forearm (decision 60-70) instead of 95 % turns tshirt_26 from 0/14 into 9-11/14.
The translation-only expert is at least as good as the full one, so the useful part of the expert is a route,
not a cuff rotation: hover above the finger, then drive the opening along the forearm to past the elbow. A
translation-only student can imitate it.

Handoff states drift out of r1's state distribution as the threshold rises (median 10-NN distance to 109k
archived r1 decision states, standardized privileged features, same garment: 0.53 at 0.6, 0.89 at 0.8, 1.30
at 0.95), which matches the timing effect at the population level. Within a condition the distance does not
predict a unit's outcome (AUC 0.52), so it is not a per-state handoff rule.

## Method: phase-composed teacher distillation

The pretrained policy fails in one phase (entry) and is strong in the next (upper arm, 81 % success once
reached). Post-training therefore composes a teacher from (i) a privileged translation-only route for the weak
phase and (ii) the pretrained policy itself for the strong phase, handing off early enough that the policy
receives states it knows, and distills the composite into the single point-cloud student:

1. Collect composite rollouts (translation-only expert entry, r1 from 60 % forearm) on training bodies
   (`composite_train_20261004`, 28 training bodies x 5 garments, running).
2. Fine-tune r1 on the accepted composite episodes (all states: the expert's entry translations and r1's own
   continuation from composite states), `finetune_fmvp_bc.py`, init r1, 3 epochs.
3. Evaluate the student alone (no expert) on development bodies 1-14, then once on test bodies 15-41.
4. If the student loses part of the entry, DAgger: run the student, relabel entry-phase states with the
   expert's translation, aggregate.

The translation-only teacher beats r1 by 20 : 1 on the 69 development units (64 vs 45; flow 48). The method claim stands only if the distilled student, without privileged input, keeps a clear part of
that gain on the held-out test bodies.

### First distilled student: plain BC loses the entry (2026-10-04)

Composite rollouts on the first 28 training bodies (all from body regions 1-2): 132 of 138 accepted, 30,686
states (8,957 expert entry, 19,089 r1 continuation, 2,640 hold). Fine-tuned from r1 (trust 0.5, 3 epochs,
`pctd_20261004/model`). On held-out training bodies the trunk follows the expert's entry direction (median
cosine 0.93, r1 as is 0.68) and keeps r1's own actions; weaker trust (0.05, 10 epochs) reaches 0.96.

The student alone (no expert), development bodies 1-14 x 5 garments (`pctd_eval_dev_20261004`): **31/69,
below r1's 45/69**. 30 of its 38 failures never cover 60 % of the forearm, i.e. the entry fails. Bodies 1-7
(regions 1-4) 23/35 (r1 29), bodies 8-14 (regions 4-8) 8/34 (r1 16): the student generalizes poorly beyond the
two body regions it was trained on, and open-loop imitation of the expert's states leaves it without
corrections once it drifts.

Next (running, `dagger_chain.sh` in the session scratchpad):

1. Composite rollouts on one training body from each of regions 3-27 (25 bodies), student v2 on all 53 bodies
   (trust 0.1, 8 epochs).
2. DAgger round: v2 executes on 25 further training bodies (one per region 3-27) with the expert running in
   shadow (`--variants dagger_entry`); entry states before the 60 % handoff are labelled with the expert's
   translation (`finetune_fmvp_bc.py --expert-dagger`, kind 7), accepted episodes add their own continuation.
3. Student v3 on the aggregate, evaluated alone on development bodies 1-14.
