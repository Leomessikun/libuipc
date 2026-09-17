# Review of the data-efficient dressing RL research plan — 2026-09-18

Status: offline measurements on saved FQL artifacts plus literature surveys;
nothing trained or simulated. It reviews the plan in
[the FQL record](2026-09-17-fql-pretraining.md#research-plan-after-the-completed-pilot)
and the owner's eight-point version of it (central question: turn existing
manipulation experience into a policy that handles difficult contact
transitions with little extra simulation and total time). Scripts:
`output/uipc_manip/sleeve_path_audit_20260917/fql_vs_teacher.py`,
`teacher_recoveries.py`, `teacher_stalls.py`.

## Measurements on the 30,000-update FQL pilot

**1. The withheld-body gap is mostly the teacher's gap.** Under the corrected
reward the teacher's 25 reconstructed episodes pass coverage-and-grasp 9 times:
7 of 15 on the training bodies 14045–14047 and 2 of 10 on the withheld bodies
14048–14049. On the four evaluated cells:

| Cell | Teacher on that cell | FQL actor, two rounds |
|---|---|---|
| tshirt_26 / 14046 (training body) | pass | pass, pass |
| tshirt_68 / 14046 (training body) | pass | pass, pass |
| tshirt_26 / 14048 (withheld) | **fail**: 0.35 upper arm, 3.3 cm tracking error | fail, fail (never on the upper arm) |
| tshirt_68 / 14049 (withheld) | pass | fail (7.8 cm lateral, 2.4 cm tracking), pass |

"4/4 on training bodies, 1/4 on withheld" therefore does not show overfitting:
one of the two withheld cells is a cell the data generator cannot do. The
withheld bodies are the harder bodies for the teacher. A generalization claim
needs cells chosen by the teacher's outcome, not only by body identity.

**2. The actor does not copy the teacher in the contact phase.** Mean absolute
translation-action difference to the teacher on the same cell, by 50-decision
block: 0.05, 0.2–0.5, 0.4–0.5, 0.1–0.3, 0.0, 0.0 (normalized action units),
on training and withheld cells alike, including the ones it completes.

**3. The training set has no alternative at a stall.** Eight of the twelve
failed teacher episodes stall from decision 125–153 on: the opening moves less
than 1 cm per 20 decisions for about 130 decisions while the translation
command stays saturated. Stalled decisions are 1,340 of 7,500 (18 %); 1,045 of
them carry a saturated command. Every one of those states has one action label,
"keep pushing". A critic fit to this set cannot rank an action that was never
tried there. The thirteen "progress losses later regained" in the set are
reading discontinuities of the ray metric, not deliberate recoveries.

**4. `early_turn` is inherited.** The teacher triggers it in 24 of its 25
episodes; all eight FQL rollouts trigger it. It measures the scripted route.

**5. Outcomes from a restored state are noisy** (measured earlier, see
[the repeatability record](2026-09-17-dressing-research-redesign.md)): identical
commands from one restored snapshot give final coverages 0.230, 0.246, 0.197
after 12 decisions, with up to 14.5 cm of cloth-coordinate spread. A single
continuation is one sample of a distribution; comparing two actions at a state
needs several continuations each.

## Prior art for "selective recovery experience" (survey, primary sources)

Every component is published; the closest work is rigid-body only.

| Work | Selects | Criterion | Restore | Equal-cost evidence |
|---|---|---|---|---|
| Tavakoli et al., "Exploring Restart Distributions", [1811.11298](https://arxiv.org/abs/1811.11298) | visited states | uniform, TD error, or return | yes | yes: 10 % of the same total steps |
| RFCL, Tao et al., ICLR 2024, [2405.03379](https://arxiv.org/abs/2405.03379) | per-demonstration reset state and initial states | success frontier, PLR-style rank | yes | yes: beats RLPD, JSRL, Cal-QL and uniform demo-state resets at 1–2 M steps from ≤ 5 demos |
| SCOUT, Aphale, Singh, July 2026, [2607.26417](https://arxiv.org/abs/2607.26417) | (context, scaffold level) cells | binary success EMA with an anti-stall rule | yes | yes: shared budget, includes a random-scaffold baseline; names the two questions "scaffold access" and "scaffold allocation" |
| AuxSS, Mehra et al., 2025, [2507.04606](https://arxiv.org/abs/2507.04606) | demonstration states | early episode termination | yes | yes: 300k steps for all |
| VDS, Zhang et al., [2006.09641](https://arxiv.org/abs/2006.09641) | goals | Q-ensemble disagreement | no | partial |
| PLR, Jiang et al., [2010.03934](https://arxiv.org/abs/2010.03934) | levels | value loss and staleness | no | yes |
| Go-Explore, [2004.12919](https://arxiv.org/abs/2004.12919) | archive cells | visit counts | optional | not established; billions of frames |
| ThriftyDAgger, [2109.08273](https://arxiv.org/abs/2109.08273) | states to query a human | novelty or `1 − Q̂` under a label budget | no | approximate |
| PLD, [2511.00091](https://arxiv.org/abs/2511.00091) | failure regions probed by a residual actor | — | no | abstract only |
| VinePPO, ICML 2025, [2410.01679](https://arxiv.org/abs/2410.01679) | every intermediate state (language) | none | yes | wall-clock; states that intermediate resets are "rare in typical RL settings" |
| OmniReset, ICLR 2026, [2603.15789](https://arxiv.org/abs/2603.15789) | reset distributions, rigid contact-rich assembly | diversity only, no adaptation | yes | no |

Read at full text by the survey agent: the rows above except PLD. Not found
after targeted search: visited-state reset selection by critic unreliability;
branching several continuations from selected restored states to build
Monte-Carlo critic targets in robotics; any reset curriculum for deformable
manipulation; any reset paper that charges the restore operation to the
budget. The modified start distribution changes the objective (Tavakoli et al.
say so and do not solve it); evaluation must be from ordinary task starts.

A local precedent also constrains the idea: the
[recovery-teacher experiment](2026-09-17-recovery-teacher.md) found a recovery
route from a failed state (0.98 coverage) and supervising with it did not
improve the policy (3/8 against 2/8). Acquiring recovery data is not enough;
the open part is how the evidence enters policy improvement.

## What limits withheld-body success (survey, primary sources)

- **Configuration diversity, not episodes per configuration.** Lin et al.
  (ICLR 2025, [2410.18647](https://arxiv.org/abs/2410.18647), full text):
  generalization follows a power law in the number of distinct training
  objects or environments and shows no such law in demonstrations per
  configuration once a threshold is met; 8 objects give a normalized score
  above 0.8 and 32 above 0.9; their recipe is 32 configurations at 50
  demonstrations each. Mediratta et al. (ICLR 2024,
  [2312.05742](https://arxiv.org/abs/2312.05742), full text): at a fixed
  1 M transitions, raising training levels from 200 to 100k raises test
  return and shrinks the gap, while 10 M transitions at 200 levels does not;
  behavior cloning beats offline RL on unseen levels. Gao et al. (RSS 2024,
  [2403.05110](https://arxiv.org/abs/2403.05110)): a factor present in the
  task but not varied in the data drops real success from 77.5 % to 2.5 %;
  vary each factor along a "stair", not the full cross product.
- **The reference pipeline's scale.** One Policy to Dress Them All (full
  text): 27 pose regions × 50 poses (45 train) × 5 garments = 6,750
  configurations, body shape, height and gender resampled per pose; 1e6
  steps per method; no GPU hours stated anywhere in that paper or in FCVP or
  FMVP. Its stated reason a single RL policy fails is multi-task RL
  optimization (uneven learning speed, conflicting gradients), which does not
  apply to cloning from a scripted expert. FMVP later cut poses per region
  from 45 to 5, added four body sizes, and replaced on-policy distillation by
  offline behavior cloning on more than 8,000 filtered simulated trajectories.
- **Teacher to student.** Where DAgger was ablated against plain cloning
  (ExpertGen [2603.15956](https://arxiv.org/abs/2603.15956), VIRAL
  [2511.15200](https://arxiv.org/abs/2511.15200)) it improved recovery from
  the student's own mistakes; FMVP and TraKDis show filtered offline cloning
  can suffice for deformables. DAgger fixes covariate shift, not coverage.
- **Automated generation across assets** is the cost lever: DexGarmentLab
  retargets one demonstration across 2,500 garments by structural
  correspondence; FoldNet++ ([2609.12433](https://arxiv.org/abs/2609.12433),
  abstract) generates 120k episodes over 1,000 T-shirts rule-based and reports
  above 90 % on unseen real T-shirts; RoboCasa measures 28.8 % → 47.6 % from
  generated data.

Our training set has three body values on the withheld axis, two orders of
magnitude below every calibration point above. This does not contradict
measurement 1: the teacher's own failures and the tiny body count act
together, and the teacher is also the generator that must be scaled.

## Which learner belongs next to FQL (survey, primary sources)

- **FQL is no longer the top line on OGBench**, and the claims that beat it
  disagree with each other: chunking (QC-FQL [2507.07969](https://arxiv.org/abs/2507.07969);
  Adaptive Q-Chunking [2605.05544](https://arxiv.org/abs/2605.05544), an
  independent PKU table: FQL 37 → 58 offline-to-online against QC-FQL 38 → 86,
  AQC 62 → 96, and plain RLPD 67 online with no offline phase), expanded-MDP
  flow RL (RQL [2606.17551](https://arxiv.org/abs/2606.17551), same lab as
  FQL, offline only, 100 M-transition headline tasks), and a modernized
  Gaussian-style behavior-regularized actor-critic (ReBRAC-v2
  [2608.01205](https://arxiv.org/abs/2608.01205): 74.8 against the next
  52.3; its cloning term is 88 % of its performance; it did not rerun FQL).
  Every one of these numbers is at 1–100 M transitions with state inputs.
- **Nothing in the flow-Q family is demonstrated at our scale** (7,500
  transitions, 25 episodes, point clouds). The small-narrow-data evidence is
  negative for expressive flow policies: on D4RL Adroit-human (about 25 demos
  per task) FQL's own table has ReBRAC 59 against FQL 52 and needs α in
  1,000–30,000; Value Flows ([2510.07650](https://arxiv.org/abs/2510.07650))
  reports that swapping the flow behavior policy for a Gaussian one improves
  Adroit by 10 %; EXPO ([2507.07986](https://arxiv.org/abs/2507.07986),
  ICLR 2026) declines to pretrain on Adroit "due to the narrowness of the
  dataset". FQL's advantage is argued from multimodal action data; 25 episodes
  of one scripted controller are close to unimodal.
- **The bottleneck paper.** "Is Value Learning Really the Main Bottleneck?"
  ([2406.09329](https://arxiv.org/abs/2406.09329), NeurIPS 2024): the barrier
  is usually policy generalization on test-time states outside the data's
  support, not the critic; behavior-regularized policy gradient (FQL's actor
  form) beats weighted regression. Remedies are state coverage and test-time
  or online correction, not a better offline objective.
- **Methods shown at small data with high-dimensional inputs** (all outside
  the flow-Q family): DSRL (NeurIPS 2025, frozen BC policy, SAC over its
  latent noise, no offline reward labels: 10 demonstrations, 20 % → 90 % in
  under 50 real episodes); WSRL ([2412.07762](https://arxiv.org/abs/2412.07762),
  ICLR 2025: warm the buffer with the pretrained policy's own rollouts and
  drop the offline data; 17k transitions, two wrist cameras, 13/20 → 20/20 in
  18 min where RLPD stayed at 0/20); DICE-RL ([2603.10263](https://arxiv.org/abs/2603.10263),
  frozen flow policy plus residual, 40–265 demos, pixels, no code); OGPO
  ([2605.03065](https://arxiv.org/abs/2605.03065)): vanilla policy extraction
  fails with image critics and needs conservative advantages, a warning for
  point-cloud critics. The "Three Regimes" analysis
  ([2510.01460](https://arxiv.org/abs/2510.01460)) says there is no uniform
  winner: anchor to the pretrained policy (WSRL) when it is better than the
  data, to the data (RLPD) when the data are better.
- **Shortlist for our setting**, in order: a tuned behavior-cloning arm (the
  flow prior FQL already trains; Mediratta et al. and the Adroit rows say it
  is the thing to beat); RLPD (no cloning term, so no narrow-data trap;
  needs the point-cloud encoder); WSRL; DSRL; then QC-FQL as the cheapest
  upgrade of FQL itself (one flag; chunking suits a 300-decision contact
  task). Keep RQL, AQC (no code) and ReBRAC-v2 as references. floq
  ([2509.06863](https://arxiv.org/abs/2509.06863)) reports a flow-matching
  critic 2× better and far more robust to noisy TD targets than FQL's
  monolithic critic, which matters for a critic fit to 7,500 noisy rows.
- Two measurements to place ourselves before choosing: the cloning policy's
  return against the dataset's return (the Three Regimes test), and whether
  the action distribution at a state is multimodal at all.

## What this implies for the plan

0. **On the learner.** FQL is a defensible baseline, not the newest and not
   one shown at our data scale. The matched comparison the plan asks for
   should include a tuned cloning arm, RLPD and WSRL, and QC-FQL as FQL's
   cheap upgrade; the choice between anchoring to the policy or to the data
   follows from the Three Regimes measurement, not from a leaderboard.
1. **Where a contribution can be.** Not in the mechanism's parts. In the
   measured answer to two questions under an expensive simulator, on
   deformable contact: which visited states deserve restored continuations,
   and how branched outcomes should enter the critic and the actor.
   Measurements 3 and 5 make branching load-bearing here: bootstrapped TD has
   no alternative action to bootstrap from at a stall, and single outcomes are
   too noisy to rank actions. Budgets must be total simulator time including
   restore, which no surveyed paper reports.
2. **Baselines the mechanism must beat at equal total simulator cost**:
   ordinary-start online fine-tuning (RLPD or the FQL continuation); RFCL with
   the same demonstrations; TD-error-prioritized visited-state resets; uniform
   visited-state resets. If critic-based selection does not beat TD-error
   selection the method reduces to Tavakoli et al.
3. **Ablations**: branched evidence to the critic only against critic and
   actor; one against several continuations per state; budget in wall time
   including restore against environment steps; evaluation from ordinary
   starts.
4. **Evaluation cells must be stratified by the teacher's outcome**
   (teacher passes, teacher stalls, teacher loses the grasp), on bodies outside
   training, or the generalization question stays confounded (measurement 1).
5. **Scale the generator before the learner.** Every survey calibration
   point says the withheld axis needs tens of bodies, not more episodes on
   three. The scripted expert is queryable at any state and any body, so
   this is simulator time only: 32 bodies × 10–20 episodes is 320–640
   episodes, about 2.6–5 h at the measured 733 s per 25. The teacher's own
   success rate caps what cloning can reach, which is why the elbow repair
   comes first.
6. **The first experiment that needs no new learner**: from the eight stalled
   teacher states, restore and run K continuations each of a few scripted
   alternatives (back off, lift, lateral re-centre, keep pushing) and record
   the outcome distributions and the restore cost. It answers whether better
   continuations exist at those states, how many samples separate them given
   measurement 5, and what a restore costs; all three are prerequisites of the
   mechanism and none is known.
