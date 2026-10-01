# Garment generalisation and moving-arm baselines (2026-10-01)

After round 2 of IPC post-training saturated on the five Cloth3D garments
(`2026-09-28-policy-eval-and-clothesnet.md`: r2 127 vs r1 130 of 201; the lookahead teacher ties r1), the work
returns to the project's two goals: post-training that generalises across garment types, and dressing a moving
arm. The GPU is shared with another user's GR00T fine-tune (about 64 GiB), so jobs ran one at a time below
70 GiB through a queue script.

## Moving arm: both policies fail under GRAB motion; the motion alone does not

Runner: `probe_arm_motion.py` of the `research/anticipatory-dressing` worktree, run unmodified (Empty-FEM body
driven by per-vertex soft position constraints, `wang_live_arm` observation, tshirt_26, body 14046, onset 1 s,
GRAB clips converted at 35 % amplitude, 3 s long). Outputs in `output/uipc_manip/grab_baseline_20261001`.

| clip | r1 | fmvp_sim | hold (no robot action) |
|---|---|---|---|
| s1_mug_pass (119 mm excursion) | grasp lost at decision 13 (1.3 s) | grasp lost at decision 13 (1.3 s) | no failure over 60 decisions (6 s) |
| s1_mug_lift (63 mm excursion) | grasp lost at decision 154 (15.4 s) | grasp lost at decision 147 (14.7 s) | no failure over 200 decisions (20 s) |

No run puts the sleeve on the upper arm (maximum upper-arm ratio 0). Body tracking stays below 0.5 mm in every
run, so the drive follows the motion. With the arm still, r1 dresses this body (held success at decision 206
in the anticipatory-dressing pilot, `2026-09-26-anticipatory-dressing-pilot.md` in that worktree).

Reading: the motion does not break the grasp by itself; the hold control keeps it through both clips. Grasp loss
comes from the policy's actions combined with the motion. On mug_pass the policy keeps pulling while the arm
moves and the gripper's tracking limit is exceeded 0.3 s after onset; on mug_lift the motion ends at 4 s and
the policy then fails to dress the displaced arm, losing the grasp after about 11 s of trying. Both
checkpoints fail the same way, so this is a property of policies trained on a static arm, and a target for
motion-aware post-training. One body and two clips: failure modes, not rates.

## Garment generalisation: the lookahead teacher ties r1 on ClothesNet training garments too

Teacher check on 8 hung ClothesNet training garments x the first 7 test bodies: plain r1
(`cn_train_r1_20261001`) against r1 with the one-step lookahead (round 2's labeller, ring-distance fix;
`cn_train_teacher_h1_20261001` for the first garments, `cn_train_teacher_h1b_20261001` for tcsc_083 and
tcsc_top558 to reach the informative garments sooner; the first driver was stopped after 3 garments).

Plain r1 reaches 11 of 55 units: tcsc_083 6/7, tcsc_top558 3/7, and 2 of 41 on the six collared jackets
(shirt007, suit007, jacket112, jacket081, tcnc_jacket143, model2_054). On the 34 units both ran, the teacher
reaches 10 vs r1's 9 (1:0 discordant); on the two garments where r1 succeeds at all it is 6 vs 6 and 4 vs 3.

Reading: on new garment types the one-step labeller is no better than r1 either, so collecting its labels on
these garments would repeat round 2. The collared jackets fail for both (mostly no threading or grasp loss
early), which no short lookahead changes. All 16 hung ClothesNet training garments are collared (TCLO, TCNC,
TCSC); the categories where r1 transfers best on the held-out set (cn_tcsc_model2_*, cn_tnlc_*) have no hung
training garments yet.
