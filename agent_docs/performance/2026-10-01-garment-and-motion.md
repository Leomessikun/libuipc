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

## M2 pilot: one planner under three motion beliefs (interim, mug pass, 2026-10-01)

`scripts/wang_transfer/motion_lookahead_probe.py` (ca8463c6) runs r1 with an IPC lookahead on the GRAB-driven
body of the anticipatory-dressing worktree (used unmodified). The snapshot covers the IPC world, the env state and
the motion state (clock, body target, target joints, cells, meshes); candidates are scored with the arm landmarks
at the end of their own rollout. Every decision from 0.3 s before onset to 1 s after the motion ends, 12
candidates (nominal, half, stop, no rotation, +-0.25 per axis, nominal + believed fingertip shift, shift only),
each held 4 decisions; score = ring progress - excess load, infeasible on grasp loss, tracking or stretch. The
three conditions differ only in the human motion assumed inside the candidate rollouts (and the fingertip shift):
current pose held, causal constant-velocity extrapolation from the last decision, true GRAB future. The executed
episode always follows the real motion. Body 14046, tshirt_26, onset 1 s, 450 decisions, success = interior
armhole endpoint held 21 states with a valid grasp. Outputs in `output/uipc_manip/m2_pilot_20261001`.

| s1_mug_pass | outcome | plans / changed | most chosen |
|---|---|---|---|
| r1 alone | grasp lost at decision 13 | - | - |
| r1 + GICP (this run) | grasp lost at decision 13 | - | - |
| planner, current pose | grasp lost at decision 13 | 6 / 0 | nominal every time |
| planner, causal | success, held from decision 133 | 44 / 39 | +y 25, no rotation 5, +z 4, follow 4 |
| planner, true future | success, held from decision 145 | 44 / 42 | +y 29, +z 8, follow 2 |

Under the current-pose belief every candidate looks safe and the planner keeps the policy's action, which fails
exactly like r1; with either motion belief it moves the gripper off the policy's action (mostly +0.25 on y) and
the episode both keeps the grasp and dresses the arm. The explicit follow-the-arm candidates rarely win.

Reproducibility caveat: the anticipatory-dressing pilot ran r1 + GICP with the same arguments, checkpoint, hang,
motion and motion code (`pilot_pass_continuation_20260927`) and it succeeded (held from decision 156); here it lost
the grasp at decision 13. The only changed file is the policy bridge, whose change adds a flow-policy path that r1
does not use. Single episodes under motion therefore do not reproduce run to run, and none of the rows above is a
rate. Repeats of GICP with the same arguments: alone, grasp lost at decision 150; r1 then GICP in one process as in the
pilot, r1 at 13 and GICP at 153; alone again, 13. Over five identical runs GICP ends at 13, 13, 150, 153 and one
success (the pilot's), while r1 alone fails at 13 in every run so far. GICP sits near a threshold where small
run-to-run differences (shared GPU, solver tolerance) decide whether the grasp survives the motion, so every
condition needs repeats; two more runs per planner condition are queued (`scratchpad/queue_m2b.sh`).

Mug lift, one run each (motion ends at 4 s; the planner stops at decision 50): planner with causal motion succeeds
(held from decision 131); planner with the true future reaches armhole fraction 0.65 and then loses the grasp at
decision 156, after planning has ended; r1 + GICP loses it at 189; r1 alone lost it at 154 (fmvp_sim at 147); the
current-pose planner loses it at 168. On this clip the failures come after the motion, where all conditions run the
same unplanned policy, so the run-to-run variance seen for GICP can decide them; repeats are needed here as well.
