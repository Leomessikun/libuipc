# Paired test evaluation of fmvp_sim, r1 and flow; ClothesNet onboarding (2026-09-28)

## Paired evaluation on held-out bodies

`scripts/wang_transfer/eval_policies_batched.py` runs every policy on every (garment, test body) unit of
the body-split index (`output/uipc_manip/dressing_dataset_20260927/index.json`, 41 test bodies x 5
garments) from the same start: v5 collection settings (armhole-aligned start, body-fit filter, garment
sizing, interior-sections success, 750 decisions + 20-decision hold), seed 2026092700, one replica,
7-body batched worlds under MPS. A body with no legal start at the first offset moves to the second;
201 of 205 units have a legal start (tshirt_392 has 4 without).

Policies: `fmvp_sim.pt` (released PyBullet fine-tune, zero force), r1 (behaviour cloning + one IPC
lookahead DAgger round, frozen encoder), and the flow-matching chunk policy
(`dressing_dataset_20260927/flow_r1_h16/flow_policy_eval.pt`: frozen r1 encoder features + gripper
load, 16-action chunks, trained 30k steps on 1,807 accepted v4/v5 episodes; native actions, gripper
load as force input).

| garment | units | fmvp_sim | r1 | flow |
|---|---|---|---|---|
| tshirt_26 | 41 | 18 | 22 | 25 |
| tshirt_68 | 41 | 14 | 26 | 24 |
| tshirt_4 | 41 | 28 | 31 | 32 |
| tshirt_392 | 37 | 26 | 32 | 33 |
| hospital_gown | 41 | 14 | 19 | 24 |
| all | 201 | 100 (50 %) | 130 (65 %) | 138 (69 %) |

Exact McNemar on discordant pairs: r1 vs fmvp_sim 37:7 (p = 5e-6); flow vs fmvp_sim 45:7 (p = 7e-8);
flow vs r1 16:8 (p = 0.15). Failures are mostly lost grasp (flow 53, r1 65 of their failures).

Reading:
* The IPC fine-tune is a real gain over the released checkpoint under identical settings, concentrated
  in tshirt_68 (+12) and the gown (+5); none on tshirt_4.
* Flow is at least r1 but not measurably better; it imitates r1's accepted rollouts, and it changes
  architecture, force input and data at once, so a later gain would not be attributable to one of them.
* Absolute rates are inflated: every test body has an accepted episode on some garment in the index.
  Paired differences are unaffected. Paired starts are not bit-identical (cloth settles differently by
  0.03-4.8 mm between runs, measured by the anticipatory-dressing audit), so single pairs are samples.
* Most of the rise from the pre-fix rates (0 to a third per garment before 2026-09-25) is the
  environment and criterion fixes (body filter, sizing, sections-wrap, armhole alignment), not training.

Outputs: `output/uipc_manip/policy_eval_test_20260927/` (r1, flow) and
`policy_eval_test_fmvp_20260927/` (fmvp_sim), each with `results.jsonl`; `--summary` prints the table.

## ClothesNet garments in the pipeline

ClothesNet tops (`/home/ge47gax/kun/ClothesNet/extract/.../ClothesNetM/Tops`, 395 sleeved tops in four
categories; some entries are mislabelled, e.g. gloves among long sleeves) ship as 4-18 unwelded sewing
panels in centimetres, Y up; panel seams coincide within 0.5 mm.
`scripts/clothesnet/prepare_garment.py` welds them (skipping a panel that would make an edge
non-manifold), keeps the largest piece, converts to metres in Wang's raw frame, and writes Wang-style
tables to `output/uipc_manip/clothesnet_assets/index.json`:
* cuff: the boundary loop farthest out at -X (Wang's tables use the -X sleeve);
* armhole seam: the welded vertices between the cuff's panels and the rest (exact, from the panels);
  `shoulder_polygon` is six of them around the ring;
* grasp patch: the 150 vertices nearest the shoulder-ridge point 13 cm medial of the armhole centre
  (stepping in when a wide neckline leaves no ridge there); pickers 4.5 cm medial and 5 cm above it;
  alignment line along the ridge. These copy where Wang's tables sit on tshirt_26/68/392.

`scripts/clothesnet/garment_idx_clothesnet.py` merges them with Wang's tables
(`UIPC_MANIP_GARMENT_INDEX_MODULE`); the drape bake takes the pull schedule from the index module for
garments outside its built-in table (`dressing_bake.pull_schedule`, sac-stability). The collector's
`--sleeve-template` gives the physical-sleeve test the prepared mesh.

Pilot, one garment per category: all four weld with no skipped panel, the physical-sleeve sections
find the right cuff and three closed interior sections, the libuipc bake accepts the welded meshes,
and the gravity hangs settle (opening 24-34 cm from the picker, 55-61 degrees down; tshirt_26:
21.6 cm, 55 degrees). The zero-shot evaluation (`output/uipc_manip/clothesnet_eval_pilot_20260928/`,
14 test bodies x 4 garments x 3 policies) is running; first batches (7 bodies each, identical starts):

| garment | fmvp_sim | r1 | flow |
|---|---|---|---|
| cn_tnsc_043 (short sleeve) | 3/7 | 4/7 | 5/7 |
| cn_tnlc_010 (long sleeve) | 3/7 | 5/7 | 4/7 |

Every body had a legal start. Rendered frames of an accepted cn_tnsc_043 episode show the sleeve on the
upper arm; a failed one has the hand beside the opening and the garment sliding off.

## Data scaling of the flow policy (offline)

Same trainer, 30k steps, fixed validation split (255 episodes), training on a fixed random subset of
training bodies (`train_flow_policy.py --train-fraction`):

| training bodies | training states | best val first-action MSE (sum of 4 dims) |
|---|---|---|
| 25 % | 146k | 0.0185 |
| 50 % | 278k | 0.0185 |
| 100 % | 571k | 0.0191 |

Imitation error does not fall with more trajectories of the five Cloth3D garments; the current data
saturate this model. Closed loop agrees: flow matches r1 on the training garments but loses to it on
unseen ClothesNet garments (below), so what is missing is garment diversity, not episode count.

## Interim zero-shot result on held-out ClothesNet garments (2026-09-28 14:00)

8 of 20 test garments, 62 units paired across the three policies: fmvp_sim 13, r1 21, flow 15.
r1 vs fmvp_sim 9:1 (p = 0.02); flow vs r1 0:6 (p = 0.03). Interim look, three comparisons: treat as
provisional. Open-front jackets and the coarse cn_tnsc_top236 (16 mm edges, grasp lost within 20
decisions) fail for every policy and are start/asset failures rather than policy differences.
Cuff-aimed starts (`--align-target cuff`) lost to armhole-aimed ones on the pilot garments
(9/14 vs 3/14, 6:0 discordant, p = 0.031), though they rescued one long-sleeve group (0/7 -> 3/7);
armhole aiming stays the default.

## Where r1's gain comes from: BC-only controls (final, 201 test units, 2026-09-29)

r1 was trained in two steps from `fmvp_sim.pt` (frozen encoder, trunk fine-tuned): round 0 = behaviour
cloning on 157 accepted fmvp_sim episodes (68,929 states, 8 epochs; `fmvp_ipc_bc_20260924`), then 8 more
epochs from round 0 on those states plus the 6,857 states of one IPC DAgger round (75,786 states). Only
1,505 of those 6,857 carry an IPC-lookahead label; the other 5,352 are behaviour-cloning states of the
accepted episodes added (`features/episodes.json` lists 177 episodes, 157 of round 0 plus 20: the 8 DAgger
rollouts of `fmvp_dagger_r1_20260924`, which carry the labels and all succeeded, and 12 fmvp_sim successes from
`fmvp_dataset_multiregion_500_20260924` that finished after round 0 was built). All of r1's data is tshirt_26:
r1 predates the multi-garment collector (2026-09-25). The
control `bc_cont` continues round 0 for the same 8 epochs, learning rate, trust and hold weight on the
68,929 BC states only (`fmvp_bc_continued_control_20260929`). All four on the same 201 units:

| model | accepted |
|---|---|
| fmvp_sim | 100 |
| bc0 (round 0) | 118 |
| bc_cont (round 0 + 8 epochs, no IPC labels) | 105 |
| r1 (round 0 + 8 epochs with one IPC DAgger round) | 130 |

Exact McNemar, discordant pairs: bc0 vs fmvp_sim 32:14 (p = 0.011); bc_cont vs bc0 10:23 (p = 0.035);
r1 vs bc_cont 33:8 (p = 1e-4); r1 vs bc0 22:10 (p = 0.050).

Reading: behaviour cloning on the checkpoint's own successes helps; training further on the same successes
hurts (back to the released level); training further with the IPC-labelled states instead improves on
round 0. One IPC DAgger round (its labels plus the episodes it steered) is the difference between the two
continuations, at an equal training budget; a labels-only control is still needed to credit the labels alone.
Their gain over round 0 is borderline (p = 0.05); the controlled contrast is r1 vs bc_cont.

## All control policies on the 201 test units (final, 2026-09-29)

Same 201 units, same starts, exact McNemar on discordant pairs (`multi_way.py` over the six result dirs):

| policy | what it is | accepted |
|---|---|---|
| fmvp_sim | released PyBullet fine-tune | 100 |
| bc_cont | round 0 + 8 epochs BC only | 105 |
| flow_fmvp_all | flow (frozen encoder) on all 447 accepted fmvp_sim episodes | 97 |
| flow_fmvp | flow (frozen encoder) on 236 fmvp_sim successes, matched units | 104 |
| bc0 | round-0 BC on fmvp_sim successes | 118 |
| r1 | round 0 + 8 epochs with one IPC DAgger round | 130 |
| flow_r1m | flow on 236 r1 successes, same units as flow_fmvp | 132 |
| flow_e2e | end-to-end point encoder, all r1 data (force-rotation fix) | 137 |
| flow | frozen-encoder flow, all r1 data | 138 |

Key contrasts: r1 vs bc_cont 33:8 (p = 1e-4); flow_fmvp vs flow_r1m 12:40 (p = 1e-4); flow_fmvp vs fmvp_sim
18:14 (p = 0.60); flow_r1m vs r1 16:14 (p = 0.86); flow vs flow_r1m 15:9 (p = 0.31); flow_e2e vs flow 13:14
(p = 1.0); flow_e2e vs r1 20:13 (p = 0.30).

Reading: a flow student lands at its teacher's level (fmvp_sim data -> fmvp_sim level; r1 data -> r1 level),
so the flow head and the encoder (frozen vs end-to-end) do not change closed-loop success on these garments;
the gain comes from the teacher, and the teacher's gain over the equal-budget BC control comes from the IPC
labels. More r1 data (all vs matched 236 episodes) leans positive but is not significant. Unseen-garment
results for flow_e2e are in `clothesnet_eval_test_e2e_20260929` (running).

More fmvp_sim data does not lift the student past its teacher either: `flow_fmvp_all`
(`fmvp_flow_all_20260929`, 447 accepted fmvp_sim episodes from `fmvp_sim_rollouts_scale_20260929` and the
earlier collection, about 1.9x the 236 of flow_fmvp) reaches 97/201, vs fmvp_sim 15:18 (p = 0.73), vs r1 6:39
(p < 1e-4), vs flow 7:48 (p < 1e-4). Doubling the teacher's own successes leaves the flow policy at the
teacher's level, so the r1-data flow policies' gain is not a data-volume effect.

## Held-out ClothesNet garments: the end-to-end encoder does not generalise (2026-09-29)

20 held-out garments (`clothesnet_testset_20260928.json`), first 7 test bodies, same starts for every policy.
flow_e2e (encoder trained from scratch on the five Cloth3D garments) against the frozen-FMVP-encoder
policies, on the 132 units all four completed:

| policy | accepted |
|---|---|
| flow_e2e | 24 |
| fmvp_sim | 40 |
| flow (frozen encoder) | 44 |
| r1 | 47 |

flow_e2e vs flow 2:22 (p < 1e-4), vs r1 3:26 (p < 1e-4), vs fmvp_sim 5:21 (p = 0.003). On the training
garments the same checkpoint matches flow (137 vs 138 of 201). Its unseen-garment failures are mostly
750-decision timeouts with partial progress, not early grasp loss: the scratch encoder fits the five
garments' geometry, while the FMVP encoder (pretrained in FleX on many garments) carries the transfer.
Keep the pretrained representation for garment generalisation.

The same evaluation for fmvp_sim, r1 and flow (240 units so far, second body group still running):
fmvp_sim 51, r1 66, flow 67; r1 vs fmvp_sim 23:8 (p = 0.011), flow vs fmvp_sim 23:7 (p = 0.005),
flow vs r1 13:12 (p = 1.0).

## Held-out ClothesNet garments: final (20 garments x 14 test bodies, 2026-09-29)

All 840 planned results are in (no-legal-start units excluded; lost jobs from the two MPS teardowns rerun).
254 units paired across the three policies: fmvp_sim 52 (20 %), r1 68 (27 %), flow 70 (28 %).
r1 vs fmvp_sim 25:9 (p = 0.009), flow vs fmvp_sim 25:7 (p = 0.002), flow vs r1 14:12 (p = 0.85).

| garment | units | fmvp_sim | r1 | flow |
|---|---|---|---|---|
| cn_tcsc_model2_013 | 14 | 10 | 9 | 12 |
| cn_tnlc_normal_model_034 | 13 | 7 | 9 | 10 |
| cn_tnlc_jacket021 | 14 | 7 | 10 | 9 |
| cn_tcsc_model2_080 | 14 | 7 | 7 | 9 |
| cn_tclo_027 | 14 | 8 | 7 | 5 |
| cn_tnsc_model2_020 | 14 | 5 | 8 | 7 |
| cn_tnsc_model2_004 | 13 | 4 | 5 | 4 |
| cn_tcnc_jacket144 | 11 | 2 | 4 | 5 |
| cn_tnlc_036 | 14 | 1 | 2 | 5 |
| cn_tclo_suit003 | 14 | 1 | 3 | 2 |
| cn_tcsc_top610 | 14 | 0 | 2 | 1 |
| cn_tnsc_top065 | 14 | 0 | 1 | 1 |
| cn_tnlc_062 | 14 | 0 | 1 | 0 |
| cn_tclo_048, cn_tclo_022, cn_tcsc_top483, cn_tcsc_top228, cn_tnlc_top315, cn_tnsc_top369, cn_tnsc_top236 | 71 | 0 | 0 | 0 |

The IPC fine-tune's gain over the released checkpoint holds on unseen garments; the flow student keeps its
teacher's level. Seven garments fail for every policy (never threaded, early grasp loss on a coarse mesh,
a cuff as wide as the armhole, or mostly no legal start): garment/start compatibility, not policy choice,
dominates the unseen-garment rate. Success on unseen garments (20-28 %) is about 40 % of the training-garment
rate (50-69 %).

### flow_fmvp_all on the held-out ClothesNet garments (first 7 test bodies, 2026-09-29)

The flow policy trained on all 447 accepted fmvp_sim episodes (`clothesnet_eval_test_flow_fmvp_all_20260929`),
paired on the 132 units that the flow_e2e run shares (same 20 garments, first 7 test bodies, same starts):

| policy | training data | accepted of 132 |
|---|---|---|
| flow_e2e | r1 rollouts, end-to-end point encoder | 24 |
| flow_fmvp_all | 447 fmvp_sim episodes, frozen encoder | 33 |
| fmvp_sim | released checkpoint | 40 |
| flow | r1 rollouts, frozen encoder | 44 |
| r1 | one IPC DAgger round | 47 |

flow_fmvp_all vs fmvp_sim 1:8 (p = 0.039); vs flow 3:14 (p = 0.013); vs r1 3:17 (p = 0.003).
On unseen garments the student of the uncorrected teacher lands slightly below that teacher, while the student
of the IPC-corrected teacher stays at its teacher's level; neither student exceeds its teacher. With the 201
Cloth3D units (97 vs 100), the data volume of the teacher's own successes is ruled out as the source of the
flow-on-r1 gain on both garment sets.

## Round 2: IPC labels on all five garments do not add to r1 (final, 201 test units, 2026-09-30)

Round 2 rolled out r1 with the batched one-step IPC lookahead (every 3 decisions from decision 40, gripper
load > 15 N, 10 candidates) on the 5 Cloth3D garments and 66 training bodies (`dagger_r2_cloth3d_20260929`,
357 episodes, 21,322 labelled states, 14x round 1, no test bodies). Three models, each 8 epochs from r1 with
r1's settings (lr 1e-4, trust 0.5, hold weight 2):

| model | data beyond r1's 75,786 states | accepted of 201 |
|---|---|---|
| r2 | + 21,322 IPC-labelled states (weight 3), no whole episodes (`fmvp_ipc_dagger_r2_20260930`) | 127 |
| r1c | nothing: r1 continued on its own data (`fmvp_r1_continued_control_20260930`) | 130 |
| si | + 202 of r1's own accepted episodes on the same 5 garments, 64,089 states = the labels' weight (`fmvp_ipc_selfimit_r2ctrl_20260930`) | 117 |

Exact McNemar on the same 201 units (`policy_eval_test_r2_20260930`): r2 vs r1 14:17 (p = 0.72), r2 vs r1c
14:17 (p = 0.72), r2 vs si 25:15 (p = 0.15), r1 vs si 22:9 (p = 0.029), r1c vs si 24:11 (p = 0.041), r2 vs
fmvp_sim 39:12 (p = 2e-4). Without test body 3047 (196 units): r2 vs r1 14:17, r1 vs si 21:9 (p = 0.043).

Reading: the second round of IPC labels, 14 times the first and spread over all five garments, leaves the policy
where round 1 put it (127 vs 130). Self-imitation of r1's own successes at the same training weight is the one
continuation that hurts (117, back to bc0's 118), so the labels are not harmful where self-imitation is, but
they no longer add. Label statistics match round 1 (32 % keep the nominal action, median candidate progress
spread 0.79 mm); round 2 picks the stop candidate in 25 % of labels against 14 % in round 1. Round 2 was scored
before the ring-distance fix of 29bde384, which changes 0.6 % of Cloth3D states. Neither label fix changes this (`policy_eval_test_r2fix_20260930`, same units): r2f, r1 data plus only the
round-2 labels that keep the nominal action or gain >= 1 mm (12,164 of 21,681), reaches 129 (vs r1 13:14,
p = 1.0; vs si 23:11, p = 0.058); r2w1, all labels at weight 1 instead of 3, reaches 121 (vs r1 11:20,
p = 0.15). Weak labels and label weight are not what holds round 2 back. The ClothesNet held-out eval of r2,
r1c and si, and a teacher check (r1 with the online lookahead at H = 1 and H = 4, fixed score) are running.

### Round 2 on the held-out ClothesNet garments (final, 254 units, 2026-09-30)

The same three models on the 20 held-out garments x 14 test bodies (`clothesnet_eval_test_r2_20260930`), paired
with the earlier run on the same 254 units: fmvp_sim 52, si 58, r2 62, r1c 66, r1 68, flow 70. r2 vs r1 9:15
(p = 0.31), r2 vs si 11:7 (p = 0.48), r1 vs si 18:8 (p = 0.076), r2 vs fmvp_sim 17:7 (p = 0.064). The Cloth3D
reading holds on unseen garments: nothing trained on top of r1 improves it, round-2 labels sit between r1 and
self-imitation, and self-imitation is the weakest continuation. Seven garments stay at zero for every model.

### Teacher check: the one-step lookahead is no better than r1 (2026-09-30)

A DAgger student cannot exceed its labeller, so r1 was run with the lookahead active at test time (round 2's
labeller: every 3 decisions from decision 40, load > 15 N, 10 candidates, ring-distance fix of 29bde384) on the
first 14 test bodies x 5 garments (62 units with a legal start; `policy_eval_test_teacher_h1_20260930`). The
teacher reaches 41 vs r1's 43 on the same units (2:4, p = 0.69); r2 39, r1c 41, si 43, fmvp_sim 38, flow 45.
The one-step labeller does not choose better actions than r1 already takes, which explains why 14x more of its
labels leave the student at r1's level. Most remaining failures are grasp loss, which a one-decision hold
cannot see. The same check at H = 4 is running (`policy_eval_test_teacher_h4_20260930`, `..._h4b_...`).
