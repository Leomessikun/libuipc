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
