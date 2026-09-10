# Camera rigs for the dressing observation

Date: 2026-09-10. Branch `agent/camera-rigs` from `cloth-cable-manip-rl` at
`0b9ccfc9`. The question was whether a camera setup like Hello Robot's
Stretch 3, with a head view and a wrist view, would give the policy a better
observation and help training. This record builds the rigs as
`DressingObsConfig.mode` options (`python/uipc_manip/dressing_obs.py`, trainer
flag `--obs-mode`) and measures them on identical recorded expert states.

## Stretch 3 cameras

| Camera | Sensor | Field of view | Range | Mount |
|---|---|---|---|---|
| Head depth | RealSense D435if, Stretch 3's upgrade of the D435i with an IR-pass filter | 87 x 58 degrees | 0.3 to 3 m ideal | Pan-tilt head. `joint_head` sits 1.33 m up the mast. Mounted in portrait: at zero pan and tilt the camera's image-width axis runs along the mast (composed from the SE3 URDF's joint rotations), so 87 degrees span the vertical |
| Head wide-angle | Colour only | 140 degrees | none | Pan-tilt head, for teleoperation |
| Gripper | RealSense D405 in Stretch Gripper 3, pointed at the fingertips, which carry ArUco markers | 87 x 58 degrees | 7 to 50 cm ideal | `gripper_camera_joint` at (0, 0.044, 0.042) m in the gripper body; `joint_grasp_center` at 0.23 m. The camera sits about 19 cm behind the grasp centre and 4.4 cm off the gripper axis |

Head pan spans -3.9 to 1.5 rad, tilt -1.53 to 0.79 rad. The lift travels
1.1 m, and the telescoping arm has four 0.13 m stages (SE3 URDF limits).

Sources:
- [Stretch 3, what's new](https://hello-stretch3.com/stretch-3-whats-new)
- [D435if specifications](https://www.intel.com/content/www/us/en/products/sku/233194/intel-realsense-depth-camera-d435if/specifications.html)
- [D405](https://www.intelrealsense.com/depth-camera-d405/)
- [stretch_urdf, `SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf`](https://github.com/hello-robot/stretch_urdf)

What related dressing and assistive work learns from:

- **Wang et al., RSS 2023.** This is the protocol the port pretrains on. It uses a Sawyer arm and one RealSense D435i. The arm cloud is captured before dressing, under a static-pose assumption. The garment cloud is live.
- **[FMVP](https://arxiv.org/abs/2509.12741).** This is the preset behind `visible_dual`, with the same Sawyer and a single D435i. The arm moves, so only its visible part is observed. The limitations section names single-camera occlusion and proposes more cameras or active sensing. Its worst arm motion, lowering the arm, is put down to the garment covering the arm from the camera.
- **[Bodies Uncovered](https://arxiv.org/abs/2109.04930), on a Stretch RE1.** Three fixed external cameras over and beside the bed are fused by ICP. It does not use the robot's own head camera.
- **Hello Robot's [Stretch AI](https://github.com/hello-robot/stretch_ai).** Its learning-from-demonstration stack builds on LeRobot and needs the gripper D405.

This search found no dressing policy trained from Stretch's own head and
gripper depth views.

## Wang's observation against its code

This section reads `pointcloud_3` in `/home/ge47gax/kun/dressing/dress_env.py` and the pretraining launcher `curl/launch_train_curl.py`.

- **Camera.** One camera, `default_camera`, is fixed in the FleX world (y up) at (-0.7, 1.7, 1.2). Its angles are (-80, -35, 0) degrees and it renders 360 x 360.
  - The compiled FleX scene places the human, so its pose relative to the camera cannot be read from the Python repository.
  - `wang_static_arm` therefore keeps the port's front-oblique camera: about 1 m from the arm's midpoint and 0.35 m above it.
- **Arm cloud.** `voxelized_partial_right_arm_pc` is built once, in `_reset` (lines 1197-1201):
  - it starts from the reset-time depth capture of the human, voxelised at 6.25 cm;
  - it keeps the voxels within one voxel of the right-arm mesh;
  - the garment never hides it.
- **Garment cloud.** It comes from `flex_get_occluded_cloth_rgbd` (`util.py` line 117):
  - the garment is rendered alone, and its pixels are blanked wherever the human's depth is nearer;
  - so the body hides the garment and the garment hides itself;
  - the picker is hidden while rendering.
- **Gripper point.** One point at the gripper completes the cloud, and the cloud is centred there.
- **Randomisation.** The launcher turns every observation randomisation off: `randomize_camera`, `randomize_cloth_observation`, `randomize_gripper_pos`, erosion, dilation and `full_obs_guide`. The voxel is 0.0625 m.

`visible_dual` differs from Wang in five ways:
- It uses two cameras instead of one.
- The sleeve hides the arm as it goes: FMVP's visible arm instead of a static capture.
- The body does not hide the garment.
- The cameras follow the arm's landmarks instead of staying fixed in the world.
- Jitter and dropout are on by default.

## The rigs

| Mode | Cameras | Arm | Occluders | Jitter |
|---|---|---|---|---|
| `visible_dual` (default, unchanged) | Front-oblique and side-overhead, 70 degrees, 96 px | Live, hidden by the sleeve | Arm and garment | 3 cm, dropout 0.1 |
| `wang_static_arm` | The front-oblique camera alone | Captured without the garment | Arm, garment, body | 3 cm |
| `stretch3_head` | D435if head (details below) | Live | Arm, garment, body | 3 cm |
| `stretch3_head_wrist` | The head plus a D405 on the tool (details below) | Live | Arm, garment, body | Head 3 cm, wrist 5 mm, dropout 0.1 |

The Stretch cameras in detail:

- **D435if head.** Portrait, 58 x 87 degrees, 56 x 96 px, 0.3 to 3 m. It stands 0.8 m from the arm's midpoint toward the person's front-right, 1.30 m above the floor, and is aimed at the arm's midpoint.
- **D405 on the tool.** 87 x 58 degrees, 48 x 28 px, 7 to 50 cm. It sits 19 cm behind and 4.5 cm above the grasp point, aimed 10 cm past it. At the reset the gripper points along the forearm toward the elbow, and the grip rotation carries the camera.

Setting `static_arm=True` gives any rig Wang's static arm.

How the rigs render occlusion:
- Every body point is an occluder, except points within 1.5 cm of the arm.
- Each segment keeps its own depth buffer. A point is hidden by another segment 5 mm in front of it, such as skin under a sleeve, but by its own segment only 2 cm in front.
- Each point covers the pixels its 1 cm radius spans at its depth, so a garment sampled at its vertices stays opaque 10 cm in front of the wrist camera.

The environment passes the rigs' extra inputs through `RigInputs.from_cells`: the elbow, the shoulders' lateral direction, the floor, the body points, the tool point and the grip rotation. The legacy modes ignore every new field and keep their observations bit for bit, which `test_rig_fields_leave_the_legacy_modes_untouched` holds.

`pretrain_wang.py` forwards `--obs-mode` to the trainer: `prepare(["teacher", "--region", "13", "--obs-mode", "stretch3_head_wrist"])` gives that mode in `dressing_config`.

## What each rig sees

**Setup.**
- **States.** The scripted expert was recorded for 300 decisions of six steps, from a frozen snapshot of `0b9ccfc9`, on 24 cells:
  - tshirt_26 and tshirt_68 on bodies 0-7;
  - the same two garments on region-13 bodies 14006, 14007, 14013 and 14015.
  - That makes 7,200 cell-decisions.
- **Rendering.** Every rig re-renders the same states offline, without augmentation.
- **What the fractions count.** They are of arm points, assigned to the hand, forearm or upper arm by their nearest limb segment, and of garment vertices.
- **GPU.** The observation time is for one 16-cell batch on the shared RTX PRO 6000, one run each, with other jobs on the GPU.

**Caveat.** `visible_dual` and `visible_single` keep the legacy depth buffer: no body occluders, and a 2 cm tolerance between arm and garment. Skin up to 2 cm under the sleeve therefore counts as seen, so their arm fractions are upper bounds against the rigs.

| Rig | Obs ms | Hand | Forearm | Upper arm | Opening ring | Arm within 15 cm of opening | Garment within 10 cm of tool | Voxels arm / garment | Voxel IoU, step to step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `visible_dual` | 30.0 | 0.76 | 0.31 | 0.53 | 0.49 | 0.29 | 0.57 | 62 / 180 | 0.91 |
| `visible_single` | 28.5 | 0.73 | 0.21 | 0.35 | 0.34 | 0.19 | 0.44 | 52 / 151 | 0.90 |
| `wang_static_arm` | 52.1 | 0.91 | 0.30 | 0.47 | 0.29 | 0.45 | 0.39 | 62 / 139 | 0.91 |
| `stretch3_head` | 28.5 | 0.53 | 0.33 | 0.25 | 0.30 | 0.23 | 0.49 | 49 / 134 | 0.90 |
| `stretch3_head_wrist` | 31.3 | 0.58 | 0.35 | 0.27 | 0.39 | 0.28 | 0.60 | 52 / 139 | 0.90 |
| `stretch3_head_wrist`, `static_arm` | 35.8 | 0.71 | 0.44 | 0.41 | 0.39 | 0.51 | 0.60 | 63 / 139 | 0.91 |
| `xray` | 42.8 | 1 | 1 | 1 | 1 | 1 | 1 | 82 / 228 | 0.95 |

By expert stage (sample counts: approach 962, middle 2,187, align_yaw 655, elbow_hook 856, last 1,183, done 1,136):

| Rig | Opening ring: approach / middle / align / hook / last / done | Arm within 15 cm of opening: same stages |
|---|---|---|
| `visible_dual` | 0.68 / 0.30 / 0.44 / 0.51 / 0.52 / 0.67 | 0.24 / 0.25 / 0.25 / 0.28 / 0.31 / 0.45 |
| `wang_static_arm` | 0.35 / 0.13 / 0.30 / 0.33 / 0.38 / 0.40 | 0.49 / 0.38 / 0.39 / 0.37 / 0.49 / 0.63 |
| `stretch3_head` | 0.45 / 0.35 / 0.20 / 0.37 / 0.23 / 0.16 | 0.15 / 0.22 / 0.23 / 0.25 / 0.19 / 0.36 |
| `stretch3_head_wrist` | 0.56 / 0.45 / 0.33 / 0.48 / 0.30 / 0.24 | 0.31 / 0.25 / 0.27 / 0.33 / 0.22 / 0.36 |
| `stretch3_head_wrist`, `static_arm` | 0.56 / 0.45 / 0.33 / 0.48 / 0.30 / 0.24 | 0.63 / 0.48 / 0.42 / 0.57 / 0.44 / 0.60 |

**The wrist camera adds what it was placed to add, but cannot see through the sleeve.**
- Against the head alone, it lifts the opening ring from 0.30 to 0.39 and the garment by the tool from 0.49 to 0.60.
- The gain is largest on the forearm pull: the ring rises from 0.35 to 0.45 during `middle`.
- The arm near the opening rises only from 0.23 to 0.28.

**The Stretch head sees less of the upper arm than the port's two cameras, and loses the opening late.**
- It sees 0.25 of the upper arm, against 0.53.
- The opening ring falls to 0.23 in `last` and 0.16 in `done`.
- From the front-right, 1.3 m up, the torso and the stretched garment stand between the head and the shoulder.

**Only a static arm restores the arm near the opening.** That is the information the sleeve hides: 0.45 with Wang's one camera, and 0.51 with the Stretch pair.

## Decodability

**Probe design.**
- **Model.** The actor's PointNet++ encoder and extra vector, with a 256-256 MLP head.
- **Data.** Observations packed as the environment packs them, every second decision:
  - training on 2,400 from bodies 0-5, 14006 and 14007, for 20 epochs;
  - testing on 1,200 from bodies 6, 7, 14013 and 14015.
- **Targets.** Relative to the tool: the opening centroid, the opening normal and the garment centroid, plus the forearm and upper-arm ratios.
- **Seeds.** Three, reported as mean and standard deviation.
- **Overfitting.** Training error on the opening is about 2 cm, so the probe overfits this set. Differences under about 0.5 cm are within the seed spread.

| Rig | Opening, cm | Normal, deg | Garment, cm | Forearm ratio RMSE | Upper-arm ratio RMSE |
|---|---:|---:|---:|---:|---:|
| blind: goal and tool points only | 6.46 +- 0.27 | 27.3 +- 0.3 | 10.48 +- 0.69 | 0.44 | 0.31 |
| `visible_dual` | 4.08 +- 0.17 | 15.1 +- 0.4 | 4.87 +- 0.29 | 0.30 +- 0.02 | 0.27 +- 0.02 |
| `wang_static_arm` | 4.53 +- 0.28 | 17.1 +- 1.4 | 5.65 +- 0.22 | 0.29 +- 0.02 | 0.26 +- 0.02 |
| `stretch3_head` | 4.72 +- 0.35 | 18.5 +- 1.0 | 6.29 +- 0.23 | 0.35 +- 0.04 | 0.31 +- 0.04 |
| `stretch3_head_wrist` | 4.46 +- 0.51 | 19.5 +- 1.0 | 5.98 +- 0.20 | 0.36 +- 0.03 | 0.31 +- 0.04 |
| `stretch3_head_wrist`, `static_arm` | 4.68 +- 0.16 | 18.3 +- 0.4 | 6.01 +- 0.26 | 0.35 +- 0.05 | 0.31 +- 0.04 |
| `xray`: every point | 4.36 +- 0.41 | 15.1 +- 1.4 | 5.39 +- 0.10 | 0.30 +- 0.02 | 0.26 +- 0.03 |

**No rig decodes better than `visible_dual`.**
- Every rig beats blind by a wide margin, so the cloud carries the state.
- None beats `visible_dual`, and neither does `xray`, which sees everything. Here the 6.25 cm voxel and the encoder bound what is read, not the cameras.
- The Stretch rigs are 0.4 to 0.6 cm worse on the opening, about 1 cm worse on the garment, and 0.05 worse on both progress ratios. The wrist camera's visibility gain does not survive the voxel.

## Decision

- **Default.** It stays `visible_dual`. No RL A/B is queued: a rig change is not expected to speed training in simulation.
- **Wang-faithful pretraining.** Use `--obs-mode wang_static_arm --no-obs-augment`.
  - It is Wang's observation: one camera, the arm captured before dressing, the garment hidden by the body, and no observation randomisation.
  - It decodes within 0.5 cm of `visible_dual`, and keeps the arm near the opening in view.
  - Its camera pose is the port's, not Wang's world-fixed one.
- **Sim-to-real on a Stretch 3.** Train with `stretch3_head_wrist`, so the observation is the one the robot will produce.
  - It costs about 0.4 cm on the opening and 1 cm on the garment.
  - `static_arm=True` needs the arm scanned before the gripper approaches, which the pan-tilt head can do.
  - If the real robot is another platform, match its actual camera; Wang and FMVP both used one external D435i.

**Reproduce.** The scripts are in the session scratchpad, under `camera/`:
- `record_expert.py`;
- `analyze_rigs.py`;
- `probe_rigs.py`;
- `rigs_common.py`.

They write `rig_metrics.json`, `rig_probe.json` and `rig_probe_seeds12.json`.
