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
