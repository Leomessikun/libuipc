# Wang's released policy in our simulator: it reads the scene and then leaves the arm

Date: 2026-09-22. Checkpoint `vision_based_policy.pt` (750,137 steps, best test return 0.750),
bridge `python/uipc_manip/wang_bridge.py` and `wang_client.py` (commit 09d30fca).

The reference trains 5,000,000 steps per region over 27 regions, which at our 14,500
transitions/hour is 345 hours for one region. Running their released policy in our environment
instead would skip the pretraining phase entirely. This is what happened when we tried.

## The bridge works

Their checkpoint cannot enter our agent -- the networks differ in module names and shapes -- but it
enters theirs, which is pure PyTorch and torch_geometric and needs no simulator. Strict
`load_state_dict` accepts it with nothing missing and nothing extra, which is what confirms every
encoder hyperparameter read off their launcher. The actor is the flow one: the trunk runs per point
and the gripper's row is the action.

Four conventions were matched: the 3-way one-hot over arm, garment and gripper against our four
flags; centring on the gripper, which both do; the vertical, their y-up against our z-up; and point
density, their 6.25 cm voxel against our 768-point cloud, which matters because the encoder's
ball-query radii are absolute. The rotation about the vertical is fixed nowhere in their code, so it
was measured rather than guessed.

## The policy reads our observation

Two checks, both on frames taken along the scripted expert's path rather than at reset alone:

* The commanded translation aligns with the direction from the gripper to the shoulder, with a
  clean unimodal peak: 0.669 at yaw 0, 0.909 at 330, 0.706 at 300. The scene is not being read
  through a wrong frame.
* Shuffling which points are arm and which are garment, keeping the same positions and the same
  single gripper point, moves the action by 1.243 on average and drops the alignment from 0.909 to
  0.415. The policy is reading the segmentation, not emitting a constant.

That rules out the failure mode where a policy ignores its input and the yaw sweep measures nothing
but the average goal direction.

## In closed loop it never dresses anything

Three action configurations, all at yaw 330:

| | cells | policy decisions | upper-arm ratio |
|---|---:|---:|---:|
| as-is, 8.66 mm per decision | 25 | 150 | 0.000 |
| translation held 12 steps (104 mm, theirs is 100), rotation divided back | 5 | 75 | 0.000 |
| the same with rotation left repeated | 1 | in progress | 0.000 |

Their `max_translation` is 0.1 m per decision against our 8.66 mm, so one of their decisions is
about twelve of ours; holding the action that long matches the displacement without touching our
speed limit or the solver. Holding repeats the rotation too, which is why it is divided back in the
second row and left alone in the third.

The single-cell diagnostic puts the policy and the scripted expert in *one world* on the same cell,
same seed, same solver, so only the controller differs. Distance from the gripper to the shoulder,
decomposed into the arm axis and the perpendicular:

```
rotation=scaled        rotation=full
k=  0  along +0.03  off-axis 10.8 cm     k=  0  along +0.03  off-axis 10.8 cm
k=100  along +0.83  off-axis 20.2 cm     k=100  along +0.41  off-axis 49.8 cm
k=400  along +0.32  off-axis 20.2 cm
k=800  along +0.37  off-axis 53.4 cm
```

The gripper does travel up the arm at first -- 0.03 to 0.83 of the fingertip-to-shoulder span by
decision 100 -- which is why the open-loop alignment measured well. It then leaves: by decision 800
it is 53 cm off the arm axis, further than the arm is long (44.5 cm). Leaving the rotation repeated
makes it leave sooner, so the rotation is not what was holding it back.

The winding test never fires. `opening_threaded` asks whether the fingertip-to-shoulder axis passes
through the garment's opening ring; for the policy it is 0 at every logged step of every run. The
sleeve is carried alongside the arm, never around it.

## The control says the metric and the cell are fine

The scripted expert, in the same world on the same cell, reaches `threaded` 1.0, forearm ratio 1.00
and **upper-arm ratio 0.987**. So the winding test can be satisfied under our contact model, and
this cell is dressable. The policy's three numbers are 0.0, 0.00 and 0.000.

## What this closes and what it leaves

Closed: using their released checkpoint as a teacher or as a source of demonstrations, by any route
that goes through our simulator. It reads the scene correctly and still does not dress.

Not established: *why*. The open-loop direction is right and the closed-loop behaviour is not, which
is the signature of states leaving the distribution the policy was trained on -- their particle
cloth against our mesh IPC -- but we did not separate that from a remaining interface difference.
`PickerMultipleParticleNoSphere`, the class that defines what their six numbers mean to the gripper,
is not in this softgym checkout, so the action semantics could not be read, only inferred from
`max_translation`. Anyone reopening this should start there.

Their simulator is not an option for settling it: PyFlex is built here but its binary wants
`cudaSetupArgument`, removed in CUDA 10.1, and the GPU is Blackwell, which CUDA 10 cannot target at
all. Two earlier blockers were cleared on the way (a missing `libSDL2` already present in the `curl`
environment, and `__powf_finite`, which a four-line shim supplies).

The bridge itself stays useful: any other checkpoint of theirs, including the two FMVP ones, can be
driven from here without repeating the work.
