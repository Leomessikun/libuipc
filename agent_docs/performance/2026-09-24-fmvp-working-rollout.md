# Working FMVP rollout route, 2026-09-24

Use the unchanged `fmvp_sim.pt` actor in Genesis+IPC with the complete SMPL-X body. No critic, network fine-tuning, or post-training algorithm is needed for this collection step.

The legacy seven-ring validator mislabeled the armhole seam as the free cuff and inspected the shirt torso. Its zero-success verdict was incorrect. `scripts/wang_transfer/physical_sleeve.py` instead follows the actual free cuff and three connected sections of the sleeve mesh. It keeps the original upper-arm progress threshold of 0.7 and verifies that all four sections remain around the arm throughout the hold. The native Genesis viewer can show these sections with `--show-physical-sleeve`.

## Reproduce the collector

```bash
/home/ge47gax/kun/genesis-world/.venv/bin/python scripts/wang_transfer/collect_better_rollouts.py \
  --hang output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz --hang-key k300 \
  --preflight-manifest output/uipc_manip/fmvp_ipc_rot_strain10_preflight_20260923.json \
  --bodies 14046 14047 14049 --variants baseline --replicas 3 --seed 1006 \
  --steps 300 --hold 20 --success .7 --success-geometry physical_sleeve \
  --yaw 267 --rotation fmvp --collision-geometry full_body --placement-offset-mm 0 5 0 \
  --abort-gripper-force 1000 --cloth-density 750 --cloth-strain-rate 10 \
  --out output/uipc_manip/fmvp_ipc_verified_collection_NEW_RUN
```

The output directory must be new. `accepted.json` lists complete, held, grasp-valid simulation episodes; failed attempts remain saved separately in the same run for diagnosis. One action is 0.1 simulated seconds, so `--hold 20` checks a two-second endpoint. The main run uses the historical bridge voxel and yaw settings; the optional single-voxel/matched-yaw experiment did not improve the combined pilot and is not the collection setting.

Each NPZ has aligned `obs[t]`, `actions[t]`, `obs[t+1]`, proposed checkpoint actions, controller IDs, cloth vertices, full-body mesh, tool positions, forces, grasp checks, and physical sleeve geometry. External holding has controller ID 2; checkpoint-driven steps have controller ID 0. These are virtual-gripper Cartesian actions, not Franka joint trajectories. Repeats of one body/start are correlated and should not be counted as distinct body/pose coverage.

The roughly 0.20 kg shirt and strain coefficient 10 are experimental simulator settings. Geometry completion does not calibrate force, prove real textile behavior, or qualify hardware deployment. Record deformation and simulated loads alongside each trajectory rather than discarding genuine sleeve completion using the wrong garment region.

## View the actual saved motion

```bash
/home/ge47gax/kun/genesis-world/.venv/bin/python scripts/wang_transfer/view_rollouts_genesis.py \
  output/uipc_manip/fmvp_ipc_mass020_three_controllers_20260923/body_14046_seed_1004/baseline.npz \
  output/uipc_manip/fmvp_ipc_mass020_three_controllers_20260923/body_14047_seed_1004/baseline.npz \
  --show-physical-sleeve
```

Space pauses, arrows step, Home restarts, and Tab switches episodes. Red is the real free cuff; yellow lines are connected sections between the cuff and armhole. No Matplotlib is used.

## Unattended larger collection

`scripts/wang_transfer/collect_dataset_background.py` targets 500 accepted
episodes without agent or API calls. It varies 16 body IDs and small initial
placement offsets, then tries the baseline, half-speed, and quarter-speed
checkpoint controllers. It independently audits every saved episode with a
20-decision hold and rejects failed grasps or extreme deformation (empirical
guards: edge p99 <=2.25 and maximum <=4 relative to state zero). These are
simulation dataset guardrails, not calibrated textile limits.

The job directory is `output/uipc_manip/fmvp_dataset_500_20260924`.
`status.json` gives live progress; `manifest.json` lists accepted episodes;
`attempts.jsonl` retains every verdict. The attempt budget is 2000. A low-disk
condition, three consecutive process errors, or a `STOP` file ends collection
without relaxing the acceptance criteria. Creating `STOP` waits for the
current batch to finish. The PID is saved in `runner.pid`. This script can
resume from the same output directory and seed-run arguments.
