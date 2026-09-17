# Sleeve-path and reward audit of the existing dressing data — 2026-09-17

Status: offline analysis of saved data, CPU only. No simulation, training,
solver, reward or policy change. It tests, on data already on disk, the premise
of the [sleeve-motion transfer proposal](2026-09-17-dressing-transfer-direction.md)
and the success protocol of its first experiment. The owner's stated goal for
the new direction (2026-09-17): a dressing policy that trains fast and easily on
one workstation; the per-region reinforcement-learning teachers of the reference
pipeline are hard to scale and need a cluster.

Scripts and outputs: `output/uipc_manip/sleeve_path_audit_20260917/`
(`sleeve_paths.py` → `sleeve_paths.txt`, `reward_cliffs.py` →
`reward_cliffs.json`). Run from the repository root with
`/home/ge47gax/kun/genesis-world/.venv/bin/python`; both finish in seconds.

## Data

`output/uipc_manip/expert_r13_heldout_s0`: 25 episodes of the seven-stage
scripted expert (`dressing_heuristic.py`), five garments × bodies 14045–14049,
300 decisions each, with the 35-D arm-frame geometry of
`dressing_privileged.py` and the 6-D action at every decision. The arm
centreline is the polyline finger → elbow → shoulder in that frame; a point's
*arc* is the normalized arc length of its closest centreline point (elbow at
0.57 on average), its *lateral* the distance to it.

`output/uipc_manip/abl_dense_s1/checkpoints/replay_latest`: the ordinary SAC
run's 125,016 transitions (24 slots, horizon 300). It stores observations,
actions and rewards only: `priv_dim` is 0, so **this replay carries no sleeve
geometry** and cannot label sleeve goals or an executor without re-simulation.

## 1. The opening is a different control variable from the tool

The offset between the opening centroid and the tool averages 19.3 cm and moves
by 11.2 cm (standard deviation of the offset vector) inside one episode; by
garment the mean offset runs from 14.0 cm (tshirt_68) to 29.6 cm (hospital
gown). A command that moves the tool by δ does not move the opening by δ.

Across the 11 episodes that end above 0.7 upper-arm ratio (five garments, four
bodies), at equal decision index:

| Path in centreline coordinates | Across-episode std of arc | Across-episode std of lateral |
|---|---:|---:|
| Tool | 0.083 | 8.45 cm |
| Opening centroid | 0.063 | 1.20 cm |

**Confound, stated plainly**: from stage 2 on, the expert already steers the
*opening centroid* (not the gripper) toward fixed arm-frame waypoints, with the
gripper moved at constant speed along `waypoint − opening` (an identity
Jacobian). The tight opening path is therefore partly a property of the data
generator. What the numbers support without that confound: the tool path that
realises one sleeve path differs by garment by 14–30 cm, so recorded commands
are garment-specific labels while the sleeve path is not. Any comparison of
action-sequence imitation with sleeve-goal control on these trajectories
inherits the expert's factorization; it speaks to the interface within this
expert's target family, not to dressing strategies in general.

The hospital gown's ring has a 15.9 cm mean radius with 16.1 cm between its
largest and smallest radius: a centre/normal/radius goal does not describe
that opening. It is the concrete case for the proposal's caveat that the full
contour may be needed.

## 2. The expert's failures are one executor failure, already in hand

Ten of 25 episodes never reach 0.7. In all ten the opening's arc stops at
0.61–0.79, just past the elbow; in nine the tool ends 0.14–0.27 of the arm
length ahead of it (tool arc 0.77–1.00). The eight t-shirt failures end 3.0–5.4
cm off the centreline against 0.9–3.2 cm for the successes, nine of the ten
exceed the 2 cm grasp limit (maximum tracking error 1.8–3.2 cm), and the
expert's final stage is `elbow_hook` in eight. By cell: tshirt_392 (the smallest ring, 7.4 cm
radius) fails on all five bodies, body 14049 fails with four of five garments.

| Opening arc bin | 0.5 (elbow) | 0.6 | 0.7 |
|---|---:|---:|---:|
| Median lateral, successes | 4.3 cm | 1.7 cm | 1.1 cm |
| Median lateral, never reached 0.7 | 4.4 cm | 3.6 cm | 4.1 cm |

This is the case the proposal hypothesized — the command keeps moving and the
sleeve does not — and it is a failure of the *executor*: the identity-Jacobian
servo has no stall detection, no back-off and no lateral regulation, and it
advances its stage on a step budget whether or not the opening followed. The
goal (slide the opening along the centreline) was right in every one of them.

## 3. Four "failures" are complete dressings scored zero

Four episodes reach an upper-arm ratio of 1.00 and end at 0.00. In each, the
ratio falls from 1.00 to 0.00 in **one decision while the opening centroid
moves less than 1 mm**, with the centroid at or 2–5 cm beyond the shoulder
point. `wang_progress` casts the upper-arm ray from the shoulder toward the
elbow and keeps hits in front of its origin, so an opening pulled past the
shoulder intersects neither ray; the task term then switches from
`forearm_len + 5 × upperarm_distance` (about +1.9) to minus the fingertip's
distance to the opening (about −0.6). A comment in the expert's last stage
documents the same cliff, and the expert stops on the reading to dodge it; the
metric and the reward still have it. Counted at the peak, the expert dresses
15 of 25, not 11 of 25.

## 4. The SAC reward is dominated by branch switches

In the ordinary SAC replay (rewards divided by the 0.5 reward scale, reset rows
dropped): the median one-decision reward change is 0.0032 and the 99th
percentile 0.049, but 128 decisions switch from an on-arm branch to the
off-arm branch with a median drop of 0.76 (62 from the upper-arm branch, 66
from the forearm branch; 0.31 per episode). Switches in either direction carry
**91 % of the summed squared one-decision reward change**. 36.5 % of decisions
are on the upper-arm branch and 0.06 % in the success zone. This quantifies the
discontinuity the [SAC audit](2026-09-17-normal-sac-rollout-audit.md) observed
on one rollout (0.371 → 0 over 3.78 mm). It is reported for the protocol below,
not as a reason to reopen SAC work, and it does not by itself explain why the
policy parks at 0.51–0.67.

## Consequences for the proposed first experiment

1. **Success rule.** "Sustained completion at the episode's end" penalizes a
   policy that dresses fully and keeps moving. Fix the rule before any run:
   success is reaching at least 0.7 with a valid grasp and the opening staying
   on or beyond the upper arm, with a stop rule or with past-the-shoulder
   counted as complete. Otherwise the metric decides the study.
2. **Labels.** Truncate the four overshoot episodes at their peak; do not drop
   them and do not use their tail as goals.
3. **Data.** Geometry-labelled native data is these 25 episodes plus the 12
   reconstructed ones (six action sequences replayed twice) and the audit
   captures. The SAC replay has no geometry. With six admitted source
   sequences on one body, a two-seed comparison of two imitation interfaces
   will be decided by noise. Twenty-five expert episodes cost 733 s of
   simulation; collecting a few hundred with logged geometry and observations
   is hours on one GPU and should come first.
4. **Teacher.** The imitation ceiling is the teacher's success: 6 of 25 with
   the grasp limit, 15 of 25 at the peak. Section 2 shows what to repair: the
   executor at the elbow for tight sleeves. A teacher with stall detection,
   back-off and lateral centring, tested on the same 25 cells (about 12
   minutes), is the cheapest decisive test of whether sleeve feedback fixes
   the dominant failure. If it does not, the sleeve-goal route is refuted
   before any training.
5. **Transfer.** No cross-simulator result can come from existing data: the
   Newton buffer has no geometry. The first screen is within IPC, across
   garments and bodies. A transfer study needs Newton rollouts re-run with
   geometry logging.
6. **Cost on one workstation.** Measured here: 3,000 supervised updates in
   86.8 s with no simulator call, 3,600 reconstructed transitions in 318.6 s,
   native evaluation at about 18 transitions/s (19,200 decisions ≈ 18 min).
   The closed IPC-label SAC arms were the opposite: the simulator and a
   derivative sidecar inside every update, 1.5 times SAC's wall time per
   transition for the replay arm.
