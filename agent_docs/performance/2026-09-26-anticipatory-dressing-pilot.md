# Anticipatory dressing: GRAB motion pilot

## Scope and decision

The owner authorized implementation and bounded experiments on
`research/anticipatory-dressing` on 2026-09-26. The isolated worktree is
`/home/ge47gax/kun/libuipc-anticipatory-dressing`, based on `a02afbe6`.
The original workspace and the `9eae3121` collection jobs are preserved.
This pilot uses one sleeve of `tshirt_26` to test the value of future motion
information. It does not establish bilateral dressing, unseen-garment
generalization, a learned teacher/student, or real-robot performance.

Proceed with the pilot, with two separate gates: useful future information and
predictability from causal observations. A successful privileged controller
does not establish that a student can infer its inputs. Failure of one pause
heuristic does not disprove anticipation in general.

## Primary-source check

| Work | Relevant evidence and boundary |
| --- | --- |
| [Sun et al., Dressing in Motion (2026)](https://arxiv.org/abs/2609.04759) | Static demonstrations, point-cloud diffusion policy, PDE arm representation and region registration for reactive correction. The PDF reports 0.77 for phone use and 0.50 for receiving an object, and discusses pre-contact motion and control-frequency limitations. These motivate a test; they do not identify the unique cause of failure. |
| [Hao et al., Force-Modulated Visual Policy (2025)](https://arxiv.org/abs/2509.12741) | The authors describe unstable/limited-fidelity actuated human simulation and inaccurate simulated force, then adapt with real data. This is not a proof that FleX cannot simulate any moving arm. The local r1 checkpoint is a direct action actor, not a generative flow-matching policy. |
| [Takase, Onda and Yamazaki (2025)](https://www.jstage.jst.go.jp/article/jrobomech/37/3/37_710/_article/-char/en) | Uses time-series depth images to predict the robot end-effector target, combined with force control for dressing. This differs from predicting a future arm trajectory, but rules out claiming the first dressing system using motion prediction. |
| [Clegg et al., Learning Human Behaviors for Robot-Assisted Dressing (2017)](https://arxiv.org/abs/1709.07033) | Learns simulated recipient behavior for sleeve insertion, explicitly motivated by anticipating human motion. Different actor and objective, but relevant prior art. |
| [GRAB](https://github.com/otaheri/GRAB) | Whole-body object interactions represented with SMPL-X. These are motion sources, not dressing demonstrations or measurements of how a recipient responds to cloth. |

This is a targeted prior-art check, not an exhaustive novelty certification.
Teacher distillation, privileged future inputs, motion prediction, and
diffusion/flow policies are established ingredients. A contribution would need
an explicit mechanism and controlled gains beyond their combination.

## Candidate mechanism after the pilot

Let $h_t$ be the observed point-cloud/action history, $m_{t:t+H}$ a future
human motion, and $u$ a candidate robot action chunk. An IPC lookahead teacher
can score consequences $C(s_t,u,m)$ using the actual cloth state $s_t$.
An oracle uses the recorded future. A deployable policy must use a forecast
distribution $p_\phi(m\mid h_t)$, rather than assume access to that recording.

A candidate extension is to train on actions that preserve sleeve threading
and recoverability across plausible futures, with a sampled risk objective:

$$
u^* = \arg\min_u \left[\frac{1}{K}\sum_{k=1}^K C(s_t,u,m_k)
  + \lambda\,\operatorname{CVaR}_{\alpha}\{C(s_t,u,m_k)\}_{k=1}^K\right],
\qquad m_k\sim p_\phi(m\mid h_t).
$$

Here $K$ is the motion sample count, $\lambda\geq0$ the weight on bad outcomes,
and $\alpha\in(0,1)$ the upper-tail probability level. This is a proposed
risk-aware target, not a new CVaR algorithm or an implemented result. Useful
costs include physical sleeve progress, loss of threading, grasp error and
elapsed time. Simulated contact force is not assumed calibrated. Teacher
sampling avoids relying on the unvalidated long-horizon physics gradients.

The substantive research question is whether selecting wait/follow/realign
actions for multiple possible contact outcomes improves distillation under
partial observations at affordable simulation cost. A single deterministic
future target can give conflicting labels to indistinguishable histories.
DAgger does not remove that information limit.

Before training, compare history-only action chunks with prediction-conditioned
chunks using identical dynamic demonstrations, network capacity, decision
frequency, observation history and total data/query budget. Hold out GRAB
subjects and whole source sequences; adjacent windows from a recording must
not cross train/test splits. Keep static/dynamic data, prediction/no-prediction,
and true/predicted future as separate ablations. Measure forecast calibration
and action gains, not only forecast point error.

Garment diversity is a separate axis. Replacing the actor with flow matching
cannot create successful demonstrations for shapes the teacher cannot dress.
ClothesNet supplies geometry; new garments still need valid sleeve/grasp
semantics, physical starts and successful recovery supervision. Evaluate
garment identities and motion sources independently, including their joint
holdout. The present known-garment pilot cannot support that claim.

## Motion conversion implemented

`python/uipc_manip/grab_motion.py` and
`scripts/wang_transfer/prepare_grab_motion.py` read trusted local GRAB NPZ
files and export a pickle-free motion archive. Right collar, shoulder, elbow
and wrist rotations are transferred as local SO(3) deltas to the existing
recipient. Shape, root, scale, ground offset and seated torso remain fixed;
the source object and finger motion are not replayed. Skinning can still move
nearby surface vertices. Frame zero must match `generate_body` within 10 µm.

Local data are at `/home/ge47gax/kun/GRAB/data/GRAB/grab`; `/kun` is not the
actual path on this workstation. The three prepared s1 clips use source
seconds 1–4, 30 Hz, rotation amplitude 0.35, a 0.3 s ramp, and recipient 14046.
They are perturbed motion patterns, not full-amplitude replay of the person.

| Source clip | Frames | Maximum target vertex speed | Initial maximum coordinate error |
| --- | ---: | ---: | ---: |
| `mug_pass_1.npz` | 91 | 0.270 m/s | 2.23e-16 m |
| `mug_lift.npz` | 91 | 0.178 m/s | 2.23e-16 m |
| `phone_call_1.npz` | 91 | 0.538 m/s | 2.23e-16 m |

No wave-named sequence was found; these clips must not be relabeled as waving.
Archives are under ignored `output/anticipatory_dressing/motions/`. The saved
metadata includes source hash, recipient configuration and retarget settings.
Geometry arrays are stored as float32; the reported initial error above is
measured before serialization.

```bash
cd /home/ge47gax/kun/libuipc-anticipatory-dressing
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/wang_transfer/prepare_grab_motion.py \
  /home/ge47gax/kun/GRAB/data/GRAB/grab/s1/mug_pass_1.npz \
  --body 14046 --start 1 --duration 3 --fps 30 --amplitude .35 \
  --out output/anticipatory_dressing/motions/pass_example.npz
```

Use a new output path; conversion refuses to overwrite an existing archive.

## Experiment contract

The runner being validated is `scripts/wang_transfer/probe_arm_motion.py`.
It reads the existing r1 checkpoint through the Python 3.9 policy bridge in
the residual worktree, while importing the environment from this new branch.
Code and input hashes are saved. Inference runs on CPU with zero force input.

The body surface is an Empty FEM driven by per-vertex soft position targets
inside the IPC solve. It is not teleported into a fixed collider. Maximum
actual-versus-target error is checked every physics substep (default 2 mm);
target jumps above 15 mm/substep invalidate the run. Human self-contact is
disabled, as for the previous fixed body; this is not a human biomechanics
model. A soft target drive and IPC alone do not certify exact human motion.

Use the same initial garment, recipient, random seed, 10 Hz decisions, speed
limits and full decision budget for all methods. The camera stays at its
initial position. `wang_live_arm` includes live cloth/body occlusion;
`wang_static_arm` is an explicitly privileged bare-arm visibility diagnostic.
Changing visibility can affect the pretrained policy, so a static-body run
under the chosen observation mode is required before interpreting motion.

| Controller | Information and action |
| --- | --- |
| `r1` | Current visible point cloud; original action. |
| `gicp` | Same actor plus bounded registration correction from previous/current visible arm ROIs. |
| `oracle_pause` | Same correction; suppress actor advance if the true finger displacement over the next 0.4 s exceeds 2 cm. |
| `causal_pause` | Same correction; pause using exact past/current prescribed finger velocity. This is a privileged-present control with no future information, stronger than noisy visual velocity. |
| `yoked_pause` | Shift the oracle schedule by 20 decisions, keeping exactly its full-budget pause count. Offline diagnostic, not deployable. |
| `hold` | No robot action, used to validate body motion. |

The GICP module uses local covariance-weighted nearest-neighbor registration,
an observed spatial ROI and a 1 cm bound on point displacement. Sun et al. use
a PDE-derived ROI, their own diffusion policy/chunk execution and a bounded
SE(3) correction. This local comparison is a correction-module proxy, **not a
complete Sun et al. reproduction**. Registration failures return identity
and are counted. A future rigorous comparison must implement those missing
parts or use the authors' released pipeline.

The primary endpoint is a physically wrapped sleeve and upper-arm ratio at
least 0.7 throughout the final 20 decisions, with valid grasp throughout.
Also report whether a held success occurred earlier, progress, pause count,
registration failures, grasp error, actual human tracking error and wall time.
All methods get 350 decisions, including pauses. Solver failure is invalid
physics, not a policy failure; reset states must never be appended as terminal
outcomes. Retain failed runs and aligned observations/actions. Full human and
cloth mesh states are saved for geometry inspection.

Start with a short hold-only smoke run, then static-body r1, then paired
motion trials. Onset time alone does not prove a contact phase: classify it
from the actual sleeve/arm geometry in the saved trajectory. Include both
pre-contact and already-threaded starts when feasible.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/wang_transfer/probe_arm_motion.py \
  --motion output/anticipatory_dressing/motions/s1_mug_pass_body14046.npz \
  --out output/anticipatory_dressing/smoke_example \
  --methods hold --steps 40 --onset 0 --wait-for-gpu 1800
```

`--preflight-only` validates assets, initial placement and schedules on CPU.
Choose a fresh output directory for each run. The GPU guard waits for existing
compute clients and never stops them. Desktop clients/MPS server are ignored;
the existing Python collectors are not. A new external job may still start
after a guard check, so this is an advisory check, not a shared scheduler lock.

## Evidence at this stage

- Four motion conversion/interpolation tests pass, including rotation wrap,
  recipient preservation, clipping and safe archive round trips.
- Three controller tests pass: geometric registration recovery, failed
  registration/bounded correction, and causal versus future pause schedules.
- Existing camera-rig tests include the added mode: 27 selected CPU tests pass
  in total (2026-09-26). No CUDA tests are included in that count.
- CPU placement preflight succeeds for recipient 14046: 4.655 mm minimum
  full-body gap, garment yaw 350 degrees. The first run caught and fixed a
  semantic dictionary-key mismatch; its failure artifact is retained.
- For the pass clip at onset 1 s: oracle 21, causal 23 and shifted-oracle 21
  pauses out of 350 decisions. These are schedule counts, not outcomes.
- GPU physics and policy comparisons are pending the `9eae3121` collection
  window. No policy improvement, motion tracking result or novelty result is
  established by the CPU checks.

The force reproducibility, gradient horizon and landmark-error numbers in
the proposed plan came from earlier limited audits. They should not be
restated as universal guarantees. This pilot avoids force inputs and records
its own geometric validity evidence.
