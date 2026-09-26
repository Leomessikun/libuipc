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
| [Erickson et al., Deep Haptic Model Predictive Control (ICRA 2018)](https://arxiv.org/abs/1709.09735) | Learns action-conditioned cloth force predictions in simulation and uses MPC for real dressing. Predicting contact consequences for dressing is also established; this work predicts forces, not arbitrary future recipient motion. |
| [Cai et al., Privileged Information in Partially Observable RL (NeurIPS 2024)](https://arxiv.org/abs/2412.00985) | Analyzes failure of generic expert distillation and conditions for successful use of privileged information. It reinforces the need to test what the student's observations identify; it does not directly analyze this dressing system. |
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

### Existing FMVP rollouts are the initial action supervision

The owner emphasized reusing the already collected checkpoint rollouts. A
read-only inspection of the original workspace on 2026-09-26 confirms that
`output/uipc_manip/fmvp_scaled_multigarment_v4_20260925/manifest.json` contains
810 entries marked accepted. Its collection configuration covers `tshirt_26`,
`tshirt_68`, `tshirt_4`, `tshirt_392` and `hospital_gown`, using the r1 checkpoint.
This is a manifest count, not a fresh independent geometry audit or a count of
independent recipients. Older dataset versions can overlap and need deduplication.

One inspected hospital-gown episode contains 270 observations, 269 six-axis
action commands, 270 cloth meshes and tool poses, a static full-body mesh,
and per-transition grasp validity. Its controller IDs identify 249 FMVP
decisions followed by 20 hold decisions. The alignment is
`obs[t] -> actions[t] -> obs[t+1]`. These recordings already supply supervised
pull-up behavior; GRAB supplies recipient motion, not robot actions.

The recommended training order is to first establish a history-conditioned
flow/diffusion imitation baseline from existing valid rollouts. For example,
three past observations can condition the next eight recorded action commands;
these window lengths are proposed hyperparameters. Match the deployment action
frame, scale and decision period, mask episode ends, balance hold segments,
and split by whole episodes/bodies/garment identities before extracting windows.
Distinguish `policy_actions` (raw proposals), `actions` (commands submitted to
the environment), and `executed_translation` (accepted anchor displacement).
The virtual-gripper Cartesian commands are not robot joint trajectories.

This baseline does not require new dynamic demonstrations or an IPC solve per
gradient update. The current actor's visual encoder may be reusable, but a new
flow action head is not the same network as the FMVP actor. The existing BC
scripts offer data-contract references;
they fine-tune the original actor and do not implement action-chunk flow matching.

If the dynamic pilot supports further work, use the pretrained policy or r1
as an action proposal in moving-body simulations, collect teacher-corrected
recovery decisions, and train with both existing and new data. Superimposing
GRAB motion on an old successful robot trajectory does not make its old actions
valid dynamic supervision: contact outcomes must be recomputed. Failed actions
must not be treated as successful imitation targets. New-garment supervision
still depends on obtaining physically valid behavior on those garments.

### Static flow baseline: data and training entry prepared

`python/uipc_manip/rollout_chunks.py` now reads the v4 manifest's job-relative
paths, checks the recorded acceptance/grasp flags, array alignment, finite
values, normalized command bounds and consistent observation/action contracts.
It hashes the observation/command arrays for exact training-data deduplication
and writes compact memory-mapped arrays, leaving source geometry untouched.
This is a recording audit, not independent mesh/contact success validation.

The prepared snapshot is
`output/anticipatory_dressing/flow_bc_v4_data/manifest.json` in the isolated
worktree. All 810 entries pass these checks, with no exact training-array
duplicates. The cache occupies 5.07 GiB. The seed is 20260926 and the validation
fraction is 0.2, with body IDs split before constructing any windows:

| Split | Episodes | Body/pose IDs | Commands |
| --- | ---: | ---: | ---: |
| Train | 661 | 122 | 204,401 |
| Validation | 149 | 31 | 47,505 |

All five garments appear on both sides; body IDs do not overlap. This is a
student imitation holdout, not proof of unseen people or garments: the source
FMVP teacher may have trained on these bodies. Total controller counts are
232,910 ordinary FMVP, 16,200 hold, 1,592 tracking-limited FMVP and 1,204
IPC-improved FMVP commands. Thus the labels include recorded collector
corrections. Two rotation axes are identically zero; six stored action
components do not imply demonstrations covering arbitrary 6-DoF rotation.

`python/uipc_manip/train_flow_bc.py` adds a static imitation baseline using
the existing FQL `FlowPolicy.vector` API, tool-point encoder and residual
network utilities. It has no Q objective or simulator calls. Three ordered
observations, with an initial-history mask, condition an eight-command joint
flow. Future padding is masked; windows never cross episodes. Sampling is
uniform over body IDs, then episodes, then decisions, including recorded holds.
The compact PointNet++ encoder starts from random weights. FMVP visual-weight
transfer is a possible later experiment, not an implemented feature.
The saved checkpoint carries the data hash and the command/observation
contract. The initial deployment plan executes one command and replans every
0.1 s; predicting eight commands does not require executing all eight blindly.

The six CPU tests in `test_rollout_chunks.py` pass. They cover temporal
alignment and masking, split leakage, duplicate removal, malformed/invalid
recordings, parameter updates and exact checkpoint inference round trips.
A real-data CPU smoke completed two updates with batch size 2, hidden width
32 and two flow integration steps, under
`output/anticipatory_dressing/flow_bc_cpu_smoke`. Its final validation flow loss
is 1.4462 and sampled-command MSE is 0.6520 (zero-command MSE 0.04575).
These tiny-sample diagnostics only check the training path; the smoke
checkpoint has not learned a usable policy. No full training or closed-loop
flow-policy evaluation has run. The unrelated six GPU collection workers
were still active during preparation and were not interrupted.

The next bounded training run, after a GPU window is available, is:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.train_flow_bc \
  --data output/anticipatory_dressing/flow_bc_v4_data \
  --out output/anticipatory_dressing/flow_bc_v4_train \
  --device cuda --steps 5000 --batch-size 16 --history 3 --horizon 8
```

This command must run from the isolated worktree. Output directories must be
new. Five thousand updates are an initial budget, not a convergence claim.
The immediate milestone is a flow policy that can reproduce the existing
single-sleeve pull-up behavior in IPC. Add the flow-checkpoint inference
adapter and compare against r1 on matched held-out body/garment starts,
with identical control periods, grasp checks and the collector's physical
sleeve/proximal endpoint plus hold window. Report success, lost grasp and
completion time per garment; offline MSE is insufficient. A small paired
smoke should establish the interface before expanding the evaluation cohort.
Only after this static baseline is working should dynamic teacher corrections
and future-prediction ablations become the main training experiment.

### First static training run and paired evaluation adapter

The owner approved proceeding. The launch waited for the active collection
workers, then completed the configured 5,000 updates with batch size 16 in
56.1 s process wall time (53.95 s to the final logged update). Artifacts are
`output/anticipatory_dressing/flow_bc_v4_train/{run.json,metrics.jsonl,best.pt,latest.pt}`;
`flow_bc_v4_launch.json` records the launch and return code zero. The checkpoint
selected by validation flow loss is update 4,500: loss 0.12349, sampled command
MSE 0.02404, versus zero-command MSE 0.04300 on the fixed 64 validation windows.
Update 5,000 has flow loss 0.12504 and command MSE 0.02355. These normalized
offline errors do not establish dressing competence or convergence.

`flow_client.py` provides CPU inference in a separate process using this
worktree's model code, while the collector keeps the original external FMVP
environment package. The environment file hash still matches the v4 recordings.
Each slot has an independent causal history and seeded Gaussian stream; reset
clears both. The client checks observation mode, point budget, collision mode,
command scales and decision period. It executes the first predicted command
and replans. The collector's `flow` variant consumes the full flat observation
and bypasses FMVP's rotation conversion, since that conversion is already in
the training labels. Both controllers use the recorded translation/world-yaw
command space (the two unsupported rotation axes are zero).

Eight CPU tests pass, including separate slot histories, reset reproducibility,
contract mismatch rejection and an exact subprocess/direct-inference comparison.
`eval_static_flow.py` prepares one validation start per garment, reusing the
source hang, body, placement, material, seed and sleeve/armhole success rule.
It checks source environment, hang and checkpoint hashes before launching.
The initial cohort is body 10040 for `tshirt_26`, `tshirt_4`, `tshirt_68` and
`hospital_gown`, and body 25040 for `tshirt_392`. These are held out from student
training, but selected from accepted demonstrations and potentially familiar
to the FMVP teacher. Treat the result as a reproduction diagnostic, not an
unbiased success-rate estimate or unseen-garment result.

Each fresh collector world contains a baseline slot and a flow slot with the
same start. Both use the original external completion hold, 750 search
decisions and a 20-decision hold window. The runner retains failures and
reports the actual initial cloth/tool/observation differences. This does not
test autonomous stopping. The first run uses seeded Gaussian noise, not the
optional zero-noise diagnostic. GPU jobs are checked between cases; the
runner never terminates unrelated processes. Evaluation results are pending.

The five-case runner has been launched at
`output/anticipatory_dressing/flow_bc_v4_eval/status.json`. It is currently
`waiting_for_gpu`: immediately after training, the original workspace started
the continuous v5 collection (five active workers at inspection). The current
runner permits a 3,600 s initial GPU wait and 7,200 s evaluation wall budget.
The owner's earlier no-overlap preference remains in effect; a question about
allowing shared GPU execution is pending. No paired rollout has run yet.

A further CPU audit of the selected checkpoint uses all 149 validation
episodes, five prescribed positions per episode and torch seed 2026092607.
The positions are 0, 10, T//2, T-21 and T-1 (the final 20 commands are the
recorded hold). The first predicted command is compared with the recorded
command; observations remain from the demonstration, not policy rollouts.
`flow_bc_v4_train/offline_phase_audit.json` records the checkpoint hash and
all metrics. Translation errors below are RMS command-vector errors using
the 8.660254 mm per-axis scale, not measured cloth/gripper tracking errors:

| Position | Seeded Gaussian flow (mm) | Zero-noise flow (mm) | Zero command (mm) |
| --- | ---: | ---: | ---: |
| Start | 4.210 | 3.342 | 5.332 |
| Early (10) | 2.967 | 2.495 | 4.741 |
| Middle | 1.763 | 1.172 | 3.459 |
| Before hold | 1.620 | 0.692 | 3.438 |
| Final hold | 3.063 | 3.704 | 0.000 |

The initial policy has appreciable starting-action error and has not learned
reliable stopping on these recorded terminal observations. Zero initial noise
is a diagnostic; it is not the generative distribution's mean and is not the
selected evaluation protocol. Keep the external hold explicit in any result.
These findings justify inspecting closed-loop behavior before treating the
offline loss reduction as a usable dressing policy or starting dynamic training.

### GRAB conversion

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
The motion runner uses a 0.01 m/s Newton velocity tolerance (CLI override
available), compared with 0.1 m/s in the existing static preset. This change
follows the failed first smoke described below; its benefit still needs the
queued rerun. Other methods in a paired comparison use exactly the same setting.

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

While the current state meets the physical endpoint, every controller
suppresses further actor advance; its own registration correction remains
available. The actor resumes if progress is lost. This common completion hold
prevents continued pulling after completion from invalidating the baseline.
Both this hold and the experimental pause schedule are saved separately.
The direct runner defaults to 350 decisions; the queued matrix uses 750,
matching the current collector's cap. These budgets must not be pooled.

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
For policy methods it also runs an actual CPU r1 inference with the packed
observation, including its gripper and goal points. This is an interface check,
not a rollout or a success measurement.
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
- The real CPU r1 probe succeeds with 63 visible arm points and 200 visible
  cloth points and returns a finite six-dimensional action. An earlier manual
  probe omitted the required gripper point; using the actual observation
  packer fixed that probe. The rollout runner already uses that packer.
- The first GPU hold smoke ran in a gap between collection batches. Its body
  moved 68.38 mm, but at 1.95 s tracking error reached 3.4024 mm, exceeding the
  unchanged 2 mm gate. Nineteen complete decisions were retained in
  `output/anticipatory_dressing/smoke_pass/hold.npz`; this is invalid physics
  for policy comparison. The last valid state is at 1.9 s. The old summary
  counter was cleared by the environment's automatic error reset; the runner
  now reports the last valid trajectory counters instead of reset counters.
- The failed smoke used the static preset's 0.1 m/s Newton velocity tolerance.
  Backend inspection shows an absolute displacement convergence test of
  `velocity_tol * dt`. The new 0.01 m/s setting is a targeted hypothesis to
  retest, not yet a demonstrated fix. No tracking bound has been relaxed.
- `scripts/wang_transfer/run_motion_pilot.py` is queued in
  `output/anticipatory_dressing/pilot_20260926`. It waits at most two hours for
  an initial GPU window and then has a two-hour total execution budget. It
  first tests all three moving bodies for 40 decisions, then static r1 for
  750 decisions. Only if those gates pass does it run the five controllers
  on all three clips at onsets 1 and 8 s. Status and case logs are retained;
  failure stops subsequent cases. There is no training launch.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ge47gax/kun/genesis-world/.venv/bin/python \
  scripts/wang_transfer/run_motion_pilot.py \
  --motions output/anticipatory_dressing/motions/s1_mug_pass_body14046.npz \
            output/anticipatory_dressing/motions/s1_mug_lift_body14046.npz \
            output/anticipatory_dressing/motions/s1_phone_call_1_body14046.npz \
  --out output/anticipatory_dressing/pilot_example \
  --steps 750 --onsets 1 8 --wait-for-gpu 7200 --wall-budget 7200
```

Read `status.json` in that output directory before launching another copy.
No policy improvement or novelty result has been established.

### Rerun outcome

The queued runner has now stopped at its motion gate. With the tighter Newton
setting, the pass clip completes all 40 hold decisions with maximum body error
0.314 mm; the lift clip completes 40 with maximum body error 0.0985 mm. Both
pass the 2 mm target-tracking check. The phone clip stops after its fifth
decision because garment grasp tracking exceeds the grasp-validity limit;
maximum recorded body tracking error is 0.754 mm. This last failure is a
stationary-gripper task failure, not an observed violation of the human
tracking limit. The current runner conservatively stops on it; separating
motion-physics validity from this legitimate controller failure remains open.
Static r1 and the five-controller comparisons have not run. These smoke checks
produce no evidence of anticipation gains.

The force reproducibility, gradient horizon and landmark-error numbers in
the proposed plan came from earlier limited audits. They should not be
restated as universal guarantees. This pilot avoids force inputs and records
its own geometric validity evidence.
