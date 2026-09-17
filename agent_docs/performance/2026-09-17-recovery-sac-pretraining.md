# Verified IPC recovery guidance inside SAC pretraining

## Owner's objective and actual change

The owner clarified that IPC should improve the RL pretraining algorithm while
the existing policy interface and subsequent training remain usable. The earlier
recovery experiment ran behavior cloning alone. It did not test this integration.

`pretrain_wang --recovery-source-dirs DIR --recovery-weight 1` now passes an
optional recovery supervisor into the ordinary `SACAgent.update` call. The actor
uses one combined loss and optimizer step:

$$
L_\pi = \mathbb{E}_{s\sim D, a\sim\pi}[\alpha\log\pi(a|s)-\min_i Q_i(s,a)]
+ \lambda\mathbb{E}_{(s,a^*)\sim D_{\mathrm{IPC}}}
  [\operatorname{mean}_{j\in A}(\mu_j(s)-a^*_j)^2].
$$

Here $D$ is ordinary transition replay; $D_{\mathrm{IPC}}$ holds only admitted
full-continuation recovery observations/actions; $A$ contains the active command
axes (translation and y/z rotation in this dressing controller). $\mu$ is the
squashed deterministic actor output. The SAC critic, temperature and target
updates retain their existing rules. The recovery batch is independent and
present at every scheduled actor update, so it cannot disappear in the much
larger ordinary replay. Its arrays and sampling stay on the training device.
Sampling has a separate update-indexed RNG and does not consume SAC action noise.

The option is confined to the pretraining launcher. Each update explicitly
receives supervision; a loaded checkpoint and `update(replay)` use ordinary SAC.
The actor architecture, observations, reward, horizon, controller, and simulator
are unchanged. The student needs no IPC search or teacher at inference. This
does not establish real-world observation/actuation compatibility or sim-to-real
performance.

Only completed `verified_full_continuation_v1` directories with admission passed
are accepted. Rejected episodes are excluded. Physics/observation/action contracts
must agree, including JSON tuple/list equivalence, and recovery training bodies
must not overlap pretraining evaluation bodies. Observation/action-only files
are never treated as Bellman transitions with invented rewards or successors.

This is an auxiliary supervised actor update within SAC, not a differentiated
IPC actor gradient, new critic objective, solved offline RL method, or established
novel algorithm. See the [prior-art audit](2026-09-17-recovery-teacher.md).

## Fixed pilot, specified before execution

Both arms start from `dressing_redesign_20260916/warm_sac/checkpoints/`:
`checkpoint_00127416.pt` and its paired `replay_latest`. Actor, trained twin
critics/targets, temperature, optimizers and historical replay are restored.
The BC checkpoint's untrained critic is not used. Existing dense/plain network
architecture is preserved rather than silently changing the saved network.

Each arm runs 1,200 additional live IPC transitions (four copies, 300 decisions),
one ordinary gradient update per collected transition, batch 64, actor frequency
four, seed 1. Pool: tshirt_26 and tshirt_68 on body 14046. Withheld development
evaluation: both garments on body 14047. No checkpoint is selected by evaluation;
the fixed final checkpoint is compared. Source pretraining cost is shared, not
zero and not counted as training from scratch.

Control: ordinary SAC. Treatment: identical SAC plus the 360 previously verified
recovery rows from tshirt_68/14046, active-axis MSE weight 1, no weight sweep.
These targets were verified from BC-visited states; they are not new labels at
the current SAC actor's visited states. This limits what one round can establish.
This pilot does not implement automatic failure detection or repeated relabeling.

The existing pretrainer evaluates before/after training. An additional matched
evaluation uses the two training configurations, two rounds, both fixed final
policies, full 300-decision episodes, and the usual controller without added caps.
Assess final upper-arm coverage, sustained coverage, grasp validity and errors;
teacher-action MSE is only a fitting diagnostic. No long run or tuning sweep
follows automatically if this bounded pilot fails.

Artifacts: `output/uipc_manip/recovery_sac_20260917/`. Implementation tests and
status follow below.

## Final status: baseline implemented, comparison stopped

The owner challenged this as insufficiently novel and requested deeper algorithm
research. The launcher and its sole active ordinary-SAC child were stopped with
SIGTERM. The treatment and final paired evaluation never started. The source
policy's initial two-episode evaluation completed (.247 mean coverage, 0/2
successes, approximately 109 s). No completed continuation checkpoint or
training-comparison result exists; partial work must not be presented as a
negative or positive policy result. No training job remains running.

The opt-in supervised SAC implementation remains available as a conventional
comparison baseline. Seventy-four focused tests pass across the SAC agent,
pretraining protocol, demonstration loader and recovery loader, including zero
weight equivalence, unchanged first-step critic/temperature updates, active-axis
masking, admission/leakage checks and plain-SAC checkpoint reuse. A real 360-row
dataset loads onto CUDA with the saved SAC protocol. An older BC smoke-test
fixture was updated to the existing environment metadata and stage contract;
the current change does not rename the BC stage.

The [subsequent research design](2026-09-17-ipc-policy-gradient-research.md)
explores simulation allocation for return-based gradient correction. Neither
the auxiliary imitation loss nor its stopped pilot establishes a novel algorithm.
