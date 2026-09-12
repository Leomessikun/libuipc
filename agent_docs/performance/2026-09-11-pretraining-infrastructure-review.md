# The pretraining infrastructure, reviewed from first principles

Wall-clock time to a trained policy is the number of transitions a recipe needs, times the seconds each
transition costs, divided by the number of transitions produced in parallel. Wang's recipe (point-cloud
SAC, regional teachers distilled into a student) fixed the first factor for PyFleX. This port swapped
the simulator for libuipc IPC and kept the recipe. Every issue below is a term in that product. Figures
come from relaunch 3 of the region-13 teacher (`wang_teacher_r13_s1`) unless stated otherwise;
unmeasured claims say so.

## The budget is below Wang's own numbers

- Relaunch 3 produces 11.6k transitions per hour on one RTX PRO 6000 Blackwell with 24 environments,
  so a 600k teacher takes 2.2 days and the chain of three teachers and a student about nine.
- Wang's teachers reached their best checkpoints between 0.52M and 1.96M transitions, median 0.75M.
  The chain's 600k is below that median. At this rate the median is 2.7 days per teacher, the top of
  the range seven, and Wang's 23.5M teacher total 84 days.
- Faithful execution is therefore under-budgeted by construction. The 300k checkpoint is where the
  budget is either raised or traded for a deviation that needs fewer transitions.

## Where a transition's time goes

| Part | Share of wall time |
|---|---:|
| Simulation (IPC step, observation, reward) | 68% |
| Evaluation | 18% |
| Network updates | 15% |
| World rebuilds | 1% |

- **The parts run one after another** in a single process on a single GPU. Any other GPU job slows
  training directly: the AL-release probes of the same day took the teacher from 3.8 s to between 6
  and 9 s per vector step.
- **Evaluation follows Wang's protocol.** It runs one deterministic episode for each of the 25
  held-out configurations every 10k transitions, as `curl/train.py` does. Cutting it is a deviation.
  Running it on another process or GPU against saved checkpoints keeps the protocol and takes 18% off
  the critical path.
- **The teachers run in series** although they are independent. Three GPUs would cut the teacher phase
  to a third without any algorithmic change. This is the largest lever.
- **Copies per world.** Going from 16 to 32 copies raised throughput from 6.8 to 10.5 transitions per
  second ([dressing correctness](2026-09-09-dressing-correctness.md)). The teacher runs 24 copies in
  29 of 96 GB. The gain from 24 to 32 has not been measured.
- **Overlap.** Overlapping updates with simulation is capped at 15%. Whether IPC leaves the GPU
  headroom for it has not been measured.

## Fast pretraining in Newton, fine-tuning in IPC, is not worth it

The idea was to spend most transitions in a cheaper simulator (Newton's VBD cloth) and fine-tune in IPC.
The measurements do not support it:
- **The speed gap is mostly a different MDP.** The quoted 4.4 times compared Newton at one simulation
  step per decision with this port at six. At six steps per decision, Newton's 40 environments give
  11.5 transitions per second against 10.5 for this port's 32 copies: 1.10 times. At 16 copies the gap
  is 1.70 times ([dressing correctness](2026-09-09-dressing-correctness.md)).
- **VBD differs exactly where the task is hard.** VBD cloth stretches and penetrates slightly, so the
  scripted expert reaches the 0.7 threshold in 20 of 35 cells there. It also collapses a closed cuff,
  which is why Newton needs pre-baked states ([pretraining](2026-09-08-uipc-manip-pretraining.md)).
  A policy that learns to get past the elbow by stretching the sleeve does not transfer.
- **Newton's own SAC did not solve the task.** Its 33 evaluated runs peak at one success in 40
  episodes after 2.98M transitions.
- **The cost is two simulators.** Garments, bodies, observations, actions and rewards must match
  exactly, and the critic's values are simulator-specific.

Sim-to-sim pretraining pays when the cheap simulator is an order of magnitude faster and agrees on the
task-critical contact. Here it is 1.1 to 1.7 times faster and disagrees at the elbow.

## The grasp is the structural weak point

Wang's PyFleX picker attaches particles kinematically. This port holds 48 anchors with a soft position
constraint and bounds them with the 6 cm vertex tether ([solver stall](2026-09-11-dressing-solver-stall.md)).
- **Before the tether,** held vertices trailed their targets by up to 179 mm. This caused the slow
  tail that stalled relaunches 1 and 2.
- **The tether drops a whole move** whenever a held vertex would trail by more than 6 cm. The policy
  cannot observe why its action had no effect. How much that costs learning has not been measured;
  counting dropped moves per episode would measure it.
- **The watchdog.** By 84k transitions it tripped three times, at episode steps 225 to 279. The last
  two trips ran with no other GPU job. Each trip drops 24 transitions from the slowest,
  contact-hardest states. The rate is about one per 13k; at more than about one per 10k the replay
  would under-sample the elbow.
- **The real fix is not available.** It is the penalty-free moving boundary of AL-IPC (Zheng, Luo and
  Li, section 5.3), which no libuipc version implements. This is a cost of choosing IPC, not a bug.

## Sample efficiency: what the recipe leaves unused

- **The expert is not used.** The scripted expert dresses 22 of 40 cells in this port with the tether.
  Wang's agent is `SAC_AWAC`, and his launcher has a `demo_buffer_dir` option, set to `None` in the
  final configuration. Seeding the replay with expert episodes targets the first few hundred thousand
  transitions, where Newton's working runs showed no upper-arm progress through 280k. Not measured
  here.
- **The teachers need not see point clouds.** They exist only to be distilled. The asymmetric critic
  on a 35-float privileged state is built and makes updates 2.1 times faster (2.85 times in bf16).
  Its effect on sample efficiency has not been measured in this port.

## Contact forces are available but not wired in

- **Arm contact force.** The 0.0.28 wheel exposes `uipc.core.ContactSystemFeature`, with contact
  energy, gradient and Hessian per primitive type: point-triangle, point-edge, point-point, edge-edge
  and point-half-plane, each split into normal (`+N`) and friction (`+F`). The force on an arm vertex
  is the negated sum of its gradients.
  - The gradient is in the solver's incremental-potential units. Converting it to newtons, probably by
    dividing by dt squared, needs a calibration.
  - Calibration test: a sleeve resting on the arm must press on it with its own weight.
  - Wang's PyFleX signal was the contact multipliers λ, which depend on iteration count and step size.
    His force-penalty threshold of 500 is empirical. A barrier gradient on a penetration-free state
    is the cleaner signal.
  - This port's reward has no force term (`dressing_reward.py`). Wang's penalises force above 500
    with weight 0.001.
- **Grasp pull force.** It is the hold spring times the hold gap. This needs no API, but its size is
  set by the chosen constraint strength.
- **Robot coupling force.** Genesis's `two_way_soft_constraint` coupling feeds a robot link
  `translation_strength * mass * (x_ipc - x_aim)` (`ipc_coupler/utils.py`). That is a spring
  reading, a step behind and stiffness-dependent. It is usable as a coarse wrist-force signal once the
  Stretch 3 is in the loop.

## What is not the lever

- **The solver.** IPC is not time-of-impact locked on this scene: it takes 4 to 6 Newton iterations
  per frame. AL-IPC is 2.2 times slower here, and the authors' release adds nothing upstream lacks
  ([solver stall](2026-09-11-dressing-solver-stall.md)). Contact parameters and step size were not
  levers.
- **The SAC hyperparameters match Wang's launcher** (`curl/launch_train_curl.py`): batch 64
  (`get_batch_size` for `pointcloud_3`), an actor update every fourth critic update, learning rates
  1e-4, and evaluation every 10k. Neither side clips gradients. Time limits bootstrap on both sides.
  The one difference is the temperature learning rate, 5e-5 here against 1e-4.

## Relaunch 3's first evaluations

| Transitions | Held-out upper-arm ratio | Forearm ratio | Successes | Simulator errors |
|---:|---:|---:|---:|---:|
| 50.4k | 0.083 | 0.56 | 0 of 25 | 0 |
| 64.2k | 0.253 | 0.64 | 1 of 25 | 0 |
| 76.8k | 0.085 | 0.36 | 0 of 25 | 0 |
| 83.5k | 0.000 | 0.07 | 0 of 25 | 25 |
| 90.7k | 0.000 | 0.38 | 0 of 25 | 0 |
| 105.1k | 0.220 | 0.60 | 0 of 25 | 0 |
| 112.3k | 0.128 | 0.44 | 0 of 25 | 25 |
| 125.0k | 0.283 | 0.60 | 1 of 25 | 0 |
| 131.8k | 0.000 | 0.03 | 0 of 25 | 25 |
| 146.2k | 0.114 | 0.26 | 0 of 25 | 25 |
| 152.7k | 0.000 | 0.05 | 0 of 25 | 25 |

- **The 90.7k evaluation is valid and ends on no upper arm.** Forearm ratio by garment: 0.00 hospital
  gown, 0.45 tshirt_26, 0.54 tshirt_68, 0.73 tshirt_4, 0.16 tshirt_392. Three evaluations in a row
  sit below 64.2k.
- **At 105.1k it is back to 0.22** (forearm 0.60, no success). Forearm ratio by garment: 0.00
  hospital gown, 0.80 tshirt_26, 0.20 tshirt_68, 1.00 tshirt_4, 1.00 tshirt_392. From 50.4k to
  105.1k the held-out score swung between 0 and 0.25 without a trend.
- **At 125.0k it is the best so far, 0.283, and `best.pt` moved there.**
  - 14 of 25 configurations end on the upper arm, as at 64.2k; 7 reach 0.5 or more, against 9.
  - Its one success, tshirt_26 on body 14046 at 0.71, sits on the threshold, as the 64.2k success
    did before its re-evaluation. It is not re-evaluated yet: a probe beside the teacher slows it and
    risks another voided evaluation.
  - Coverage is wider: all five tshirt_4 configurations and the first hospital gown (body 14047,
    0.54) end on the upper arm.
- **The 131.8k evaluation was voided too, and the rounds are getting slower.** Evaluation seconds ran
  691, 838, 740, 1073 and 2038 over the last five rounds; the watchdog's budget is eight times the
  median of the last 64 decisions, so a world that slows raises its own budget and takes longer to
  trip. This round tripped early, at a forearm ratio of 0.03. The 146.2k and 152.7k rounds went the
  same way, so 5 of 16 rounds have measured nothing, four of the last five, and
  1.4 GPU-hours went into rounds that measured nothing. The evaluation signal is
  effectively gone until the teacher restarts on watchdog-free evaluation worlds; its checkpoints
  are all kept, so those rounds can be replayed offline. Only the teacher held the GPU, and the CPU tests that ran
  in that window pass `--device cpu`.
- **The run slows as the policy pushes deeper.** Environment seconds per vector step, by 1,000-step
  window: 4.22, 4.30, 3.91, 3.46, 4.94, 5.99. Measured between log rows, that is 16.0k, 16.0k, 18.3k,
  19.5k, 15.1k and 12.8k transitions per hour, and the rows before the seventh watchdog trip read
  9.0k. That trip fired on a 59 s budget against 30 to 36 s earlier, the budget being eight times the
  median of the last 64 decisions. The 11.6k per hour these budgets use is the run's average
  including evaluation, and deeper sleeves cost more, not less.
- **The 83.5k evaluation measured nothing.** In `dressing_env.py`, the `RuntimeError` handler of `step`
  turns a watchdog trip into a simulator error for every slot of the world. One slow configuration
  therefore ends all 25 episodes at once, and they are scored at their last-seen ratios.
  - That evaluation ran from about 20:00 to 20:14. Two re-evaluation probes started on the same GPU at
    20:02, and their contention is the likely trigger.
  - The 112.3k evaluation tripped the same way with no other job on the GPU, only the teacher and
    desktop processes, so the teacher's own slow decisions are enough. Two of 12 evaluations
    so far measured nothing.
  - Wang's evaluation has no watchdog: in `curl/train.py` a simulator error ends only its own episode.
- **The failure mode grows as the policy improves.** Deeper sleeves mean slower contact decisions,
  and any other GPU job slows every decision. Under a world-level watchdog, an evaluation world of 25
  configurations is all-or-nothing.
  - Fix options: exempt evaluation worlds from the watchdog, or re-run an evaluation that tripped.
    Either takes effect only after the teacher restarts.
  - Until then, re-evaluate saved checkpoints offline before choosing a teacher for distillation.
- **The deterministic evaluation of one configuration flips between rounds.** For example, tshirt_68
  on body 14046 scored 0.50, 0.46, 0 and 0 over the evaluations from 43.2k to 76.8k. One deterministic
  episode per configuration, on a threshold task with a GPU solve that is not bitwise repeatable, is a
  noisy estimate.
- **Training rollouts did not collapse.** The replay stores rewards at half scale; on the upper arm the
  raw reward is the forearm length plus five times the upper-arm distance. Reconstructed from it:
  - 83 of 264 training episodes (one per slot) reached the upper arm, 6 went well up it, and none
    reached the 0.7 success line.
  - Per 24-slot episode, the upper-arm count rose from 0 in the first episode to 11 to 13 at 21k to
    36k, and to 14 at 69.6k to 76.8k.
  - In the episode the third trip cut at 83.5k, 8 slots reached the upper arm and 18 of 24 the forearm.
- **Evaluation episodes so far.** Of 200:
  - 75 ended on the upper arm;
  - 10 ended at 0.5 or more;
  - 1 reached the 0.7 success line: tshirt_26 on body 14048 at 64.2k. Bodies 14045 to 14049 are Wang's
    held-out poses 45 to 49, never trained on.

- **Offline re-evaluation.** Each checkpoint ran once more in fresh evaluation worlds, on the same 25
  configurations with deterministic actions, after the teacher's own evaluation (scratch
  `perf/stall_probe.py --deterministic`). A second pass was stopped on purpose to keep the GPU free for
  the teacher's next evaluation.

  | Checkpoint | Teacher's evaluation | Re-evaluation | Configurations within 0.05 |
  |---:|---:|---:|---:|
  | 64.2k | 0.253, 1 of 25 | 0.239, 0 of 25 | 22 of 25 |
  | 76.8k | 0.085, 0 of 25 | 0.085, 0 of 25 | 25 of 25 |

  - The single 64.2k success does not hold: tshirt_26 on body 14048 scored 0.87, then 0.63. Two
    tshirt_68 configurations moved the other way and back (0.53 to 0, and 0 to 0.37).
  - The 76.8k drop is the policy, not evaluation noise. Per garment, the forearm ratio moved from
    0.00, 1.00, 0.40, 0.99 and 0.78 (hospital gown, tshirt_26, tshirt_68, tshirt_4, tshirt_392) at 64.2k
    to 0.39, 1.00, 0.00, 0.40 and 0.00 at 76.8k. The policy traded three garments for one rather than
    getting uniformly worse.

## Wang's regional teachers

His launcher (`curl/launch_train_curl.py:319-345`) lists the checkpoints the student distils from. Each
file name carries the step and the held-out mean upper-arm ratio of that best evaluation.
- **Training.** There are 27 teachers, one per arm-pose region, each trained by pure online SAC without
  demonstrations for up to 5M steps.
- **Best steps.** They range from 0.52M to 1.96M, median 0.75M, and sum to 23.5M.
- **Scores.** 23 file names carry one: 0.646 to 0.880, median 0.78, with 20 of them between 0.70 and
  0.85. The other four are plain checkpoints between 0.74M and 1.52M.
- **The chain's regions:**

  | Region | Best step | Held-out upper-arm ratio |
  |---:|---:|---:|
  | 4 | 920k | 0.845 |
  | 13 | 1.96M | 0.740 |
  | 22 | 520k | 0.782 |

  Region 13 is the slowest of all 27. At this port's rate its best step is seven days away, and the
  chain gives it 600k transitions.
- **The student.** Distilled from the teachers, it reaches a held-out upper-arm ratio of 0.68, against
  0.34 for one policy trained directly (Table II).

