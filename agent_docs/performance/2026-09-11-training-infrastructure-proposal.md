# Training infrastructure proposal

Date: 2026-09-11. Status: proposal, gated on two measurements; nothing is implemented yet.

The question: what infrastructure gets a working region-13 teacher in days, not months, on the
hardware we have, and still yields a point-cloud student that runs on a Stretch 3?

The answer: keep the IPC simulator. Change what the teacher sees, a 35-float privileged state
instead of point clouds, and where it starts, from the scripted expert's episodes instead of from
scratch. Train the point-cloud student with Wang's own distillation recipe on its own rollouts. Five
research tracks converge on this: the dressing literature, sample-efficient learning, fast cloth
simulators, an audit of Wang's repository, and the expert as a data source.

Tags: [RA] reported by the authors, [MI] measured in this port, [E] our estimate.

## The current path costs one to two GPU-weeks per teacher

- Wang's region-13 teacher peaked at 1.96M transitions with 0.740 [RA] (`curl/launch_train_curl.py`,
  the teacher list at 319-345). At relaunch 3's 11.6k transitions per hour [MI] that is 7 GPU-days.
- Wang's episodes are 150 decisions (`curl/launch_train_curl.py:241`, changed from 300 on
  2022-08-22). Ours are 300 decisions of 0.1 s, a recorded deviation: under the 0.15 m/s cap the
  expert finishes at decisions 212 to 274 and dresses no body within 150
  (`2026-09-09-dressing-correctness.md`, "The episode length also caps the expert").
  - If the need scales with episodes rather than transitions, region 13 takes 14 GPU-days [E].
  - Wang's per-decision translation bound lives in his softgym fork, which is not in the checkout.
    His `easy_config` only picks initial states with smaller joint angles
    (`imitation/launch_train_dagger.py:132`), not a shorter approach. The range stays at 7 to 14.
- All 27 regions at that pace: 84 GPU-days or more [E].
- Relaunch 3 at 88k transitions: held-out upper-arm ratio 0.253, then 0.085, both reproduced in
  fresh worlds [MI] (`2026-09-11-pretraining-infrastructure-review.md`, "Offline re-evaluation").

## The simulator is not the lever

No candidate documents a tenfold per-transition speedup over GPU IPC on cloth-against-body contact
together with evidence that its policies transfer to real dressing.

| Candidate | Speed | Fidelity or transfer | Verdict |
|---|---|---|---|
| Newton VBD | 1.1 to 1.7 times at matched decimation [MI]; "over 300 times" [RA] | stretches and penetrates at the elbow [MI] | rejected (review) |
| PyFleX, Wang's simulator | no number for this scene; a third party reports under 50 ms per 50 ms step for one environment [RA], so 2 to 3 times per process [E] | the only dressing sim-to-real evidence [RA]; over 1,000 triangle intersections against under 20 for MPM in a pull-through [RA] | optional one-hour gate, not scheduled |
| Genesis PBD | no cloth number from anyone | none on cloth-body contact; not reachable through the IPC coupler | rejected |
| Isaac Lab, PhysX 5 FEM cloth | 26.6 ms per step at 40k vertices [RA, third party] | fling error 128 against 8.5 for Genesis [RA, third party] | rejected |
| MuJoCo Warp flex | mesh-flex and flex-flex collision still planned | none | rejected |
| Learned simulators (HOOD, ClothTransformer) | 200 times over GIPC [RA] | no gripper-over-body data; a policy would exploit the surrogate | rejected |

PyFleX's shipped `NvFlexReleaseCUDA_x64.a` holds sm_30 code and PTX only. On sm_120 it would run by
driver JIT, which nobody has reported.

## What Wang's code offers

- **No demonstration or planning pipeline fed his final teachers.** `curl/train.py` reads neither
  `demo_buffer_dir` nor any `pretrain_*` key, and the agent is `curl_sac`
  (`curl/launch_train_curl.py:215`). The linear-demo, BC, DAgger, CEM and TOPDM experiments all date
  from before the horizon change and look abandoned rather than rejected.
- **CEM and TOPDM** cost 6,000 simulated transitions per executed decision
  (`cem_plan/launch_cem.py:65-68`), about 3 GPU-days per episode here [E].
- **No results are in the checkout.** It holds no logs, replay or checkpoints, and `paper.txt` is
  FMVP, not RSS 2023.
- **His student** trained on its own stochastic rollouts, one replay per region, with the SAC actor
  loss plus 0.002 times a mean and square-root-sigma match to that region's teacher
  (`curl/SAC_AWAC.py:1025-1039`, `curl/launch_train_curl.py:302-308`). The submitted student scored
  0.70 at 300k steps (`curl/launch_eval.py:34-35`) [RA].
- **FMVP** (the same group, CoRL 2025) replaced that with filtered BC. It rolled out the RL policy
  over 8,000 times across 27 regions, 5 poses and 5 garments, kept 2,514 rollouts with an upper-arm
  ratio of at least 0.7 and no early turn, and trained a PointNet++ on them by BC
  (`paper.txt:1080-1107`) [RA].

## What the literature says

- **Nobody trained a dressing policy from scratch on less than about a million steps.** Clegg 2017
  used 50 to 100M samples, Clegg 2020 28 to 60M, and Assistive Gym 10M PPO steps for 26% success
  [RA]. The escapes changed paradigm: learned dynamics with MPC (Erickson 2018, RoBE), filtered BC
  (FMVP), or fine-tuning a policy that already works (Clegg 2020, FMVP's IQL) [RA].
- **Demonstration-seeded off-policy RL** (RLPD, IBRL, ResFiT, HIL-SERL) reports budgets of 10k to
  500k samples [RA]. On simulated cloth, DDPG with 20 demonstrations and an asymmetric critic
  reached 77 to 90% success after about 80k transitions (Matas 2018) [RA].
- **Those gains are on sparse rewards.** Ours is dense, and plain SAC already reaches the upper arm
  in 83 of 264 training episodes [MI]. Here demonstrations buy the elbow turn, not finding the
  reward, so expect a smaller multiplier than the headlines.
- **No dressing policy runs on a Stretch.** The closest is RoBE, blanket manipulation on a Stretch
  RE1 [RA].

## The proposal

### Teacher: privileged state, anchored on the expert

- **Actor and critic read the 35-float privileged state** (`dressing_privileged.py:22-34`): arm
  geometry in the arm frame, tool pose and clearance, the opening ring, the garment centroid,
  progress, grip error and the garment one-hot. Today only the critic reads it; the actor always
  reads the point cloud (`sac.py`, `_update_critic`).
- **Why it is faster.** An update costs 45 ms today (3,837 s over 84,823 updates) [MI]. MLP updates
  make an update-to-data ratio of 4 affordable, and a teacher that reads no point cloud need not
  render one. Neither saving is measured yet. On cloth, SAC on reduced state clearly beats SAC on
  images (SoftGym) [RA].
- **Seed the replay with every expert episode**, failed ones included with their rewards. Draw half
  of each batch from the expert and half online, with LayerNorm critics and 3-step returns (RLPD,
  ResFiT).
- **Probe A, a bounded residual.** The action is the expert's action plus a bounded correction from
  an MLP on the privileged state and the expert's stage. It starts at the expert's level; its
  ceiling is the expert's strategy (ResFiT [RA]).
- **Probe B, RLPD sampling with a JSRL roll-in.** The expert drives the first h decisions, and h
  shrinks over training.
- **What the expert cannot do.** It is a stage machine whose stages never go back and whose
  timeouts count decisions (`dressing_heuristic.py`), so its actions are wrong away from its own
  trajectory. That rules out DAgger with the expert as labeller, and IBRL's action proposals need a
  re-entrant expert first.

### Student: Wang's recipe on the Stretch 3 cameras

- **The recipe.** A PointNet++ on `stretch3_head_wrist` clouds plus proprioception, trained on its
  own rollouts with the SAC loss plus the teacher term. A privileged teacher can label any state the
  student reaches. Warm-start it with FMVP's filtered BC on teacher rollouts; `collect_rollouts.py`
  already has the filter.
- **Keep the RL term.** Wang's and FMVP's teachers saw point clouds; this teacher does not. Where its
  action depends on what the cameras cannot see, pure imitation fails. Examples are the sleeve
  interior, grip slack, and moves the tether dropped. FLASH is the one precedent that distils a
  scripted privileged cloth teacher into a vision student [RA].
- **The check** is the per-configuration gap between student and teacher. Large gaps locate those
  dependencies.
- **Environment gap.** `observation()` renders one camera mode (`dressing_env.py:676-694`). The
  student loop needs Stretch clouds and privileged rows every step.
- **Onboard compute.** The Stretch 3's NUC has no GPU [RA]; PointNet++ latency on it is unmeasured.

## Step 0, gates and budget

Before any training:

1. **End the expert's last stage on progress, not on a point past the shoulder.**
   - Its last target sits 10 cm past the shoulder (`dressing_heuristic.py:84`). An opening pushed
     past the shoulder is invisible to the progress rays.
   - The reward then falls from the forearm length plus five times the upper-arm distance to minus
     the finger's distance from the opening centre (`dressing_reward.py:125`, `:153`), from about
     +1.8 to about -0.6 per decision [E].
   - Of the four tshirt_26 successes in `2026-09-09-dressing-correctness.md`, body 3 did this: 1.000
     at decision 200, then 0 from decision 225 to the end, 75 decisions. Bodies 2, 5 and 7 held
     their reading.
   - A demonstration that pays for finishing with negative reward teaches the wrong thing on
     whichever cells it hits, and the final ratio is what evaluation scores. A target exactly at the
     shoulder may stop the pull short, hence a progress condition.
2. **Measure the expert on region 13 under the evaluation metric**, the final upper-arm ratio.
   - Run the 25 held-out configurations (7.5k transitions) and the 225 training ones (67.5k): 2 to 6
     GPU-hours [E].
   - The 22 of 40 is the highest reading per episode on bodies 0 to 7, not region 13.
   - On region 13 only tshirt_68 was measured, at 2 to 4 of 8 (`2026-09-11-dressing-solver-stall.md`,
     "The expert does not stall in production mode").
3. **Log privileged state, actions and rewards while doing so.** No path does this today:
   `--policy heuristic` reaches only the evaluation branch of `train_sac.py`, and
   `collect_rollouts.py` stores no privileged state.

| Stage | Pass | Cost [E] |
|---|---|---|
| Gate 1: expert baseline | sets the bar: the expert's held-out mean final upper-arm ratio | 2 to 6 GPU-hours |
| Gate 2: probe A or B at 100k learner transitions | held-out mean above the expert's | about 9 GPU-hours each |
| Teacher to completion | Wang's 0.74 as the reference | 250k to 500k transitions, 1 to 2 days |
| Student | per-configuration gap to the teacher | 150k to 250k transitions, 0.5 to 1 day |

The first go or no-go comes within about one GPU-day of starting Step 0. If both gates pass, teacher
and student take 3 to 5 days [E], against 7 to 14 GPU-days for the current teacher alone.

## Protocol levers, not taken now

- **0.2 s decisions, 150 per episode** (`action_repeat` 12): the same simulated seconds per episode
  for half the transitions, and the same episodes per hour. A protocol change, not tuning.
- **Ending an episode on sustained success** would save the idle 9 to 29% of successful episodes,
  but it changes the task.
- **One privileged teacher for several regions.** The state is in the arm frame, so one teacher
  might replace several of the 27. Test this after region 13 passes.

## Implementation, in order

No GPU is needed until Step 0; about 1 to 2 days of code [E].

1. Expert: the shoulder-bounded target, and a collection path that writes privileged state,
   actions, rewards, dones and final ratios.
2. A privileged MLP actor; a demonstration buffer with half-and-half sampling; LayerNorm, n-step
   returns and an update-to-data setting.
3. The residual wrapper, with the expert's stage as input.
4. Environment: Stretch 3 clouds and privileged rows each step; the student loop with the teacher
   term.
5. Exempt evaluation worlds from the watchdog (the review's 83.5k evaluation).

### Built on 2026-09-12

The teacher kept running throughout; none of this reaches it until it restarts.

- The expert's last stage ends on the progress reading rather than on its target past the shoulder
  (`dressing_heuristic.py`), and the environment exposes that reading (`GenesisIPCDressingEnv.progress`).
- `DressingConfig.decision_watchdog` switches the watchdog off per world, and
  `pretrain_wang.build_world(..., watchdog=False)` builds the evaluation worlds that way.
- `expert_baseline.py` runs the expert over a region's held-out or training configurations, writes
  one episode file of privileged rows, actions and rewards, and summarises them with the records
  `train_sac.evaluate` writes, so the bar and an evaluation round are the same numbers.
- 163 CPU tests pass, including new ones for the finish rule and the episode tape; two GPU smoke
  tests wait for the stop (`test_pretrain_wang_cuda.py`, `test_expert_baseline_cuda.py`).

Still to build: the privileged MLP actor, the demonstration buffer with half-and-half sampling, and
the residual wrapper.

## GPU schedule

Step 0 and the probes need the GPU that relaunch 3 holds. Relaunch 3 reaches 300k transitions
around 2026-09-12 15:00, and that decision now includes this switch.
- **Pause it at an episode end**, from where it resumes.
- **Or share the GPU.** Two processes measured 1.16 times the aggregate throughput, each 64% slower
  (`2026-09-08-uipc-manip-pretraining.md`, "Running a second training process").

## Sources

- Wang et al., RSS 2023: https://arxiv.org/abs/2306.12372
- FMVP, CoRL 2025: https://arxiv.org/abs/2509.12741 ; FCVP: https://arxiv.org/abs/2311.04390
- Clegg 2017: https://arxiv.org/abs/1709.07033 ; Clegg 2020: https://arxiv.org/abs/1909.06682 ;
  Assistive Gym: https://arxiv.org/abs/1910.04700
- Erickson 2018: https://arxiv.org/abs/1709.09735 ; RoBE: https://arxiv.org/abs/2304.04822
- RLPD: https://arxiv.org/abs/2302.02948 ; IBRL: https://arxiv.org/abs/2311.02198 ;
  JSRL: https://arxiv.org/abs/2204.02372 ; HIL-SERL: https://arxiv.org/abs/2410.21845
- ResFiT: https://arxiv.org/abs/2509.19301 ; Policy Decorator: https://arxiv.org/abs/2412.13630
- Matas et al. 2018: https://arxiv.org/abs/1806.07851 ; SoftGym: https://arxiv.org/abs/2011.07215
- FLASH: https://arxiv.org/abs/2604.17513 ; realizable students: https://arxiv.org/abs/2505.09546 ;
  DAgger: https://arxiv.org/abs/1011.0686
- Right-Side-Out: https://arxiv.org/abs/2509.15953 ; GAUGE: https://arxiv.org/abs/2608.05948 ;
  RGBench: https://arxiv.org/abs/2511.06434 ; MuJoCo Warp flex:
  https://github.com/google-deepmind/mujoco_warp/issues/1362
- Stretch 3 hardware: https://docs.hello-robot.com/0.3/hardware/hardware_guide_stretch_3/
