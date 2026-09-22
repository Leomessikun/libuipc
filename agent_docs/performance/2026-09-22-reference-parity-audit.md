# Reference parity: every hyperparameter matches, the budget is 40x

Date: 2026-09-22. Reference: `/home/ge47gax/kun/dressing/curl` (Wang et al., RSS 2023,
arXiv:2306.12372), launcher `launch_train_curl.py`, agent `SAC_AWAC.py`.

Written because a collapse diagnosis nearly produced two changes that the reference contradicts.

## The critic: two classes, and only one is used

`SAC_AWAC.py` defines both `Critic` (line 246) and `FlowCritic` (line 319). Reading the first and
stopping is a mistake this audit corrects.

Selection, `SAC_AWAC.py:591`:

```python
if ('flow' in self.encoder_type and self.args.flow_q_mode != 'concat_latent') or \
   ('flow' not in self.encoder_type and self.args.vector_q_mode == 'repeat_action'):
    critic_class = FlowCritic
else:
    critic_class = Critic
```

The dressing launcher sets `encoder_type = 'pointcloud_flow'` (`launch_train_curl.py:466`) and
`flow_q_mode = 'repeat_gripper'` (line 472), so **`FlowCritic` is the critic of the dressing task**.
`Critic` is dead code for this configuration.

`FlowCritic.forward`:

```python
new_obs.x = torch.cat([new_obs.x, action], dim=-1)    # action onto every point
obs, indices = self.encoder(new_obs, detach=detach_encoder)
q1 = self.Q1(obs)                                      # heads see no action
q2 = self.Q2(obs)
```

with `new_args.pc_feature_dim += action_shape[0]` widening the per-point feature. This is exactly
`--critic-action-mode dense` (`models.py::_broadcast_action`), so that change was well founded.

Two further facts about it:

- **One encoder feeds both Q heads.** `Critic` and `FlowCritic` both do this, so a shared encoder is
  the reference's design, not a port defect. Giving Q1 and Q2 separate encoders would be inventing a
  deviation.
- **The heads are plain MLPs**, `Linear-ReLU-Linear-ReLU-Linear`. `--trunk-style residual` is ours,
  not the reference's. It measured better here (0.417/0.288 against 0.283/0.150 on region 13, two
  seeds each, re-scored at +-0.01) and is worth keeping on that evidence alone.

## Hyperparameters

`default_config.py` is overridden by the launcher; reading the former alone gives wrong answers
(it says `critic_lr: 1e-3`, `batch_size: 128`, `discount: 0.99`, `num_train_steps: 2000000`).
The launcher's values, against ours:

| | reference | ours | |
|---|---:|---:|---|
| `critic_lr` | 1e-4 | 1e-4 | match |
| `actor_lr` | 1e-4 | 1e-4 | match |
| `alpha_lr` | 1e-4 | 1e-4 | match |
| `actor_update_freq` | 4 | 4 | match |
| `batch_size` | 64 | 64 | match |
| `critic_tau` | 0.01 | 0.01 | match |
| `encoder_tau` | 0.05 | 0.05 | match |
| `critic_target_update_freq` | 2 | 2 | match |
| `init_temperature` | 0.1 | 0.1 | match |
| `hidden_dim` | 1024 | 1024 | match |
| `encoder_feature_dim` | 50 | 50 | match |
| `discount` | 0.99 at horizon 150 | 0.995 at horizon 300 | horizon-matched |
| `use_curl` | False | absent | match |
| `lr_decay` | None | absent | match |
| **`num_train_steps`** | **5,000,000** | **125,000** | **40x** |

`encoder_lr = 1e-6` (line 500) is a red herring: the optimizers that use it are created only under
`if self.use_curl:` (`SAC_AWAC.py:665`), and `use_curl` is False (line 479). The critic's encoder
trains inside `critic_optimizer` at `critic_lr`, as ours does. `num_train_steps = 50000` at line 218
is the `debug` branch.

Per episode the gap is wider than 40x: theirs is 5,000,000 / 150 = 33,333 episodes, ours
125,000 / 300 = 417, so **80x fewer episodes**.

## What this leaves

Architecture and optimisation are at parity. The late fall of every arm -- peak then decline with
`q1_mean` rising -- has been diagnosed at 2.5% of the reference's budget, and the one run that went
further (the 270k baseline) was the weaker latent+plain configuration. dense+residual has never been
run past 125k: the resume that would have done so was killed at 134k.

Two claims made during that diagnosis and withdrawn here: that the reference's released critic is
latent-Q (it is `FlowCritic`, dense), and that our learning rates are 10x below the reference's
(they are identical; the 10x came from the overridden `default_config.py`).

At 14,500 transitions/hour a 5,000,000-transition run is 345 hours, so matching the budget outright
is not available. What is available is one long run -- resume dense+residual to 400-500k, 20-35
hours -- which separates "the recipe tops out near 0.43" from "the curve resumes climbing".
