"""Native IPC counterfactual collection and small offline representation experiments.

This is a fixed-mesh cloth_drag diagnostic with synthetic partial observations,
not a dressing dataset, rendered RGB-D benchmark or transferable SAC checkpoint.
Collection records full-state labels but the encoder sees only partial position
histories and past commands. Episode groups are split before augmentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .counterfactual import response_objective
from .models import EncoderConfig, PointNet2Encoder, trunk_layers


def compact_encoder():
    return EncoderConfig(sa_ratio=[.5, .5], sa_neighbors=[8, 8],
                         sa_radius=[.08, .16], sa_mlp=[[16, 24], [24, 32], [32, 48]],
                         linear_mlp=[48], output_dim=32)


class ResponseModel(nn.Module):
    def __init__(self, history=True):
        super().__init__()
        self.use_history = history
        self.encoder = PointNet2Encoder(0, compact_encoder())
        self.history = nn.GRU(35, 32, batch_first=True)
        self.decoder = trunk_layers(38, 64, 3, "residual", 2)

    def encode(self, points, past_actions):
        if not self.use_history:
            points, past_actions = points[:, -1:], past_actions[:, -1:]
        b, h, n, _ = points.shape
        p = points.reshape(b * h, n, 3)
        z = self.encoder(p, p[..., :0], torch.ones(p.shape[:2], dtype=torch.bool, device=p.device))
        z = z.reshape(b, h, -1)
        if not self.use_history:
            return z[:, -1]
        return self.history(torch.cat((z, past_actions), -1))[0][:, -1]

    def respond(self, h, action, query):
        b, n, _ = query.shape
        return self.decoder(torch.cat((h[:, None].expand(b, n, -1),
                                       action[:, None].expand(b, n, -1), query), -1))


def collect(args):
    from .iaql_env import IAQLClothEnv, IAQLEnvConfig
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "data.npz").exists():
        raise FileExistsError("refusing to overwrite collected data")
    cfg = IAQLEnvConfig(num_slots=args.slots, horizon=args.length,
                        export_mode="converged_raw", tangent_device=args.device,
                        friction=args.friction, friction_chain=args.friction > 0)
    env = IAQLClothEnv(root / "world", cfg)
    rng = np.random.default_rng(args.seed)
    arrays = {k: [] for k in ("points", "past_actions", "action", "response", "tangent",
                              "episode", "velocity", "reward", "goal", "probe_plus",
                              "probe_minus", "probe_direction", "probe_valid", "remaining_time")}
    validity, repeatability = [], []
    # Static material coordinates are decoder queries, never hidden current positions.
    query = env.rests[0] - env.origins[0]
    # Identical IDs across variants; labels remain dense, independent of visibility.
    visible_ids = rng.choice(env.n, min(args.points, env.n), replace=False)
    for episode in range(args.episodes):
        env.reset(args.seed + episode)
        ph, ah = [], []
        previous_action = np.zeros((env.N, 3))
        episode_start = len(arrays["action"])
        for step in range(args.length):
            x = env.all_positions().copy()
            local = x - env.origins[:, None]
            ph.append(local[:, visible_ids])
            ah.append(previous_action.copy())
            ph, ah = ph[-args.history:], ah[-args.history:]
            # Repeat initial observation with zero preceding command, consistently across arms.
            history = np.stack([ph[0]] * (args.history - len(ph)) + ph, axis=1)
            actions_history = np.stack([np.zeros_like(previous_action)] * (args.history - len(ah)) + ah, axis=1)
            action = rng.uniform(-.7, .7, (env.N, 3))
            snap = env.snapshot()
            center = env.step(action if env.N > 1 else action[0], capture=True)
            y = env.all_positions().copy() - x
            d = np.asarray(center["tangent"]).reshape(env.N, env.obs_dim, 3)[:, :env.n * 3]
            d = d.reshape(env.N, env.n, 3, 3) * cfg.state_scale
            velocity = env.all_velocities().mean(1)
            v = rng.normal(size=(env.N, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            plus = minus = np.zeros_like(y)
            probe = step in (args.history, args.length - 2)
            if probe:
                for epsilon in args.epsilons:
                    outcomes = []
                    for sign in (1, -1):
                        env.restore(snap)
                        perturbed = action + sign * epsilon * v
                        env.step(perturbed if env.N > 1 else perturbed[0])
                        outcomes.append(env.all_positions().copy() - x)
                    dv = np.einsum("bnca,ba->bnc", d, v)
                    for sign, actual in zip((1, -1), outcomes):
                        error = np.linalg.norm((actual - y - sign * epsilon * dv).reshape(env.N, -1), axis=1)
                        signal = np.linalg.norm((actual - y).reshape(env.N, -1), axis=1)
                        for slot in range(env.N):
                            validity.append(dict(episode=episode, slot=slot, step=step, epsilon=epsilon,
                                                 sign=sign, error=float(error[slot]), signal=float(signal[slot]),
                                                 relative_error=float(error[slot] / (signal[slot] + 1e-9))))
                    if np.isclose(epsilon, args.epsilon):
                        plus, minus = outcomes
                # Replay nominal once; never dump a perturbed state over the anchor snapshot.
                env.restore(snap)
                env.step(action if env.N > 1 else action[0])
                replay_error = env.all_positions() - x - y
                repeatability.append(dict(episode=episode, step=step,
                    max_error_m=float(np.max(np.abs(replay_error))),
                    error_norm_m=np.linalg.norm(replay_error.reshape(env.N, -1), axis=1).tolist()))
                if np.max(np.abs(replay_error)) > 1e-5:
                    raise RuntimeError("nominal replay differs by more than 10 micrometers")
            values = dict(points=history, past_actions=actions_history, action=action,
                          response=y, tangent=d, episode=np.full(env.N, episode), velocity=velocity,
                          reward=np.asarray(center["reward"]).reshape(env.N),
                          goal=env.goals - env.origins, probe_plus=plus, probe_minus=minus,
                          probe_direction=v, probe_valid=np.full(env.N, probe),
                          remaining_time=np.full((env.N, 1), 1 - step / args.length))
            for key, value in values.items():
                arrays[key].append(value)
            previous_action = action
        print(json.dumps({"collected_episode": episode, "decisions": len(arrays["action"]) - episode_start}), flush=True)
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    # Monte Carlo values under the random collection policy, finite horizon, gamma=.99.
    rewards = data["reward"].reshape(args.episodes, args.length, env.N)
    returns = np.zeros_like(rewards)
    carry = np.zeros((args.episodes, env.N))
    for step in reversed(range(args.length)):
        carry = rewards[:, step] + .99 * carry
        returns[:, step] = carry
    data.update(query=query, visible_ids=visible_ids, task_ids=env.held.copy(), returns=returns.reshape(-1))
    np.savez_compressed(root / "data.npz", **data)
    (root / "linearization.json").write_text(json.dumps(validity, indent=2))
    (root / "repeatability.json").write_text(json.dumps(repeatability, indent=2))
    meta = dict(schema=1, config=vars(args), environment=env.describe(), response_units="meters",
                action_units="normalized [-1,1]", derivatives="meters per normalized action",
                observation="fixed subset of material vertices; synthetic partial cloud, not RGB-D",
                probe_epsilon=args.epsilon)
    (root / "dataset.json").write_text(json.dumps(meta, indent=2, default=str))


def split(data, seed):
    episodes = np.unique(data["episode"])
    if len(episodes) < 4:
        raise ValueError("need at least four independent episode groups")
    episodes = np.random.default_rng(seed).permutation(episodes)
    heldout = episodes[:max(1, len(episodes) // 4)]
    test = np.isin(data["episode"], heldout)
    return np.flatnonzero(~test), np.flatnonzero(test)


def sample_targets(tangent, task_ids, mode, count, rng):
    """Privileged *training* sampling only; validation always covers every vertex."""
    rows = []
    for d in tangent:
        n = len(d)
        sensitivity = np.linalg.norm(d.reshape(n, -1), axis=1)
        probability = sensitivity / sensitivity.sum() if sensitivity.sum() > 0 else None
        if mode == "uniform":
            ids = rng.choice(n, count)
        elif mode == "sensitivity":
            ids = np.r_[rng.choice(n, count // 2, p=probability), rng.choice(n, count - count // 2)]
        elif mode == "task":
            # These are grasp vertices in cloth_drag, not dressing contact annotations.
            if len(task_ids) == 0:
                raise ValueError("task sampling requires recorded grasp IDs")
            k = count // 4
            ids = np.r_[rng.choice(n, 2*k, p=probability), rng.choice(task_ids, k), rng.choice(n, count-3*k)]
        else:
            raise ValueError(mode)
        rows.append(ids)
    return np.stack(rows)


def ridge_probe(train_x, train_y, test_x, test_y):
    # Train-only feature standardization and fixed ridge, no test-selected regularizer.
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-4)
    x, z = (train_x - mean) / std, (test_x - mean) / std
    x = torch.cat((x, torch.ones_like(x[:, :1])), -1).double()
    z = torch.cat((z, torch.ones_like(z[:, :1])), -1).double()
    y = train_y.double()
    w = torch.linalg.solve(x.T @ x + torch.eye(x.shape[1], device=x.device) * 1., x.T @ y)
    return float(F.mse_loss(z @ w, test_y.double()))


def run(args):
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("refusing to overwrite experiment results")
    with np.load(root / "data.npz") as loaded:
        data = {k: loaded[k] for k in loaded.files}
    metadata = json.loads((root / "dataset.json").read_text())
    if not np.isclose(args.epsilon, metadata["probe_epsilon"]):
        raise ValueError("training epsilon must match the stored true-response probe")
    train_idx, test_idx = split(data, args.split_seed)
    ood_digest = None
    if args.ood_data:
        ood_root = Path(args.ood_data)
        ood_meta = json.loads((ood_root / "dataset.json").read_text())
        if not np.isclose(args.epsilon, ood_meta["probe_epsilon"]):
            raise ValueError("OOD probe epsilon differs")
        with np.load(ood_root / "data.npz") as loaded:
            ood = {k: loaded[k] for k in loaded.files}
        for key in ("query", "visible_ids"):
            if not np.array_equal(data[key], ood[key]):
                raise ValueError(f"OOD dataset has incompatible {key}")
        base_rows = len(data["action"])
        ood["episode"] = ood["episode"] + int(data["episode"].max()) + 1
        for key in data:
            if key not in {"query", "visible_ids", "task_ids"}:
                data[key] = np.concatenate((data[key], ood[key]))
        test_idx = np.arange(base_rows, len(data["action"]))
        with (ood_root / "data.npz").open("rb") as handle:
            ood_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    device = torch.device(args.device)
    tensors = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in data.items()
               if k not in {"visible_ids", "episode", "query", "task_ids"}}
    query = torch.as_tensor(data["query"], device=device, dtype=torch.float32)
    # One scalar train-only RMS keeps spatial and action-direction anisotropy intact.
    scale = tensors["response"][train_idx].square().mean().sqrt().clamp_min(1e-5)
    tensors["response"] /= scale; tensors["tangent"] /= scale
    tensors["probe_plus"] /= scale; tensors["probe_minus"] /= scale
    all_results = []
    for seed in args.seeds:
        for variant in args.variants:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            target_rng = np.random.default_rng(seed + 10000)
            model = ResponseModel(history=variant not in {"no_history", "counterfactual_no_history"}).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            mode = {"no_history": "difference", "counterfactual_no_history": "counterfactual", "random_encoder": "response"}.get(variant, variant)
            started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step in range(0 if variant == "random_encoder" else args.steps):
                idx = rng.choice(train_idx, args.batch_size)
                h = model.encode(tensors["points"][idx], tensors["past_actions"][idx])
                q = query[None].expand(len(idx), -1, -1)
                y, d = tensors["response"][idx], tensors["tangent"][idx]
                if args.target_sampling != "dense":
                    ids = sample_targets(d.detach().cpu().numpy(), data.get("task_ids", np.array([], dtype=int)), args.target_sampling,
                                         args.target_points, target_rng)
                    rows = torch.arange(len(idx), device=device)[:, None]
                    ids = torch.as_tensor(ids, device=device)
                    q, y, d = q[rows, ids], y[rows, ids], d[rows, ids]
                # Matched per-seed batch/direction streams across objectives.
                v = torch.as_tensor(rng.normal(size=(len(idx), 3)), device=device, dtype=torch.float32)
                v = F.normalize(v, dim=-1)
                loss, stats = response_objective(lambda a: model.respond(h, a, q),
                    tensors["action"][idx], y, d, v,
                    mode=mode, epsilon=args.epsilon, weight=args.weight)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            metrics = evaluate(model, tensors, query, data["visible_ids"], train_idx, test_idx, scale, args)
            metrics.update(variant=variant, seed=seed, training_s=elapsed, peak_cuda_bytes=peak)
            torch.save(dict(model=model.state_dict(), variant=variant, response_scale=float(scale),
                            config=vars(args)), out / f"{variant}_s{seed}.pt")
            all_results.append(metrics)
            (out / "results.json").write_text(json.dumps(all_results, indent=2))
            print(json.dumps(metrics), flush=True)
    with (root / "data.npz").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    (out / "protocol.json").write_text(json.dumps(dict(config=vars(args), dataset_sha256=digest,
        ood_dataset_sha256=ood_digest, train_episodes=np.unique(data["episode"][train_idx]).tolist(),
        test_episodes=np.unique(data["episode"][test_idx]).tolist(), response_scale=float(scale)), indent=2))


def evaluate(model, t, query, visible_ids, train_idx, test_idx, scale, args):
    model.eval()
    latents, predictions, derivatives, occluded = [], [], [], []
    for start in range(0, len(t["action"]), args.batch_size):
        idx = slice(start, start + args.batch_size)
        with torch.no_grad():
            points = t["points"][idx]
            h = model.encode(points, t["past_actions"][idx])
            h_occluded = model.encode(points[:, :, :max(8, points.shape[2] // 2)], t["past_actions"][idx])
            a = t["action"][idx]; q = query[None].expand(len(a), -1, -1)
            predictions.append(model.respond(h, a, q))
            occluded.append(model.respond(h_occluded, a, q))
            latents.append(h)
        # JVP measured at held-out actions; derivative is not detached during training.
        _, jvp = torch.autograd.functional.jvp(lambda act: model.respond(h, act, q), a,
                                               t["probe_direction"][idx])
        derivatives.append(jvp.detach())
    h, pred, jvp = torch.cat(latents), torch.cat(predictions), torch.cat(derivatives)
    target = torch.einsum("bnca,ba->bnc", t["tangent"], t["probe_direction"])
    test = torch.as_tensor(test_idx, device=h.device)
    p, d = jvp[test].flatten(1), target[test].flatten(1)
    result = dict(response_mse=float(F.mse_loss(pred[test], t["response"][test])),
                  response_rmse_m=float(F.mse_loss(pred[test], t["response"][test]).sqrt() * scale),
                  jvp_relative_rmse=float((p-d).square().sum().sqrt() / d.square().sum().sqrt().clamp_min(1e-8)),
                  jvp_cosine=float(F.cosine_similarity(p, d).mean()),
                  occluded_response_mse=float(F.mse_loss(torch.cat(occluded)[test], t["response"][test])))
    result["constant_response_mse"] = float((t["response"][test] - t["response"][train_idx].mean(0)).square().mean())
    result["constant_velocity_mse"] = float((t["velocity"][test] - t["velocity"][train_idx].mean(0)).square().mean())
    constant_d = t["tangent"][train_idx].mean(0)
    constant_jvp = torch.einsum("nca,ba->bnc", constant_d, t["probe_direction"][test]).flatten(1)
    result["constant_jvp_cosine"] = float(F.cosine_similarity(constant_jvp, d).mean())
    result["constant_jvp_relative_rmse"] = float((constant_jvp-d).square().sum().sqrt() / d.square().sum().sqrt().clamp_min(1e-8))
    # Decoder discarded: probe only frozen h and action, not the response teacher.
    features = torch.cat((h, t["action"]), -1)
    result["frozen_velocity_mse"] = ridge_probe(features[train_idx], t["velocity"][train_idx],
                                                 features[test], t["velocity"][test])
    result["frozen_tangent_mse"] = ridge_probe(features[train_idx], t["tangent"][train_idx].flatten(1),
                                               features[test], t["tangent"][test].flatten(1))
    result["constant_tangent_mse"] = float((t["tangent"][test] - constant_d).square().mean())
    commands = torch.cat((t["past_actions"].flatten(1), t["action"]), -1)
    result["commands_only_velocity_mse"] = ridge_probe(commands[train_idx], t["velocity"][train_idx],
                                                        commands[test], t["velocity"][test])
    probe_idx = test[t["probe_valid"][test] > .5]
    with torch.no_grad():
        q = query[None].expand(len(probe_idx), -1, -1)
        a, v = t["action"][probe_idx], t["probe_direction"][probe_idx]
        errors = []
        for sign, key in ((1, "probe_plus"), (-1, "probe_minus")):
            actual_pred = model.respond(h[probe_idx], a + sign * args.epsilon * v, q)
            errors.append(F.mse_loss(actual_pred, t[key][probe_idx]))
        result["true_perturbed_response_mse"] = float(torch.stack(errors).mean())
    # Offline dense Q probes consume frozen h; response conditioning has its own matched arm.
    for use_response in (False, True):
        torch.manual_seed(args.probe_seed)
        qnet = DenseValueProbe(use_response).to(h.device)
        optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr)
        rng = np.random.default_rng(args.probe_seed)
        y = t["returns"][:, None]
        mean, std = y[train_idx].mean(), y[train_idx].std().clamp_min(1e-4)
        normalized_y = (y - mean) / std
        for _ in range(args.probe_steps):
            idx = rng.choice(train_idx, args.batch_size)
            value = qnet(t["points"][idx, -1], h[idx], t["action"][idx], t["goal"][idx], pred[idx][:, visible_ids], t["remaining_time"][idx])
            loss = F.mse_loss(value, normalized_y[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            value = qnet(t["points"][test, -1], h[test], t["action"][test], t["goal"][test], pred[test][:, visible_ids], t["remaining_time"][test])
            result["response_q_mse" if use_response else "dense_q_mse"] = float(F.mse_loss(value, normalized_y[test]))
            result["constant_q_mse"] = float(normalized_y[test].square().mean())
    return result


class DenseValueProbe(nn.Module):
    """Diagnostic dense action-conditioned value network, not the production SAC critic.

    Same parameter count with/without predicted per-point cloth response; zero channels
    in the control. Target is return under the collection policy, not optimal Q.
    """
    def __init__(self, use_response):
        super().__init__()
        self.use_response = use_response
        self.encoder = PointNet2Encoder(9, compact_encoder())
        self.head = trunk_layers(65, 64, 1, "residual", 2)

    def forward(self, points, h, action, goal, response, remaining_time):
        extra = torch.cat((action, goal), -1)[:, None].expand(-1, points.shape[1], -1)
        feat = torch.cat((extra, response if self.use_response else torch.zeros_like(response)), -1)
        valid = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        return self.head(torch.cat((h, self.encoder(points, feat, valid), remaining_time), -1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--episodes", type=int, default=8)
    c.add_argument("--length", type=int, default=20)
    c.add_argument("--slots", type=int, default=4)
    c.add_argument("--history", type=int, default=4)
    c.add_argument("--points", type=int, default=64)
    c.add_argument("--friction", type=float, default=0.)
    c.add_argument("--seed", type=int, default=123)
    c.add_argument("--device", default="cuda")
    c.add_argument("--epsilons", nargs="+", type=float, default=[.005, .02, .05, .1])
    c.add_argument("--epsilon", type=float, default=.05)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ood-data", help="evaluate on a separate same-mesh corpus; never train on it")
    r.add_argument("--steps", type=int, default=500)
    r.add_argument("--probe-steps", type=int, default=200)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--weight", type=float, default=1.)
    r.add_argument("--target-sampling", choices=["dense", "uniform", "sensitivity", "task"], default="dense")
    r.add_argument("--target-points", type=int, default=64)
    r.add_argument("--epsilon", type=float, default=.05)
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--split-seed", type=int, default=2026)
    r.add_argument("--probe-seed", type=int, default=500)
    r.add_argument("--device", default="cpu")
    r.add_argument("--variants", nargs="+", default=["random_encoder", "response", "jacobian", "counterfactual", "difference", "no_history"],
                   choices=["random_encoder", "response", "jacobian", "counterfactual", "difference", "no_history", "counterfactual_no_history"])
    args = p.parse_args()
    if args.phase == "collect":
        if args.episodes < 4 or args.length <= args.history + 1 or args.slots < 1 or args.points < 8:
            p.error("need >=4 episodes, length > history+1, positive slots and >=8 visible points")
        if not all(0 < e <= .3 for e in args.epsilons) or args.epsilon not in args.epsilons:
            p.error("probe radii must lie in (0,.3] and include epsilon")
        collect(args)
    else:
        if args.steps < 1 or args.probe_steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.epsilon <= 0 or args.target_points < 1:
            p.error("steps, batch size, lr and epsilon must be positive")
        run(args)


if __name__ == "__main__":
    main()
