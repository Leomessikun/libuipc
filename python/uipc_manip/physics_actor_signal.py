"""The physics direction of the next state's value for every slot of a training world.

After a vector step the backend still holds the last frame's assembled system for all slots at
once (the slots share one IPC world and do not touch, so the system is block-diagonal across
them). The signal is ``g_u = (∂V/∂x')ᵀ ∂x'/∂u`` per slot: the critic's value of the next
observation, ``V(x') = min Q(s', μ(s'))``, differentiated exactly through the environment's
observation function (visibility and voxel membership captured from the environment's own call
inside ``env.step``), and one device solve against the global system with every slot's
``∂V/∂x'`` as the right-hand side; the held vertices of each slot then give its six action
components. A slot whose command was refused by more than the gate, whose capture did not
match its observation, or whose direction is not finite contributes no signal.
"""
from __future__ import annotations

import time

import numpy as np

from uipc_manip import physics_gradient_adjoint as adjoint
from uipc_manip.obs import FLAG_DEFORMABLE


def action_gradient(lam_cloth: np.ndarray, layout: dict, offsets: np.ndarray, max_translation: float,
                    max_rotation: float, clip_x_rotation: bool) -> np.ndarray:
    """Six action components from a slot's adjoint solution on its cloth (3n): translation through
    the anchor, rotation through the held offsets, per action unit."""
    lam = np.asarray(lam_cloth, dtype=np.float64).reshape(layout["n"], 3)[layout["anchor_idx"]]
    held = lam * (layout["strength"] * layout["mass"][layout["anchor_idx"]])[:, None]
    g = np.zeros(6)
    g[:3] = held.sum(axis=0) * float(max_translation)
    g[3:] = np.cross(np.asarray(offsets, dtype=np.float64), held).sum(axis=0) * float(max_rotation)
    if clip_x_rotation:
        g[3] = 0.0
    return g


class PhysicsActorSignal:
    def __init__(self, agent, gate: float = 0.5, rel_tol: float = 1e-4, max_rounds: int = 8) -> None:
        self.agent = agent
        self.gate = float(gate)
        self.rel_tol = float(rel_tol)
        self.max_rounds = int(max_rounds)
        self.env = None
        self.feature = None
        self.layouts: list[dict] = []
        self.n_arm: list[int] = []
        self._calls: list = []
        self._hooked = False
        self._original = None
        self._anchor_before = None
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ binding to a world
    def bind(self, env) -> None:
        self.env = env
        self.feature = adjoint.adjoint_feature(env)
        if self.feature is None:
            raise RuntimeError("the physics actor signal needs a build with LinearSystemAdjointFeature (use the tree's build, not the wheel)")
        self.layouts = [adjoint.cloth_layout(env, 1.0, slot=i) for i in range(env.num_envs)]
        self.n_arm = [int(np.asarray(cell.arm_points).shape[0]) for cell in env.cells]
        self.voxel_size = float(env._batched_obs.cfg.voxel_size_m)

    # ------------------------------------------------------------------ around env.step
    def before_step(self) -> None:
        from uipc_manip import dressing_obs

        self._anchor_before = np.asarray(self.env._anchor, dtype=np.float64).copy()
        self._calls = []
        original = dressing_obs.voxel_centroids_batched

        def hooked(pts, select, voxel_size, n_envs):
            self._calls.append((pts.detach().cpu().numpy().copy(), select.detach().cpu().numpy().copy()))
            return original(pts, select, voxel_size, n_envs)

        self._original = original
        dressing_obs.voxel_centroids_batched = hooked
        self._hooked = True

    def abort(self) -> None:
        self._unhook()
        self._calls = []

    def _unhook(self) -> None:
        if self._hooked:
            from uipc_manip import dressing_obs

            dressing_obs.voxel_centroids_batched = self._original
            self._hooked = False

    def after_step(self, next_obs: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(g [B, 6], valid [B], stats)`` for the step just taken; ``next_obs`` is the observation the
        environment returned for it."""
        import torch

        self._unhook()
        env, agent = self.env, self.agent
        b = int(env.num_envs)
        g = np.zeros((b, 6))
        valid = np.zeros(b, dtype=bool)
        stats = {"physics_s": 0.0, "physics_solve_s": 0.0, "physics_residual": float("nan"), "physics_capture_fail": 0}
        t0 = time.time()
        try:
            if not self._calls:
                raise RuntimeError("no observation call was captured during the step")
            pts, select = self._calls[-1]
            positions = [np.asarray(p, dtype=np.float64) for p in env.positions()]
            spec = env.spec
            flat = torch.as_tensor(np.asarray(next_obs, dtype=np.float32).reshape(b, -1), device=agent.device)
            pos, feat, valid_pts, extra = spec.unpack_torch(flat)
            pos = pos.clone()
            xs = []
            ok = np.ones(b, dtype=bool)
            for i in range(b):
                n = positions[i].shape[0]
                rows = np.arange(self.n_arm[i], self.n_arm[i] + n)
                if rows[-1] >= pts.shape[1] or not np.allclose(pts[i, rows], positions[i].astype(np.float32), atol=1e-5):
                    ok[i] = False
                    xs.append(None)
                    continue
                visible = np.nonzero(select[i, rows])[0]
                vox = np.floor(positions[i][visible].astype(np.float32) / self.voxel_size).astype(np.int64) + (1 << 15)
                key = (vox[:, 0] << 32) | (vox[:, 1] << 16) | vox[:, 2]
                uniq, inverse = np.unique(key, return_inverse=True)
                deformable = torch.nonzero(valid_pts[i] & (feat[i, :, FLAG_DEFORMABLE] > 0.5)).reshape(-1)
                if int(deformable.numel()) != int(uniq.shape[0]) or uniq.shape[0] == 0:
                    ok[i] = False
                    xs.append(None)
                    continue
                x = torch.as_tensor(positions[i], device=agent.device, dtype=torch.float32).requires_grad_(True)
                counts = torch.as_tensor(np.bincount(inverse).astype(np.float32), device=agent.device)
                sums = torch.zeros((uniq.shape[0], 3), device=agent.device, dtype=torch.float32)
                sums = sums.index_add(0, torch.as_tensor(inverse, device=agent.device), x[torch.as_tensor(visible, device=agent.device)])
                centroids = sums / counts.unsqueeze(-1)
                tool = torch.as_tensor(np.asarray(env._anchor[i], dtype=np.float32), device=agent.device)
                if not torch.allclose(pos[i, deformable], (centroids - tool[None, :]).detach(), atol=2e-5):
                    ok[i] = False
                    xs.append(None)
                    continue
                pos[i, deformable] = centroids - tool[None, :]
                xs.append(x)
            stats["physics_capture_fail"] = int((~ok).sum())
            if not ok.any():
                raise RuntimeError("no slot's capture matched its observation")
            if agent._cut_padding:
                used = valid_pts.any(dim=0).nonzero()
                m = int(used.max()) + 1 if used.numel() else 1
                pos, feat, valid_pts = pos[:, :m], feat[:, :m], valid_pts[:, :m]
            obs = (pos, feat, valid_pts, extra)
            mu, _, _, _ = agent.actor(obs, compute_pi=False, compute_log_pi=False)
            q1, q2 = agent.critic(obs, mu)
            v = torch.min(q1, q2).sum()
            grads = torch.autograd.grad(v, [x for x in xs if x is not None])
            rhs = np.zeros(int(self.feature.dof_count()))
            it = iter(grads)
            for i in range(b):
                if not ok[i]:
                    continue
                gi = next(it).detach().cpu().numpy().astype(np.float64).reshape(-1)
                lay = self.layouts[i]
                rhs[lay["dof_offset"]:lay["dof_offset"] + lay["dof_count"]] = gi
            t1 = time.time()
            lam, residual = self.feature.solve(rhs, self.rel_tol, self.max_rounds)
            lam = np.asarray(lam)
            stats["physics_solve_s"] = time.time() - t1
            stats["physics_residual"] = float(residual)
            max_t, max_r = float(env.cfg.max_translation), float(env.cfg.max_rotation)
            clip = bool(getattr(env.cfg, "clip_rotation_to_yz", False))
            anchor_after = np.asarray(env._anchor, dtype=np.float64)
            for i in range(b):
                if not ok[i]:
                    continue
                lay = self.layouts[i]
                g[i] = action_gradient(lam[lay["dof_offset"]:lay["dof_offset"] + lay["dof_count"]], lay, np.asarray(env._offsets[i]), max_t, max_r, clip)
                commanded = float(np.linalg.norm(np.asarray(actions[i, :3], dtype=np.float64))) * max_t
                moved = float(np.linalg.norm(anchor_after[i] - self._anchor_before[i]))
                fraction = moved / commanded if commanded > 1e-9 else 1.0
                valid[i] = bool(np.isfinite(g[i]).all() and np.linalg.norm(g[i]) > 0.0 and fraction >= self.gate and np.isfinite(residual))
        except Exception as exc:  # noqa: BLE001 - the signal must never take the training step down
            stats["physics_error"] = repr(exc)[:200]
            g[:] = 0.0
            valid[:] = False
        stats["physics_s"] = time.time() - t0
        stats["physics_valid_fraction"] = float(valid.mean())
        self.last_stats = stats
        return g, valid, stats
