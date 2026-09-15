"""The decision tangent of every lockstep slot on one device, as batched dense block algebra.

Slots never touch, so the exported system is block-diagonal per slot. Each substep's blocks are
scattered from the backend's triplets into a dense ``(N, 3n, 3n)`` tensor, factorised in one
batched call (single precision, then two rounds of double-precision residual refinement), and the
BDF1 inertia chain, the soft position constraint's aim term and friction's lagged blocks are batched
tensor operations: ``H_f X_f = 2M X_{f-1} - M X_{f-2} + s m daim_f - B_fric X_{f-1}`` for the three
translation axes at once. The host does nothing per slot. ``physics_gradient_adjoint.tangent_pass``
is the scalar reference this reproduces (tests).
"""
from __future__ import annotations

import numpy as np
import torch


class BatchedTangent:
    def __init__(self, layouts: list, steps: int, device="cuda", factor_dtype=torch.float32, refine: int = 2):
        self.device = torch.device(device)
        self.N, self.steps = len(layouts), int(steps)
        self.n = int(layouts[0]["n"])
        self.cnt = 3 * self.n
        self.factor_dtype, self.refine = factor_dtype, int(refine)
        d = self.device
        block_starts = np.array([lay["dof_offset"] // 3 for lay in layouts], dtype=np.int64)
        vertex_starts = np.array([lay.get("vertex_offset", lay["dof_offset"] // 3) for lay in layouts], dtype=np.int64)
        self.block_order = np.argsort(block_starts)
        self.block_starts_sorted = block_starts[self.block_order]
        self.vertex_order = np.argsort(vertex_starts)
        self.vertex_starts_sorted = vertex_starts[self.vertex_order]
        self.block_starts = torch.as_tensor(block_starts, device=d)
        self.vertex_starts = torch.as_tensor(vertex_starts, device=d)
        self.m3 = torch.as_tensor(np.stack([np.repeat(lay["mass"], 3) for lay in layouts]), dtype=torch.float64, device=d)
        held = np.asarray(layouts[0]["anchor_idx"], dtype=np.int64)
        for lay in layouts[1:]:
            if not np.array_equal(np.asarray(lay["anchor_idx"]), held):
                raise ValueError("Every slot must hold the same vertices")
        self.held_dofs = [torch.as_tensor(3 * held + k, device=d) for k in range(3)]
        self.sm_held = torch.as_tensor(np.stack([lay["strength"] * np.asarray(lay["mass"])[held] for lay in layouts]),
                                       dtype=torch.float64, device=d)
        self.begin()

    def begin(self):
        z = torch.zeros(self.N, self.cnt, 3, dtype=torch.float64, device=self.device)
        self.dx_prev, self.dx_prev2, self.frames, self.f = z, z.clone(), [], 0

    def _assign(self, ids, starts_sorted, order, starts):
        ids = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=self.device)
        pos = torch.searchsorted(torch.as_tensor(starts_sorted, device=self.device), ids, right=True) - 1
        slot = torch.as_tensor(order, device=self.device)[pos]
        return slot, ids - starts[slot]

    def dense_blocks(self, rows, cols, values) -> torch.Tensor:
        """Scatter the upper-block-triangle triplets of the exported system into per-slot dense blocks."""
        slot_r, r = self._assign(rows, self.block_starts_sorted, self.block_order, self.block_starts)
        slot_c, c = self._assign(cols, self.block_starts_sorted, self.block_order, self.block_starts)
        if not torch.equal(slot_r, slot_c):
            raise ValueError("A block couples two slots; the system is not block-diagonal per slot")
        v = torch.as_tensor(np.asarray(values, dtype=np.float64).reshape(-1, 3, 3), device=self.device)
        A = torch.zeros(self.N * self.cnt * self.cnt, dtype=torch.float64, device=self.device)
        i = torch.arange(3, device=self.device)
        base = slot_r * (self.cnt * self.cnt)
        row_idx = (3 * r)[:, None, None] + i[None, :, None]
        col_idx = (3 * c)[:, None, None] + i[None, None, :]
        A.index_add_(0, (base[:, None, None] + row_idx * self.cnt + col_idx).reshape(-1), v.reshape(-1))
        off = r != c
        if bool(off.any()):
            A.index_add_(0, (base[off][:, None, None] + col_idx[off] * self.cnt + row_idx[off]).reshape(-1), v[off].reshape(-1))
        return A.view(self.N, self.cnt, self.cnt)

    def substep(self, rows, cols, values, coupling=None):
        """One substep's response of every slot to a unit shift of its held vertices' aim per axis;
        ``coupling`` is ``(rows, cols, blocks)`` in global vertex ids (friction's ``dG/dx_prev``)."""
        A = self.dense_blocks(rows, cols, values)
        ratio = (self.f + 1) / self.steps
        rhs = (2.0 * self.m3[:, :, None] * self.dx_prev - self.m3[:, :, None] * self.dx_prev2).contiguous()
        for k in range(3):
            rhs[:, self.held_dofs[k], k] += self.sm_held * ratio
        if coupling is not None and len(coupling[0]):
            c_rows, c_cols, blocks = coupling
            slot_r, r = self._assign(c_rows, self.vertex_starts_sorted, self.vertex_order, self.vertex_starts)
            slot_c, c = self._assign(c_cols, self.vertex_starts_sorted, self.vertex_order, self.vertex_starts)
            keep = (slot_r == slot_c) & (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
            slot_r, r, c = slot_r[keep], r[keep], c[keep]
            B = torch.as_tensor(np.asarray(blocks, dtype=np.float64).reshape(-1, 3, 3), device=self.device)[keep]
            prev = self.dx_prev.reshape(self.N, self.n, 3, 3)[slot_r, c]       # (T, 3, axes)
            pulled = torch.einsum("tij,tja->tia", B, prev)                     # (T, 3, axes)
            flat = rhs.view(self.N * self.n, 3, 3)
            flat.index_add_(0, slot_r * self.n + r, -pulled)
        LU, piv = torch.linalg.lu_factor(A.to(self.factor_dtype))
        x = torch.linalg.lu_solve(LU, piv, rhs.to(self.factor_dtype)).to(torch.float64)
        for _ in range(self.refine):
            residual = rhs - torch.bmm(A, x)
            x = x + torch.linalg.lu_solve(LU, piv, residual.to(self.factor_dtype)).to(torch.float64)
        self.dx_prev2, self.dx_prev = self.dx_prev, x
        self.frames.append(x)
        self.f += 1

    def finish(self, control_jacs: np.ndarray, dt: float, state_scale: float):
        """``(dx, dv, tangent)`` as numpy: the final positions' and velocities' responses to the
        normalised action, and the observation tangent ``(N, 6n + 6, 3)``."""
        J = torch.as_tensor(np.asarray(control_jacs, dtype=np.float64), device=self.device).reshape(self.N, 3, 3)
        dx = torch.bmm(self.frames[-1], J)
        prev = torch.bmm(self.frames[-2], J) if len(self.frames) > 1 else torch.zeros_like(dx)
        dv = (dx - prev) / dt
        tangent = torch.cat([dx, dv, J, torch.zeros(self.N, 3, 3, dtype=torch.float64, device=self.device)], dim=1) / state_scale
        return dx.cpu().numpy(), dv.cpu().numpy(), tangent.cpu().numpy()
