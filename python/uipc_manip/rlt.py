"""Recurrent Looped Transformer over decision frames.

Zhang, *Recurrent Looped Transformer* (2026 technical report, upstream commit ``1bee93a9``, which
ships no code). A causal encoder ``E`` over the token prefix builds a global memory, and a decoder
``D`` advances a complete state ``H_t = (s_t, C_t)`` once per token from the encoder feature and the
previous state. Equation numbers below are the report's.

Here a token is one decision: the frame's spatial encoding, and for the actor the command that led
to it. The module owns the temporal parameters only; the spatial encoders live in the heads and feed
per-frame vectors in, so nothing here sees a point cloud.

Three execution schedules of the same computation are provided, as the report's section 2.4
requires them to agree:

* :meth:`RecurrentLoopedTransformer.run` — a window in one call: the encoder in parallel under a
  causal mask, then the decoder recurrence position by position. Returns every state and the
  caches that produced it.
* :meth:`RecurrentLoopedTransformer.branch` — alternative tokens at every position of a recorded
  run, each attending to the recorded prefix before it and to itself only. This is how a candidate
  action is scored against the recorded history at every learning position at once, and how the
  Bellman successor is advanced with the recorded command: "batch across sequences, preserve the
  recurrence within each" (4.1), applied across positions.
* :meth:`RecurrentLoopedTransformer.step` — one token at a time with an explicit
  :class:`RLTState` of cached keys and values, for collection.

Padding is left-contiguous, as ``replay.sample_sequences(pad=True)`` and ``RolloutHistory`` produce
it: the state is ``(s*, ∅)`` at the first valid position and no valid position ever reads a padded
one. Every query may always attend to its own position, so a padded row cannot produce NaN.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RLTConfig:
    dim: int = 64
    """``d``, the width of encoder features, memory values and the state."""
    layers: int = 2
    """``L_E = L_D``; the report's tied configuration needs them equal and so do we."""
    heads: int = 4
    window: int = 8
    """``W``, the decoder's causal sliding window over its own activations; ``1`` keeps no past."""
    groups: int = 1
    """``G`` memory groups: ``1`` shares one memory across decoder layers, ``layers`` gives each its own."""
    alpha: float = 0.5
    """Feedback scale of the merge (2.11); the report calls a modest nonzero value a candidate initialisation."""
    tied: bool = False
    """Share each encoder layer's attention and FFN with the decoder layer of the same depth (2.6)."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RLTConfig":
        return cls(**data)


def _rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary positional transformation of ``x [B,T,h,dh]`` at integer positions ``pos [B,T]``.

    The report leaves the positional transformation of the query/key maps open; a rotation makes
    attention depend on relative position, so a window that starts mid-episode and a stream that
    started at reset agree wherever their prefixes agree.
    """
    half = x.shape[-1] // 2
    inv = 1.0 / (10000.0 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angle = pos.to(torch.float32)[..., None] * inv
    cos, sin = angle.cos()[:, :, None, :].to(x.dtype), angle.sin()[:, :, None, :].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Projections and the attention operator with its output projection.

    Normalisations are not here: the report keeps them stage-specific, so a tied decoder can reuse
    this module while owning its own norms. ``mask [B,Tq,Tk]`` is True where a query may attend.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError("RLT dim must split into heads of even width")
        self.heads, self.head_dim = int(heads), dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.heads, self.head_dim)

    def query(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return _rotate(self._split(self.q(x)), pos)

    def kv(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _rotate(self._split(self.k(x)), pos), self._split(self.v(x))

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask[:, None])
        return self.o(out.transpose(1, 2).reshape(*q.shape[:2], -1))


def _ffn(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.attn, self.ffn = Attention(dim, heads), _ffn(dim)


class DecoderLayer(nn.Module):
    """SWA over decoder activations (2.12–2.14), cross-attention to memory (2.15), FFN (2.16).

    With ``shared`` the SWA projections and the FFN are the encoder layer's (2.6); the norms, the
    cross-attention query/output projections and the memory group index stay this layer's own.
    """

    def __init__(self, dim: int, heads: int, group: int, shared: EncoderLayer | None) -> None:
        super().__init__()
        self.group = int(group)
        self.norm_s, self.norm_m, self.norm_d = nn.RMSNorm(dim), nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.swa = shared.attn if shared is not None else Attention(dim, heads)
        self.ffn = shared.ffn if shared is not None else _ffn(dim)
        self.xq, self.xo = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.heads, self.head_dim = int(heads), dim // heads

    def memory_query(self, b: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return _rotate(self.xq(self.norm_m(b)).reshape(*b.shape[:-1], self.heads, self.head_dim), pos)

    def read_memory(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask[:, None])
        return self.xo(out.transpose(1, 2).reshape(*q.shape[:2], -1))


class Memory(nn.Module):
    """Encoder-derived memory groups (2.2): ``k = P_K(e, t)``, ``v = W_V RMSNorm_E(e)``."""

    def __init__(self, dim: int, heads: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.k = nn.ModuleList(nn.Linear(dim, dim) for _ in range(groups))
        self.v = nn.ModuleList(nn.Linear(dim, dim) for _ in range(groups))
        self.heads, self.head_dim = int(heads), dim // heads

    def forward(self, e: torch.Tensor, pos: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        en = self.norm(e)
        split = lambda x: x.reshape(*x.shape[:-1], self.heads, self.head_dim)  # noqa: E731
        return [_rotate(split(k(en)), pos) for k in self.k], [split(v(en)) for v in self.v]


class Merge(nn.Module):
    """Gated merge of the encoder feature with the previous state (2.9–2.11)."""

    def __init__(self, dim: int, alpha: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.feedback = nn.Linear(dim, dim, bias=False)
        self.alpha = float(alpha)

    def forward(self, e: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        r = self.norm(s_prev)
        g = torch.sigmoid(self.gate(torch.cat([e, r], dim=-1)))
        return e + self.alpha * g * self.feedback(r)


@dataclass
class Run:
    """Everything a window's recorded pass produced: states and the caches a branch reads."""

    s: torch.Tensor
    """``[B,T,d]`` raw states ``s_t`` (before the readout norm)."""
    valid: torch.Tensor
    pos: torch.Tensor
    first: torch.Tensor
    """``[B,T]``: the first valid position of each row, where the state was ``s*``."""
    enc_k: list[torch.Tensor]
    enc_v: list[torch.Tensor]
    mem_k: list[torch.Tensor]
    mem_v: list[torch.Tensor]
    dec_k: list[torch.Tensor]
    dec_v: list[torch.Tensor]


@dataclass
class RLTState:
    """Streaming state of ``N`` streams: the report's ``(s_t, C_t)`` plus the encoder cache and memory.

    Buffers hold ``max_len`` slots; slot ``j`` of a stream is its ``j``-th token since reset, so the
    filled prefix of a stream is ``slots < count``. Inference only: nothing here carries a graph.
    """

    s: torch.Tensor
    count: torch.Tensor
    enc_k: list[torch.Tensor]
    enc_v: list[torch.Tensor]
    mem_k: list[torch.Tensor]
    mem_v: list[torch.Tensor]
    dec_k: list[torch.Tensor]
    dec_v: list[torch.Tensor]

    @property
    def max_len(self) -> int:
        return int(self.enc_k[0].shape[1])


class RecurrentLoopedTransformer(nn.Module):
    def __init__(self, in_dim: int, cfg: RLTConfig) -> None:
        super().__init__()
        d, layers = int(cfg.dim), int(cfg.layers)
        if layers < 1:
            raise ValueError("RLT needs at least one layer")
        if int(cfg.window) < 1:
            raise ValueError("The decoder window W must be at least 1")
        if int(cfg.groups) not in (1, layers):
            raise ValueError("Memory groups must be 1 (shared) or one per decoder layer")
        self.cfg = cfg
        self.embed = nn.Linear(int(in_dim), d)
        self.enc = nn.ModuleList(EncoderLayer(d, cfg.heads) for _ in range(layers))
        self.memory = Memory(d, cfg.heads, int(cfg.groups))
        self.dec = nn.ModuleList(
            DecoderLayer(d, cfg.heads, layer if int(cfg.groups) == layers else 0, self.enc[layer] if cfg.tied else None)
            for layer in range(layers)
        )
        self.merge = Merge(d, cfg.alpha)
        self.s_star = nn.Parameter(torch.zeros(d))
        self.norm_o = nn.RMSNorm(d)
        self.out_dim = d

    # ------------------------------------------------------------------ shared pieces
    @staticmethod
    def positions(valid: torch.Tensor) -> torch.Tensor:
        """Position of each frame counted from its row's first valid frame."""
        return (valid.long().cumsum(1) - 1).clamp_min(0)

    @staticmethod
    def first_valid(valid: torch.Tensor) -> torch.Tensor:
        before = torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], dim=1)
        return valid & ~before

    def readout(self, s: torch.Tensor) -> torch.Tensor:
        """``RMSNorm_o(s_t)`` (2.6); the heads' trunks read this."""
        return self.norm_o(s)

    # ------------------------------------------------------------------ a window
    def run(self, tokens: torch.Tensor, valid: torch.Tensor) -> Run:
        """``tokens [B,T,in]``, ``valid [B,T]`` -> every state of the window under the current parameters."""
        B, T, _ = tokens.shape
        valid = valid.bool()
        pos, first = self.positions(valid), self.first_valid(valid)
        own = torch.eye(T, dtype=torch.bool, device=tokens.device)[None].expand(B, -1, -1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tokens.device))[None] & valid[:, None, :] | own
        x = self.embed(tokens)
        enc_k, enc_v = [], []
        for layer in self.enc:
            xn = layer.norm1(x)
            k, v = layer.attn.kv(xn, pos)
            x = x + layer.attn.attend(layer.attn.query(xn, pos), k, v, causal)
            x = x + layer.ffn(layer.norm2(x))
            enc_k.append(k)
            enc_v.append(v)
        mem_k, mem_v = self.memory(x, pos)
        W = int(self.cfg.window)
        s_prev = self.s_star.expand(B, -1)
        s_star = self.s_star.expand(B, -1)
        states, dec_k, dec_v = [], [[] for _ in self.dec], [[] for _ in self.dec]
        ones = torch.ones(B, 1, 1, dtype=torch.bool, device=tokens.device)
        for t in range(T):
            s_prev = torch.where(first[:, t, None], s_star, s_prev)
            z = self.merge(x[:, t], s_prev)
            mem_mask = torch.cat([valid[:, None, :t], ones], dim=2)
            lo = max(0, t - W + 1)
            swa_mask = torch.cat([valid[:, None, lo:t], ones], dim=2)
            for index, layer in enumerate(self.dec):
                zn = layer.norm_s(z)[:, None]
                k, v = layer.swa.kv(zn, pos[:, t : t + 1])
                dec_k[index].append(k)
                dec_v[index].append(v)
                past_k = torch.cat(dec_k[index][lo:], dim=1)
                past_v = torch.cat(dec_v[index][lo:], dim=1)
                b = z[:, None] + layer.swa.attend(layer.swa.query(zn, pos[:, t : t + 1]), past_k, past_v, swa_mask)
                g = layer.group
                a = b + layer.read_memory(layer.memory_query(b, pos[:, t : t + 1]), mem_k[g][:, : t + 1], mem_v[g][:, : t + 1], mem_mask)
                z = (a + layer.ffn(layer.norm_d(a)))[:, 0]
            states.append(z)
            s_prev = z
        return Run(
            s=torch.stack(states, dim=1), valid=valid, pos=pos, first=first, enc_k=enc_k, enc_v=enc_v,
            mem_k=mem_k, mem_v=mem_v, dec_k=[torch.cat(k, dim=1) for k in dec_k], dec_v=[torch.cat(v, dim=1) for v in dec_v],
        )

    def branch(self, run: Run, tokens: torch.Tensor) -> torch.Tensor:
        """States ``s'_t`` an alternative token at every position would produce after the recorded prefix.

        Position ``t`` of ``tokens`` replaces the recorded token at ``t``: it attends to the recorded
        keys before ``t`` and to itself, merges with the recorded ``s_{t-1}``, and reads the recorded
        memory before ``t`` plus its own entry. The recorded run is not changed. Returns ``[B,T,d]``.
        """
        B, T, _ = tokens.shape
        valid, pos = run.valid, run.pos
        own = torch.eye(T, dtype=torch.bool, device=tokens.device)[None].expand(B, -1, -1)
        before = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tokens.device), diagonal=-1)[None] & valid[:, None, :]
        prefix = torch.cat([before, own], dim=2)
        x = self.embed(tokens)
        for index, layer in enumerate(self.enc):
            xn = layer.norm1(x)
            k, v = layer.attn.kv(xn, pos)
            k, v = torch.cat([run.enc_k[index], k], dim=1), torch.cat([run.enc_v[index], v], dim=1)
            x = x + layer.attn.attend(layer.attn.query(xn, pos), k, v, prefix)
            x = x + layer.ffn(layer.norm2(x))
        own_k, own_v = self.memory(x, pos)
        s_prev = torch.cat([self.s_star.expand(B, 1, -1), run.s[:, :-1]], dim=1)
        s_prev = torch.where(run.first[..., None], self.s_star.expand(B, T, -1), s_prev)
        z = self.merge(x, s_prev)
        W = int(self.cfg.window)
        offsets = torch.arange(T, device=tokens.device)
        band = before & (offsets[None, :] >= offsets[:, None] - (W - 1))[None]
        swa_mask = torch.cat([band, own], dim=2)
        for index, layer in enumerate(self.dec):
            zn = layer.norm_s(z)
            k, v = layer.swa.kv(zn, pos)
            k, v = torch.cat([run.dec_k[index], k], dim=1), torch.cat([run.dec_v[index], v], dim=1)
            b = z + layer.swa.attend(layer.swa.query(zn, pos), k, v, swa_mask)
            g = layer.group
            mk, mv = torch.cat([run.mem_k[g], own_k[g]], dim=1), torch.cat([run.mem_v[g], own_v[g]], dim=1)
            a = b + layer.read_memory(layer.memory_query(b, pos), mk, mv, prefix)
            z = a + layer.ffn(layer.norm_d(a))
        return z

    # ------------------------------------------------------------------ one token at a time
    def init_state(self, num_streams: int, max_len: int, device=None) -> RLTState:
        d, h, dh = self.out_dim, self.cfg.heads, self.out_dim // self.cfg.heads
        device = self.s_star.device if device is None else device
        buf = lambda: torch.zeros(num_streams, max_len, h, dh, device=device, dtype=self.s_star.dtype)  # noqa: E731
        layers, groups = len(self.enc), len(self.memory.k)
        state = RLTState(
            s=self.s_star.detach().expand(num_streams, -1).clone(), count=torch.zeros(num_streams, dtype=torch.long, device=device),
            enc_k=[buf() for _ in range(layers)], enc_v=[buf() for _ in range(layers)],
            mem_k=[buf() for _ in range(groups)], mem_v=[buf() for _ in range(groups)],
            dec_k=[buf() for _ in range(layers)], dec_v=[buf() for _ in range(layers)],
        )
        return state

    def reset_state(self, state: RLTState, streams=None) -> None:
        """Back to ``H_0 = (s*, ∅)`` for the given streams (default all): the caches are masked by ``count``."""
        rows = slice(None) if streams is None else torch.as_tensor(streams, device=state.count.device)
        state.count[rows] = 0
        state.s[rows] = self.s_star.detach()

    @torch.no_grad()
    def step(self, state: RLTState, token: torch.Tensor) -> torch.Tensor:
        """Advance every stream by one token; returns the readout ``[N,d]`` and updates ``state`` in place."""
        N, L = token.shape[0], state.max_len
        t = state.count
        if bool((t >= L).any()):
            raise ValueError(f"A stream has run past the {L} slots of its RLT state; reset it at the episode boundary")
        rows = torch.arange(N, device=token.device)
        slots = torch.arange(L, device=token.device)[None].expand(N, -1)
        filled = (slots <= t[:, None])[:, None, :]
        pos = t[:, None]
        x = self.embed(token)[:, None]
        for index, layer in enumerate(self.enc):
            xn = layer.norm1(x)
            k, v = layer.attn.kv(xn, pos)
            state.enc_k[index][rows, t], state.enc_v[index][rows, t] = k[:, 0], v[:, 0]
            x = x + layer.attn.attend(layer.attn.query(xn, pos), state.enc_k[index], state.enc_v[index], filled)
            x = x + layer.ffn(layer.norm2(x))
        mk, mv = self.memory(x, pos)
        for g in range(len(mk)):
            state.mem_k[g][rows, t], state.mem_v[g][rows, t] = mk[g][:, 0], mv[g][:, 0]
        z = self.merge(x, state.s[:, None])
        W = int(self.cfg.window)
        band = filled & (slots >= (t[:, None] - (W - 1)))[:, None, :]
        for index, layer in enumerate(self.dec):
            zn = layer.norm_s(z)
            k, v = layer.swa.kv(zn, pos)
            state.dec_k[index][rows, t], state.dec_v[index][rows, t] = k[:, 0], v[:, 0]
            b = z + layer.swa.attend(layer.swa.query(zn, pos), state.dec_k[index], state.dec_v[index], band)
            g = layer.group
            a = b + layer.read_memory(layer.memory_query(b, pos), state.mem_k[g], state.mem_v[g], filled)
            z = a + layer.ffn(layer.norm_d(a))
        state.s = z[:, 0]
        state.count = t + 1
        return self.readout(state.s)


class RecurrentLoopedHistory(nn.Module):
    """The RLT as a history module of the heads: frame vectors in, a state per frame out.

    Token ``t`` is ``[frame_t ; a_{t-1}]``; the command before a window's first frame is unknown and
    zero, so a window that starts mid-episode and the empty history at reset carry the same token
    layout. ``action_dim = 0`` (the dense critic) leaves the commands inside the frame encodings.
    ``forward`` reads the last position, which is what the acting policy needs; ``run``, ``sequence``
    and ``branch`` expose every position for learning.
    """

    def __init__(self, frame_dim: int, action_dim: int, length: int, cfg: RLTConfig) -> None:
        super().__init__()
        self.length, self.frame_dim, self.action_dim = int(length), int(frame_dim), int(action_dim)
        if self.length < 2:
            raise ValueError("An RLT history needs at least two frames; length 1 is the single-frame policy")
        self.cfg = cfg
        self.model = RecurrentLoopedTransformer(self.frame_dim + self.action_dim, cfg)
        self.out_dim = self.model.out_dim

    def tokens(self, latent: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor | None = None) -> torch.Tensor:
        keep = valid.bool()[..., None]
        # A padded frame is an empty cloud; an encoder may hand back NaN for it, and NaN * 0 is NaN.
        parts = [torch.where(keep, latent, torch.zeros_like(latent))]
        if self.action_dim:
            if commands is None:
                raise ValueError("This history reads the commands between frames")
            previous = torch.cat([torch.zeros_like(commands[:, :1]), commands], dim=1)
            parts.append(torch.where(torch.cat([keep[:, :1] & False, keep[:, :-1]], dim=1), previous, torch.zeros_like(previous)))
        return torch.cat(parts, dim=-1)

    def run(self, latent: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor | None = None) -> Run:
        return self.model.run(self.tokens(latent, valid, commands), valid)

    def sequence(self, latent: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor | None = None) -> torch.Tensor:
        """Readout at every position, ``[B,T,d]``."""
        return self.model.readout(self.run(latent, valid, commands).s)

    def branch(self, run: Run, latent: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor | None = None) -> torch.Tensor:
        """Readout of the state an alternative frame vector at every position would produce, ``[B,T,d]``."""
        return self.model.readout(self.model.branch(run, self.tokens(latent, valid, commands)))

    def forward(self, latent: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor | None = None) -> torch.Tensor:
        """``latent [B,L,frame_dim]``, ``valid [B,L]``, ``commands [B,L-1,action_dim]`` -> ``[B,d]`` at the last frame."""
        return self.sequence(latent, valid, commands)[:, -1]


class TrajectoryPretrainingHead(nn.Module):
    """Action-conditioned trajectory prediction, a robotics adaptation of recurrent pretraining.

    The state after observation ``t`` contains only preceding commands. Successor-state and reward
    predictions additionally read the recorded current command ``a_t``. Optional command regression
    reads the state alone and is behavior cloning; its default zero weight prevents imitation of
    random or failed behavior. This continuous multitask loss is not the report's language-model
    likelihood. Each component is an elementwise mean over valid positions, with explicit weights
    needed to account for different target units. Padding neither contributes losses nor cuts the
    valid recurrent state's gradient paths.
    """

    def __init__(self, dim: int, action_dim: int, priv_dim: int = 0, reward: bool = True, *,
                 command_weight: float = 0.0, priv_weight: float = 1.0, reward_weight: float = 1.0) -> None:
        super().__init__()
        self.command_weight, self.priv_weight, self.reward_weight = map(float, (command_weight, priv_weight, reward_weight))
        if any(not math.isfinite(w) or w < 0 for w in (self.command_weight, self.priv_weight, self.reward_weight)):
            raise ValueError("Pretraining weights must be finite and nonnegative")
        self.command = nn.Linear(dim, int(action_dim))
        self.priv = nn.Linear(dim + int(action_dim), int(priv_dim)) if int(priv_dim) > 0 else None
        self.reward = nn.Linear(dim + int(action_dim), 1) if reward else None

    def forward(self, state: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor,
                next_priv: torch.Tensor | None = None, rewards: torch.Tensor | None = None) -> dict:
        """``state [B,T,d]`` readouts, ``valid [B,T]``, ``commands [B,T,A]`` taken at each position,
        ``next_priv [B,T,P]`` reached after it, ``rewards [B,T,1]``. Returns unweighted per-target
        diagnostics and their weighted sum under ``"loss"``. Disabled command diagnostics are
        detached and supply no imitation gradient."""
        valid = valid.bool()
        state = torch.where(valid[..., None], state, torch.zeros_like(state))
        commands = torch.where(valid[..., None], commands, torch.zeros_like(commands))
        dynamics = torch.cat([state, commands], dim=-1)

        def mse(pred, target):
            # Select before arithmetic: a missing padded target may contain NaN or Inf.
            selected = pred[valid]
            return (selected - target[valid]).square().mean() if selected.numel() else pred.sum() * 0.0

        command_loss = mse(torch.tanh(self.command(state)), commands)
        out = {"next_command": command_loss if self.command_weight else command_loss.detach()}
        total = state.sum() * 0.0
        if self.command_weight:
            total = total + self.command_weight * command_loss
        if self.priv is not None:
            if next_priv is None:
                raise ValueError("This head predicts the privileged state; the batch carries none")
            out["next_priv"] = mse(self.priv(dynamics), next_priv)
            total = total + self.priv_weight * out["next_priv"]
        if self.reward is not None:
            if rewards is None:
                raise ValueError("This head predicts the reward; the batch carries none")
            out["reward"] = mse(self.reward(dynamics), rewards)
            total = total + self.reward_weight * out["reward"]
        out["loss"] = total
        return out


def pretraining_step(actor, head: TrajectoryPretrainingHead, batch, unpack) -> dict:
    """One pretraining loss on a padded ``SequenceBatch`` for an actor with an RLT history.

    The actor's own stack — spatial encoder, tokens ``[frame_t ; a_{t-1}]``, recurrent state —
    is what pretrains, so RL then starts from the weights that predicted the trajectories.
    ``unpack`` turns flat observations into the encoder's ``(pos, feat, valid, extra)``.
    """
    length = batch.actions.shape[1]
    frames = unpack(batch.obs[:, :length].reshape(-1, batch.obs.shape[-1]))
    valid = batch.valid.bool()
    state = actor.history.sequence(actor._frame_latent(frames).reshape(*valid.shape, -1), valid, batch.actions[:, :-1])
    next_priv = batch.priv[:, 1:] if getattr(batch, "priv", None) is not None else None
    return head(state, valid, batch.actions, next_priv, batch.rewards)
