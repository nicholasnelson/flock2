"""Rung 3 of the ladder (``docs/flock-2/m1-map-world-model.md``): a world model whose state has
spatial shape, with every operation learned and no geometry built in.

The structure is the RSSM's (``flock.wm``): a deterministic memory h, a stochastic state z with a
prior p(z_t | h_t) and a posterior q(z_t | h_t, o_t), trained by reconstructing the view and felt
numbers, predicting the reward, and the two KL terms. What changes is the shape of the state:

- **h** is a G x G grid of cells (``cell`` numbers each) plus a global vector (``glob``), and
  **z** is one categorical (``classes`` values) per cell plus ``n_glob_z`` global categoricals.
- **Seeing (posterior).** The view is encoded into ``cols`` column features. Each cell asks, by
  attention, what the columns say: its query comes from its memory and a learned cell embedding,
  the keys from each column's features and a learned column embedding. A learned "nothing" key lets
  a cell take nothing in. The cell's posterior comes from its memory and what it took in. The felt
  numbers go to the global part.
- **Moving (transition).** A convolutional GRU (3 x 3) over the grid, with the previous z and the
  action as input at every cell; the global vector has its own GRU. How the grid should shift or
  turn when the sheep moves is not given: it must be learned, if it is useful at all.
- **Visualising (decoder).** Each column of the view asks the cells, by attention, what it should
  show (learned column queries; keys from each cell's h, z and learned embedding). Each column's
  answer is decoded into 8 rows of features, and transposed convolutions paint the 64 x 256 strip.
- Felt numbers and reward are predicted from the global part and the grid's mean.

Nothing tells the model which cell means "ahead-left" or how a step moves things: cell and column
embeddings are learned parameters, not positions. If a map emerges, it is the sheep's own.

The model keeps ``flock.wm.WorldModel``'s interface (``embed_obs``, ``transition``, ``prior``,
``posterior``, ``observe``, ``decode``, ``decode_intero``, ``reward_head``, ``initial``), with h and z
flattened, so ``wm.loss_fn`` and the hybrid scripts use it unchanged.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from flock import wm


class SpatialConfig(NamedTuple):
    h: int = 64
    w: int = 256
    n_classes: int = 3  # RGB
    rgb: bool = True
    n_actions: int = 8
    grid: int = 16
    cell: int = 32  # deterministic numbers per cell
    classes: int = 8  # values of each cell's categorical (and of each global one)
    glob: int = 256  # global deterministic size
    n_glob_z: int = 16  # global categoricals
    cols: int = 32  # view columns the encoder produces
    col_dim: int = 128
    key_dim: int = 64
    hidden: int = 384
    felt_dim: int = 128
    intero_weight: float = wm.INTERO_WEIGHT
    recon_weight: float = 1.0
    reward_weight: float = wm.REWARD_WEIGHT
    recon_balance: float = 0.0
    body_dims: tuple = ()
    kl_dyn: float = wm.KL_DYN
    ms_weight: float = 0.0
    ms_horizons: tuple = (2, 4, 8)
    ms_stride: int = 8

    @property
    def n_cells(self) -> int:
        return self.grid * self.grid

    @property
    def deter(self) -> int:
        return self.n_cells * self.cell + self.glob

    @property
    def stoch(self) -> int:
        return self.n_cells + self.n_glob_z

    @property
    def feat(self) -> int:
        return self.deter + self.stoch * self.classes

    @property
    def embed(self) -> int:
        return self.cols * self.col_dim + self.felt_dim


SPATIAL = SpatialConfig()
_act = jax.nn.silu


def _conv_any(conv, x):
    """Apply a 2-D conv to (..., H, W, C) with any leading dims."""
    lead = x.shape[:-3]
    y = conv(x.reshape((-1,) + x.shape[-3:]))
    return y.reshape(lead + y.shape[1:])


class ConvGRU(nnx.Module):
    def __init__(self, inp: int, size: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(inp + size, 3 * size, (3, 3), padding="SAME", use_bias=False, rngs=rngs)
        self.norm = nnx.LayerNorm(3 * size, rngs=rngs)

    def __call__(self, x, h):
        parts = self.norm(_conv_any(self.conv, jnp.concatenate([x, h], -1)))
        reset, cand, update = jnp.split(parts, 3, -1)
        cand = jnp.tanh(jax.nn.sigmoid(reset) * cand)
        update = jax.nn.sigmoid(update - 1.0)
        return update * cand + (1 - update) * h


class ColumnEncoder(nnx.Module):
    """(..., 64, 256, 3) colour centred on 0 -> (..., cols, col_dim) column features."""

    def __init__(self, rngs: nnx.Rngs, c: SpatialConfig):
        ch = (3, 32, 64, 96)
        self.convs = nnx.List([nnx.Conv(ch[i], ch[i + 1], (4, 4), strides=2, padding="SAME", rngs=rngs) for i in range(3)])
        self.norms = nnx.List([nnx.LayerNorm(k, rngs=rngs) for k in ch[1:]])
        self.c = c
        self.out = nnx.Linear((c.h // 8) * 96, c.col_dim, rngs=rngs)

    def __call__(self, x):
        lead = x.shape[:-3]
        y = x.reshape((-1,) + x.shape[-3:])
        for conv, norm in zip(self.convs, self.norms):
            y = _act(norm(conv(y)))  # -> (N, 8, 32, 96)
        y = y.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1)  # (N, cols, 8 * 96)
        return self.out(y).reshape(lead + (self.c.cols, self.c.col_dim))


class StripDecoder(nnx.Module):
    """(..., cols, d) per-column features -> (..., 64, 256, 3) colour centred on 0."""

    def __init__(self, rngs: nnx.Rngs, c: SpatialConfig, d: int):
        self.c = c
        self.inp = nnx.Linear(d, (c.h // 8) * 96, rngs=rngs)
        ch = (96, 64, 32, 3)
        self.deconvs = nnx.List([nnx.ConvTranspose(ch[i], ch[i + 1], (4, 4), strides=(2, 2), padding="SAME", rngs=rngs) for i in range(3)])
        self.norms = nnx.List([nnx.LayerNorm(k, rngs=rngs) for k in ch[1:-1]])

    def __call__(self, f):
        lead = f.shape[:-2]
        y = self.inp(f.reshape((-1,) + f.shape[-2:]))  # (N, cols, 8 * 96)
        y = y.reshape(y.shape[0], self.c.cols, self.c.h // 8, 96).transpose(0, 2, 1, 3)  # (N, 8, 32, 96)
        for i, dc in enumerate(self.deconvs):
            y = dc(y)
            if i < len(self.norms):
                y = _act(self.norms[i](y))
        return y.reshape(lead + (self.c.h, self.c.w, 3))


class GlobalHead(nnx.Module):
    """A prediction from the global part of the state and the grid's mean (felt numbers, reward)."""

    def __init__(self, rngs: nnx.Rngs, c: SpatialConfig, out: int):
        self.c = c
        self.mlp = wm.MLP(c.glob + c.n_glob_z * c.classes + c.cell + c.classes, out, rngs, c.hidden)

    def __call__(self, feat):
        return self.mlp(_global_summary(self.c, feat))


def _split(c: SpatialConfig, h=None, z=None):
    out = []
    if h is not None:
        out += [h[..., : c.n_cells * c.cell].reshape(h.shape[:-1] + (c.grid, c.grid, c.cell)), h[..., c.n_cells * c.cell :]]
    if z is not None:
        zz = z.reshape(z.shape[:-1] + (c.stoch, c.classes))
        out += [zz[..., : c.n_cells, :].reshape(z.shape[:-1] + (c.grid, c.grid, c.classes)), zz[..., c.n_cells :, :].reshape(z.shape[:-1] + (c.n_glob_z * c.classes,))]
    return out


def _global_summary(c, feat):
    h, z = feat[..., : c.deter], feat[..., c.deter :]
    grid, g, zgrid, zg = _split(c, h, z)
    return jnp.concatenate([g, zg, grid.mean((-3, -2)), zgrid.mean((-3, -2))], -1)


class SpatialWorldModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, n_intero: int, reward: bool = True, cfg: SpatialConfig = SPATIAL):
        c = self.cfg = cfg
        self.n_intero = n_intero
        self.ensemble = None
        self.encoder = ColumnEncoder(rngs, c)
        self.felt_in = nnx.Linear(n_intero, c.felt_dim, rngs=rngs)
        self.cell_emb = nnx.Param(0.1 * jax.random.normal(rngs.params(), (c.n_cells, c.key_dim)))
        self.col_emb = nnx.Param(0.1 * jax.random.normal(rngs.params(), (c.cols, c.key_dim)))
        self.null_key = nnx.Param(0.1 * jax.random.normal(rngs.params(), (1, c.key_dim)))
        self.null_val = nnx.Param(jnp.zeros((1, c.col_dim)))
        # seeing: cell queries over column keys
        self.q_cell = nnx.Linear(c.cell + c.key_dim, c.key_dim, rngs=rngs)
        self.k_col = nnx.Linear(c.col_dim + c.key_dim, c.key_dim, rngs=rngs)
        self.post_cell = wm.MLP(c.cell + c.col_dim, c.classes, rngs, 128)
        self.prior_cell = wm.MLP(c.cell, c.classes, rngs, 128)
        self.post_glob = wm.MLP(c.glob + c.felt_dim + c.col_dim, c.n_glob_z * c.classes, rngs, c.hidden)
        self.prior_glob = wm.MLP(c.glob, c.n_glob_z * c.classes, rngs, c.hidden)
        # moving
        self.grid_in = nnx.Conv(c.classes + c.n_actions, c.cell, (3, 3), padding="SAME", rngs=rngs)
        self.grid_gru = ConvGRU(c.cell, c.cell, rngs)
        self.glob_in = wm.MLP(c.n_glob_z * c.classes + c.classes + c.n_actions, c.hidden, rngs, c.hidden)
        self.glob_gru = wm.GRU(c.hidden, c.glob, rngs)
        # visualising: column queries over cell keys
        self.q_dec = nnx.Param(0.1 * jax.random.normal(rngs.params(), (c.cols, c.key_dim)))
        self.k_dec = nnx.Linear(c.cell + c.classes + c.key_dim, c.key_dim, rngs=rngs)
        self.v_dec = nnx.Linear(c.cell + c.classes, c.col_dim, rngs=rngs)
        self.decoder = StripDecoder(rngs, c, c.col_dim + c.glob)
        self.intero_head = GlobalHead(rngs, c, n_intero)
        self.reward_head = GlobalHead(rngs, c, wm.REWARD_BINS) if reward else None
        if reward:
            self.reward_head.mlp.b.kernel[...] = jnp.zeros_like(self.reward_head.mlp.b.kernel[...])

    # --- the RSSM interface -----------------------------------------------------------------------
    def obs_input(self, views):
        return views.astype(jnp.float32) / 255.0 - 0.5

    def embed_obs(self, views, intero=None):
        c = self.cfg
        cols = self.encoder(self.obs_input(views))
        return jnp.concatenate([cols.reshape(cols.shape[:-2] + (-1,)), self.felt_in(intero)], -1)

    def initial(self, batch: int):
        return jnp.zeros((batch, self.cfg.deter)), jnp.zeros((batch, self.cfg.stoch * self.cfg.classes))

    def transition(self, h, z, a_onehot):
        c = self.cfg
        grid, g, zgrid, zg = _split(c, h, z)
        a_grid = jnp.broadcast_to(a_onehot[..., None, None, :], grid.shape[:-1] + (c.n_actions,))
        x = _act(_conv_any(self.grid_in, jnp.concatenate([zgrid, a_grid], -1)))
        grid = self.grid_gru(x, grid)
        g = self.glob_gru(_act(self.glob_in(jnp.concatenate([zg, zgrid.mean((-3, -2)), a_onehot], -1))), g)
        return jnp.concatenate([grid.reshape(grid.shape[:-3] + (-1,)), g], -1)

    def prior(self, h):
        c = self.cfg
        grid, g = _split(c, h)
        cell = self.prior_cell(grid).reshape(grid.shape[:-3] + (c.n_cells, c.classes))
        glob = self.prior_glob(g).reshape(g.shape[:-1] + (c.n_glob_z, c.classes))
        return jnp.concatenate([cell, glob], -2)

    def posterior(self, h, embed):
        c = self.cfg
        grid, g = _split(c, h)
        cols = embed[..., : c.cols * c.col_dim].reshape(embed.shape[:-1] + (c.cols, c.col_dim))
        felt = embed[..., c.cols * c.col_dim :]
        cells = grid.reshape(grid.shape[:-3] + (c.n_cells, c.cell))
        q = self.q_cell(jnp.concatenate([cells, jnp.broadcast_to(self.cell_emb[...], cells.shape[:-1] + (c.key_dim,))], -1))
        k = self.k_col(jnp.concatenate([cols, jnp.broadcast_to(self.col_emb[...], cols.shape[:-1] + (c.key_dim,))], -1))
        k = jnp.concatenate([k, jnp.broadcast_to(self.null_key[...], k.shape[:-2] + (1, c.key_dim))], -2)
        v = jnp.concatenate([cols, jnp.broadcast_to(self.null_val[...], cols.shape[:-2] + (1, c.col_dim))], -2)
        att = jax.nn.softmax(jnp.einsum("...qd,...kd->...qk", q, k) / jnp.sqrt(c.key_dim), -1)
        seen = jnp.einsum("...qk,...kd->...qd", att, v)  # (..., cells, col_dim)
        cell = self.post_cell(jnp.concatenate([cells, seen], -1))
        glob = self.post_glob(jnp.concatenate([g, felt, cols.mean(-2)], -1)).reshape(g.shape[:-1] + (c.n_glob_z, c.classes))
        return jnp.concatenate([cell, glob], -2)

    def observe(self, views, actions, key, intero=None) -> wm.Step:
        B, T = views.shape[:2]
        nA = self.cfg.n_actions
        embed = self.embed_obs(views, intero)
        a_prev = jnp.concatenate([jnp.zeros((B, 1, nA)), jax.nn.one_hot(actions[:, :-1], nA)], 1)
        h0, z0 = self.initial(B)

        def body(carry, x):
            h, z = carry
            e, a, k = x
            h = self.transition(h, z, a)
            prior = self.prior(h)
            post = self.posterior(h, e)
            z = wm._sample(post, k)
            return (h, z), wm.Step(h, z, prior, post)

        _, out = jax.lax.scan(body, (h0, z0), (embed.swapaxes(0, 1), a_prev.swapaxes(0, 1), jax.random.split(key, T)))
        return jax.tree.map(lambda x: x.swapaxes(0, 1), out)

    def decode(self, s: wm.Step):
        c = self.cfg
        grid, g, zgrid, _ = _split(c, s.h, s.z)
        cells = jnp.concatenate([grid, zgrid], -1).reshape(grid.shape[:-3] + (c.n_cells, c.cell + c.classes))
        k = self.k_dec(jnp.concatenate([cells, jnp.broadcast_to(self.cell_emb[...], cells.shape[:-1] + (c.key_dim,))], -1))
        att = jax.nn.softmax(jnp.einsum("qd,...kd->...qk", self.q_dec[...], k) / jnp.sqrt(c.key_dim), -1)
        cols = jnp.einsum("...qk,...kd->...qd", att, self.v_dec(cells))  # (..., cols, col_dim)
        cols = jnp.concatenate([cols, jnp.broadcast_to(g[..., None, :], cols.shape[:-1] + (c.glob,))], -1)
        return self.decoder(cols)

    def decode_intero(self, s: wm.Step):
        return self.intero_head(jnp.concatenate([s.h, s.z], -1))

    def reward(self, feat):
        return wm.symexp(jnp.sum(jax.nn.softmax(self.reward_head(feat), -1) * wm.BINS, -1))
