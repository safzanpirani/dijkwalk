"""Policies for GraphWalk. All mask invalid neighbor slots using the valid
flag baked into the observation (slot feature 0), so no separate action-mask
plumbing is needed in the PPO loop.

encode_observations/decode_actions split follows pufferlib.models.Default so
the LSTMWrapper composes. The mask is stashed on self during encode; the
(B*T) flattening in LSTMWrapper preserves row order, so it lines up in decode.
"""
import torch
import torch.nn as nn

import pufferlib.models
import pufferlib.pytorch

from .env import K, F, N_GLOBAL

MASK_VALUE = -1e9


def split_obs(observations):
    """-> (slot features [B,K,F], global features [B,N_GLOBAL])"""
    obs = observations.float()
    return obs[:, :K * F].view(-1, K, F), obs[:, K * F:]


class Policy(nn.Module):
    """MLP baseline: shared per-slot encoder, flatten, mask invalid logits."""

    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.slot_enc = nn.Sequential(nn.Linear(F, 32), nn.GELU())
        self.body = nn.Sequential(
            nn.Linear(K * 32 + N_GLOBAL, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, K), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)
        self._invalid = None

    def encode_observations(self, observations, state=None):
        slots, glob = split_obs(observations)
        self._invalid = slots[:, :, 0] < 0.5
        x = self.slot_enc(slots).flatten(1)
        return self.body(torch.cat([x, glob], dim=1))

    def decode_actions(self, hidden):
        logits = self.actor(hidden).masked_fill(self._invalid, MASK_VALUE)
        return logits, self.value(hidden)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        super().__init__(env, policy, input_size, hidden_size)


class Transformer(nn.Module):
    """Transformer encoder over neighbor-slot tokens + a global token.
    Per-slot logits come from the slot token outputs; value from the global."""

    def __init__(self, env, hidden_size=128, layers=4, heads=4, ff=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.slot_proj = nn.Linear(F, hidden_size)
        self.glob_proj = nn.Linear(N_GLOBAL, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=ff,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.slot_head = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)
        self._invalid = None
        self._slot_out = None

    def encode_observations(self, observations, state=None):
        slots, glob = split_obs(observations)
        invalid = slots[:, :, 0] < 0.5
        self._invalid = invalid
        tokens = torch.cat(
            [self.glob_proj(glob).unsqueeze(1), self.slot_proj(slots)], dim=1)
        pad = torch.cat(
            [torch.zeros_like(invalid[:, :1]), invalid], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=pad)
        self._slot_out = out[:, 1:]
        return out[:, 0]

    def decode_actions(self, hidden):
        logits = self.slot_head(self._slot_out).squeeze(-1)
        logits = logits.masked_fill(self._invalid, MASK_VALUE)
        return logits, self.value(hidden)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


POLICIES = {'Policy': Policy, 'Transformer': Transformer}
RNNS = {'Recurrent': Recurrent}
