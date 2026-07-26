"""Transparent approximate compute accounting for the Q1 comparison."""
from __future__ import annotations

from dijkwalk.env import F, K, N_GLOBAL

RL_ARCHITECTURES = {
    'lstm': {
        'hidden_size': 128,
        'paired_supervised_size': 'small',
    },
    'transformer': {
        'hidden_size': 128,
        'layers': 4,
        'heads': 4,
        'ff': 256,
        'paired_supervised_size': 'large',
    },
}


def rl_forward_macs_per_transition(architecture):
    """Approximate one policy forward pass in multiply-accumulates."""
    spec = RL_ARCHITECTURES[architecture]
    hidden = spec['hidden_size']
    if architecture == 'lstm':
        feedforward = (
            K * F * 32
            + (K * 32 + N_GLOBAL) * hidden
            + hidden * hidden
            + hidden * K
            + hidden
        )
        recurrent = 4 * (
            hidden * hidden
            + hidden * hidden
        )
        return feedforward + recurrent

    tokens = K + 1
    projections = K * F * hidden + N_GLOBAL * hidden
    per_layer = (
        4 * tokens * hidden**2
        + 2 * tokens**2 * hidden
        + 2 * tokens * hidden * spec['ff']
    )
    heads = K * hidden + hidden
    return projections + spec['layers'] * per_layer + heads


def estimate_rl_train_macs(architecture, transitions):
    """Approximate rollout plus one-epoch PPO training compute.

    Each transition has one rollout forward pass. Training adds one forward
    and an approximately two-forward-cost backward pass, for four forward
    equivalents total. Optimizer and environment work are excluded.
    """
    return 4 * transitions * rl_forward_macs_per_transition(architecture)
