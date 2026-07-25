#!/usr/bin/env bash
# Experiments from PLAN-pufferlib.md. Pass extra flags through, e.g.:
#   ./run_ablations.sh --train.device cuda --train.total-timesteps 50000000 --wandb
set -euo pipefail
PY=${PY:-.venv/bin/python}
EXTRA=("$@")

# 1. Shaping ablation (headline) — LSTM policy throughout
$PY train.py train --rnn-name Recurrent --tag shaping-beta1 --env.beta 1.0 "${EXTRA[@]}"
$PY train.py train --rnn-name Recurrent --tag shaping-sparse --env.beta 0.0 "${EXTRA[@]}"
$PY train.py train --rnn-name Recurrent --tag shaping-binary --env.binary-penalty True --env.penalty 1.0 "${EXTRA[@]}"

# 2. Policy ablation — beta=1 throughout
$PY train.py train --tag policy-mlp "${EXTRA[@]}"
$PY train.py train --policy-name Transformer --tag policy-transformer "${EXTRA[@]}"
# (policy-lstm == shaping-beta1 above)
