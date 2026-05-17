# RL Cliff Walking — Q-Learning vs SARSA

Implements and compares two reinforcement learning algorithms — Q-Learning and SARSA — on the OpenAI Gymnasium CliffWalking-v0 environment.

## Algorithms

**Q-Learning** (off-policy) — updates Q-values using the maximum future reward regardless of the actual next action taken, leading to more aggressive optimal-path discovery.

**SARSA** (on-policy) — updates Q-values based on the action actually taken next, making it more conservative and safer near the cliff edge.

## Key Parameters

| Parameter | Value |
|---|---|
| Episodes | 30,000 |
| Learning Rate | 0.1 |
| Discount Factor | 0.99 |
| Epsilon Decay | 0.999 |

## Setup

```bash
pip install gymnasium numpy
python Q_learning.py
python SARSA.py
```

Trained Q-tables are saved as `.pkl` files and can be loaded for evaluation without retraining.
