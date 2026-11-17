"""Frozen PPO policy as a NumPy forward pass (no torch). Shared by the simulator tests and
the real aioquic controller so both pick actions with exactly the same code.

Deterministic action = argmax of the policy logits (what SB3 predict(deterministic=True) does).
"""
import numpy as np


class NumpyPolicy:
    def __init__(self, weights):
        w = np.load(weights) if isinstance(weights, str) else weights
        self.w1, self.b1 = w["w1"].astype(np.float64), w["b1"].astype(np.float64)
        self.w2, self.b2 = w["w2"].astype(np.float64), w["b2"].astype(np.float64)
        self.w3, self.b3 = w["w3"].astype(np.float64), w["b3"].astype(np.float64)

    def logits(self, state):
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        h = np.tanh(self.w1 @ x + self.b1)
        h = np.tanh(self.w2 @ h + self.b2)
        return self.w3 @ h + self.b3

    def action(self, state):
        return int(np.argmax(self.logits(state)))
