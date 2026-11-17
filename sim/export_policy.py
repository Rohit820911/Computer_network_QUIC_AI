"""Export a trained PPO policy to plain numpy so the real aioquic controller needs no torch.

    python sim/export_policy.py models/ppo_v1          # writes models/ppo_v1.npz

Only the deterministic actor is exported: Linear(16,64) tanh Linear(64,64) tanh Linear(64,5),
action = argmax of the 5 outputs (exactly what model.predict(..., deterministic=True) does).
Self-test: 5000 random states must give the same action in numpy as in Stable Baselines3.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stable_baselines3 import PPO  # noqa: E402

from common.features import STATE_DIM  # noqa: E402


def numpy_logits(w, x):
    h = np.tanh(w["W0"] @ x + w["b0"])
    h = np.tanh(w["W1"] @ h + w["b1"])
    return w["Wa"] @ h + w["ba"]


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_v1"
    base = base if os.path.isabs(base) else os.path.join(ROOT, base)
    base = base[:-4] if base.endswith(".zip") else base
    model = PPO.load(base, device="cpu")
    pol = model.policy
    net = pol.mlp_extractor.policy_net
    lin = [m for m in net if m.__class__.__name__ == "Linear"]
    assert len(lin) == 2, f"expected 2 hidden layers, got {len(lin)}"
    w = {"W0": lin[0].weight, "b0": lin[0].bias, "W1": lin[1].weight, "b1": lin[1].bias,
         "Wa": pol.action_net.weight, "ba": pol.action_net.bias}
    w = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in w.items()}
    assert w["W0"].shape[1] == STATE_DIM, "state size differs from common/features.py"
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(5000, STATE_DIM)).astype(np.float32)
    ref, _ = model.predict(X, deterministic=True)
    mine = np.array([int(np.argmax(numpy_logits(w, x.astype(np.float64)))) for x in X])
    agree = float(np.mean(ref == mine))
    print(f"self-test: numpy and Stable Baselines3 choose the same action on {agree * 100:.2f}% of 5000 states")
    if agree < 0.999:
        sys.exit("export FAILED the self-test; not writing the file")
    np.savez(base + ".npz", **w)
    print("wrote", base + ".npz")


if __name__ == "__main__":
    main()
