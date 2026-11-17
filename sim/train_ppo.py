"""Phase 5: train PPO in the simulator (never on the real link).

    python sim/train_ppo.py --steps 400000 --name ppo_v1
    python sim/train_ppo.py --steps 400000 --name ppo_v2 --weights 1,0.5,2 --seed 1

Saves models/<name>.zip, results/train_curve_<name>.csv and .png.
Every episode draws new link conditions (sim/link_sim.py: TRAIN), so the policy cannot
memorize one network. The curve shows mean reward PER STEP; reference lines are the
measured scores of simple policies from tests/audit_sim.py (oracle, fixed40, random).
A good-looking curve is not enough: run sim/evaluate_policy.py and READ the behavior plot.
"""
import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from sim.env import DEFAULT_WEIGHTS, CCEnv  # noqa: E402

REFERENCES = {"oracle (cwnd = BDP)": 0.97, "fixed 40 packets": 0.51, "random actions": 0.27}


class Curve(BaseCallback):
    def __init__(self, steps_per_episode):
        super().__init__()
        self.n, self.rows = steps_per_episode, []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf:
            self.rows.append((self.model.num_timesteps, float(np.mean([e["r"] for e in buf])) / self.n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--name", default="ppo_v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--weights", default=None, help="reward weights a,b,c (default 1,0.5,2)")
    p.add_argument("--rtt-noise", default=None,
                   help="lo,hi in ms: each episode draws a constant RTT offset in this range that "
                        "corrupts the observed RTT (sim/noise.py). Policy v2 used 0,4. Default: clean.")
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--ent-coef", type=float, default=0.01)
    args = p.parse_args()
    torch.set_num_threads(1)  # tiny network: extra threads only thrash (3x slower when two runs share a CPU)
    w = tuple(float(x) for x in args.weights.split(",")) if args.weights else DEFAULT_WEIGHTS

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    noise = tuple(float(x) for x in args.rtt_noise.split(",")) if args.rtt_noise else None
    env = make_vec_env(CCEnv, n_envs=args.n_envs, seed=args.seed,
                       env_kwargs=dict(weights=w, rtt_noise_ms=noise))
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=256, n_epochs=10, gamma=args.gamma,
                learning_rate=3e-4, ent_coef=args.ent_coef, policy_kwargs=dict(net_arch=[64, 64]),
                seed=args.seed, verbose=0, device="cpu")
    cb = Curve(CCEnv().max_steps)
    t0 = time.time()
    print(f"training {args.name}: {args.steps} steps, weights={w}, gamma={args.gamma}, "
          f"ent_coef={args.ent_coef}, seed={args.seed}, rtt_noise_ms={noise}", flush=True)
    model.learn(total_timesteps=args.steps, callback=cb)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    path = os.path.join(ROOT, "models", args.name)
    model.save(path)

    rows = np.asarray(cb.rows)
    np.savetxt(os.path.join(ROOT, "results", f"train_curve_{args.name}.csv"), rows, delimiter=",",
               header="timesteps,mean_reward_per_step", comments="")
    k = max(1, len(rows) // 10)
    print("curve (timesteps: reward/step): " + "  ".join(
        f"{int(rows[i, 0])}: {rows[i, 1]:.2f}" for i in range(0, len(rows), k)), flush=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#fcfcfb")
    ax.plot(rows[:, 0], rows[:, 1], color="#1baf7a", linewidth=1.8, label="PPO (learned)")
    for (name, y), ls in zip(REFERENCES.items(), (":", "--", "-.")):
        ax.axhline(y, color="#52514e", linestyle=ls, linewidth=1, label=name)
    ax.set_xlabel("training steps (intervals)")
    ax.set_ylabel("mean reward per step")
    ax.set_title(f"Training curve: {args.name}", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e4e3df")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", f"train_curve_{args.name}.png"), dpi=150)


if __name__ == "__main__":
    main()
