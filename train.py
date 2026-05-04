import argparse
import numpy as np
import torch

from env import make_env
from rl_model import RLModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--n_iterations", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--reward_scale", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--save", default="model.pt")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = [make_env() for _ in range(args.n_envs)]

    model = RLModel(obs_dim=8, num_actions=4, hidden=args.hidden,
                    reward_scale=args.reward_scale, gamma=args.gamma)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ep_return = np.zeros(args.n_envs, dtype=np.float32)
    completed = []

    for it in range(args.n_iterations):
        rollout = model.collect_rollouts(envs, args.n_steps)

        for t in range(args.n_steps):
            ep_return += rollout["rewards"][:, t]
            for i in range(args.n_envs):
                if rollout["is_done"][i, t]:
                    completed.append(float(ep_return[i]))
                    ep_return[i] = 0.0

        model.train()
        optimizer.zero_grad()
        losses = model.loss_a2c(rollout, entropy_coef=args.entropy_coef, value_coef=args.value_coef)
        losses["loss"].backward()
        optimizer.step()

        if it % args.log_every == 0:
            recent = completed[-50:] if completed else [0.0]
            print(f"iter {it:5d}  loss {losses['loss'].item():7.3f}  "
                  f"pi {losses['policy_loss'].item():6.3f}  v {losses['value_loss'].item():6.3f}  "
                  f"H {losses['entropy'].item():5.3f}  "
                  f"return {np.mean(recent):7.2f} (last {len(recent)} eps)", flush=True)

        if it > 0 and it % args.save_every == 0:
            model.save(args.save)

    model.save(args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
