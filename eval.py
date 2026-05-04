import argparse
import numpy as np

from env import make_env
from agent import Agent


def run_episode(env, agent):
    obs, _ = env.reset()
    total = 0.0
    reward, terminated, truncated, info = 0.0, False, False, {}
    while True:
        action = agent.choose_action(obs, reward=reward, terminated=terminated,
                                     truncated=truncated, info=info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--no_render", action="store_true")
    args = p.parse_args()

    env = make_env(render_mode="human" if not args.no_render else None)
    agent = Agent(env)

    returns = [run_episode(env, agent) for _ in range(args.n_episodes)]
    env.close()

    for i, r in enumerate(returns):
        print(f"episode {i:2d}: {r:7.2f}")
    print(f"mean over {args.n_episodes} runs: {np.mean(returns):.2f} +- {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
