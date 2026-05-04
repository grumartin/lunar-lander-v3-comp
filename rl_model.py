import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class RLModel(nn.Module):
    """Minimal actor-critic model for discrete actions.

    Two 1-hidden-layer MLPs: a policy head producing logits and a value head
    producing V(s). No target network, no replay buffer, no observation stacking,
    no RNN state. Trained with n-step A2C using `loss_a2c`.
    """

    def __init__(self, obs_dim, num_actions, hidden=64, reward_scale=10.0, gamma=0.99):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.reward_scale = reward_scale
        self.gamma = gamma

        self.policy = mlp(obs_dim, num_actions, hidden)
        self.value = mlp(obs_dim, 1, hidden)

    def forward(self, observations):
        """observations: tensor of shape (..., obs_dim).
        Returns logits (..., num_actions) and values (...)."""
        logits = self.policy(observations)
        values = self.value(observations).squeeze(-1)
        return logits, values

    @torch.inference_mode()
    def collect_rollouts(self, env_list, num_steps):
        """Roll out B = len(env_list) envs for num_steps each, auto-resetting on done.

        Continues from where the previous call left off (persistent last-obs
        stored on the model). Action selection samples from the Categorical
        policy; if epsilon > 0, overrides with a uniform-random action
        independently per env.

        Returns a dict of numpy arrays shaped (B, T, ...):
            observations: (B, T, obs_dim) — obs fed to the policy at step t
            actions:      (B, T)          — action taken at step t
            rewards:      (B, T)          — reward received for that action
            is_done:      (B, T)          — 1.0 if the episode ended at step t
            logits:       (B, T, num_actions) — policy logits at step t
            last_obs:     (B, obs_dim)    — obs after the last step, for
                                            bootstrapping the n-step return
        """
        B = len(env_list)
        T = num_steps
        device = next(self.parameters()).device

        if not hasattr(self, "_rollout_last_obs") or self._rollout_last_obs.shape[0] != B:
            self._rollout_last_obs = np.stack(
                [env.reset()[0] for env in env_list]
            ).astype(np.float32)

        obs_buf = np.zeros((B, T, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((B, T), dtype=np.int64)
        rew_buf = np.zeros((B, T), dtype=np.float32)
        done_buf = np.zeros((B, T), dtype=np.float32)
        logit_buf = np.zeros((B, T, self.num_actions), dtype=np.float32)

        was_training = self.training
        self.eval()

        for t in range(T):
            obs_t = self._rollout_last_obs
            logits, _ = self.forward(torch.from_numpy(obs_t).to(device))

            ## TODO:
            # Compute the action index using the model forward function:
            # actions = ...
            raise NotImplementedError("")

            next_obs = np.zeros_like(obs_t)
            for i, env in enumerate(env_list):
                o, r, term, trunc, _ = env.step(int(actions[i]))
                done = bool(term or trunc)
                if done:
                    o, _ = env.reset()
                next_obs[i] = o
                rew_buf[i, t] = r
                done_buf[i, t] = float(done)

            obs_buf[:, t] = obs_t
            act_buf[:, t] = actions
            logit_buf[:, t] = logits.cpu().numpy()
            self._rollout_last_obs = next_obs

        if was_training:
            self.train()

        return {
            "observations": obs_buf,
            "actions": act_buf,
            "rewards": rew_buf,
            "is_done": done_buf,
            "logits": logit_buf,
            "last_obs": self._rollout_last_obs.copy(),
        }

    def loss_a2c(self, rollout, entropy_coef=0.01, value_coef=0.5):
        """Compute A2C loss from a rollout dict.

        Uses n-step returns bootstrapped with V(last_obs).
        Returns a dict with the scalar `loss` tensor and detached diagnostics.
        """
        device = next(self.parameters()).device
        obs = torch.from_numpy(rollout["observations"]).to(device)
        actions = torch.from_numpy(rollout["actions"]).long().to(device)
        rewards = torch.from_numpy(rollout["rewards"]).to(device) / self.reward_scale
        is_done = torch.from_numpy(rollout["is_done"]).to(device)
        last_obs = torch.from_numpy(rollout["last_obs"]).to(device)

        B, T = actions.shape

        logits, values = self.forward(obs)  # (B, T, A), (B, T)

        with torch.no_grad():
            _, last_value = self.forward(last_obs)  # (B,)

        ## TODO:
        # 1) Compute the future returns with discount factor gamma
        # Warning: The episode ends when is_done is True
        # returns = ...
        #
        # 2) We will implement the A2C actor critic algorithm.
        # Compute the following losses
        # value_loss = ...
        # policy_loss = ...
        #
        # 3) compute the term of entropy regularizationo of the action distribution:
        # entropy = ...
        raise NotImplementedError("")

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "mean_return": returns.mean().detach(),
        }

    def save(self, path=None):
        if path is None:
            path = "model.pt"
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        if path is None:
            path = "model.pt"
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
