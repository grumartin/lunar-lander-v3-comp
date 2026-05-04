import os
import numpy as np
import torch

from rl_model import RLModel


class Agent:
    def __init__(self, env=None, player_name=None):
        path = "model.pt"
        self.model = RLModel(obs_dim=8, num_actions=4)
        if os.path.exists(path):
            self.model.load(path)
        self.model.eval()

    @torch.inference_mode()
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False,
                      info=None, action_mask=None):
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32))[None, :]
        logits, values = self.model(obs)

        ## TODO:
        # Compute the action index using the model forward function:
        # action_index = ...
        raise NotImplementedError()

        return action_index
