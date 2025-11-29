# envs/isaac_randomized_env.py

"""
Gym-style wrapper around RandomizedManipulationTask.

For training simplicity, we expose a SINGLE env API:
- reset() -> obs, info
- step(action) -> obs, reward, done, info

Internally, we just use env index 0 of the Isaac-style task.
"""

import numpy as np
import torch

from tasks.randomized_manipulation_task import RandomizedManipulationTask, RandomizedManipulationTaskCfg


class IsaacRandomizedEnv:
    def __init__(self, cfg: RandomizedManipulationTaskCfg):
        assert cfg.num_envs >= 1, "Use at least 1 env."
        self.cfg = cfg
        self.task = RandomizedManipulationTask(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = cfg.obs_dim
        self.act_dim = cfg.act_dim

        # TODO: if you have a global Isaac Lab simulation object, keep a handle here.
        # self.sim = ...

    def reset(self):
        """
        Reset env 0. Returns:
          obs: (obs_dim,)
          info: dict with 'goal' and 'dyn_params'
        """
        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        self.task._reset_idx(env_ids)
        self.task._update_obs()

        obs_all = self.task.get_observations()["obs"]  # (num_envs, obs_dim)
        goals_all = self.task.get_goals()
        dyn_all = self.task.get_dynamics_params()

        obs = obs_all[0].cpu().numpy()
        info = {
            "goal": goals_all[0].cpu().numpy(),
            "dyn_params": dyn_all[0].cpu().numpy(),
        }
        return obs, info

    def step(self, action: np.ndarray):
        """
        Step env 0 with given action: (act_dim,).
        Returns:
          obs: (obs_dim,)
          reward: float
          done: bool
          info: dict
        """
        assert action.shape == (self.act_dim,)
        action_torch = torch.tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, act_dim)

        # Pre-physics
        self.task.pre_physics_step(action_torch)

        # TODO: integrate physics in Isaac Lab
        # if you have a sim.step() or similar, call it here.
        # self.sim.step()

        # Post-physics
        self.task.post_physics_step()

        obs_all = self.task.get_observations()["obs"]
        rew_all = self.task.get_rewards()
        reset_all = self.task.get_resets()
        goals_all = self.task.get_goals()
        dyn_all = self.task.get_dynamics_params()
        success_all = self.task.success_buf

        obs = obs_all[0].cpu().numpy()
        reward = float(rew_all[0].cpu().item())
        done = bool(reset_all[0].cpu().item())
        info = {
            "goal": goals_all[0].cpu().numpy(),
            "dyn_params": dyn_all[0].cpu().numpy(),
            "success": bool(success_all[0].cpu().item()),
        }
        return obs, reward, done, info
