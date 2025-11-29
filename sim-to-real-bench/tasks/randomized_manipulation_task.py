# tasks/randomized_manipulation_task.py

"""
Isaac-style task skeleton for dynamics-randomization benchmark
(loosely inspired by Peng et al. "Sim-to-Real with Dynamics Randomization").

This is NOT using real Isaac Lab APIs yet — it's a structure you will:
- plug into Isaac Lab task system, OR
- just keep as a stand-alone logic layer that calls Isaac Lab from TODOs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class RandomizedManipulationTaskCfg:
    num_envs: int = 1               # start with 1 for simplicity
    episode_length: int = 100
    obs_dim: int = 32               # robot state + object state (+ maybe goal)
    act_dim: int = 7                # robot action dimension
    goal_dim: int = 2               # e.g., puck target position (x, y)

    # Domain randomization ranges
    link_mass_scale_range: tuple = (0.5, 1.5)
    joint_damping_scale_range: tuple = (0.5, 2.0)
    friction_scale_range: tuple = (0.5, 1.5)
    controller_gain_scale_range: tuple = (0.5, 2.0)
    dt_scale_range: tuple = (0.8, 1.2)
    obs_noise_std_range: tuple = (0.0, 0.01)


class RandomizedManipulationTask:
    """
    Task logic:
    - keeps per-env state, obs, rewards, resets
    - handles dynamics randomization
    - defines obs, reward, success criteria
    """

    def __init__(self, cfg: RandomizedManipulationTaskCfg):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Per-env counters and buffers
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, cfg.obs_dim, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Goals and dynamics params
        self.goals = torch.zeros(self.num_envs, cfg.goal_dim, device=self.device)
        # μ = [mass_scale, damping_scale, friction_scale, gain_scale, dt_scale, obs_noise_std]
        self.dyn_params = torch.zeros(self.num_envs, 6, device=self.device)

        # TODO: hook actual Isaac Lab scene/robot/object here
        # self._setup_scene()

    # ------------------------------------------------------------------
    # Simulation step hooks (conceptually like Isaac Lab RLTask)
    # ------------------------------------------------------------------

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Called before physics integration.
        actions: (num_envs, act_dim)
        """
        # TODO: send actions to robot in Isaac Lab, e.g. joint targets or torques
        # Example:
        # self.robot.apply_action(actions)
        pass

    def post_physics_step(self) -> None:
        """
        Called after physics integration.
        Updates obs, rewards, reset flags, etc.
        """
        self.episode_length_buf += 1

        self._update_obs()
        self._update_rew_and_done()

        # Reset selected envs
        env_ids = torch.nonzero(self.reset_buf, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._reset_idx(env_ids)

    # ------------------------------------------------------------------
    # Reset & randomization
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Reset selected envs, sample new goals, randomize dynamics.
        """
        # Reset counters
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.success_buf[env_ids] = False

        # New goals
        self.goals[env_ids] = self._sample_goals(num=env_ids.numel())

        # Randomized dynamics
        dyn = self._sample_dynamics(num=env_ids.numel())
        self._apply_dynamics(env_ids, dyn)
        self.dyn_params[env_ids] = dyn

        # TODO: reset robot and object states for these envs in Isaac Lab
        # Example:
        # self.robot.reset(env_ids)
        # self.object.reset(env_ids)
        pass

    def _sample_goals(self, num: int) -> torch.Tensor:
        """
        Sample random goal positions (e.g., target puck positions).
        """
        low = torch.tensor([-0.1, -0.1], device=self.device)
        high = torch.tensor([0.1, 0.1], device=self.device)
        return low + (high - low) * torch.rand(num, self.cfg.goal_dim, device=self.device)

    def _sample_dynamics(self, num: int) -> torch.Tensor:
        """
        Sample μ for each env: [mass_scale, damping_scale, friction_scale,
                                gain_scale, dt_scale, obs_noise_std]
        """
        cfg = self.cfg
        device = self.device

        def sample_range(lo_hi):
            lo, hi = lo_hi
            return lo + (hi - lo) * torch.rand(num, 1, device=device)

        mass_scale = sample_range(cfg.link_mass_scale_range)
        damp_scale = sample_range(cfg.joint_damping_scale_range)
        fric_scale = sample_range(cfg.friction_scale_range)
        gain_scale = sample_range(cfg.controller_gain_scale_range)
        dt_scale = sample_range(cfg.dt_scale_range)
        noise_std = sample_range(cfg.obs_noise_std_range)

        dyn = torch.cat([mass_scale, damp_scale, fric_scale,
                         gain_scale, dt_scale, noise_std], dim=-1)
        return dyn  # (num, 6)

    def _apply_dynamics(self, env_ids: torch.Tensor, dyn: torch.Tensor) -> None:
        """
        Apply dynamics params to Isaac Lab simulation for selected envs.
        dyn: (num_envs_to_reset, 6)
        """
        # TODO: implement mapping to Isaac Lab:
        # mass_scale = dyn[:, 0]
        # damp_scale = dyn[:, 1]
        # fric_scale = dyn[:, 2]
        # gain_scale = dyn[:, 3]
        # dt_scale = dyn[:, 4]
        # obs_noise_std = dyn[:, 5]
        #
        # Apply to robot/articulation/physics for env_ids.
        pass

    # ------------------------------------------------------------------
    # Observations & rewards
    # ------------------------------------------------------------------

    def _update_obs(self) -> None:
        """
        Build observation vector from Isaac Lab state + add noise.
        """
        # TODO: read from Isaac:
        # joint positions, joint velocities, end-effector pose,
        # object pose, etc.
        # For now, placeholder zeros with right shape:
        joint_state = torch.zeros(self.num_envs, 14, device=self.device)  # q & dq example
        obj_state = torch.zeros(self.num_envs, 6, device=self.device)     # pos + vel example

        base_obs = torch.cat([joint_state, obj_state], dim=-1)            # (N, 20) example
        assert base_obs.shape[1] <= self.cfg.obs_dim

        self.obs_buf[:, :base_obs.shape[1]] = base_obs

        # Option: include goals inside obs
        # self.obs_buf[:, -self.cfg.goal_dim:] = self.goals

        # Add observation noise
        noise_std = self.dyn_params[:, -1]  # obs_noise_std
        noise = torch.randn_like(self.obs_buf) * noise_std.unsqueeze(-1)
        self.obs_buf += noise

    def _update_rew_and_done(self) -> None:
        """
        Sparse reward: 0 if success, -1 otherwise (as in Peng).
        """
        achieved = self._achieved_goal_from_obs(self.obs_buf)  # (N, goal_dim)
        goal = self.goals

        dist = torch.norm(achieved - goal, dim=-1)
        success = dist < 0.02

        self.rew_buf[:] = -1.0
        self.rew_buf[success] = 0.0

        timeout = self.episode_length_buf >= self.cfg.episode_length

        self.success_buf = success
        self.reset_buf = torch.logical_or(success, timeout)

    def _achieved_goal_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract achieved goal from obs, e.g. puck position.
        For now, assume last 2 dims are object (x, y).
        """
        return obs[:, -2:]

    # ------------------------------------------------------------------
    # Accessors used by the wrapper / trainer
    # ------------------------------------------------------------------

    def get_observations(self) -> Dict[str, torch.Tensor]:
        return {"obs": self.obs_buf}

    def get_rewards(self) -> torch.Tensor:
        return self.rew_buf

    def get_resets(self) -> torch.Tensor:
        return self.reset_buf

    def get_dynamics_params(self) -> torch.Tensor:
        return self.dyn_params

    def get_goals(self) -> torch.Tensor:
        return self.goals
