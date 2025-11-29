# train_peng_benchmark.py

"""
Training skeleton for Peng-style dynamics-randomization benchmark.

- Uses IsaacRandomizedEnv (single env wrapped)
- RecurrentActor / RecurrentCritic
- EpisodeReplayBuffer
- Very rough TD + policy gradient (you'll refine to RDPG + HER).
"""

import numpy as np
import torch
import torch.optim as optim

from tasks.randomized_manipulation_task import RandomizedManipulationTaskCfg
from envs.isaac_randomized_env import IsaacRandomizedEnv
from rl.models import RecurrentActor, RecurrentCritic
from rl.replay_buffer import EpisodeReplayBuffer, Episode


def train():
    # ------------------- config -------------------
    cfg = RandomizedManipulationTaskCfg(
        num_envs=1,    # keep 1 for now; you can extend later
        obs_dim=32,
        act_dim=7,
        goal_dim=2,
        episode_length=100,
    )

    env = IsaacRandomizedEnv(cfg)

    obs_dim = cfg.obs_dim
    act_dim = cfg.act_dim
    dyn_dim = 6     # μ dimension; must match task.dyn_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    num_episodes = 2000
    gamma = 0.99
    batch_size = 8
    lr_actor = 1e-4
    lr_critic = 1e-3
    buffer_capacity = 1000

    actor = RecurrentActor(obs_dim=obs_dim, act_dim=act_dim).to(device)
    critic = RecurrentCritic(obs_dim=obs_dim, act_dim=act_dim, dyn_dim=dyn_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)

    replay_buffer = EpisodeReplayBuffer(capacity=buffer_capacity)

    for ep in range(num_episodes):
        obs, info = env.reset()
        goal = info["goal"]
        dyn = info["dyn_params"]

        obs_list = [obs]
        act_list = []
        rew_list = []
        done_list = []
        goal_list = [goal]

        prev_action = np.zeros(act_dim, dtype=np.float32)

        total_reward = 0.0

        for t in range(cfg.episode_length):
            # ------------- select action -------------
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            prev_act_t = torch.tensor(prev_action, dtype=torch.float32, device=device).view(1, 1, -1)

            hidden_a = actor.init_hidden(batch_size=1, device=device)
            with torch.no_grad():
                mu, std, _ = actor(obs_t, prev_act_t, hidden_a)
                # simple Gaussian sampling
                act = mu[0, 0].cpu().numpy()
                # you can add exploration noise manually here if desired

            next_obs, reward, done, info = env.step(act)

            obs_list.append(next_obs)
            act_list.append(act)
            rew_list.append(reward)
            done_list.append(done)
            goal_list.append(info["goal"])

            prev_action = act
            obs = next_obs
            dyn = info["dyn_params"]
            total_reward += reward

            if done:
                break

        T = len(act_list)
        dyn_arr = np.repeat(dyn[None, :], T, axis=0)  # (T, dyn_dim)

        episode = Episode(
            obs=np.array(obs_list, dtype=np.float32),         # (T+1, obs_dim)
            actions=np.array(act_list, dtype=np.float32),     # (T, act_dim)
            rewards=np.array(rew_list, dtype=np.float32),     # (T,)
            dones=np.array(done_list, dtype=bool),            # (T,)
            goals=np.array(goal_list, dtype=np.float32),      # (T+1, goal_dim)
            dyn_params=dyn_arr.astype(np.float32),            # (T, dyn_dim)
        )
        replay_buffer.add(episode)

        # ---------------- train ----------------
        if len(replay_buffer) >= batch_size:
            episodes = replay_buffer.sample_batch(batch_size)

            # For simplicity assume equal lengths; in practice pad & mask.
            T_max = max(len(ep.actions) for ep in episodes)

            obs_batch = []
            act_batch = []
            rew_batch = []
            done_batch = []
            dyn_batch = []

            for ep_data in episodes:
                ep_T = len(ep_data.actions)
                pad_len = T_max - ep_T

                obs_pad = np.pad(ep_data.obs[:-1], ((0, pad_len), (0, 0)))
                act_pad = np.pad(ep_data.actions, ((0, pad_len), (0, 0)))
                rew_pad = np.pad(ep_data.rewards, (0, pad_len))
                done_pad = np.pad(ep_data.dones.astype(np.float32), (0, pad_len))
                dyn_pad = np.pad(ep_data.dyn_params, ((0, pad_len), (0, 0)))

                obs_batch.append(obs_pad)
                act_batch.append(act_pad)
                rew_batch.append(rew_pad)
                done_batch.append(done_pad)
                dyn_batch.append(dyn_pad)

            obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)   # (B, T, obs_dim)
            act_batch = torch.tensor(act_batch, dtype=torch.float32, device=device)   # (B, T, act_dim)
            rew_batch = torch.tensor(rew_batch, dtype=torch.float32, device=device)   # (B, T)
            done_batch = torch.tensor(done_batch, dtype=torch.float32, device=device) # (B, T)
            dyn_batch = torch.tensor(dyn_batch, dtype=torch.float32, device=device)   # (B, T, dyn_dim)

            B, T_b, _ = obs_batch.shape

            # -------- critic update (rough TD target) --------
            critic_opt.zero_grad()

            hidden_c = critic.init_hidden(batch_size=B, device=device)
            q_pred, _ = critic(obs_batch, act_batch, dyn_batch, hidden_c)  # (B, T, 1)
            q_pred = q_pred.squeeze(-1)  # (B, T)

            with torch.no_grad():
                G = torch.zeros_like(rew_batch)
                G[:, -1] = rew_batch[:, -1]
                for t in reversed(range(T_b - 1)):
                    G[:, t] = rew_batch[:, t] + (1.0 - done_batch[:, t]) * gamma * G[:, t + 1]

            critic_loss = ((q_pred - G) ** 2).mean()
            critic_loss.backward()
            critic_opt.step()

            # -------- actor update (maximize Q) --------
            actor_opt.zero_grad()

            prev_act_batch = torch.zeros_like(act_batch)  # you can shift actions if you want true a_{t-1}
            hidden_a = actor.init_hidden(batch_size=B, device=device)
            mu, std, _ = actor(obs_batch, prev_act_batch, hidden_a)

            q_for_actor, _ = critic(obs_batch, mu, dyn_batch,
                                    critic.init_hidden(batch_size=B, device=device))
            q_for_actor = q_for_actor.squeeze(-1)
            actor_loss = -q_for_actor.mean()
            actor_loss.backward()
            actor_opt.step()

        # ---------------- logging ----------------
        if (ep + 1) % 10 == 0:
            print(f"[Episode {ep+1}] return = {total_reward:.2f}, steps = {T}")

    print("Training finished.")


if __name__ == "__main__":
    train()
