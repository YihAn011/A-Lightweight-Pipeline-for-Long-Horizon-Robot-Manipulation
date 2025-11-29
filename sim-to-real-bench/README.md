In tasks/randomized_manipulation_task.py
1. _setup_scene()

(Optional)

Load robot articulation

Load object (puck/block)

Store handles for robot & object

2. _apply_dynamics(env_ids, dyn)

Must implement:

Scale link masses

Scale joint damping

Scale surface friction

Scale controller gains

Adjust dt / action repeat if supported

Save noise level (obs_noise_std) for _update_obs()

3. _update_obs()

Must implement:

Read robot joint positions/velocities from Isaac

Read object pose (pos + maybe orientation)

Build a 1D observation vector

Add Gaussian observation noise

Optionally append goal to the observation

4. _achieved_goal_from_obs(obs)

Must implement:

Extract object position from obs
(e.g., last 2 or 3 dims)

5. _reset_idx(env_ids)

Must implement:

Reset robot state in Isaac

Reset object pose

Sample a new goal

Sample new dynamics → call _apply_dynamics()

Reset step counters

In envs/isaac_randomized_env.py
1. Replace every # TODO: self.sim.step()

with your actual Isaac Lab call:

self.sim.step()   # or world.step()


This must appear in both:

reset()

step()