# import numpy as np
# import os
# import gymnasium as gym
# from gymnasium import utils
# from gymnasium.envs.mujoco import MujocoEnv
# from gymnasium.spaces import Box

# DEFAULT_CAMERA_CONFIG = {
#     "distance": 1.5,
#     "azimuth": 70,
#     "elevation": -20,
# }

# class ReachEnv(MujocoEnv, utils.EzPickle):
#     metadata = {
#         "render_modes": [
#             "human",
#             "rgb_array",
#             "depth_array",
#         ],
#         "render_fps": 100,
#     }
    
#     def __init__(self, **kwargs):
#         # Find the path to the XML file
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         parent_dir = os.path.dirname(os.path.dirname(current_dir))
#         self.model_path = os.path.join(parent_dir, "xml", "reach_target_scene_custom.xml")
        
#         # Set up the workspace boundaries for the target
#         self.target_bounds = {
#             'x': (0.0, 0.3),   # Forward/back
#             'y': (-0.2, 0.2),  # Left/right
#             'z': (0.05, 0.3)   # Up/down
#         }
        
#         observation_space = Box(
#             low=-np.inf, 
#             high=np.inf, 
#             shape=(15,),  # 6 joint pos + 6 joint vel + 3 target pos
#             dtype=np.float32
#         )
        
#         # Initialize the MujocoEnv
#         MujocoEnv.__init__(
#             self,
#             model_path=self.model_path,
#             frame_skip=5,
#             observation_space=observation_space,
#             default_camera_config=DEFAULT_CAMERA_CONFIG,
#             **kwargs
#         )
        
#         # The action space is determined by the number of actuators
#         self.action_space = Box(
#             low=-1.0,
#             high=1.0,
#             shape=(self.model.nu,),
#             dtype=np.float32
#         )
        
#         # Initialize EzPickle
#         utils.EzPickle.__init__(self, **kwargs)
        
#         # Find the wrist site for tracking the end effector
#         self.wrist_site_name = None
#         for i in range(self.model.nsite):
#             site_name = self.model.site(i).name
#             if 'wrist_site' in site_name:
#                 self.wrist_site_name = site_name
#                 break
        
#         if self.wrist_site_name is None:
#             # If no wrist site, try to find any suitable site
#             for i in range(self.model.nsite):
#                 site_name = self.model.site(i).name
#                 if 'site' in site_name and 'target' not in site_name:
#                     self.wrist_site_name = site_name
#                     break
    
#     def _get_obs(self):
#         # For the arm positions, use the actual number of DoF from the model
#         qpos = np.zeros(6, dtype=np.float32)
#         qvel = np.zeros(6, dtype=np.float32)
        
#         # Try to get joint states
#         if self.data.qpos.size >= 13:
#             qpos = self.data.qpos.flat.copy()[7:13]
#         elif self.data.qpos.size >= 6:
#             qpos = self.data.qpos.flat.copy()[:6]
            
#         if self.data.qvel.size >= 12:
#             qvel = self.data.qvel.flat.copy()[6:12]
#         elif self.data.qvel.size >= 6:
#             qvel = self.data.qvel.flat.copy()[:6]
        
#         # Get target position
#         target_pos = self.data.body('target').xpos.copy()
        
#         # Combine into observation
#         obs = np.concatenate([qpos, qvel, target_pos]).astype(np.float32)
#         return obs

#     def compute_reward(self, achieved_goal=None, desired_goal=None, info=None):
#         if self.wrist_site_name:
#             hand_pos = self.data.site(self.wrist_site_name).xpos
#         else:
#             # Fallback in case we couldn't find a site
#             hand_pos = self.data.body('so100_Fixed_Jaw').xpos
            
#         target_pos = self.data.body('target').xpos
        
#         # Calculate distance between hand and target
#         distance = np.linalg.norm(hand_pos - target_pos)
        
#         # Reward is negative distance (closer is better)
#         reward = -distance
        
#         # Bonus for being close
#         if distance < 0.05:
#             reward += 1.0
            
#         return reward

#     def _sample_target_position(self):
#         # Sample a random position within the workspace
#         # Use the environment's seeded random generator instead of np.random directly
#         x = self.np_random.uniform(self.target_bounds['x'][0], self.target_bounds['x'][1])
#         y = self.np_random.uniform(self.target_bounds['y'][0], self.target_bounds['y'][1])
#         z = self.np_random.uniform(self.target_bounds['z'][0], self.target_bounds['z'][1])
        
#         return np.array([x, y, z])

#     def reset_model(self):
#         # Reset the qpos and qvel
#         qpos = self.init_qpos
#         qvel = self.init_qvel
        
#         # Set the state
#         self.set_state(qpos, qvel)
        
#         # Sample a new target position with explicit randomness
#         # Reseed the random number generator to ensure variation
#         target_pos = np.array([
#             self.np_random.uniform(low=self.target_bounds['x'][0], high=self.target_bounds['x'][1]),
#             self.np_random.uniform(low=self.target_bounds['y'][0], high=self.target_bounds['y'][1]),
#             self.np_random.uniform(low=self.target_bounds['z'][0], high=self.target_bounds['z'][1])
#         ])
        
#         print(f"New target position: {target_pos}")
        
#         # Get the target body ID
#         target_body_id = self.model.body('target').id
        
#         # Set the target position using mujoco API method
#         self.data.body(target_body_id).xpos[:] = target_pos
        
#         # Forward the model to apply the position changes
#         import mujoco
#         mujoco.mj_forward(self.model, self.data)
        
#         # Ensure the target actually moved by checking the position again
#         actual_pos = self.data.body('target').xpos.copy()
#         print(f"Actual target position: {actual_pos}")
        
#         # Run a few simulation steps to settle
#         for _ in range(5):
#             self.do_simulation(np.zeros(self.model.nu), self.frame_skip)
        
#         # Return observation
#         return self._get_obs()

#     def step(self, action):
#         # Clip action to valid range
#         action = np.clip(action, -1.0, 1.0)
        
#         # Apply action scaling
#         if self.model.nu > 0:
#             ctrl_range = self.model.actuator_ctrlrange
#             bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
#             weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
#             self.data.ctrl[:] = bias + weight * action
        
#         # Advance simulation
#         self.do_simulation(action, self.frame_skip)
        
#         # Get observation
#         obs = self._get_obs()
        
#         # Calculate reward
#         reward = self.compute_reward()
        
#         # Check if done (no termination condition yet)
#         terminated = False
#         truncated = False
        
#         # Additional info
#         info = {}
        
#         # Rendering is handled by MujocoEnv parent class automatically
        
#         return obs, reward, terminated, truncated, info 

#     def reset(self, *, seed=None, options=None):
#         # Explicitly seed and reset the environment
#         if seed is not None:
#             self._np_random, seed = gym.utils.seeding.np_random(seed)
#         else:
#             # Create a new random seed if none provided
#             import time
#             seed = int(time.time() * 1000) % 10000
#             self._np_random, seed = gym.utils.seeding.np_random(seed)
        
#         # Force initialization of np_random
#         self.np_random = self._np_random
        
#         # Call parent reset
#         super().reset(seed=seed, options=options)
        
#         # Call reset_model which now uses self.np_random for randomization
#         obs = self.reset_model()
        
#         # Ensure the renderer is updated - don't close viewers
#         if self.render_mode is not None:
#             self.render()
        
#         return obs, {} 

#     def close(self):
#         if hasattr(self, "_viewers"):
#             for key, viewer in self._viewers.items():
#                 if hasattr(viewer, "close"):
#                     viewer.close()
#             self._viewers = {}
#         return super().close() 



import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 70,
    "elevation": -20,
}

class ReachEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        **kwargs,
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        self.model_path = os.path.join(parent_dir, "xml", "reach_target_scene_custom.xml")
        utils.EzPickle.__init__(
            self,
            **kwargs,
        )

        self.target_bounds = {
            'x': (-0.3, 0.3),  # Left/right 
            'y': (-0.38, 0.2),  # Forward/back
            'z': (0.05, 0.5)   # Up/down
        }
        
        # Observation space: 6 arm joints pos, 6 arm joints vel, 3 target pos
        observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(15,),  # Explicitly 15 based on _get_obs structure
            dtype=np.float32
        )
        
        # Initialize self.target_pos with a placeholder, it will be set in reset_model
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_pos = [.1, -0.2, 0.1]
        
        MujocoEnv.__init__(
            self,
            self.model_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )
                
        # Find the wrist site for tracking the end effector
        self.wrist_site_name = None
        for i in range(self.model.nsite):
            site_name = self.model.site(i).name
            if 'wrist_site' in site_name:
                self.wrist_site_name = site_name
                break
        
        if self.wrist_site_name is None:
            # If no wrist site, try to find any suitable site
            for i in range(self.model.nsite):
                site_name = self.model.site(i).name
                if 'site' in site_name and 'target' not in site_name:
                    self.wrist_site_name = site_name
                    break

    def _get_obs(self):
        # Arm joints are typically the first `nu` elements
        arm_qpos = self.data.qpos.flat[:self.model.nu].copy()
        arm_qvel = self.data.qvel.flat[:self.model.nu].copy()

        # Target position is the first 3 elements of the free joint's qpos state
        target_joint_qpos_idx = self.model.nu
        target_position = self.data.qpos.flat[target_joint_qpos_idx : target_joint_qpos_idx + 3].copy()

        obs = np.concatenate(
            (
                arm_qpos[:6], # Take first 6 even if model.nu is different (unlikely for this arm)
                arm_qvel[:6],
                target_position[:3],
            )
        ).astype(np.float32) # Ensure correct dtype
        # print(f"[_get_obs] obs: {obs}")
        return obs
        
    def compute_reward(self):
        if self.wrist_site_name:
            hand_pos = self.data.site(self.wrist_site_name).xpos
            
        target_pos = self.data.body('target').xpos
        
        # Calculate distance between hand and target
        distance = np.linalg.norm(hand_pos - target_pos)
        # print(f"[compute_reward] distance: {distance}")
        
        # Reward is negative distance (closer is better)
        reward = -distance
        
        # Bonus for being close
        if distance < 0.05:
            reward += 1.0
            
        return reward

    def step(self, action):
        # print(f"[step] action: {action}")
        # print(f"[step] observation before simulation: {self._get_obs()}")
        self.do_simulation(action, self.frame_skip)
        # print(f"[step] observation after simulation: {self._get_obs()}")
        observation = self._get_obs()
        reward = self.compute_reward()
        # print(f"[step] reward: {reward}")
        terminated = False

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, {}

    def reset_model(self):
        # Get the initial qpos and qvel arrays
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Determine the starting index of the target free joint in qpos
        target_joint_qpos_idx = self.model.nu
        target_joint_qvel_idx = self.model.nu

        # Ensure the indices are within bounds
        if target_joint_qpos_idx + 7 > len(qpos) or target_joint_qvel_idx + 6 > len(qvel):
             raise ValueError("Model structure mismatch: Cannot find target joint indices in qpos/qvel.")

        # Randomize the robot arm's joint positions using the actual joint limits from the XML
        # These match the range values from so_arm100.xml
        joint_ranges = [
            (-2.2*0.9, 2.2*0.9),       # Rotation joint (y-axis)
            (-3.14158*0.3, 3.14158*0.3),   # Pitch joint (x-axis)
            (0.0, 3.14158*0.3),    # Elbow joint (x-axis)
            (-2.0*0.3, 1.8*0.3),       # Wrist_Pitch joint (x-axis)
            (-3.14158*0.9, 3.14158*0.9), # Wrist_Roll joint (y-axis)
            (-0.2*0.9, 2.0*0.9),       # Jaw joint (z-axis)
        ]
        
        # Apply random positions for each joint within its limits
        for i in range(min(self.model.nu, len(joint_ranges))):
            low, high = joint_ranges[i]
            qpos[i] = self.np_random.uniform(low=low, high=high)
        
        # # Sample a new target position
        self.target_pos = np.array([0,-0.3,0.1
            # self.np_random.uniform(low=self.target_bounds['x'][0], high=self.target_bounds['x'][1]),
            # self.np_random.uniform(low=self.target_bounds['y'][0], high=self.target_bounds['y'][1]),
            # self.np_random.uniform(low=self.target_bounds['z'][0], high=self.target_bounds['z'][1])
        ])

        # Set the position part of the free joint in qpos
        qpos[target_joint_qpos_idx : target_joint_qpos_idx + 3] = self.target_pos
        # Set orientation quaternion to identity (1, 0, 0, 0)
        qpos[target_joint_qpos_idx + 3 : target_joint_qpos_idx + 7] = [1.0, 0.0, 0.0, 0.0]

        # Set the velocity part of the free joint in qvel to zero
        qvel[target_joint_qvel_idx : target_joint_qvel_idx + 6] = 0.0

        # Set the full state
        self.set_state(qpos, qvel)

        # Return the observation based on the new state
        observation = self._get_obs()
        return observation
