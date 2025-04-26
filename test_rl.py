import numpy as np
from helper_burr_piece_creator import create_floor, define_all_burr_pieces, FLOOR_INDEX_OFFSET
from helper_geometric_functions import get_feasible_motions, get_unsupported_pids, apply_mate
from helper_file_mgmt import load_mates_list
from helper_display import display_pieces
from datetime import datetime
import gymnasium as gym
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

NUM_PIECES = 6
TARGET_OFFSETS = np.array([[0,0,4],[-1,0,3],[1,0,3],[0,1,3],[0,-1,3],[0,0,2]])
START_OFFSETS = np.array([[4,-8,1],[-8,0,1],[8,0,1],[-4,8,3],[-4,-8,3],[4,8,1]])

class BurrPuzzleEnv(gym.Env):
    def __init__(self, mates_list):
        super().__init__()

        self.mates_list = mates_list
        self.action_space = gym.spaces.Discrete(len(self.mates_list))
        self.max_held = 2
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_PIECES*3,), dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(len(self.mates_list),), dtype=np.uint8)
        })

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.floor = create_floor()
        self.pieces = define_all_burr_pieces(TARGET_OFFSETS)
        self.pieces_augmented = self.pieces + [self.floor]
        self.active_pids = [FLOOR_INDEX_OFFSET]
        self.timestep = 0
        obs = self._get_obs()
        return (obs, {})

    def step(self, action):
        prev_cost = cost_function(self.pieces, START_OFFSETS)
        mate = self.mates_list[action]
        self.pieces_augmented = apply_mate(self.pieces_augmented, mate)
        moved_id = mate[0][0]
        if moved_id not in self.active_pids:
            self.active_pids.append(moved_id)

        new_cost = cost_function(self.pieces, START_OFFSETS)
        reward = prev_cost - new_cost
        reward -= 0.01 # per time step
        if new_cost < 0.1:
            reward += 100 # Completion reward
        if np.mod(self.timestep, 10) == 0:
            print(f"Received Reward {reward:.2f} at timestep {self.timestep}")
        self.timestep += 1
        done = (new_cost < 0.1) or self.timestep > 100
        obs = self._get_obs()
        return obs, reward, done, False, {}
    
    def _get_obs(self):
        piece_states = np.array([
            piece['mesh'].bounding_box.centroid for piece in self.pieces
        ])
        return {
            "observation": piece_states.flatten(),
            "action_mask": self._get_action_mask()
        }
    
    def _get_action_mask(self):
        # First get a headcount of who's unsupported, except the floor
        pieces_not_floor = [piece for piece in self.pieces if piece['id'] != FLOOR_INDEX_OFFSET]
        unsupported_ids = get_unsupported_pids(pieces_not_floor)
        # If we're at the limit, only these unsupported pieces can get moved. Otherwise, carry on.
        movable_ids = [i for i in range(NUM_PIECES)]
        if len(unsupported_ids) >= self.max_held:
            movable_ids = unsupported_ids
        
        active_pieces = [piece for piece in self.pieces_augmented if piece['id'] in self.active_pids]
        valid_mates_set = set()
        for pid in movable_ids:
            this_piece = next(p for p in self.pieces if p['id'] == pid)
            feasible = get_feasible_motions(this_piece, active_pieces, mates_list=self.mates_list, steps=20, check_path=True)
            for (p1c1, p2c2, _) in feasible:
                valid_mates_set.add((p1c1, p2c2))

        # Create binary mask
        mask = np.zeros(len(self.mates_list), dtype=np.int8)
        for i, mate in enumerate(self.mates_list):
            if mate in valid_mates_set:
                mask[i] = 1
        return mask

def cost_function(pieces, target_offsets):
    offsets = [piece['mesh'].bounding_box.centroid for piece in pieces]
    diff = offsets - target_offsets
    dist = [np.linalg.norm(dif) for dif in diff]
    return sum(dist)

def mask_fn(env: gym.Env):
    return env._get_action_mask()

def env_train(folder):
    mates_list = load_mates_list()
    def make_env():
        env = BurrPuzzleEnv(mates_list)
        env = ActionMasker(env, mask_fn)
        return env
    # env = make_env()
    vec_env = SubprocVecEnv([make_env for _ in range(24)])
    model = MaskablePPO('MultiInputPolicy', vec_env, verbose=1,tensorboard_log="./ppo_burr_tensorboard/")
    model.learn(total_timesteps=1000)
    model.save("puzzle_agent")
    return model

def env_test(model, folder):
    mates_list = load_mates_list()
    test_env = BurrPuzzleEnv(mates_list)
    obs, _ = test_env.reset()

    done = False
    total_reward = 0
    step_count = 0
    while not done:
        # MaskablePPO still expects {"observation": ..., "action_mask": ...}
        action, _states = model.predict(obs, deterministic=True)
        
        # Step with the predicted action
        obs, reward, done, _, info = test_env.step(action)
        flattened_pos = obs['observation']
        piece_positions = np.reshape(flattened_pos, (6, 3))
        # display_pieces(step_count, offsets=piece_positions, folder=folder)
        # save_simulation_state(step_count, pieces_augmented, active_pids, [], folder=folder)
        step_count += 1
        total_reward += reward

    print("Test episode reward:", total_reward)

if __name__=="__main__":
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = f"results/{timestamp}"
    # folder = f"results/2025-04-24_12-45-55" # Existing folder
    folder_sim = f"{folder}/states"
    folder_img = f"{folder}/frames"
    model = env_train(folder_sim)
    model = MaskablePPO.load("puzzle_agent")
    env_test(model, folder_sim)