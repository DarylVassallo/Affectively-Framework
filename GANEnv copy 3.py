import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from GANWrapper import GANWrapper

from AstarAgent import AstarAgent





import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import os

import keyboard

class GANLevelEnv(gym.Env):
    def __init__(self, gan_model=GANWrapper(), level_dim=32):
        super().__init__()

        self.episode_count = 0

        # self.step_max_count = 11
        self.step_max_count = 11

        self.gan = gan_model
        self.level_dim = level_dim

        self.action_list = deque(maxlen=5)
        self.observation = None

        # self.segment_buffer = deque(maxlen=self.step_max_count)
        self.segment_buffer = deque(maxlen=3)

        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level_dim + 3,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level_dim,), dtype=np.float32)

        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level_dim * 5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=((self.level_dim * 5) + 3,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.level_dim,), dtype=np.float32)  # input vector for GAN

        self.segment_count = 0

        self.initialise_results_log()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_count += 1

        self.segment_buffer.clear()
        self.segment_count = 0

        self.action_list.clear()
        self.observation = None     

        self.total_score = 0

        self.score_count = 0
        self.total_enemy_count = 0

        while len(self.action_list) <= 0:
            self.current_action = np.random.uniform(-1, 1, size=self.level_dim)
            self.current_segment = self.gan.generate(self.current_action)

            self.playable = self.is_playable(self.current_segment)

            if self.playable:    
                # self.observation = np.concatenate([self.current_action, side_info])         

                # self.observation = self.current_action

                for i in range(5):
                    self.action_list.append(self.current_action)
                side_info = self.segment_side(self.segment_count).astype(np.float32)
                self.observation = np.concatenate([np.concatenate(self.action_list),side_info]).astype(np.float32)

                # print("Observation " + str(self.observation) + " added")
                # print("")

        self.segment_buffer.append(self.current_segment)

        info = {"segment": self.current_segment,
                "full_level": self.segment_buffer}
        
        return self._get_observation_stack(), info
        # return self.observation, info

    def step(self, action):
        # print("Action mean:", np.mean(action), "std:", np.std(action))
        self.current_action = action
        terminated = False
        truncated = False

        self.segment_count += 1

        score = 0
        if self.playable and self.segment_count < self.step_max_count:
            self.current_segment = self.gan.generate(self.current_action)

            # print("self.current_segment: " + str(self.current_segment))

            self.segment_buffer.append(self.current_segment)

            if self.current_segment is None or len(self.current_segment) == 0:
                terminated = True
                # reward = -1.0
                reward = 0

                
                # self.observation = np.concatenate([self.current_action, side_info])

                # self.observation = self.current_action
                # return self.observation, reward, terminated, False, info

                self.action_list.append(self.current_action)
                side_info = self.segment_side(self.segment_count).astype(np.float32)
                self.observation = np.concatenate([np.concatenate(self.action_list),side_info]).astype(np.float32)

                return self._get_observation_stack(), reward, terminated, False, info

            self.playable = self.is_playable(self.segment_buffer)

            score, enemy_count = self.reward(self.current_segment, self.segment_count)

            self.score_count += score
            self.total_enemy_count += enemy_count

            # side_info = self.segment_side((len(self.segment_buffer) - 1))
            # self.observation = np.concatenate([self.current_action, side_info])

            # self.observation = self.current_action  
            self.action_list.append(self.current_action)   
            side_info = self.segment_side(self.segment_count).astype(np.float32)
            self.observation = np.concatenate([np.concatenate(self.action_list),side_info]).astype(np.float32)

        info = {"segment": self.current_segment,
                "full_level": self.segment_buffer}
        
        if not self.playable or self.segment_count >= self.step_max_count:
            with open(self.playable_file, "a", encoding="utf-8") as f:
                f.write("Episode " + str(self.episode_count) + " : " + str(self.playable) + " : " + str(self.segment_count) + "\n")

            with open(self.enemy_count_file, "a", encoding="utf-8") as f:
                f.write("Episode " + str(self.episode_count) + " : " + str(self.total_enemy_count) + "\n")

            terminated = True

            if not self.playable:
                print("NOT PLAYABLE WITH " + str(self.segment_count) + " SEGMENTS")

                # score = score - ((11 - self.segment_count) + 1)
                # self.score_count = self.score_count - ((11 - self.segment_count) + 1)

                self.score_count -= score
                score = 0
                

                # score = 0

            else:
                print("PLAYABLE WITH " + str(self.segment_count) + " SEGMENTS")

            print("LEVEL SCORE: " + str(self.score_count))
            print("==============================")
            print("==============================")

        self.total_score += score

        if terminated == True:
            with open(self.reward_file, "a", encoding="utf-8") as f:
                f.write("Episode " + str(self.episode_count) + " : " + str(self.score_count) + "\n")

        # return self.observation, score, terminated, truncated, info
        return self._get_observation_stack(), score, terminated, truncated, info

    def initialise_results_log(self):
        self.log_dir = "./ExperimentLogs/"
        os.makedirs(self.log_dir, exist_ok=True)

        existing_logs = [
            d for d in os.listdir(self.log_dir)
            if d.startswith("ResultsLog_") and os.path.isdir(os.path.join(self.log_dir, d))
        ]

        if existing_logs:
            existing_numbers = [
                int(d.split("_")[-1]) for d in existing_logs if d.split("_")[-1].isdigit()
            ]
            next_log_number = max(existing_numbers) + 1
        else:
            next_log_number = 1

        self.log_dir = os.path.join(self.log_dir, f"ResultsLog_{next_log_number}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.playable_file = os.path.join(self.log_dir, "playable_results.txt")
        self.enemy_count_file = os.path.join(self.log_dir, "enemy_count.txt")
        self.reward_file = os.path.join(self.log_dir, "reward.txt")

    def segment_side(self, segment_idx):
        if segment_idx < 6:
            return np.array([1, 0, 0], dtype=np.float32)  # left
        else:
            return np.array([0, 0, 1], dtype=np.float32)  # right
        
    def is_playable(self, segment):
        return True

        agent = AstarAgent() 
        playable, playable_distance = agent.AStarRun(segment)

        return playable

    def reward(self, segment, index): 
        num_enemies = sum(np.count_nonzero(arr == 5) for arr in segment)

        side = self.segment_side(index)

        reward = 0
        if np.array_equal(side, np.array([1, 0, 0], dtype=np.float32)):
            reward = 1 / (num_enemies + 1)
        elif np.array_equal(side, np.array([0, 0, 1], dtype=np.float32)):
            reward = num_enemies / 7
        else:
            print("SIDE: " + str(side))
            keyboard.wait("space")

        print("reward: " + str(reward))
        # reward = 0
        # if index < 6:
        #     reward = num_enemies / 7
        # else:
        #     reward = (1 / (num_enemies + 1))

        # print("reward: " + str(reward))
        return reward, num_enemies
    
    def _get_observation_stack(self):
        # print("self.action_list: " + str(self.action_list))
        # stacked = np.stack(self.action_list, axis=0).flatten()
        # return stacked.astype(np.float32)

        return self.observation