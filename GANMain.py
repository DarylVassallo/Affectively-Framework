import os
import numpy as np

from GANEnv import GANLevelEnv
from stable_baselines3 import PPO
from GANGenerate import CNet

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch
import torch.nn as nn

from scipy.signal import savgol_filter

def create_new_segment_folder(base_path="GeneratedLevelSegments"):
    os.makedirs(base_path, exist_ok=True)

    existing = [
        f for f in os.listdir(base_path)
        if f.startswith("SegmentsV") and f[9:].isdigit()
    ]

    if not existing:
        next_num = 1
    else:
        nums = [int(f[9:]) for f in existing]
        next_num = max(nums) + 1

    new_folder = os.path.join(base_path, f"SegmentsV{next_num}")
    os.makedirs(new_folder)

    return new_folder

class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input = np.prod(observation_space.shape)
        self.linear = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    def forward(self, obs):
        return self.linear(obs.flatten(start_dim=1))
    
def generate_graph_using_module():
    env = GANLevelEnv()
    # model = PPO.load("GANArousalAgents/MaxEnemy/cnn_ppo_solid_optimize_1_extended.zip", env=env)
    # model = PPO.load("GANArousalAgents/Playable/cnn_ppo_solid_onlyplayable_optimize_1_extended.zip", env=env)
    # obs, info = env.reset()

    episode_nums = []
    avg_enemy_counts = []

    episode_num = 100
    while episode_num <= 700:
        print("###################################")
        print("###################################")
        print("###################################")
        print("###################################")
        print("episode_num: " + str(episode_num))
        # model = PPO.load("GANArousalAgents/MaxEnemy/cnn_ppo_optimize_1_" + str(episode_num) + "_steps", env=env)
        model = PPO.load("GANArousalAgents/PPO/cnn_ppo_optimize_1_" + str(episode_num) + "_steps", env=env)
        average_num_enemy = 0
        for j in  range(10):
            print("ATTEMPT " + str(j))
            obs, info = env.reset()
            num_enemies = 0

            terminated = False
            for i in range(11):
                print("SEGMENT " + str(i))
                if not terminated:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)

                    full_level = info["full_level"]
                    num_enemies = sum(np.count_nonzero((arr == 5)) for arr in full_level)

            episode_nums.append(episode_num)
            avg_enemy_counts.append(num_enemies)

            average_num_enemy += num_enemies
            print(str(j) + ": " + str(num_enemies))

        average_num_enemy = average_num_enemy / 10
        # episode_nums.append(episode_num)
        # avg_enemy_counts.append(average_num_enemy)

        print("")
        print(str(average_num_enemy))
        print("====================")

        # if episode_num == 100:
        #     episode_num += 900
        # else:
        #     episode_num += 500
        episode_num += 300

    plt.figure()
    plt.plot(episode_nums, avg_enemy_counts, marker='o')
    plt.xlabel("Training Steps")
    plt.ylabel("Average Number of Enemies")
    plt.title("Enemy Count vs Training Progress")
    plt.grid(True)
    plt.show()
    
def generate_multiple_enemies_graph():
    # log_file = "RecordedLogs\MultipleValuesRecordedWithoutPlayability\enemy_count.txt"
    log_file = os.path.join( "ExperimentLogs", "ResultsLog_1", "enemy_count.txt" )

    values_1 = []
    values_2 = []
    values_3 = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(":")]
            if len(parts) != 4:
                continue

            v1, v2, v3, v4 = parts
            values_1.append(float(v2))
            values_2.append(float(v3))
            values_3.append(float(v4))

    x = range(len(values_1))

    window = 41
    poly = 2

    smooth_1 = savgol_filter(values_1, window, poly)
    smooth_2 = savgol_filter(values_2, window, poly)
    smooth_3 = savgol_filter(values_3, window, poly)

    plt.figure()
    plt.plot(x, smooth_1, label="Left Section")
    plt.plot(x, smooth_2, label="Middle Section")
    plt.plot(x, smooth_3, label="Right Section")

    plt.xlabel("Episode")
    plt.ylabel("Enemy Count")
    plt.title("Smoothed Progression Of Enemy Count Per Episode")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

def generate_graph():
    # Read values from file
    # log_file = "RecordedLogs\MaxEnemyV2Logs\scores.txt"

    # log_file = os.path.join( "ExperimentLogs", "Maximum_Enemy_Count_Logs", "enemy_count.txt" )
    # log_file = os.path.join( "ExperimentLogs", "Minimum_Enemy_Count_Logs", "enemy_count.txt" )
    # log_file = os.path.join( "ExperimentLogs", "Maximum_Enemy_Count_Logs", "enemy_count.txt" )
    log_file = os.path.join( "ExperimentLogs", "Maximum_Arousal_Logs", "arousal_count.txt" )

    values = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(":")]
            if line:
                if len(parts) != 2:
                    continue

                segment_count = int(parts[1])
                # if parts[1] == True:
                #     segment_count = segment_count + 1
                
                # if parts[1] == False:
                #     segment_count = segment_count - 1
                values.append(int(segment_count))

                # values.append(float(parts[1]))


    # X-axis: index (step, episode, etc.)
    x = range(len(values))

    window = 41
    poly = 2

    smooth = savgol_filter(values, window, poly)

    # Plot
    plt.figure()
    plt.plot(x, smooth)
    plt.xlabel("Episode")
    plt.ylabel("Arousal Value")
    plt.title("Smoothed Progression of Arousal Value (generated per step) per Episode")
    plt.show()

def read_state_values(log_file, state_index=-5):
    values = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(":")]
            if len(parts) != 2:
                continue

            state_parts = [s.strip() for s in parts[1].split(",")]

            # pick which value in the state array you want
            values.append(float(state_parts[38]))
# 2, 3, 4, 5
# 24, 25, 26, 30, 31, 32, 33, 34, 
    return values

def generate_state_graph():
    log_file_1 = os.path.join("ExperimentLogs", "ResultsLog_26", "state_list.txt")
    log_file_2 = os.path.join("ExperimentLogs", "ResultsLog_27", "state_list.txt")

    values1 = read_state_values(log_file_1, state_index=0)
    values2 = read_state_values(log_file_2, state_index=0)

    # Make same length (crop to shortest)
    min_len = min(len(values1), len(values2))
    values1 = values1[:min_len]
    values2 = values2[:min_len]

    x = range(min_len)

    window = 8
    poly = 2

    # window must be odd and <= length
    if window > min_len:
        window = min_len if min_len % 2 == 1 else min_len - 1

    smooth1 = savgol_filter(values1, window, poly)
    smooth2 = savgol_filter(values2, window, poly)

    plt.figure()
    # plt.plot(x, values1, label="ResultsLog_24")
    # plt.plot(x, values2, label="ResultsLog_25")
    plt.plot(x, smooth1, label="ResultsLog_24")
    plt.plot(x, smooth2, label="ResultsLog_25")

    plt.xlabel("Episode")
    plt.ylabel("State[0] Value")
    plt.title("Smoothed Progression of State[0] per Episode")
    plt.legend()
    plt.show()

def generate_graph_with_average():
    # Group 1 logs
    log_files_1 = [
        os.path.join("ExperimentLogs", "Max_Arousal_1", "reward_average.txt"),
        os.path.join("ExperimentLogs", "Max_Arousal_2", "reward_average.txt"),
        os.path.join("ExperimentLogs", "Max_Arousal_3", "reward_average.txt"),
    ]

    # Group 2 logs (CHANGE THESE PATHS)
    log_files_2 = [
        os.path.join("ExperimentLogs", "Min_Arousal_1", "reward_average.txt"),
        os.path.join("ExperimentLogs", "Min_Arousal_2", "reward_average.txt"),
        os.path.join("ExperimentLogs", "Min_Arousal_3", "reward_average.txt"),
    ]

    def load_runs(log_files, invert):
        all_values = []

        for log_file in log_files:
            values = []
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = [p.strip() for p in line.split(":")]
                    if len(parts) != 2:
                        continue

                    if invert == True:
                        segment_count = -float(parts[1])
                    else:
                        segment_count = float(parts[1])

                    values.append(segment_count)

            all_values.append(values)

        return all_values

    # Load both groups
    all_values_1 = load_runs(log_files_1, False)
    all_values_2 = load_runs(log_files_2, True)

    # Crop BOTH groups to same length so they align on the plot
    min_len = min(
        min(len(v) for v in all_values_1),
        min(len(v) for v in all_values_2)
    )

    all_values_1 = [v[:min_len] for v in all_values_1]
    all_values_2 = [v[:min_len] for v in all_values_2]

    # Convert to numpy arrays
    data_1 = np.array(all_values_1)
    data_2 = np.array(all_values_2)

    # Mean and std
    mean_1 = np.mean(data_1, axis=0)
    std_1 = np.std(data_1, axis=0)

    mean_2 = np.mean(data_2, axis=0)
    std_2 = np.std(data_2, axis=0)

    # Smooth
    window = 41
    poly = 2

    if window > min_len:
        window = min_len if min_len % 2 == 1 else min_len - 1

    mean_1_smooth = savgol_filter(mean_1, window, poly)
    std_1_smooth = savgol_filter(std_1, window, poly)

    mean_2_smooth = savgol_filter(mean_2, window, poly)
    std_2_smooth = savgol_filter(std_2, window, poly)

    x = np.arange(min_len)

    # Plot
    plt.figure()

    # Line 1
    plt.plot(x, mean_1_smooth, label="Maximum Arousal")
    plt.fill_between(
        x,
        mean_1_smooth - std_1_smooth,
        mean_1_smooth + std_1_smooth,
        alpha=0.25,
        label="±1 Std Dev"
    )

    # Line 2
    plt.plot(x, mean_2_smooth, label="Minimum Arousal")
    plt.fill_between(
        x,
        mean_2_smooth - std_2_smooth,
        mean_2_smooth + std_2_smooth,
        alpha=0.25,
        label="±1 Std Dev"
    )

    plt.xlabel("Episode")
    plt.ylabel("Arousal")
    plt.title("Smoothed Average Arousal Per Segment (Mean ± Std Dev)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def evaluate_model():
    env = GANLevelEnv()

    # policy_kwargs = dict(
    #     features_extractor_class=CustomMLPExtractor,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch = dict(pi=[256, 256], vf=[256, 256]),
    #     activation_fn = torch.nn.ReLU,
    # )
    
    # model = PPO(
    #     policy="MlpPolicy",
    #     policy_kwargs=policy_kwargs,
    #     env=env,
    #     verbose=1,
    #     n_steps=64,
    #     tensorboard_log="./Tensorboard/CNN/",
    #     device='cuda',
    # )

    model = PPO.load("GANArousalAgents/Maximum_Arousal_Agents/cnn_ppo_optimize_1_extended.zip", env=env)

    for j in range(100):
        print("EPISODE " + str(j))
        obs, info = env.reset()
        
        all_actions = []
        for i in range(11):
            print("i number: " + str(i))

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)


def generate_action_value_graph():
    env = GANLevelEnv()

    # policy_kwargs = dict(
    #     features_extractor_class=CustomMLPExtractor,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch = dict(pi=[256, 256], vf=[256, 256]),
    #     activation_fn = torch.nn.ReLU,
    # )
    
    # model = PPO(
    #     policy="MlpPolicy",
    #     policy_kwargs=policy_kwargs,
    #     env=env,
    #     verbose=1,
    #     n_steps=64,
    #     tensorboard_log="./Tensorboard/CNN/",
    #     device='cuda',
    # )

    model = PPO.load("GANArousalAgents/MaxEnemyV2/cnn_ppo_optimize_1_extended.zip", env=env)

    obs, info = env.reset()
    
    all_actions = []
    for i in range(11):
        print("i number: " + str(i))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        all_actions.append(action)

    all_actions = np.array(all_actions)
    timesteps = np.arange(all_actions.shape[0])

    plt.figure()

    # Plot each action dimension separately
    for dim in range(all_actions.shape[1]):
        plt.scatter(
            timesteps,
            all_actions[:, dim],
            label=f"Action dim {dim}",
            alpha=0.7,
        )

    plt.xlabel("Timestep")
    plt.ylabel("Action Value")
    plt.title("Scatter Plot of PPO Actions Over Time (Using New Untrained Model with Deterministic = False)")
    plt.legend()
    plt.show()

def unity_generate_level():
    results_path = create_new_segment_folder()

    env = GANLevelEnv()

    # policy_kwargs = dict(
    #     features_extractor_class=CustomMLPExtractor,
    #     features_extractor_kwargs=dict(features_dim=256),
    #     net_arch = dict(pi=[256, 256], vf=[256, 256]),
    #     activation_fn = torch.nn.ReLU,
    # )
    
    # model = PPO(
    #     policy="MlpPolicy",
    #     policy_kwargs=policy_kwargs,
    #     env=env,
    #     verbose=1,
    #     n_steps=64,
    #     tensorboard_log="./Tensorboard/CNN/",
    #     device='cuda',
    # )

    model = PPO.load("GANArousalAgents\MinArousal\cnn_ppo_optimize_1_14000_steps", env=env)

    obs, info = env.reset()
    
    # print("reset obs: " + str(obs))
    segment = info["segment"]

    file_path = os.path.join(results_path, f"segment_{1}.csv")
    np.savetxt(file_path, segment, fmt="%d", delimiter=",")
    
    all_actions = []
    for i in range(10):
        # print("i: " + str(i))
        # action = np.random.uniform(-1, 1, size=64)
        # action = np.zeros(32) 
        # print("Action " + str(i) + ": " + str(action))

        # print("obs: " + str(obs))
        action, _ = model.predict(obs, deterministic=True)
        # print(str(i) + ": action: " + str(action))
        # obs, reward, terminated, truncated, info = env.step(i, action)
        obs, reward, terminated, truncated, info = env.step(action)

        # print("Generated Level (info):")
        # print(info)
        # print("Reward:", reward)

        # if i >= 6 and i <= 8:
        segment = info["segment"]         # numpy array
        # enemy_count = info["enemy_count"]

        # print("")
        # print("Segment " + str(i + 1) + ": " + str(segment))
        # print("====================")

        # print("i: " + str(i) + ", enemy_count: " + str(enemy_count))
        # if enemy_count > 0:
        # print("segment:")
        # print(str(segment))
            
        print("results_path: " + str(results_path))
        file_path = os.path.join(results_path, f"segment_{i+1 + 1}.csv")
        np.savetxt(file_path, segment, fmt="%d", delimiter=",")

        all_actions.append(action)

        print("----------")

    all_actions = np.array(all_actions)
    timesteps = np.arange(all_actions.shape[0])

    plt.figure()

    # Plot each action dimension separately
    for dim in range(all_actions.shape[1]):
        plt.scatter(
            timesteps,
            all_actions[:, dim],
            label=f"Action dim {dim}",
            alpha=0.7,
        )

    plt.xlabel("Timestep")
    plt.ylabel("Action Value")
    plt.title("Scatter Plot of PPO Actions Over Time (Using Maximum Enemy Count Model with Deterministic = True)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    unity_generate_level()
    # generate_graph()
    # evaluate_model()
    # generate_graph_with_average()
    # generate_state_graph()