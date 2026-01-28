from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np

from GANEnv import GANLevelEnv
from GANGenerate import CNet

import os
import shutil
import platform
import sys
import yaml
import subprocess
import sys
# print("SHOWING")
# print(sys.executable)

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

def load_config(config_file_path):
    """
    Loads parameters from a YAML configuration file.

    Args:
        config_file_path (str): The full path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration parameters.
    """
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file '{config_file_path}': {exc}")
        return None

def make_env(worker_id):
    def _init():
        env = GANLevelEnv(worker_id=worker_id)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":    
    run = 1
    weight = 0

    env = GANLevelEnv()
    env = Monitor(env)

    eval_env = GANLevelEnv()
    eval_env = Monitor(env)

    label = 'optimize' if weight == 0 else 'arousal' if weight == 1 else 'blended'

    checkpoint_callback = CheckpointCallback(
                                                save_freq=500,
                                                save_path="./GANArousalAgents/PPO/MinEnemy1/",
                                                name_prefix=f"cnn_ppo_{label}_{run}"
                                            )
    
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3,
        min_evals=3,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./GANArousalAgents/PPO/MinEnemy1/best_model/",
        log_path="./GANArousalAgents/PPO/MinEnemy1/eval_logs/",
        eval_freq=500,
        n_eval_episodes=5,
        deterministic=True,
        callback_after_eval=stop_train_callback
    )
    
    callbacks = CallbackList([
                                ProgressBarCallback(),
                                checkpoint_callback,
                                eval_callback
                            ])


    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch = dict(pi=[256, 256], vf=[256, 256]),
        activation_fn = torch.nn.ReLU,
    )

    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        verbose=1,
        n_steps=64,
        tensorboard_log="./Tensorboard/CNN/",
        device='cuda',
    )

    # checkpoint_path = "./GANArousalAgents/PPO/cnn_ppo_optimize_1_6700_steps.zip"

    # model = PPO.load(
    #     checkpoint_path,
    #     env=env,
    #     device="cuda",
    #     tensorboard_log="./Tensorboard/CNN/"
    # )
    # model.verbose = 1

    # remaining_steps = 20000 - model.num_timesteps
    remaining_steps = 20000000

    model.learn(total_timesteps=remaining_steps, callback=callbacks, reset_num_timesteps=False)
    model.save(f"./GANArousalAgents/PPO/MinEnemy1/cnn_ppo_{label}_{run}_extended")