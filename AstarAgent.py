# fro÷ßm sb3_contrib import RecurrentPPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
from agents.game_obs.Rainbow_DQN import RainbowAgent
import torch

import subprocess
import sys

import time
import random
from collections import deque

try:
    import pathfinding
    import keyboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pathfinding"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
    import pathfinding
    import keyboard

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.core.node import Node
from pathfinding.finder.a_star import AStarFinder
import numpy as np

import heapq
from node import AStarNode
from helper import Helper

import matplotlib.pyplot as plt

def __init__(self):
    self.main_save_load_num = 0
    self.last_save_num = 0

def extract_plan(time_limit_count, search_count, actions_data, best_pos, require_replanning):
    nodes = []
    actions = []
    save_nums = []
    pos_x = []
    
    if time_limit_count < 250 and search_count < 50:
        current = best_pos
        while current.parent != None:
            for i in range(current.repetitions):
                nodes.append(current)
                actions.append(current.action)
                save_nums.append(current.save_num)
                pos_x.append(current.pos_x)

            if current.state[19] > 0 or current.state[37] > 0:
                require_replanning = True
            
            current = current.parent

        nodes.reverse()
        actions.reverse()
        save_nums.reverse()
        pos_x.reverse()

    return nodes, actions, save_nums, require_replanning

def start_search(pos_pool, visited_states, env, dist_x, dist_y, damage, death, starting_state, starting_save_num, latest_save_num, starting_repetitions):
    if death > 0 or damage > 0:
        print("death: " + str(death))
        print("damage: " + str(damage))

    if len(pos_pool) == 0:
        start_pos = AStarNode   (  
                                    env=env,
                                    parent=None,
                                    dist_x=dist_x,
                                    dist_y=dist_y,
                                    damage=damage,
                                    death=death,
                                    repetitions=starting_repetitions,
                                    action=None,
                                    save_num=starting_save_num
                                )
        
        start_pos.initialize_root(starting_state, starting_save_num)
    
        children, latest_save_num = start_pos.generate_children(env, starting_save_num, latest_save_num)

        for child in children:
            heapq.heappush(pos_pool, (child.calculate_cost(), child))
        
        current_starting_pos_x = start_pos.pos_x

        best_pos = start_pos
        furthest_pos = start_pos
    else:
        current_starting_pos_x = dist_x
        best_pos, pos_pool = pick_best_pos(pos_pool)
        furthest_pos = best_pos

    return best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num

def pick_best_pos(pos_pool):
    best_pos_pool = None
    best_pos_cost = float("inf")

    for i, current_pos_pool in enumerate(pos_pool):
        current_cost = current_pos_pool[1].calculate_cost()
        if current_cost < best_pos_cost:
            best_pos_pool = current_pos_pool
            best_pos_cost = current_cost
            best_index = i
    
    best_pos_pool = pos_pool.pop(best_index)
    best_pos = best_pos_pool[1]
    return best_pos, pos_pool

def visited(x, y, t, visited_states):
    visited_states.append((x, y, t))

    return visited_states

def is_in_visited(x, y, t, visited_states):
    time_diff = 5
    x_diff = 20
    y_diff = 20

    for v in visited_states:
        if abs(v[0] - x) < x_diff and abs(v[1] - y) < y_diff and abs(v[2] - t) < time_diff and t >= v[2]: 
            return True
        
    return False

def search(time_limit_count, actions_data, env, pos_pool, best_pos, furthest_pos, current_starting_pos_x, latest_save_num, visited_states, require_replanning, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count):
    
    current = best_pos
    current_good = False
    max_right = 20
    search_count = 0

    while best_pos.reached_end_count == 0 and search_count <= 50 and time_limit_count <= 250 and (len(pos_pool) != 0 and (((best_pos.pos_x - current_starting_pos_x) < max_right) or not current_good) and env.episode_length < 600):        
        if (search_count % 50) == 0:
            print("search_count limit count: " + str(search_count))
        
        if (time_limit_count % 50) == 0:
            print("time limit count: " + str(time_limit_count))
            
        current, pos_pool = pick_best_pos(pos_pool)
        
        if current == None:
            return None
        
        current_good = False
        real_remaining_time, latest_save_num = current.simulate_pos(env, latest_save_num, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count, best_pos.remaining_time_estimated)

        check_condition = -1

        if is_in_visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states):
            current.penalty += Helper.visited_list_penalty
        
        if real_remaining_time < 0:
            check_condition = 1
            continue
        elif current.damage > 0 or current.death > 0:
            check_condition = 2
            current.penalty += (Helper.visited_list_penalty * 3)
            heapq.heappush(pos_pool, (current.calculate_cost(), current))
        elif not current.is_in_visited_list and is_in_visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states):
            check_condition = 3
            current_good = True
            current.is_in_visited_list = True
            heapq.heappush(pos_pool, (current.calculate_cost(), current))
        else:
            check_condition = 4
            current_good = True
            visited_states = visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states)

            children, latest_save_num = current.generate_children(env, current.save_num, latest_save_num)

            for child in children:
                heapq.heappush(pos_pool, (child.calculate_cost(), child))

        if current_good:
            if current.damage == 0 and current.death == 0:
                if best_pos.remaining_time_estimated > current.remaining_time_estimated or current.reached_end_count > 0:
                    best_pos = current

                    # print("CURRENT BEST POS: " + str(best_pos.action))
                    # n, a, s, _ = extract_plan(time_limit_count, actions_data, best_pos, require_replanning)
                    # print("EXTRACT ACTIONS: " + str(a))
                    # print("EXTRACT SAVE NUMS: " + str(s))
                    # print("---")
                    search_count = 0

            if current.pos_x > furthest_pos.pos_x:
                furthest_pos = current

        search_count+=1  
        time_limit_count+=1             

    if (current.pos_x - current_starting_pos_x) < max_right and furthest_pos.pos_x > best_pos.pos_x + 20:
        best_pos = furthest_pos

    # print("search time_limit_count: " + str(time_limit_count))
    # print("search search_count: " + str(search_count))
    # print("search best_pos: " + str(best_pos))
    # print("search furthest_pos: " + str(furthest_pos))
    # print("search pos_pool: " + str(pos_pool))
    # print("search visited_states: " + str(visited_states))
    # print("search latest_save_num: " + str(latest_save_num))
    return time_limit_count, search_count, best_pos, furthest_pos, pos_pool, visited_states, latest_save_num

def optimise(time_limit_count, pos_pool, visited_states, actions_data, env, original_state, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count):
    plan_ahead = 2
    steps_per_search = 1

    require_replanning = False
    latest_save_num = original_save_num

    state = original_state

    best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num = start_search(pos_pool, visited_states, env, original_dist_x, original_dist_y, original_damage, original_death, state, latest_save_num, latest_save_num, steps_per_search)

    if state[37] > 0:
        best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num = start_search(pos_pool, visited_states, env, original_dist_x, original_dist_y, original_damage, original_death, original_state, original_save_num, latest_save_num, steps_per_search)

    time_limit_count, search_count, best_pos, furthest_pos, pos_pool, visited_states, latest_save_num = search(time_limit_count, actions_data, env, pos_pool, best_pos, furthest_pos, current_starting_pos_x, latest_save_num, visited_states, require_replanning, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count)
    
    nodes_list, actions_list, save_list, require_replanning = extract_plan(time_limit_count, search_count, actions_data, best_pos, require_replanning)

    return nodes_list, actions_list, save_list, time_limit_count, search_count, best_pos, latest_save_num, pos_pool, visited_states

class AstarAgent:    
    # cwd: c:\Users\vassa\Documents\GitHub\Affectively-Framework
    # conda_env: affect-envs
    # script_path: ./train.py
    # runs: 5
    # use_gpu: 0
    # weight: 0.5
    # cluster: 0
    # target_arousal: 1
    # preference: 1
    # classifier: 1
    # game: platform
    # period_ra: 0
    # cv: 0
    # headless: 1
    # discretize: 0
    # grayscale: 0
    # output_dir: ./results/
    # algorithm: PPO
    # policy: MlpPolicy

    def AStarRun(self, segments, game_env, segment_count, step_max_count) :
        # game_env = None
        
        main_actions_data = {
            "stay_still": {"action": (1, 0, 0), "score": 0},
            "move_left": {"action": (0, 0, 0), "score": 0},
            "move_right": {"action": (2, 0, 0), "score": 0},
            "jump_straight": {"action": (1, 1, 0), "score": 0},
            "jump_left": {"action": (0, 1, 0), "score": 0},
            "jump_right": {"action": (2, 1, 0), "score": 0},
        }

        # if game_env is not None:
        #     game_env.customSideChannel.tiles_ready = False
        #     game_env.step(main_actions_data["stay_still"]["action"], 1, False)
            
        # print("AStarRun")
        # keyboard.wait("space")

        weight = 0.5
        target_arousal = 1
        cluster = 0
        period_ra = 0
        headless = 1
        grayscale = 0
        discretize = 0
        use_gpu = 0
        classifier = 1
        preference = 1
        
        temp_state = None
        # main_env = PiratesEnvironmentGameObs(
        #     id_number=1,
        #     weight=weight,
        #     graphics=True, # Pirates is bugged in headless, prevent it manually for now
        #     cluster=cluster,
        #     target_arousal=target_arousal,
        #     period_ra=period_ra,
        #     discretize=discretize,
        #     classifier=classifier,
        #     preference=preference,
        # )
        print("game_env: " + str(game_env))
        if game_env is None:
            main_env = PiratesEnvironmentGameObs(
                id_number=1,
                weight=weight,
                graphics=True, # Pirates is bugged in headless, prevent it manually for now
                cluster=cluster,
                target_arousal=target_arousal,
                period_ra=period_ra,
                discretize=discretize,
                classifier=classifier,
                preference=preference,
            )

            self.main_save_load_num = 0
            main_env.build_segment(segments)

            main_env.reached_termination = False
            main_env.reached_end_door = False
            main_env.customSideChannel.levelEnd = False
            self.obs = main_env.reset()
        else:
            self.main_save_load_num = self.last_save_num

            game_env.reached_termination = False
            game_env.reached_end_door = False
            game_env.customSideChannel.levelEnd = False
            # temp_state = game_env.semi_reset()
            temp_state = game_env.reset()
            
            main_env = game_env

            # if main_env.reached_termination == True or main_env.reached_end_door == True or temp_state[37] > 0:
            #     print("main_env.reached_termination: " + str(main_env.reached_termination))
            #     print("main_env.reached_end_door: " + str(main_env.reached_end_door))
            #     print("temp_state[37]: " + str(temp_state[37]))
            #     keyboard.wait("space")
    
            main_env.build_segment(segments)

            main_env.reached_termination = False
            main_env.reached_end_door = False
            main_env.customSideChannel.levelEnd = False
            # self.obs = main_env.semi_reset()
            self.obs = main_env.reset()
        
        # main_save_load_num = 0

        self.main_save_load_num += 1

        # if game_env is None:
        # main_env.step(main_actions_data["stay_still"]["action"], -main_save_load_num, False)
        
        # while True and game_env is not None:
        #     main_env.step(main_actions_data["stay_still"]["action"], 1, False)
            
        #     main_env.reached_termination = False
        #     main_env.reached_end_door = False
        #     main_env.customSideChannel.levelEnd = False
        #     self.obs = main_env.reset()

        #     if main_env.customSideChannel.tiles_ready == True:
        #         break

        # if game_env is not None:
        #     main_env.step(main_actions_data["stay_still"]["action"], -main_save_load_num, False)

        game_env = main_env

        # print("AStarRun BEGIN")
        # keyboard.wait("space")
        
        # print("Starting Load, num: " + str(main_save_load_num))
        main_arousal, main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], self.main_save_load_num, False)
        # keyboard.wait("space")

        main_dist_x = 0
        main_dist_y = 0
        main_damage = 0
        main_death = 0
        main_score = 0
        main_kill_count = 0
        main_time_limit_count = 0
        main_search_count = 0

        main_pos_pool = []
        main_visited_states = []

        while True:   
            main_new_save_load_num = self.main_save_load_num

            main_arousal, main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], self.main_save_load_num, False)

            main_nodes_list, main_actions_list, main_save_list, main_time_limit_count, main_search_count, main_best_pos, main_new_save_load_num, main_pos_pool, main_visited_states = optimise(main_time_limit_count, main_pos_pool, main_visited_states, main_actions_data, main_env, main_state, main_new_save_load_num, main_dist_x, main_dist_y, main_damage, main_death, main_score, main_kill_count)

            if main_search_count >= 50 or main_search_count == 0 or main_time_limit_count >= 250:
                playable = False
                print("Close 1")
                state_list = []
                game_env = None
                main_env.env.close()
                # main_env.step(main_actions_data["stay_still"]["action"], 1, False)
                # main_env.reached_termination = False
                # main_env.reached_end_door = False
                # main_env.customSideChannel.levelEnd = False
                # main_env.reset()
                # keyboard.wait("space")  
                return playable, main_dist_x, 0, game_env, state_list
            
            main_arousal, main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], self.main_save_load_num, False)

            main_arousal, main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_best_pos.action, main_best_pos.save_num, False)
                
            main_dist_x = main_best_pos.pos_x
            main_dist_y = main_best_pos.pos_y
            main_damage += main_best_pos.damage
            main_death += main_best_pos.death

            main_score = main_best_pos.score_difference
            main_kill_count = main_best_pos.kill_count_difference

            # keyboard.wait("space")   

            if main_reached_termination:
                playable = True

                main_env.can_end = True

                main_env.reached_termination = False
                main_env.reached_end_door = False
                # main_env.semi_reset()
                main_env.reset()
                main_env.customSideChannel.levelEnd = False

                action_count = 0
                arousal_counter = 0

                tick_counter = 0

                state_list = []
                can_record_state = False

                for i, part_node in enumerate(main_nodes_list):
                    if part_node.save_num:
                        main_arousal = -1
                        
                        if action_count == 0:
                            main_env.reached_termination = False
                            main_env.reached_end_door = False
                            main_env.customSideChannel.levelEnd = False

                            main_arousal, main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(part_node.action, 1, True)
                            self.last_save_num = 1

                            main_env.reached_termination = False
                            main_env.reached_end_door = False
                            main_reached_termination = False
                            main_env.customSideChannel.levelEnd = False
                        else:
                            main_arousal, main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(part_node.action, part_node.save_num, True)
                            self.last_save_num = part_node.save_num

                        tick_counter += 1
                        # state_list.append(main_state)

                        if can_record_state == True:
                            state_list.append(main_state[:38])

                        if main_state[0] < -20:
                            can_record_state = True
                        # if main_state[0] < -20:
                        # print("main_state[:38]: " + str(main_state[:38]))
                        # keyboard.wait("space")

                        # print("MAIN AROUSAL: " + str(main_arousal))
                        # arousal_counter += main_arousal
                        if main_arousal == 0:
                            arousal_counter -= 1
                        elif main_arousal == 1:
                            arousal_counter += 1
                        # print("AROUSAL: " + str(arousal_counter))

                        # if main_arousal > 0:
                        #     print("MAIN AROUSAL: " + str(main_arousal))
                        #     keyboard.wait("space") 

                        action_count += 1

                        main_dist_x += main_state[0]
                        main_dist_y += main_state[1]
                        main_damage += main_state[19]
                        main_death += main_state[37] 

                        main_score = main_state[7]
                        main_kill_count = main_state[23]

                        # keyboard.wait("space")    

                        # if main_reached_termination or main_death > 0:
                        #     # print("TERMINATION 2")
                        #     # print("main_reached_termination: " + str(main_reached_termination))
                        #     # print("main_death: " + str(main_death))
                        #     playable = True

                        #     if main_reached_end_door == False:
                        #         playable = False

                        #     main_env.env.close()

                        #     print("FINAl AROUSAL: " + str(arousal_counter))
                        #     print("FINAL TICK: " + str(tick_counter))
                        #     # keyboard.wait("space") 
                        #     # main_env.step(main_actions_data["stay_still"]["action"], 1, False)
                        #     # main_env.reached_termination = False
                        #     # main_env.reached_end_door = False
                        #     # main_env.customSideChannel.levelEnd = False
                        #     # main_env.reset()
                        #     # keyboard.wait("space")  
                        #     return playable, main_dist_x, arousal_counter, game_env

                print("END OF RUN")
                print("main_reached_termination: " + str(main_reached_termination))
                # keyboard.wait("space")
                print("---------------")

                playable = True
                main_search_count = 0
                main_time_limit_count = 0

                # if main_reached_end_door == False:
                #     playable = False

                # if not playable or (segment_count + 1) >= step_max_count:
                #     game_env = None
                #     main_env.env.close()
                game_env = None
                main_env.env.close()

                print("FINAl AROUSAL: " + str(arousal_counter))
                print("FINAL TICK: " + str(tick_counter))
                # keyboard.wait("space") 
                # main_env.step(main_actions_data["stay_still"]["action"], 1, False)
                # main_env.reached_termination = False
                # main_env.reached_end_door = False
                # main_env.customSideChannel.levelEnd = False
                # main_env.reset()
                # keyboard.wait("space")  
                return playable, main_dist_x, arousal_counter, game_env, state_list

            main_new_save_load_num += 1  

            main_arousal, main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], -main_new_save_load_num, False)
            
            # if main_reached_termination or main_death > 0:
            #     playable = True

            #     if main_reached_end_door == False:
            #         playable = False

            #     print("Close 4")
            #     main_env.env.close()

            #     return playable, main_dist_x
            
            self.main_save_load_num = main_new_save_load_num

            test_arousal, test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = main_env.step(main_actions_data["stay_still"]["action"], self.main_save_load_num, False) 
            
            # if main_reached_termination and main_death > 0:
            #     playable = True

            #     if main_reached_end_door == False:
            #         playable = False

            #     print("Close 5")
            #     main_env.env.close()

            #     return playable, main_dist_x
            
            print("Distance Travelled: " + str(main_dist_x))

            # if main_reached_termination and main_death > 0:
            #     playable = True

            #     if main_reached_end_door == False:
            #         playable = False

            #     print("Close 6")
            #     main_env.env.close()

            #     return playable, main_dist_x
            
            print("MOVED")
            print("******************************************")
            # keyboard.wait("space")

        # 0 - (transform.position - previousPosition).x
        # 1 - (transform.position - previousPosition).y
        # 2 - _corgiController.Speed.x
        # 3 - _corgiController.Speed.y
        # 4 - _corgiController._movementDirection
        # 5 - _health.CurrentHealth
        # 6 - _health.hasPowerUp
        # 7  - Player Score
        # 8 - Player Has Collisions
        # 9 - Player Is Colliding Above
        # 10 - Player Is Colliding Below
        # 11 - Player Is Colliding Left
        # 12 - Player Is Colliding Right
        # 13 - Player Is Falling
        # 14 - Player Is Grounded
        # 15 - Player Is Jumping
        # 16 - Player Speed X
        # 17 - Player Speed Y
        # 18 - Player Health
        # 19 - Player Damaged
        # 20 - Player Point Pickup
        # 21 - Player Power Pickup
        # 22 - Player Has Powerup
        # 23 - Player Kill Count
        # 24 - Bots Visible
        # 25 - Bot Has Collisions
        # 26 - Bot Is Colliding Below
        # 27 - Bot Is Colliding Left
        # 28 - Bot Is Colliding Right
        # 29 - Bot Is Falling
        # 30 - Bot Is Grounded
        # 31 - Bot Speed X
        # 32 - Bot Speed Y
        # 33 - Bot Health
        # 34 - Bot Player Distance
        # 35 - Pick Ups Visible
        # 36 - Pick Up Player Disctance
        # 37 - Player Death


# cd C:\Users\Admin\Documents\GitHub\Affectively-Framework
# conda activate unity_gym
# python PPOAgent.py
# 

# cd C:\Users\Admin\Documents\GitHub\Affectively-Framework\Tensorboard\CNN
# conda activate unity_gym
# tensorboard --logdir=C:\Users\Admin\Documents\GitHub\Affectively-Framework\Tensorboard\CNN
# 

# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework
# conda activate unity_gym
# python PPOAgent.py
# 

# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework\Tensorboard\CNN
# tensorboard --logdir=C:\Users\vassa\Documents\GitHub\Affectively-Framework\Tensorboard\CNN
# 

# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework
# conda activate unity_gym
# python AstarAgent.py
# 

# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework
# conda activate unity_gym
# python GANMain.py
# 

if __name__ == "__main__":  
    test = 1
    # env = GANLevelEnv()
    # model = PPO.load("GANArousalAgents/PPO/cnn_ppo_solid_optimize_1_extended.zip", env=env)
    # obs, info = env.reset()

    # segment_buffer = deque(maxlen=11)

    # actions = []
    # for i in range(11):
    #     action, _ = model.predict(obs, deterministic=True)
    #     actions.append(action.copy())
    #     print("action: " + str(action))
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     # keyboard.wait("space")

    #     segment = info["segment"]
    #     segment_buffer.append(segment)
    
    # actions = np.array(actions)

    # num_steps, action_dim = actions.shape

    # x = np.tile(np.arange(action_dim), num_steps)
    # y = actions.flatten()

    # plt.figure(figsize=(10, 5))
    # plt.scatter(x, y, alpha=0.6)
    # plt.xlabel("Action dimension index")
    # plt.ylabel("Action value")
    # plt.title("Scatter plot of all action values")
    # plt.grid(True)
    # plt.show()
    # # agent = AstarAgent()
    # # agent.AStarRun(segment_buffer) 