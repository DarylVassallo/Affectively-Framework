from helper import Helper
import keyboard
class AStarNode():
    def __init__(self, env, parent, dist_x, dist_y, damage, death, repetitions, action, save_num):
        self.actions_data = {
            "stay_still": {"action": (1, 0, 0), "cost": 0},
            "move_left": {"action": (0, 0, 0), "cost": 0},
            "move_right": {"action": (2, 0, 0), "cost": 0},
            "jump_straight": {"action": (1, 1, 0), "cost": 0},
            "jump_left": {"action": (0, 1, 0), "cost": 0},
            "jump_right": {"action": (2, 1, 0), "cost": 0},
        }
        
        self.reached_end_count = 0
        self.parent = parent

        self.penalty = 0
        self.reward = 0

        self.kill_count_difference = 0
        self.score_difference = 0

        self.remaining_time = 0
        
        self.state = None
        self.save_num = save_num

        self.action = action
        self.repetitions = repetitions
        
        self.time_elapsed = (env.episode_length / 600)

        if self.parent != None:
            self.damage = self.parent.damage + damage
            self.death = self.parent.death + death
            
            self.calculate_distance_from_origin((self.parent.distance_from_origin + dist_x))

            self.pos_x = self.parent.pos_x + dist_x
            self.pos_y = self.parent.pos_y + dist_y
        else:
            self.damage = damage
            self.death = death
            
            self.calculate_distance_from_origin(dist_x)

            self.pos_x = dist_x
            self.pos_y = dist_y

        self.has_been_hurt = False
        self.is_in_visited_list = False

        self.calculate_distance_from_origin(dist_x)

    def initialize_root(self, state, save_num):
        if self.parent == None:
            self.state = state

            self.save_num = save_num

    def calculate_distance_from_origin(self, dist_x):
        self.distance_from_origin = dist_x
        # self.remaining_time_estimated = (1 - (self.distance_from_origin / 600))
        test_time = max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000)))
        if test_time <= 0:
            print("self.distance_from_origin: " + str(self.distance_from_origin))
            print("(self.distance_from_origin / 1000): " + str((self.distance_from_origin / 1000)))
            print("1 - (self.distance_from_origin / 1000): " + str(1 - (self.distance_from_origin / 1000)))
            print("min(1.0, 1 - (self.distance_from_origin / 1000)): " + str(min(1.0, 1 - (self.distance_from_origin / 1000))))
            print("max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000))): " + str(max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000)))))
            keyboard.wait("space")

        self.remaining_time_estimated = max(0.0, min(1.0, 1 - (self.distance_from_origin / 1000)))

        self.calculate_cost()
    
    def calculate_cost(self):
        if self.damage > 0 or self.death > 0:
            self.cost = 9999
        else:
            self.cost = ((self.remaining_time_estimated + self.time_elapsed * 0.9) + self.penalty)
        return self.cost
    
    def extract_plan(self):
        actions = []
        pos_x_list = []
        if self.parent == None:
            return actions, pos_x_list
        
        current = self.parent
        while current.parent != None:
            for i in range(current.repetitions):
                actions.append(current.action)
                pos_x_list.append(current.pos_x)
            
            current = current.parent

        actions.reverse()
        pos_x_list.reverse()
        return actions, pos_x_list

    def simulate_pos(self, env, latest_save_num, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count, best_remaining_time_estimated):
        self.state = self.parent.state

        self.reached_end_count = 0
        self.pos_x = self.parent.pos_x
        self.calculate_distance_from_origin(self.parent.pos_x)
        self.pos_y = self.parent.pos_y
        self.damage = self.parent.damage
        self.death = self.parent.death

        action_plan, pos_x_plan = self.extract_plan()

        if self.parent == None:
            arousal, raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, original_save_num, False)
        
            if reached_end_door:
                self.reached_end_count += 1

            self.pos_x += self.state[0]
            self.pos_y += self.state[1]
            self.damage += self.state[19]
            self.death += self.state[37]

        else:

            arousal, raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.action, self.parent.save_num, False)
        
            if reached_end_door:
                self.reached_end_count += 1

            self.pos_x += self.state[0]
            self.pos_y += self.state[1]
            self.damage += self.state[19]
            self.death += self.state[37]

        self.calculate_distance_from_origin(self.pos_x)

        latest_save_num += 1
        temp_end_save_num = latest_save_num

        arousal, raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], -latest_save_num, False)
        
        if reached_end_door:
            self.reached_end_count += 1
        
        self.damage += self.state[19]
        self.death += self.state[37]

        arousal, raw_grid, self.state, reached_termination, reached_end_door, reward, done, info = env.step(self.actions_data["stay_still"]["action"], 0, False)
        
        if reached_end_door:
            self.reached_end_count += 1
        
        self.damage += self.state[19]
        self.death += self.state[37]

        test_arousal, test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = env.step(self.actions_data["stay_still"]["action"], temp_end_save_num, False)

        damage = Helper.get_damage(self, self.parent)
        self.remaining_time =  (1 -(env.episode_length / 600))

        if self.is_in_visited_list:
            self.penalty += Helper.visited_list_penalty

        self.has_been_hurt = damage != 0

        return self.remaining_time, latest_save_num

    def __lt__(self, other):
        return self.cost < other.cost

    def generate_children(self, env, save_num, latest_save_num):
        children = []
        possible_actions =  Helper.create_possible_actions(self)
        
        if self.is_leaf_node():
            possible_actions = []

        for i in range(len(possible_actions)):
            action = possible_actions[i]
            child_damage = 0
            child_death = 0
            
            arousal, raw_grid, state, reached_termination, reached_end_door, reward, done, step_info = env.step(action, save_num, False)

            child_damage += state[19]
            child_death += state[37]
            
            latest_save_num += 1

            arousal, raw_grid, state, reached_termination, reached_end_door, reward, done, step_info = env.step(action, -latest_save_num, False)

            child_damage += state[19]
            child_death += state[37]

            children.append(AStarNode(env, self, state[0], state[1], child_damage, child_death, self.repetitions, action, latest_save_num))

        return children, latest_save_num


    def is_leaf_node(self):
        if self.death <= 0:
            return False
        return self.death > 0     