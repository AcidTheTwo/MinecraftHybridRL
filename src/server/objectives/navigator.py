import numpy as np
import random
import math

'''
NAVIGATOR (LEVEL 1):
Navigates to a specified target block in the shortest time possible
'''

INPUTS = ['''player_pos, target_pos, player_vel, surr_area''']  # TODO: make into a config dict

ACTIONS = [
    'STOP', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT',
    'JUMP', 'SNEAK', 'SPRINT'
    ]

class Objective:   # TODO: do something with reset_target, make it so that target is not random ig idk bru
    def __init__(self,INPUTS):
        self.self_pos = INPUTS[0]
        self.target_pos = INPUTS[1]
        self.prev_dist = 0.0
        self.steps_at_target = 0
        self.max_steps = 400
        self.current_step = 0

    def reset_target(self, player_pos):
        """Generates a new random target within 10 blocks"""
        # Random angle and distance
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(3, 10)
        
        self.target_pos = np.array([
            player_pos[0] + math.cos(angle) * dist,
            player_pos[1], # Keep Y same for flat ground training first
            player_pos[2] + math.sin(angle) * dist
        ])
        
        self.prev_dist = np.linalg.norm(self.target_pos - player_pos)
        self.current_step = 0
        print(f"🎯 New Target: {self.target_pos} (Dist: {self.prev_dist:.2f})")
        return self.target_pos

    def calculate(self, state_dict):
        """
        Input: Dictionary containing 'player_pos', 'velocity', 'is_stuck'
        Returns: (reward, done, is_success)
        """
        player_pos = np.array(state_dict['player_pos'])
        curr_dist = np.linalg.norm(self.target_pos - player_pos)
        
        reward = 0.0
        done = False
        success = False
        self.current_step += 1

        # 1. Progress Reward (The Carrot)
        # Positive if getting closer, negative if moving away
        diff = self.prev_dist - curr_dist
        reward += diff * 10.0 
        
        # 2. Time Penalty (The Stick)
        # Forces the agent to hurry
        reward -= 0.05 

        # 3. Target Reached? (within 1 block)
        if curr_dist < 1.0:
            reward += 20.0
            print("✅ Target Reached!")
            done = True
            success = True
            self.reset_target(player_pos)

        # 4. Fail State: Taking too long
        if self.current_step >= self.max_steps:
            done = True
            reward -= 5.0 # Penalty for timing out
            self.reset_target(player_pos)
            
        self.prev_dist = curr_dist
        return reward, done, success