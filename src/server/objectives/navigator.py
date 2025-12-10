import numpy as np
import random
import math

'''
NAVIGATOR (LEVEL 1) - STRAFE ONLY
'''

# 1. ACTIONS (8 Actions)
ACTIONS = [
    'STOP', 
    'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT',
    'JUMP', 'SNEAK', 'SPRINT'
]

# 2. INPUT CONFIG
INPUT_CONFIG = {
    "self_stats": True,      
    "raw_coordinates": True, 
    "local_voxels": True,    
    "entity_radar": False    
}

class Objective:
    def __init__(self):
        self.target_pos = None
        self.prev_dist = 0.0
        self.max_steps = 400
        self.current_step = 0
        
        # Initial target (Placeholder, will be reset immediately)
        self.reset_target(np.array([0, 64, 0]))

    def get_input_shape(self):
        # 5 (Self) + 3 (Vector) + 27 (Voxels)
        return 35

    def reset_target(self, player_pos):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(3, 10)
        
        # Calculate new target
        self.target_pos = np.array([
            player_pos[0] + math.cos(angle) * dist,
            player_pos[1], # Keep Y same for flat ground
            player_pos[2] + math.sin(angle) * dist
        ])
        
        self.prev_dist = np.linalg.norm(self.target_pos - player_pos)
        self.current_step = 0
        
        # --- LOGGING START & TARGET ---
        print(f"\n📍 START: {player_pos.round(1)}  -->  🎯 TARGET: {self.target_pos.round(1)}")
        print(f"   Distance: {self.prev_dist:.2f} blocks")

    def process_state(self, raw_state):
        # 1. Extract Data
        self_stats = raw_state[0:5]
        player_pos = raw_state[5:8] 
        voxels = raw_state[10:37]
        
        # 2. Calculate Relative Vector
        rel_vec = self.target_pos - player_pos
        dist = np.linalg.norm(rel_vec)
        
        if dist > 0:
            norm_vec = rel_vec / dist
        else:
            norm_vec = np.zeros(3)
            
        # 3. Formulate Brain Input
        brain_state = np.concatenate((self_stats, norm_vec, voxels))
        
        # --- SANITIZATION ---
        if self.current_step == 0 or abs(self.prev_dist - dist) > 5.0:
            self.prev_dist = dist
            return brain_state, 0.0, False, False

        # 4. Reward Calculation
        reward = 0.0
        done = False
        success = False
        self.current_step += 1
        
        # Reward: Progress
        diff = self.prev_dist - dist
        reward += diff * 2.0 
        
        # Reward: Time Penalty
        reward -= 0.05
        
        # Check Success
        if dist < 1.5:
            reward += 20.0
            # --- LOGGING SUCCESS ---
            print(f"✅ TARGET REACHED! (Steps: {self.current_step})")
            done = True
            success = True
            self.reset_target(player_pos)
            
        # Check Fail
        if self.current_step >= self.max_steps:
            done = True
            reward -= 5.0
            # --- LOGGING FAILURE ---
            print(f"❌ TIMEOUT! Failed to reach target. (Final Dist: {dist:.2f})")
            self.reset_target(player_pos)
            
        self.prev_dist = dist
        
        return brain_state, reward, done, success