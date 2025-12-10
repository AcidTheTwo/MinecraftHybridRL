class SurvivalReward:
    def __init__(self):
        self.prev_health = 20.0
        self.prev_hunger = 20.0
        self.prev_xp = 0.0

    def calculate(self, state_vector):
        # MAPPING (Based on your sensors.py order)
        # 0: Health, 1: Hunger, 12: XP Level, 13: XP Progress
        curr_health = state_vector[0] * 20.0
        curr_hunger = state_vector[1] * 20.0
        curr_xp = state_vector[12] * 30.0 + state_vector[13]

        reward = 0.0

        # Survival Logic
        if curr_health <= 1: reward -= 10.0
        
        delta_health = curr_health - self.prev_health
        if delta_health < 0: reward += delta_health * 2.0 # Ouch!
        
        delta_hunger = curr_hunger - self.prev_hunger
        if delta_hunger > 0: reward += delta_hunger * 2.0 # Yummy
        
        delta_xp = curr_xp - self.prev_xp
        if delta_xp > 0: reward += delta_xp * 10.0 # Smart
        
        # Time penalty (to prevent idling)
        reward -= 0.01

        # Update Memory
        self.prev_health = curr_health
        self.prev_hunger = curr_hunger
        self.prev_xp = curr_xp

        return reward