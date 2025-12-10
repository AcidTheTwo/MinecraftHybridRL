import sys
import minescript

# ==========================================
# CONFIGURATION STATE
# ==========================================
CURRENT_ACTION_MAP = [] 
CURRENT_MOVE_STATE = "STOP"

def set_action_map(action_list):
    """
    Receives the list of actions this model is allowed to use.
    """
    global CURRENT_ACTION_MAP
    CURRENT_ACTION_MAP = action_list
    minescript.echo(f"Actuators: Mapped {len(action_list)} actions.")

# ==========================================
# EXECUTION LOGIC
# ==========================================
def execute_action(action_idx):
    global CURRENT_MOVE_STATE
    
    # 1. Validation
    if not CURRENT_ACTION_MAP: return
    if action_idx < 0 or action_idx >= len(CURRENT_ACTION_MAP): return
        
    # 2. Decode
    action_name = CURRENT_ACTION_MAP[action_idx]
    
    # 3. Reset Momentum Keys (Impulses)
    minescript.press_key_bind("key.jump", False)
    minescript.press_key_bind("key.sprint", False)
    minescript.press_key_bind("key.sneak", False)
    
    # 4. Handle Movement (Sticky Keys)
    if action_name in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP']:
        if action_name != CURRENT_MOVE_STATE:
            _release_all_movement()
            CURRENT_MOVE_STATE = action_name
            
            if action_name == 'FORWARD': minescript.press_key_bind("key.forward", True)
            if action_name == 'BACKWARD': minescript.press_key_bind("key.back", True)
            if action_name == 'LEFT': minescript.press_key_bind("key.left", True)
            if action_name == 'RIGHT': minescript.press_key_bind("key.right", True)

    # 5. Handle Abilities (Impulses)
    elif action_name == 'JUMP':
        minescript.press_key_bind("key.jump", True)
        
    elif action_name == 'SPRINT':
        minescript.press_key_bind("key.sprint", True)
        if CURRENT_MOVE_STATE == 'FORWARD':
            minescript.press_key_bind("key.forward", True)
            
    elif action_name == 'SNEAK':
        minescript.press_key_bind("key.sneak", True)

    # --- NEW: HEAD TURNING ---
    elif action_name == 'TURN_LEFT':
        _adjust_look(-15, 0) # Rotate Left 15 degrees
        
    elif action_name == 'TURN_RIGHT':
        _adjust_look(15, 0)  # Rotate Right 15 degrees

    # (Optional: Add LOOK_UP / LOOK_DOWN later if needed)
    
    # 6. Maintain State (Prevent stuttering)
    if CURRENT_MOVE_STATE == 'FORWARD': minescript.press_key_bind("key.forward", True)
    elif CURRENT_MOVE_STATE == 'BACKWARD': minescript.press_key_bind("key.back", True)
    elif CURRENT_MOVE_STATE == 'LEFT': minescript.press_key_bind("key.left", True)
    elif CURRENT_MOVE_STATE == 'RIGHT': minescript.press_key_bind("key.right", True)

def _release_all_movement():
    minescript.press_key_bind("key.forward", False)
    minescript.press_key_bind("key.back", False)
    minescript.press_key_bind("key.left", False)
    minescript.press_key_bind("key.right", False)

def _adjust_look(yaw_delta, pitch_delta):
    """
    Safely modifies the player's orientation.
    """
    p = minescript.player()
    if not p: return

    new_yaw = p.yaw + yaw_delta
    new_pitch = p.pitch + pitch_delta
    
    # Clamp Pitch so we don't flip upside down (-90 to 90)
    new_pitch = max(-90.0, min(90.0, new_pitch))
    
    # Apply
    minescript.player_set_orientation(new_yaw, new_pitch)