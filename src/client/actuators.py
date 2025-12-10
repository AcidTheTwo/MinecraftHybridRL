'''
GLOBAL STORAGE FOR ALL POSSIBLE ACTIONS
Stores all possible actions and what to do for each action
Stored as DICTIONARY
'''

ACTIONS = [
    'STOP', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT',
    'JUMP', 'SNEAK', 'SPRINT', 'ATTACK', 'USE',
    'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN',
    'DROP', 'SWAP_HANDS',
    'SLOT_1', 'SLOT_2', 'SLOT_3', 'SLOT_4', 'SLOT_5', 
    'SLOT_6', 'SLOT_7', 'SLOT_8', 'SLOT_9'
]

import minescript

# ==========================================
# CONFIGURATION STATE
# ==========================================
CURRENT_ACTION_MAP = [] # Will be populated by Server (e.g. ['FORWARD', 'JUMP'])
CURRENT_MOVE_STATE = "STOP"

def set_action_map(action_list):
    """
    Receives the list of actions this model is allowed to use.
    Index 0 in the list corresponds to Neural Output 0.
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
    if not CURRENT_ACTION_MAP:
        return # Not configured yet
        
    if action_idx < 0 or action_idx >= len(CURRENT_ACTION_MAP):
        return # Out of bounds
        
    # 2. Decode
    action_name = CURRENT_ACTION_MAP[action_idx]
    
    # 3. Reset Momentum Keys (Impulses)
    # We release these immediately so they don't get stuck on
    minescript.press_key_bind("key.jump", False)
    minescript.press_key_bind("key.attack", False)
    minescript.press_key_bind("key.use", False)
    
    # 4. Handle Movement (Sticky Keys)
    # These keys stay pressed until a new movement command overwrites them
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
        # Sprint is usually a toggle or hold, treating as sticky modifier
        minescript.press_key_bind("key.sprint", True)
        
    elif action_name == 'SNEAK':
        minescript.press_key_bind("key.sneak", True)

    elif action_name == 'ATTACK':
        minescript.press_key_bind("key.attack", True)
        
    elif action_name == 'USE':
        minescript.press_key_bind("key.use", True)
        
    # 6. Re-assert current movement
    # (Prevents stuttering if another key release interfered)
    if CURRENT_MOVE_STATE == 'FORWARD': minescript.press_key_bind("key.forward", True)
    elif CURRENT_MOVE_STATE == 'BACKWARD': minescript.press_key_bind("key.back", True)
    elif CURRENT_MOVE_STATE == 'LEFT': minescript.press_key_bind("key.left", True)
    elif CURRENT_MOVE_STATE == 'RIGHT': minescript.press_key_bind("key.right", True)

def _release_all_movement():
    minescript.press_key_bind("key.forward", False)
    minescript.press_key_bind("key.back", False)
    minescript.press_key_bind("key.left", False)
    minescript.press_key_bind("key.right", False)
    minescript.press_key_bind("key.sprint", False)
    minescript.press_key_bind("key.sneak", False)
    
