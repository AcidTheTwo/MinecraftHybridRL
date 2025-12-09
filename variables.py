# variables.py

# ==========================================
# NETWORK CONFIGURATION
# ==========================================
HOST = '127.0.0.1'  # Localhost
PORT = 65432

# ==========================================
# AGENT CONFIGURATION
# ==========================================
BOX_LENGTH = 5   # Width of the vision box (Blocks)
BOX_WIDTH = 5    # Depth of the vision box (Blocks)

# ==========================================
# BRAIN CONFIGURATION
# ==========================================
SAVE_FILE = "brain.pth"
LEARNING_RATE = 0.001

# ==========================================
# ACTION SPACE (SHARED)
# ==========================================
# This MUST be identical for both Agent and Server
ACTIONS = [
    'STOP', 
    'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT',
    'JUMP', 'SNEAK', 'SPRINT',
    'ATTACK', 'USE',
    'TURN_LEFT', 'TURN_RIGHT',
    'LOOK_UP', 'LOOK_DOWN',
    'DROP', 'SWAP_HANDS',
    'SLOT_1', 'SLOT_2', 'SLOT_3', 'SLOT_4',
    'SLOT_5', 'SLOT_6', 'SLOT_7', 'SLOT_8', 'SLOT_9',
    'OPEN_INVENTORY',
    'CLOSE_GUI',
    'LOOT_ALL',      # Take everything from chest
    'DEPOSIT_ALL',   # Put everything in chest
    'GUI_NEXT',       # Move cursor right
    'GUI_PREV',       # Move cursor left
    'GUI_UP',         # Jump up a row
    'GUI_DOWN',       # Jump down a row
    'GUI_CLICK',      # Normal Click (Pick up/Place)
    'GUI_SHIFT_CLICK' # Quick Move / Equip
    
]