import socket
import pickle
import time
import struct
import re
import hashlib
import math
import numpy as np

# ⚠️ CRITICAL: minescript must be imported LAST
import minescript 

# ==========================================
# 1. CONFIGURATION
# ==========================================
HOST = '127.0.0.1'
PORT = 65432
BOX_LENGTH = 5
BOX_WIDTH = 5

ACTIONS = [
    'STOP', 
    'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT',
    'JUMP', 'SNEAK', 'SPRINT',
    'ATTACK', 'USE',
    'TURN_LEFT', 'TURN_RIGHT',
    'LOOK_UP', 'LOOK_DOWN',
    'DROP', 'SWAP_HANDS',
    'SLOT_1', 'SLOT_2', 'SLOT_3', 'SLOT_4',
    'SLOT_5', 'SLOT_6', 'SLOT_7', 'SLOT_8', 'SLOT_9'
]

# GLOBAL STATE
CURRENT_MOVE_STATE = "STOP" 
PREV_HEALTH = 20.0
PREV_HUNGER = 20.0
PREV_XP = 0.0
PREV_ENTITY_HEALTH = 20.0
VIRTUAL_CURSOR_INDEX = 0

# ==========================================
# 2. HELPERS
# ==========================================

def get_block_hash(block_name):
    if block_name is None: return 0.0
    name = str(block_name).lower()
    if "air" in name: return 0.0
    hash_object = hashlib.md5(name.encode())
    hex_dig = hash_object.hexdigest()
    hex_chunk = hex_dig[:16]
    int_val = int(hex_chunk, 16)
    return int_val / 18446744073709551615.0

def get_nbt_value(nbt_str, key, default):
    if not nbt_str: return default
    match = re.search(rf"{key}:([\d\.]+)", nbt_str)
    if match:
        return float(match.group(1))
    return default

def get_relative_pos(player_pos, entity_pos, max_dist=20.0):
    if entity_pos is None: return [0.0, 0.0, 0.0]
    dx = (entity_pos[0] - player_pos[0]) / max_dist
    dy = (entity_pos[1] - player_pos[1]) / max_dist
    dz = (entity_pos[2] - player_pos[2]) / max_dist
    return [max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)), max(-1.0, min(1.0, dz))]

# ==========================================
# 3. SENSORS
# ==========================================

def get_world_state():
    addX = (BOX_LENGTH) // 2
    addZ = (BOX_WIDTH) // 2
    with minescript.tick_loop:
        p = minescript.player()
        pos = p.position
        min_pos = (int(pos[0]) - addX, int(pos[1]) - 1, int(pos[2]) - addZ)
        max_pos = (int(pos[0]) + addX, int(pos[1]) + 1, int(pos[2]) + addZ)
        region = minescript.get_block_region(min_pos, max_pos)
        block_vector = [get_block_hash(b) for b in region.blocks]
    return np.array(block_vector, dtype=np.float64)

def get_physical_stats():
    with minescript.tick_loop:
        p = minescript.player(nbt = True)
        health = (getattr(p, 'health', 20.0) or 20.0) / 20.0
        nbt_data = p.nbt if p.nbt else ""
        
        hunger = get_nbt_value(nbt_data, "foodLevel", 20.0) / 20.0
        saturation = get_nbt_value(nbt_data, "foodSaturationLevel", 0.0) / 20.0
        air = get_nbt_value(nbt_data, "Air", 300.0) / 300.0
        
        main_hash, off_hash = 0.0, 0.0
        hands = minescript.player_hand_items()
        if hands:
            if hands.main_hand: main_hash = get_block_hash(hands.main_hand)
            if hands.off_hand: off_hash = get_block_hash(hands.off_hand)

        inv_items = minescript.player_inventory()
        occupied = sum(1 for stack in inv_items if stack.slot < 36)
        inv_fullness = occupied / 36.0

        yaw = (p.yaw % 360) / 360.0
        pitch = (p.pitch + 90) / 180.0
        gui_state = get_block_hash(minescript.screen_name())

        on_ground = get_nbt_value(nbt_data, "OnGround", 1.0)
        vel_y = 0.0
        if p.velocity: vel_y = max(-1.0, min(1.0, p.velocity[1]))

        xp_level = get_nbt_value(nbt_data, "XpLevel", 0.0) / 30.0
        xp_progress = get_nbt_value(nbt_data, "XpP", 0.0)
        exhaustion = get_nbt_value(nbt_data, "foodExhaustionLevel", 0.0) / 4.0

        is_burning = 1.0 if get_nbt_value(nbt_data, "Fire", -20.0) > 0 else 0.0
        is_hurt = 1.0 if get_nbt_value(nbt_data, "HurtTime", 0.0) > 0 else 0.0
        is_gliding = get_nbt_value(nbt_data, "FallFlying", 0.0)
        cursor_pos = VIRTUAL_CURSOR_INDEX / 90.0

        stats = [health, hunger, saturation, air, main_hash, off_hash, inv_fullness,
                 yaw, pitch, gui_state, on_ground, vel_y, xp_level, xp_progress, 
                 exhaustion, is_burning, is_hurt, is_gliding, cursor_pos]
    return np.array(stats, dtype=np.float64)

def get_environment_state():
    info = minescript.world_info()
    if not info: return np.zeros(3, dtype=np.float64)
    return np.array([(info.day_ticks % 24000) / 24000.0, 
                     1.0 if info.raining else 0.0, 
                     1.0 if info.thundering else 0.0], dtype=np.float64)

def get_entity_state():
    VISION_RANGE = 20.0
    with minescript.tick_loop:
        p = minescript.player()
        
        # --- HOSTILES ---
        # 1. We request 'nbt=True' to get health data
        # 2. We ask for 'limit=5' so we can filter out ourselves
        hostiles = minescript.entities(
            type="zombie|skeleton|creeper|spider|witch|ender_man|slime", 
            sort="nearest", 
            limit=5, 
            max_distance=VISION_RANGE,
            nbt=True 
        )
        
        # 3. Filter: Find the first entity that is NOT me
        target_h = None
        for h in hostiles:
            # If distance > 0.5, it's probably not me. 
            # (Or check if h.name != p.name)
            dist = math.sqrt((h.position[0]-p.position[0])**2 + 
                             (h.position[1]-p.position[1])**2 + 
                             (h.position[2]-p.position[2])**2)
            if dist > 0.5:
                target_h = h
                break
        
        # 4. Construct Vector (Now Size 5: x, y, z, hash, HEALTH)
        h_vec = [0.0] * 5
        if target_h:
            h_vec[:3] = get_relative_pos(p.position, target_h.position, VISION_RANGE)
            h_vec[3] = get_block_hash(target_h.type)
            
            # 5. Extract Health (Try attribute first, then NBT)
            hp = getattr(target_h, 'health', None)
            if hp is None:
                hp = get_nbt_value(target_h.nbt, "Health", 20.0)
            h_vec[4] = hp / 20.0  # Normalize (0.0 - 1.0)

        # --- PASSIVES ---
        passives = minescript.entities(
            type="pig|cow|sheep|chicken|item", 
            sort="nearest", 
            limit=5, 
            max_distance=VISION_RANGE,
            nbt=True
        )
        
        target_p = None
        for pas in passives:
            dist = math.sqrt((pas.position[0]-p.position[0])**2 + 
                             (pas.position[1]-p.position[1])**2 + 
                             (pas.position[2]-p.position[2])**2)
            if dist > 0.5:
                target_p = pas
                break

        p_vec = [0.0] * 5
        if target_p:
            p_vec[:3] = get_relative_pos(p.position, target_p.position, VISION_RANGE)
            p_vec[3] = get_block_hash(target_p.type)
            
            hp = getattr(target_p, 'health', None)
            if hp is None:
                hp = get_nbt_value(target_p.nbt, "Health", 10.0)
            p_vec[4] = hp / 20.0

    # Total Vector Size is now larger
    return np.array(h_vec + p_vec, dtype=np.float64)

def get_full_state():
    return np.concatenate((
        get_physical_stats(), get_environment_state(), get_entity_state(), get_world_state()
    ))

def calculate_reward(state_vector):
    
    global PREV_HEALTH, PREV_HUNGER, PREV_XP, PREV_ENTITY_HEALTH
    
    curr_health = state_vector[0] * 20.0
    curr_hunger = state_vector[1] * 20.0
    curr_xp = state_vector[12] * 30.0 + state_vector[13]
    
    reward = 0.0
    if curr_health <= 1:
        reward -= 10
    delta_health = curr_health - PREV_HEALTH
    if delta_health < 0: 
        reward += delta_health * 2.0
    
    delta_hunger = curr_hunger - PREV_HUNGER
    if delta_hunger < 0: 
        reward += delta_hunger * 1.0
    elif delta_hunger > 0: 
        reward += delta_hunger * 2.0
        
    delta_xp = curr_xp - PREV_XP
    if delta_xp > 0: 
        reward += delta_xp * 10.0
        
    reward -= 0.005
    if reward !=-0.005: minescript.echo(f" reward - {reward}")

    PREV_HEALTH, PREV_HUNGER, PREV_XP = curr_health, curr_hunger, curr_xp
    return reward

# ==========================================
# 4. EXECUTORS (KEYBINDS ONLY - STABLE)
# ==========================================

def execute_action(action_idx):
    global CURRENT_MOVE_STATE, VIRTUAL_CURSOR_INDEX
    if action_idx < 0 or action_idx >= len(ACTIONS): return
    act = ACTIONS[action_idx]
    
    # 1. Reset Momentary Keys
    minescript.press_key_bind("key.jump", False)

    # 2. Movement (Sticky)
    if act in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP']:
        if act != CURRENT_MOVE_STATE:
            minescript.press_key_bind("key.forward", False)
            minescript.press_key_bind("key.back", False)
            minescript.press_key_bind("key.left", False)
            minescript.press_key_bind("key.right", False)
            minescript.press_key_bind("key.sneak", False)
            minescript.press_key_bind("key.sprint", False)
            minescript.press_key_bind("key.attack", False)
            minescript.press_key_bind("key.use", False)
            CURRENT_MOVE_STATE = act
            if act == 'FORWARD': minescript.press_key_bind("key.forward", True)
            if act == 'BACKWARD': minescript.press_key_bind("key.back", True)
            if act == 'LEFT': minescript.press_key_bind("key.left", True)
            if act == 'RIGHT': minescript.press_key_bind("key.right", True)
            
    # 3. Agility
    elif act == 'JUMP': minescript.press_key_bind("key.jump", True)
    elif act == 'SNEAK': minescript.press_key_bind("key.sneak", True)
    elif act == 'SPRINT': minescript.press_key_bind("key.sprint", True)
    
    # 4. Interaction
    elif act == 'ATTACK': minescript.press_key_bind("key.attack", True)
    elif act == 'USE': minescript.press_key_bind("key.use", True)
        
    # 5. Camera
    elif act in ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN']:
        with minescript.tick_loop:
            p = minescript.player()
            new_yaw, new_pitch = p.yaw, p.pitch
            if act == 'TURN_LEFT': new_yaw -= 15
            if act == 'TURN_RIGHT': new_yaw += 15
            if act == 'LOOK_UP': new_pitch -= 15
            if act == 'LOOK_DOWN': new_pitch += 15
            minescript.player_set_orientation(new_yaw, max(-90, min(90, new_pitch)))
            
    # 6. Inventory Keys
    elif act == 'DROP': minescript.press_key_bind("key.drop", True)
    elif act == 'SWAP_HANDS': minescript.press_key_bind("key.swapOffhand", True)
    
    # 7. Hotbar (Uses Native Keybinds)
    elif act.startswith('SLOT_'):
        slot_num = int(act.split('_')[1])
        minescript.press_key_bind(f"key.hotbar.{slot_num}", True)

    # 8. DISABLED MACROS (To prevent crashes/stuck GUI)
    

    # Re-Assert Sticky Keys
    if CURRENT_MOVE_STATE == 'FORWARD': minescript.press_key_bind("key.forward", True)
    elif CURRENT_MOVE_STATE == 'BACKWARD': minescript.press_key_bind("key.back", True)
    elif CURRENT_MOVE_STATE == 'LEFT': minescript.press_key_bind("key.left", True)
    elif CURRENT_MOVE_STATE == 'RIGHT': minescript.press_key_bind("key.right", True)

# ==========================================
# 5. MAIN LOOP
# ==========================================

def run():
    minescript.echo("🤖 Agent Started (Stable Hotbar Mode)")
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                minescript.echo("✅ Connected to Brain!")

                initial_state = get_full_state()
                s.sendall(struct.pack('>I', len(initial_state)))
                minescript.echo(f"📏 Sending Size: {len(initial_state)}")
                
                while True:
                    full_state = get_full_state()
                    packet = pickle.dumps({
                        'state': full_state,
                        'reward': calculate_reward(full_state),
                        'done': False
                    })
                    
                    s.sendall(struct.pack('>I', len(packet)))
                    s.sendall(packet)
                    
                    data = s.recv(1024)
                    if not data: break
                    execute_action(pickle.loads(data))
                    time.sleep(0.05)
                    
        except Exception as e:
            minescript.echo(f"⚠️ Error: {e}")
            time.sleep(3)

if __name__ == "__main__":
    run()