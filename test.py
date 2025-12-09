import minescript
import socket
import pickle
import time
import struct
import re
import hashlib
import math

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
    'SLOT_5', 'SLOT_6', 'SLOT_7', 'SLOT_8', 'SLOT_9',
    'OPEN_INVENTORY', 'CLOSE_GUI', 'LOOT_ALL', 'DEPOSIT_ALL',
    'GUI_NEXT', 'GUI_PREV', 'GUI_UP', 'GUI_DOWN', 'GUI_CLICK', 'GUI_SHIFT_CLICK'
]

# GLOBAL STATE
CURRENT_MOVE_STATE = "STOP" 
PREV_HEALTH = 20.0
PREV_HUNGER = 20.0
PREV_XP = 0.0
VIRTUAL_CURSOR_INDEX = 0

# ==========================================
# HELPERS
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
    # FIX: Safety check for None
    if nbt_str is None or nbt_str == "": return default
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
# SENSORS (Pure Python)
# ==========================================

def get_full_state():
    
    # 1. World State
    addX = (BOX_LENGTH) // 2
    addZ = (BOX_WIDTH) // 2
    with minescript.tick_loop:
        p = minescript.player()
        pos = p.position 
        min_pos = (int(pos[0]) - addX, int(pos[1]) - 1, int(pos[2]) - addZ)
        max_pos = (int(pos[0]) + addX, int(pos[1]) + 1, int(pos[2]) + addZ)
        region = minescript.get_block_region(min_pos, max_pos)
        world_vec = [get_block_hash(b) for b in region.blocks]

    # 2. Physical Stats
    with minescript.tick_loop:
        p = minescript.player()
        health = (getattr(p, 'health', 20.0) or 20.0) / 20.0
        
        # Ensure NBT is not None before passing
        nbt_data = p.nbt if p.nbt else ""
        
        hunger = get_nbt_value(nbt_data, "foodLevel", 20.0) / 20.0
        saturation = get_nbt_value(nbt_data, "foodSaturationLevel", 0.0) / 20.0
        air = get_nbt_value(nbt_data, "Air", 300.0) / 300.0
        
        main_hash, off_hash = 0.0, 0.0
        hands = minescript.player_hand_items()
        if hands:
            if hands.main_hand: main_hash = get_block_hash(hands.main_hand.item)
            if hands.off_hand: off_hash = get_block_hash(hands.off_hand.item)

        inv_items = minescript.player_inventory()
        occupied = sum(1 for stack in inv_items if stack.slot < 36)
        inv_fullness = occupied / 36.0

        yaw = (p.yaw % 360) / 360.0
        pitch = (p.pitch + 90) / 180.0
        gui_state = get_block_hash(minescript.screen_name())

        on_ground = get_nbt_value(nbt_data, "OnGround", 1.0)
        
        vel_y = 0.0
        if p.velocity: 
            vel_y = max(-1.0, min(1.0, p.velocity[1])) 

        xp_level = get_nbt_value(nbt_data, "XpLevel", 0.0) / 30.0
        xp_progress = get_nbt_value(nbt_data, "XpP", 0.0)
        exhaustion = get_nbt_value(nbt_data, "foodExhaustionLevel", 0.0) / 4.0

        is_burning = 1.0 if get_nbt_value(nbt_data, "Fire", -20.0) > 0 else 0.0
        is_hurt = 1.0 if get_nbt_value(nbt_data, "HurtTime", 0.0) > 0 else 0.0
        is_gliding = get_nbt_value(nbt_data, "FallFlying", 0.0)
        cursor_pos = VIRTUAL_CURSOR_INDEX / 90.0

        phys_vec = [health, hunger, saturation, air, main_hash, off_hash, inv_fullness,
                 yaw, pitch, gui_state, on_ground, vel_y, xp_level, xp_progress, 
                 exhaustion, is_burning, is_hurt, is_gliding, cursor_pos]

    # 3. Inventory State
    inv_vec = [0.0] * 36
    items = minescript.player_inventory()
    for stack in items:
        if stack.slot < 36: inv_vec[stack.slot] = get_block_hash(stack.item)

    # 4. Ender Chest
    ender_vec = [0.0] * 27
    with minescript.tick_loop:
        p = minescript.player()
        nbt_data = p.nbt if p.nbt else ""
        match = re.search(r"EnderItems:\[(.*?)\]", nbt_data)
        if match:
            item_matches = re.finditer(r"Slot:(\d+)b.*?id:\"(minecraft:[\w_]+)\"", match.group(1))
            for m in item_matches:
                slot = int(m.group(1))
                if 0 <= slot < 27: ender_vec[slot] = get_block_hash(m.group(2))

    # 5. Container State
    container_vec = [0.0] * 54
    items = minescript.container_get_items()
    if items:
        for stack in items:
            if stack.slot < 54: container_vec[stack.slot] = get_block_hash(stack.item)

    # 6. Environment
    info = minescript.world_info()
    if not info: env_vec = [0.0] * 3
    else: env_vec = [(info.day_ticks % 24000) / 24000.0, 1.0 if info.raining else 0.0, 1.0 if info.thundering else 0.0]

    # 7. Entities
    VISION_RANGE = 20.0
    with minescript.tick_loop:
        p = minescript.player()
        h_vec = [0.0]*4
        hostiles = minescript.entities(type="zombie|skeleton|creeper|spider|witch|ender_man|slime", 
                                     sort="nearest", limit=1, max_distance=VISION_RANGE)
        if hostiles:
            h_vec[:3] = get_relative_pos(p.position, hostiles[0].position, VISION_RANGE)
            h_vec[3] = get_block_hash(hostiles[0].type)

        p_vec = [0.0]*4
        passives = minescript.entities(type="pig|cow|sheep|chicken|item", 
                                     sort="nearest", limit=1, max_distance=VISION_RANGE)
        if passives:
            p_vec[:3] = get_relative_pos(p.position, passives[0].position, VISION_RANGE)
            p_vec[3] = get_block_hash(passives[0].type)
    ent_vec = h_vec + p_vec

    # Combine
    return phys_vec + inv_vec + ender_vec + container_vec + env_vec + ent_vec + world_vec

def calculate_reward(state_vector):
    global PREV_HEALTH, PREV_HUNGER, PREV_XP
    
    curr_health = state_vector[0] * 20.0
    curr_hunger = state_vector[1] * 20.0
    curr_xp = state_vector[12] * 30.0 + state_vector[13]

    reward = 0.0
    reward += 0.01 * (curr_health / 20.0)
    delta_health = curr_health - PREV_HEALTH
    if delta_health < 0: reward += delta_health * 2.0
    
    delta_hunger = curr_hunger - PREV_HUNGER
    if delta_hunger < 0: reward += delta_hunger * 0.5
    elif delta_hunger > 0: reward += delta_hunger * 1.0
        
    delta_xp = curr_xp - PREV_XP
    if delta_xp > 0: reward += delta_xp * 5.0
        
    reward -= 0.005
    PREV_HEALTH, PREV_HUNGER, PREV_XP = curr_health, curr_hunger, curr_xp
    return reward

def execute_action(action_idx):
    global CURRENT_MOVE_STATE, VIRTUAL_CURSOR_INDEX
    if action_idx < 0 or action_idx >= len(ACTIONS): return
    act = ACTIONS[action_idx]
    
    minescript.player_press_attack(False)
    minescript.player_press_use(False)
    minescript.player_press_jump(False)

    if act in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP']:
        if act != CURRENT_MOVE_STATE:
            minescript.player_press_forward(False)
            minescript.player_press_backward(False)
            minescript.player_press_left(False)
            minescript.player_press_right(False)
            CURRENT_MOVE_STATE = act
            if act == 'FORWARD': minescript.player_press_forward(True)
            if act == 'BACKWARD': minescript.player_press_backward(True)
            if act == 'LEFT': minescript.player_press_left(True)
            if act == 'RIGHT': minescript.player_press_right(True)
            
    elif act == 'JUMP': minescript.player_press_jump(True)
    elif act == 'SNEAK': minescript.player_press_sneak(True)
    elif act == 'SPRINT': minescript.player_press_sprint(True)
    elif act == 'ATTACK': minescript.player_press_attack(True)
    elif act == 'USE': minescript.player_press_use(True)
        
    elif act in ['TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN']:
        with minescript.tick_loop:
            p = minescript.player()
            new_yaw, new_pitch = p.yaw, p.pitch
            if act == 'TURN_LEFT': new_yaw -= 15
            if act == 'TURN_RIGHT': new_yaw += 15
            if act == 'LOOK_UP': new_pitch -= 15
            if act == 'LOOK_DOWN': new_pitch += 15
            minescript.player_set_orientation(new_yaw, max(-90, min(90, new_pitch)))
            
    elif act == 'DROP': minescript.player_drop_item(False)
    elif act == 'SWAP_HANDS': minescript.player_swap_hands()
    elif act.startswith('SLOT_'):
        minescript.player_set_held_item_slot(int(act.split('_')[1]) - 1)
        
    elif act == 'OPEN_INVENTORY': minescript.execute("key(key.keyboard.e)")
    elif act == 'CLOSE_GUI': minescript.execute("key(key.keyboard.escape)")
    elif act == 'LOOT_ALL':
        for i in range(54): minescript.inventory_click(i, 0, 1)
    elif act == 'DEPOSIT_ALL':
        for i in range(54, 90): minescript.inventory_click(i, 0, 1)
        
    elif act == 'GUI_NEXT': VIRTUAL_CURSOR_INDEX = (VIRTUAL_CURSOR_INDEX + 1) % 90
    elif act == 'GUI_PREV': VIRTUAL_CURSOR_INDEX = (VIRTUAL_CURSOR_INDEX - 1) % 90
    elif act == 'GUI_UP': 
        if VIRTUAL_CURSOR_INDEX >= 9: VIRTUAL_CURSOR_INDEX -= 9
    elif act == 'GUI_DOWN': 
        if VIRTUAL_CURSOR_INDEX < 81: VIRTUAL_CURSOR_INDEX += 9
    elif act == 'GUI_CLICK': minescript.inventory_click(VIRTUAL_CURSOR_INDEX, 0, 0)
    elif act == 'GUI_SHIFT_CLICK': minescript.inventory_click(VIRTUAL_CURSOR_INDEX, 0, 1)

    if CURRENT_MOVE_STATE == 'FORWARD': minescript.player_press_forward(True)
    elif CURRENT_MOVE_STATE == 'BACKWARD': minescript.player_press_backward(True)
    elif CURRENT_MOVE_STATE == 'LEFT': minescript.player_press_left(True)
    elif CURRENT_MOVE_STATE == 'RIGHT': minescript.player_press_right(True)

# ==========================================
# MAIN LOOP
# ==========================================

def run():
    minescript.echo("🤖 Agent Started (Crash-Proof Version)")
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