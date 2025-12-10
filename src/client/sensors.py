'''
GLOBAL STORAGE FOR ALL POSSIBLE INPUTS
Stores whatever possible inputs the game can provide as a config
Stored as DICTIONARY of config options
'''
import numpy as np

import math
import minescript

# ==========================================
# CONFIGURATION STATE
# ==========================================
# This dictionary is updated by the server at runtime.
CURRENT_CONFIG = {
    "local_voxels": False,   # 3x3x3 grid
    "entity_radar": False,   # Enemy positions
    "self_stats": True,      # Health, Hunger, etc.
    "raw_coordinates": True  # XYZ, Yaw, Pitch
}

def configure(new_config):
    """Updates the sensor configuration based on Server orders."""
    global CURRENT_CONFIG
    # Only update keys that exist to prevent trash data
    for key in new_config:
        CURRENT_CONFIG[key] = new_config[key]
    minescript.echo(f"Sensors Configured: {CURRENT_CONFIG}")

# ==========================================
# MAIN STATE BUILDER
# ==========================================
def get_dynamic_state():
    """
    Constructs the state vector dynamically based on CURRENT_CONFIG.
    Returns a flattened numpy array.
    """
    data_chunks = []
    
    # 1. Self Stats (Health, Velocity, etc.)
    if CURRENT_CONFIG.get("self_stats", True):
        data_chunks.append(get_physical_stats())
        
    # 2. Raw Coordinates (For Server-side math)
    if CURRENT_CONFIG.get("raw_coordinates", True):
        p = minescript.player()
        coords = np.array([p.position[0], p.position[1], p.position[2], p.yaw, p.pitch])
        data_chunks.append(coords)

    # 3. Local Voxels (Vision)
    if CURRENT_CONFIG.get("local_voxels", False):
        data_chunks.append(get_local_voxels())

    # 4. Entity Radar
    if CURRENT_CONFIG.get("entity_radar", False):
        data_chunks.append(get_entity_radar())

    # Concatenate all enabled chunks into one vector
    return np.concatenate(data_chunks).astype(np.float32)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_physical_stats():
    """Returns [Health(0-1), Hunger(0-1), VelX, VelY, VelZ, OnGround]"""
    p = minescript.player()
    
    # Velocity (Clamped)
    vx = max(-1.0, min(1.0, p.velocity[0]))
    vy = max(-1.0, min(1.0, p.velocity[1]))
    vz = max(-1.0, min(1.0, p.velocity[2]))
    
    # Health/Hunger (We use NBT or attribute if available, simplified here)
    # Note: Minescript 4.0+ allows p.health
    health = getattr(p, 'health', 20.0) / 20.0
    
    # OnGround (Approximation using velocity if NBT unavailable)
    on_ground = 1.0 if vy == 0 else 0.0
    
    return np.array([health, vx, vy, vz, on_ground])

def get_local_voxels(radius=1):
    """
    Returns a flattened 3x3x3 grid (27 floats).
    1.0 = Solid, 0.0 = Air, -1.0 = Dangerous
    """
    p = minescript.player()
    center = [int(p.position[0]), int(p.position[1]), int(p.position[2])]
    voxels = []
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                # Check block at (x+dx, y+dy, z+dz)
                # Minescript block check:
                b_pos = (center[0]+dx, center[1]+dy, center[2]+dz)
                block = minescript.get_block(b_pos)
                
                name = block.name.lower()
                val = 1.0 # Default Solid
                
                if "air" in name or "water" in name:
                    val = 0.0
                elif "lava" in name or "fire" in name or "magma" in name:
                    val = -1.0
                    
                voxels.append(val)
                
    return np.array(voxels)

def get_entity_radar(range=10):
    """
    Returns [RelX, RelY, RelZ, TypeHash, Health] of nearest hostile.
    """
    p = minescript.player()
    entities = minescript.entities(type="zombie|skeleton|creeper|spider", sort="nearest", limit=1, max_distance=range)
    
    if not entities:
        return np.zeros(5)
    
    e = entities[0]
    
    # Relative Position
    rx = (e.position[0] - p.position[0]) / range
    ry = (e.position[1] - p.position[1]) / range
    rz = (e.position[2] - p.position[2]) / range
    
    # Simple Hash (Just checking existence vs nothing)
    # You can expand this hash function later
    type_hash = 0.5 
    
    hp = getattr(e, 'health', 20.0) / 20.0
    
    return np.array([rx, ry, rz, type_hash, hp])