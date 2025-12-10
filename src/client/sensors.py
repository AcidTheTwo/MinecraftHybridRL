import numpy as np
import math

# ==========================================
# CONFIGURATION STATE
# ==========================================
CURRENT_CONFIG = {
    "local_voxels": False,   
    "entity_radar": False,   
    "self_stats": True,      
    "raw_coordinates": True  
}

# ⚠️ MINESCRIPT IMPORT LAST
import minescript

def configure(new_config):
    global CURRENT_CONFIG
    for key in new_config:
        CURRENT_CONFIG[key] = new_config[key]
    minescript.echo(f"Sensors Configured: {CURRENT_CONFIG}")

# ==========================================
# MAIN STATE BUILDER
# ==========================================
def get_dynamic_state():
    """
    Constructs the state vector dynamically.
    """
    data_chunks = []
    
    # 1. Self Stats [Size: 5]
    if CURRENT_CONFIG.get("self_stats", True):
        data_chunks.append(get_physical_stats())
        
    # 2. Raw Coordinates [Size: 5]
    if CURRENT_CONFIG.get("raw_coordinates", True):
        p = minescript.player()
        coords = np.array([p.position[0], p.position[1], p.position[2], p.yaw, p.pitch])
        data_chunks.append(coords)

    # 3. Local Voxels [Size: 27]
    if CURRENT_CONFIG.get("local_voxels", False):
        data_chunks.append(get_local_voxels())

    # 4. Entity Radar [Size: 5]
    if CURRENT_CONFIG.get("entity_radar", False):
        data_chunks.append(get_entity_radar())

    return np.concatenate(data_chunks).astype(np.float32)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_physical_stats():
    p = minescript.player()
    vx = max(-1.0, min(1.0, p.velocity[0]))
    vy = max(-1.0, min(1.0, p.velocity[1]))
    vz = max(-1.0, min(1.0, p.velocity[2]))
    
    # Simple Health Check
    health = getattr(p, 'health', 20.0) / 20.0
    on_ground = 1.0 if vy == 0 else 0.0
    
    return np.array([health, vx, vy, vz, on_ground])

def get_local_voxels(radius=1):
    p = minescript.player()
    center = [int(p.position[0]), int(p.position[1]), int(p.position[2])]
    voxels = []
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                b_pos = (center[0]+dx, center[1]+dy, center[2]+dz)
                block = minescript.get_block(b_pos[0], b_pos[1], b_pos[2])
                
                name = block.lower()
                val = 1.0 # Default Solid
                
                if "air" in name or "water" in name:
                    val = 0.0
                elif "lava" in name or "fire" in name or "magma" in name:
                    val = -1.0 # Danger
                    
                voxels.append(val)
                
    return np.array(voxels)

def get_entity_radar(range=10):
    p = minescript.player()
    entities = minescript.entities(type="zombie|skeleton|creeper|spider", sort="nearest", limit=1, max_distance=range)
    
    if not entities:
        return np.zeros(5)
    
    e = entities[0]
    rx = (e.position[0] - p.position[0]) / range
    ry = (e.position[1] - p.position[1]) / range
    rz = (e.position[2] - p.position[2]) / range
    type_hash = 0.5 
    hp = getattr(e, 'health', 20.0) / 20.0
    return np.array([rx, ry, rz, type_hash, hp])