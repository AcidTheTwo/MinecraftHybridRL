import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from collections import deque

# ==========================================
# 1. CONFIGURATION
# ==========================================
HOST = '0.0.0.0' 
PORT = 65432
SAVE_FILE = "brain.pth"

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.9      # Importance of future rewards (0.9 = high foresight)
EPSILON = 0.3    # Exploration rate (30% random actions initially)
BATCH_SIZE = 32
MEMORY_SIZE = 2000

# Action Space (Must match Agent exactly)
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

# ==========================================
# 2. NEURAL NETWORK
# ==========================================

class MinecraftBrain(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. HELPERS (Save/Load/Train)
# ==========================================

def save_brain(model, optimizer, filename=SAVE_FILE):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, filename)
    print(f"💾 Brain saved to {filename}")

def smart_load_brain(model, optimizer, input_size, filename=SAVE_FILE):
    if not os.path.exists(filename):
        print("🌱 New brain created (No save file found).")
        return

    try:
        checkpoint = torch.load(filename)
        # Check if input size matches (index 1 of weight matrix)
        saved_input_size = checkpoint['model_state']['network.0.weight'].shape[1]
        
        if saved_input_size == input_size:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"✅ Loaded saved brain (Inputs: {saved_input_size})")
        else:
            print(f"⚠️  MISMATCH! Saved: {saved_input_size} inputs, Current: {input_size}")
            print("♻️  Discarding old brain. Starting FRESH.")
            
    except Exception as e:
        print(f"❌ Error loading brain: {e}")
        print("♻️  Starting FRESH.")

memory = deque(maxlen=MEMORY_SIZE)

def train_step(brain, optimizer, loss_fn, device):
    if len(memory) < BATCH_SIZE:
        return 0.0 

    # 1. Sample Batch
    batch = random.sample(memory, BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)

    # 2. Convert to Tensors
    state_tensor = torch.stack(state_batch).to(device)
    next_state_tensor = torch.stack(next_state_batch).to(device)
    action_tensor = torch.tensor(action_batch).unsqueeze(1).to(device)
    reward_tensor = torch.tensor(reward_batch).float().unsqueeze(1).to(device)

    # 3. Compute Q(s, a)
    q_values = brain(state_tensor).gather(1, action_tensor)

    # 4. Compute Target Q
    with torch.no_grad():
        next_q_values = brain(next_state_tensor).max(1)[0].unsqueeze(1)
        target_q_values = reward_tensor + (GAMMA * next_q_values)

    # 5. Optimize
    loss = loss_fn(q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# ==========================================
# 4. MAIN SERVER LOOP
# ==========================================

def start_server():
    # Detect GPU (cuda) or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Brain Server initializing on {device}...")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(1.0) # Check for Ctrl+C every second
    
    print(f"🚀 Listening on port {PORT}...")
    print(f"📋 Action Space: {len(ACTIONS)} outputs")
    print(f"💡 Press Ctrl+C to save and exit.")

    current_brain = None
    current_optimizer = None
    current_loss = 0
    loss_fn = nn.MSELoss()

    # Tracking variables
    last_state_tensor = None
    last_action_idx = None
    step_count = 0

    try:
        while True:
            try:
                # 1. Accept Connection
                conn, addr = server.accept()
                print(f"🔗 Connected: {addr}")
                conn.settimeout(None)

                with conn:
                    # --- HANDSHAKE ---
                    data = conn.recv(4)
                    if not data: continue
                    input_size = struct.unpack('>I', data)[0]
                    print(f"📏 Handshake: Agent reports {input_size} inputs.")
                    
                    # Initialize Model
                    current_brain = MinecraftBrain(input_size, len(ACTIONS)).to(device)
                    current_optimizer = optim.Adam(current_brain.parameters(), lr=LEARNING_RATE)
                    smart_load_brain(current_brain, current_optimizer, input_size)

                    while True:
                        # --- RECEIVE PACKET ---
                        header = conn.recv(4)
                        if not header: break
                        msg_len = struct.unpack('>I', header)[0]
                        
                        data = b''
                        while len(data) < msg_len:
                            chunk = conn.recv(min(msg_len - len(data), 4096))
                            if not chunk: break
                            data += chunk
                        if len(data) != msg_len: break
                        
                        # --- PARSE STATE ---
                        packet = pickle.loads(data)
                        state_raw = packet['state']
                        reward = packet['reward']
                        
                        # Handle List vs Numpy
                        if isinstance(state_raw, list):
                            current_state_tensor = torch.tensor(state_raw).float().to(device)
                        else:
                            current_state_tensor = torch.from_numpy(state_raw).float().to(device)

                        # --- LEARNING STEP ---
                        if last_state_tensor is not None and last_action_idx is not None:
                            memory.append((last_state_tensor, last_action_idx, reward, current_state_tensor))
                            
                            # Train every 10 steps
                            if step_count % 1 == 0:
                                loss = train_step(current_brain, current_optimizer, loss_fn, device)
                                current_loss = loss
                            else:
                                current_loss = 0.0

                        # --- DECISION STEP ---
                        is_random = False
                        
                        if random.random() < EPSILON:
                            # Random Exploration
                            action_idx = random.randint(0, len(ACTIONS) - 1)
                            is_random = True
                            confidence = 0.0 
                        else:
                            # AI Inference
                            with torch.no_grad():
                                action_logits = current_brain(current_state_tensor.unsqueeze(0))
                                confidence = torch.max(action_logits).item() 
                                action_idx = torch.argmax(action_logits).item()
                        
                        # --- LOGGING ---
                        if step_count % 5 == 0:
                            # Get Health (Index 0 is Health %)
                            if isinstance(state_raw, list):
                                hp = state_raw[0] * 20.0
                            else:
                                hp = state_raw[0] * 20.0

                            act_name = ACTIONS[action_idx]
                            mode_str = "*RND*" if is_random else " AI "

                            print(f"📉 Loss: {current_loss:.4f} | Rw: {reward:+.3f} | HP: {hp:4.1f} | {mode_str} -> {act_name:<15} | Q: {confidence:.2f}")

                        # --- SEND RESPONSE ---
                        conn.sendall(pickle.dumps(action_idx))
                        
                        # Update trackers
                        last_state_tensor = current_state_tensor
                        last_action_idx = action_idx
                        step_count += 1

            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ Connection Error: {e}")
                if current_brain and current_optimizer:
                    save_brain(current_brain, current_optimizer)

    except KeyboardInterrupt:
        print("\n🛑 Stop signal received!")
        if current_brain and current_optimizer:
            save_brain(current_brain, current_optimizer)
        print("✅ Exiting gracefully.")
        
    finally:
        server.close()

if __name__ == "__main__":
    start_server()