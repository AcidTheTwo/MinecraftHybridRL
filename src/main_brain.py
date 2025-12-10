import socket
import pickle
import struct
import torch
import torch.optim as optim
import wandb
import random
import sys
import os
import argparse
import time

# --- LOCAL IMPORTS ---
sys.path.append(os.getcwd())

from src.common.config import HOST, PORT, LEARNING_RATE, GAMMA, BATCH_SIZE, MEMORY_SIZE, EPSILON
from src.server.utils.dynamic_loader import load_objective_module
from src.server.models.dqn import MinecraftBrain, smart_load_brain, save_brain
from src.server.training.trainer import ReplayBuffer, train_step

def start_server(model_name):
    # ==========================================
    # 1. SETUP & CARTRIDGE LOADING
    # ==========================================
    NAME = str(model_name).lower()
    print(f"⚙️  Loading Cartridge: {NAME}...")
    
    # Load the logic file dynamically
    cartridge = load_objective_module(model_name)
    ObjectiveClass = cartridge['class']
    active_actions = cartridge['actions']
    input_config   = cartridge['config']
    
    # Instantiate the Objective Engine
    reward_engine = ObjectiveClass()
    
    # Calculate Sizes
    input_size = reward_engine.get_input_shape() 
    output_size = len(active_actions)
    
    print(f"⚙️  Model Configured: {input_size} Inputs -> {output_size} Outputs")

    # ==========================================
    # 2. INITIALIZE BRAIN
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Brain Server initializing on {device}")
    
    current_brain = MinecraftBrain(input_size, output_size).to(device)
    current_optimizer = optim.Adam(current_brain.parameters(), lr=LEARNING_RATE)
    
    # Load weights specifically for THIS objective
    save_path = f"data/checkpoints/brain_{NAME}.pth"
    smart_load_brain(current_brain, current_optimizer, input_size, filename=save_path)

    wandb.init(project="hive-mind-rl", name=NAME, config={
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "batch_size": BATCH_SIZE,
        "architecture": "DQN",
        "task": NAME,
        "input_size": input_size,
        "action_space": len(active_actions)
    })
    
    # ==========================================
    # 3. SERVER LOOP
    # ==========================================
    loss_fn = torch.nn.MSELoss()
    memory = ReplayBuffer(MEMORY_SIZE)
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    
    print(f"🚀 Listening on port {PORT}...")

    # Trackers
    last_state = None
    last_action_idx = None
    step_count = 0
    current_loss = 0.0

    try:
        while True:
            try:
                conn, addr = server.accept()
                print(f"🔗 Connected: {addr}")
                conn.settimeout(None)

                with conn:
                    # --- HANDSHAKE START ---
                    # 1. Send Orders (Config & Actions) to Client
                    handshake_packet = {
                        'model_name': NAME,
                        'input_config': input_config,
                        'action_list': active_actions
                    }
                    data = pickle.dumps(handshake_packet)
                    conn.sendall(struct.pack('>I', len(data)))
                    conn.sendall(data)
                    print(f"📤 Sent Config: {NAME} (Waiting for Client...)")

                    # --- MAIN LOOP ---
                    while True:
                        # 1. Receive State
                        header = conn.recv(4)
                        if not header: break
                        msg_len = struct.unpack('>I', header)[0]

                        data = b''
                        while len(data) < msg_len:
                            chunk = conn.recv(min(msg_len - len(data), 4096))
                            if not chunk: break
                            data += chunk
                        
                        packet = pickle.loads(data)
                        raw_state = packet['state']

                        # 2. Server Logic (Reward & Pre-processing)
                        # The server can inject extra data (like target vector) into the state here
                        processed_state, reward, done, success = reward_engine.process_state(raw_state)

                        current_state_tensor = torch.FloatTensor(processed_state).to(device)

                        # 3. Learning Step
                        if last_state is not None and last_action_idx is not None:
                            memory.push(last_state, last_action_idx, reward, current_state_tensor, False)
                            
                            if len(memory) > BATCH_SIZE:
                                loss = train_step(current_brain, current_optimizer, loss_fn, memory, device)
                                current_loss = loss

                        # 4. Decision Step
                        is_random = False
                        confidence = 0.0
                        
                        if random.random() < EPSILON:
                            action_idx = random.randint(0, len(active_actions) - 1)
                            is_random = True
                        else:
                            with torch.no_grad():
                                q_values = current_brain(current_state_tensor.unsqueeze(0))
                                confidence = torch.max(q_values).item()
                                action_idx = torch.argmax(q_values).item()

                        # 5. Logging
                        if step_count % 10 == 0:
                            wandb.log({
                                "loss": current_loss,
                                "reward": reward,
                                "confidence": confidence,
                                "epsilon": EPSILON
                            })
                            print(f"Step {step_count} | Rw: {reward:.3f} | Conf: {confidence:.2f} | {'RND' if is_random else 'AI'}")

                        # 6. Send Action
                        conn.sendall(pickle.dumps(action_idx))

                        last_state = current_state_tensor
                        last_action_idx = action_idx
                        step_count += 1
                        
                        # Handle Episode End/Reset
                        if done:
                            last_state = None
                            last_action_idx = None
                            # Note: We don't break loop, just reset memory of last step

            except socket.error as e:
                print(f"⚠️  Client Disconnected: {e}")
            except Exception as e:
                print(f"❌ Critical Error: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n🛑 Saving and Exiting...")
        if current_brain:
            save_brain(current_brain, current_optimizer, save_path)
    finally:
        server.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Name of the objective file (e.g. navigator)')
    args = parser.parse_args()
    start_server(args.model)

