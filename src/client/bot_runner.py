import socket
import struct
import pickle
import time
import sensors   # Your sensor module
import actuators # Your actuator module
from src.common.config import HOST, PORT

import minescript

def run():
    minescript.echo("🤖 Universal Client Waiting...")
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                minescript.echo("✅ Connected! Waiting for orders...")

                # 1. RECEIVE CONFIGURATION
                header = s.recv(4)
                if not header: break
                msg_len = struct.unpack('>I', header)[0]
                
                data = b''
                while len(data) < msg_len:
                    data += s.recv(min(msg_len - len(data), 4096))
                
                config_packet = pickle.loads(data)
                
                model_name = config_packet['model_name']
                sensor_reqs = config_packet['input_config']
                action_map = config_packet['action_list']
                
                minescript.echo(f"⚙️ Configured for: {model_name}")
                
                # 2. CONFIGURE SENSORS
                # We pass the requirements to the sensor module
                sensors.configure(sensor_reqs)
                
                # 3. CONFIGURE ACTUATORS
                # We tell actuators what "Action 0" means for this session
                actuators.set_action_map(action_map)

                # 4. START LOOP
                while True:
                    # The sensor module now knows exactly what to return based on config
                    full_state = sensors.get_dynamic_state()
                    
                    # Send State
                    packet = pickle.dumps({'state': full_state})
                    s.sendall(struct.pack('>I', len(packet)))
                    s.sendall(packet)
                    
                    # Receive Action
                    data = s.recv(1024)
                    if not data: break
                    action_idx = pickle.loads(data)
                    
                    # Execute
                    actuators.execute_action(action_idx)
                    
                    time.sleep(0.05)

        except Exception as e:
            minescript.echo(f"⚠️ Error: {e}")
            time.sleep(3)