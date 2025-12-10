import random
import torch
from collections import deque
from src.common.config import GAMMA, BATCH_SIZE

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train_step(brain, optimizer, loss_fn, memory, device):
    batch = memory.sample(BATCH_SIZE)
    state, action, reward, next_state, done = zip(*batch)

    state = torch.stack(state).to(device)
    next_state = torch.stack(next_state).to(device)
    action = torch.tensor(action).unsqueeze(1).to(device)
    reward = torch.tensor(reward).float().unsqueeze(1).to(device)
    done = torch.tensor(done).float().unsqueeze(1).to(device)

    q_values = brain(state).gather(1, action)

    with torch.no_grad():
        next_q = brain(next_state).max(1)[0].unsqueeze(1)
        target_q = reward + (GAMMA * next_q * (1 - done))

    loss = loss_fn(q_values, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()