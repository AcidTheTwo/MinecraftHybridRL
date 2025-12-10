import torch
import torch.nn as nn
import os

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

def save_brain(model, optimizer, filename="data/checkpoints/brain.pth"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, filename)
    print(f"💾 Brain saved to {filename}")

def smart_load_brain(model, optimizer, input_size, filename="data/checkpoints/brain.pth"):
    if not os.path.exists(filename):
        print("🌱 New brain created.")
        return

    try:
        checkpoint = torch.load(filename)
        saved_input = checkpoint['model_state']['network.0.weight'].shape[1]
        
        if saved_input == input_size:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"✅ Loaded saved brain.")
        else:
            print(f"⚠️ Shape mismatch (Saved: {saved_input}, Current: {input_size}). Starting fresh.")
    except Exception as e:
        print(f"❌ Load failed: {e}")