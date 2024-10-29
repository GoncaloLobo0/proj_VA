import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple MLP with one hidden layer that works as an individual
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)    # First hidden layer
        self.relu = nn.ReLU()                            # ReLU activation
        self.fc2 = nn.Linear(hidden_size, output_size)   # Output layer
        
    # Forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

    def get_params_numpy(self):
        return np.concatenate([param.detach().numpy().flatten() for param in self.parameters()])

