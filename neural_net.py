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
        #self.softmax = nn.Softmax(dim=1)                      # Softmax for classifiying
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # Forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

    # Returns the parameters in a numpy array
    def get_parameters_numpy(self):
        return np.concatenate([param.detach().numpy().flatten() for param in self.parameters()])
    
    # Loads parameters from a numpy array
    def load_parameters_numpy(self, flat_params):
        # flat_params index
        index = 0
    
        for param in self.parameters():

            num_elements = param.numel()
            param_data = flat_params[index:index + num_elements]
            param_tensor = torch.tensor(param_data.reshape(param.shape), dtype=param.dtype)
            param.data = param_tensor
            index += num_elements

    # Calculates the inverted loss
    def calculate_fitness(self, X, y):
        with torch.no_grad():
            outputs = self(X)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, y)
            
            _, predicted = torch.max(outputs, dim=1)  # Get the predicted class
            correct = (predicted == y).sum().item()   # Count correct predictions
            total = y.size(0)                         # Total number of samples
            accuracy = correct / total

        return (-loss.item(), accuracy)
    
    def train_backprop(self, dataloader):
        
        size = len(dataloader.dataset)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1)
               
        self.train()
        
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = self(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        return loss