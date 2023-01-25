
import torch
import numpy as np

class linearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class NN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nbr_layers):
        super(NN, self).__init__()
        self.linear_in = torch.nn.Linear(input_size, hidden_size)
        # Create linear layers depending on nbr_layers
        for i in range(nbr_layers - 2):
            setattr(self, f"linear_{i}", torch.nn.Linear(hidden_size, hidden_size))
        self.linear_out = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.nbr_layers = nbr_layers
        
    def forward(self, x):
        x = self.relu(self.linear_in(x))
        # Add linear layers depending on nbr_layers
        for i in range(self.nbr_layers - 2):
            x = self.relu(getattr(self, f"linear_{i}")(x))
        x = self.linear_out(x)
        return x   
    
def init_weights(layer, this_weights, this_bias):
    """
    Initializes weights and bias of linear layers
    """
    if isinstance(layer, torch.nn.Linear):
        layer.weight.data = this_weights
        if layer.bias is not None:
            layer.bias.data.fill_(this_bias)
    