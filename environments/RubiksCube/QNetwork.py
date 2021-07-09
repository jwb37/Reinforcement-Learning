import torch
import torch.nn.functional as F

from Cube import Cube

class QNetwork(torch.nn.Module):
    def __init__(self, cube):
        super(QNetwork,self).__init__()

        input_size = cube.blocks.size * 6
        output_size = len(cube.actions)

        hidden_layer_sizes = [100, 64, 64, 32]

        self.input_layer = torch.nn.Linear( input_size, hidden_layer_sizes[0] )
        self.hidden_layers = torch.nn.ModuleList()
        for i, j in zip(hidden_layer_sizes, hidden_layer_sizes[1:]):
            self.hidden_layers.append( torch.nn.Linear(i, j) )
        self.output_layer = torch.nn.Linear( hidden_layer_sizes[-1], output_size )


    def to(self, device):
        self.input_layer = self.input_layer.to(device)
        for layer in self.hidden_layers:
            layer = layer.to(device)
        self.output_layer = self.output_layer.to(device)


    def forward(self, x):
        x = F.elu( self.input_layer(x) )

        for layer in self.hidden_layers:
            x = F.elu( layer(x) )

        x = self.output_layer(x)
        return x


    def action_values(self, state):
        return self.forward(state)

    def max_qval(self, state):
        return torch.max(self.forward(state), 1).values

    def best_action_idx(self, state):
        return torch.argmax(self.forward(state))

    def best_action(self, state):
        return Cube.actions[ self.best_action_idx(state).item() ]
