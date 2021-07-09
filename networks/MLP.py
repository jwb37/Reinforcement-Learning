import torch.nn as nn

from Params import Params

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = [
            nn.Sequential(
                nn.Linear( in_size, out_size ),
                nn.ReLU(),
                nn.Dropout(p=0.4)
            )
            for (in_size, out_size) in zip(layer_sizes[:-2], layer_sizes[1:-1])
        ]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]

        self.net = nn.Sequential( *layers )

    def forward(self, x):
        return self.net(x)



class MLPTail(nn.Module):
    def __init__(self, conv_enc, enc_vec_size, num_actions):
        super().__init__()
        self.enc = conv_enc
        self.mlp = MLP( [enc_vec_size] + Params.network['hidden_layers'] + [num_actions] )

    def forward(self, x):
        x = self.enc(x)
        return self.mlp(x)
