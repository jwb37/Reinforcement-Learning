import torch.nn as nn

from Params import Params
from .MLP import MLP


class ValAdvantage(nn.Module):
    def __init__(self, conv_enc, enc_vec_size, num_actions):
        super().__init__()

        self.conv = conv_enc

        self.value = MLP( [enc_vec_size, *Params.network['hidden_layers'], 1] )
        self.advantage = MLP( [enc_vec_size, *Params.network['hidden_layers'], num_actions] )


    def forward(self, state):
        enc_state = self.conv(state)
        val = self.value(enc_state)
        adv = self.advantage(enc_state)

        return val + adv - adv.mean() 
