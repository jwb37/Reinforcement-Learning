import torch.nn as nn

from Params import Params
from .MLP import MLP


class ActorCritic(nn.Module):
    def __init__(self, conv_enc, enc_vec_size, num_actions):
        super().__init__()

        self.conv = conv_enc

        self.critic_net = MLP( [enc_vec_size, *Params.network['hidden_layers'], 1] )
        self.actor_net = nn.Sequential(
            MLP( [enc_vec_size, *Params.network['hidden_layers'], num_actions] ),
            nn.LogSoftmax(dim=1)
        )

    def actor(self, state):
        enc_state = self.conv(state)
        return self.actor_net(enc_state)

    def critic(self, state):
        enc_state = self.conv(state)
        return self.critic_net(enc_state)

    def forward(self, state):
        enc_state = self.conv(state)
        a = self.actor_net(enc_state)
        c = self.critic_net(enc_state)

        return (a, c)
