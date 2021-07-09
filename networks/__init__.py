import torch.nn as nn

from Params import Params

from .Image import StdConvEncoder
from .Inceptionv1 import InceptionConvEncoder

from .MLP import MLPTail
from .ActorCritic import ActorCritic
from .ValAdvantage import ValAdvantage


ImgEncoders = {
    'Standard':  StdConvEncoder,
    'Inception': InceptionConvEncoder
}

TailNetworks = {
    'ValAdvantage': ValAdvantage,
    'ActorCritic':  ActorCritic,
    'MLP':          MLPTail
}


def load_network(env):
    num_actions = len(env.actions)

    if len(env.state_dims) == 1:
        # Linear state vector
        encoder = nn.Identity()
        enc_vec_size = env.state_dims[0]

    elif len(env.state_dims) == 3:
        # State is an image

        in_ch = env.state_dims[0]
        spatial_dims = env.state_dims[1:3]

        if not 'encoder' in Params.network:
            err_string = "Error: Environments which output an image require a defined encoder network"
            err_string += "Your params file should contain a network dict containing an 'encoder' key"
            err_string += f"\nPlease edit your parameters file ({Params.params_file_path})"
            raise ValueError(err_string)

        if not Params.network['encoder'] in ImgEncoders:
            err_string = f"Error: Network encoder '{Params.network['encoder']}' not recognised"
            err_string += "\nParams file must define a network encoder that matches one of the following:\n"
            err_string += "\n".join(map(lambda k: f"\t{k}", ImgEncoders.keys()))
            err_string += f"\nPlease edit your parameters file ({Params.params_file_path})"
            raise ValueError(err_string)

        encoder = ImgEncoders[Params.network['encoder']](in_ch)
        enc_vec_size = encoder.find_output_size(spatial_dims)


    if not 'tail' in Params.network:
        err_string = "Error: Your params file must include a network dict containing a 'tail' key"
        err_string += f"\nPlease edit your parameters file ({Params.params_file_path})"
        raise ValueError(err_string)

    if not Params.network['tail'] in TailNetworks:
        err_string = f"Error: Network encoder '{Params.network['tail']}' not recognised"
        err_string += "\nParams file must define a tail encoder that matches one of the following:\n"
        err_string += "\n".join(map(lambda k: f"\t{k}", TailNetworks.keys()))
        err_string += f"\nPlease edit your parameters file ({Params.params_file_path})"
        raise ValueError(err_string)

    tail_net_class = TailNetworks[
        Params.network['tail']
    ]
    return tail_net_class(encoder, enc_vec_size, num_actions)
