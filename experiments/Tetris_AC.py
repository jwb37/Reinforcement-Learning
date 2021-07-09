import torch


save_name = "Tetris_AC"


num_episodes = 40000
discount_rate = 0.999
device = torch.device('cuda')


environment = 'Tetris'
model = 'ActorCritic'
network = {
    'encoder': 'Standard',
    'tail': 'ActorCritic',
    'hidden_layers': [512]
}

image_dims = (128, 128)
greyscale = True


initial_lr = 5e-5


# Weightings for actor/critic/entropy regularization losses
lambda_act = 10.0
lambda_crt = 5.0
lambda_reg = 1.0
entropy_threshold = 0.4


# Output options
print_freq = 100
save_freq = 5000
save_best = True
min_reward_to_save = 2.0
save_video = True


RewardInfo = {
    'ActionPenalty': 0,
    'PenalisedActions': [],
    'GameOver': 0,
    'ScoreScaling': 5,
    'PieceLanded': 2,
    'HoleCreated': -3,
    'MaxHeightChange': (lambda old_height, new_height: (old_height-new_height) )
}
