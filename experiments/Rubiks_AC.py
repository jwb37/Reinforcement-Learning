import torch


save_name = "Rubiks_AC"


num_episodes = 40000
discount_rate = 0.995
device = torch.device('cuda')


environment = 'RubiksCube'
model = 'ActorCritic'
network = {
    'tail': 'ActorCritic',
    'hidden_layers': [512]
}


initial_lr = 4e-5


# Weightings for actor/critic/entropy regularization losses
lambda_act = 10.0
lambda_crt = 5.0
lambda_reg = 0.0
entropy_threshold = 10.0


# Output options
print_freq = 100
save_freq = 1000
save_best = True
min_reward_to_save = -1000.0
save_video = False


#-------------------------------------------------------
# Environment-specific options

RewardInfo = {
    'solved':   1,
    'stuck':    0,
    'move':     0
}

cube_one_hot = True     # Block colour encoding - int or one-hot
cube_side_length = 3
cube_max_steps = 100
cube_min_difficulty = 1
cube_max_difficulty = 10
