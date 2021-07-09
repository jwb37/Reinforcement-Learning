import torch
#from types import SimpleNamespace
from epsilon.Linear import LinearScheduler


save_name = "Rubiks_DeepQ"


num_episodes = 30000
discount_rate = 0.995
device = torch.device('cuda')


environment = 'RubiksCube'
model = 'DeepQ'
network = {
    'tail': 'ValAdvantage',
    'hidden_layers': [512]
}


initial_lr = 4e-5
batch_size = 40
double_dqn = True


epsilon = LinearScheduler(
    0.8,                    # Initial Epsilon Value
    0.05,                   # Final Epsilon Value
    num_episodes // 2       # Number of episodes to decay over
)

exp_buffer_size = 10000


update_target_net_freq = 200
first_target_net_update = 400
update_working_net_freq = 10
first_working_net_update = 40


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
