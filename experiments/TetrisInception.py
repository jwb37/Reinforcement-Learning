import torch
#from types import SimpleNamespace
from epsilon.Linear import LinearScheduler


save_name = "Tetris_DeepQ_Inception"


num_episodes = 30000
discount_rate = 0.995
device = torch.device('cuda')

environment = 'Tetris'
model = 'DeepQ'
network = {
    'encoder': 'Inception',
    'tail': 'ValAdvantage',
    'hidden_layers': [512]
}

image_dims = (128, 128)
greyscale = True


initial_lr = 4e-5
batch_size = 40
double_dqn = True


epsilon = LinearScheduler(
    0.8,                    # Initial Epsilon Value
    0.05,                   # Final Epsilon Value
    num_episodes // 2       # Number of episodes to decay over
)

exp_buffer_size = 10000


update_target_net_freq = 2000
first_target_net_update = 1000
update_working_net_freq = 10
first_working_net_update = 500


# Output options
print_freq = 100
save_freq = 5000
save_best = True
min_reward_to_save = -1000.0
save_video = True


RewardInfo = {
    'ActionPenalty': 0,
    'PenalisedActions': [],
    'GameOver': -30,
    'ScoreScaling': 4,
    'PieceLanded': 2,
    'HoleCreated': -5,
    'MaxHeightChange': (lambda old_height, new_height: 0 )      # Disable max height reward
}
