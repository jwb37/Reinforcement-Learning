from dataclasses import dataclass

@dataclass
class BaseParams:
    #-------------------------------------------------------------
    # General

    model_name: str
    game_name: str

    num_episodes: int
    starting_episode: int = 0

    checkpoint_path: str = "checkpoints/"

    save_best_models: bool = True
    min_reward_to_save: float = -10000.0
    save_videos: bool = False

    print_summary: bool = True
    print_every_nth_episode: int = 500

    device: str = 'cuda'


    #-------------------------------------------------------------
    # Training parameters

    discount_rate: float = 0.99
    learning_rate: float = 1e-4

    double_dqn: bool = False

    eps_decaytype: EpsilonDecay = EpsilonDecay.Linear
    eps_start: float = 0.5
    eps_final: float = 0.0
    # 0 is not a particularly sensible default value, but dataclass complains if this has no default value,
    # for aesthetic reasons, I don't want to order this away from the other values to do with epsilon values,
    # and this value gets reassigned in the __post_init__ method anyway.
    num_eps_decay_episodes: int = 0

    frame_skip: bool = False
    frame_skip_freq: int = 4
    repeat_action: bool = False

    update_target_model_freq: int = 500
    first_target_model_update: int = 500

    training_batch_size: int = 100
    training_freq: int = 5
    first_training_time: int = 100


#    Possibly implement later, but maybe keep things simple for now!
#    The whole point of writing this framework is so implementations don't HAVE to worry about internals
#    record_model_criterion: Callable[[float,float], bool] = lambda loss,reward: False


    def __post_init__(self):
        self.num_eps_decay_episodes = self.num_episodes
