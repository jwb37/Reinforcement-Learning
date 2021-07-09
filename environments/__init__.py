import importlib

def load_env(env_name):
    env_module = importlib.import_module('.RLEnv', package=f"environments.{env_name}")
    env_class = getattr(env_module, 'RLEnv')

    env = env_class()

    verify_env(
        env,
        env_name,
        req_methods = ['read_state', 'prepare_testing', 'prepare_training', 'reset', 'update'],
        req_attributes = ['actions', 'state_dims']
    )

    return env


def load_interactive_env(env_name):
    game_module = importlib.import_module(".InteractiveGame", package=f"environments.{env_name}")
    game_class = getattr(game_module, 'InteractiveGame')

    game = game_class()

    verify_env(
        game,
        env_name,
        req_methods = ['start']
    )

    return game


def verify_env(env, env_name, req_methods=[], req_attributes=[]):
    try:
        for method in req_methods:
            assert callable(getattr(env, method))
        for attr in req_attributes:
            assert getattr(env, attr)
    except AssertionError:
        print( f"Environment {env_name} must implement the following methods:" )
        print( "\n".join(map(lambda m: f"\t{m}", req_methods)) )
        print( "and the following attributes:" )
        print( "\n".join(map(lambda m: f"\t{a}", req_attributes)) )
        raise
