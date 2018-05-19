from .retro_sonic import make_sonic


def make_env(env_name, state=None, game=None, seed=-1, render_mode=False, slow=False):
    if env_name == 'sonic':
        env = make_sonic(state=state, game=game, slow=slow)
    elif env_name == 'sonic256':
        env = make_sonic(state=state, game=game, slow=slow, image_size=256)
    else:
        print("couldn't find this env")
    return env
