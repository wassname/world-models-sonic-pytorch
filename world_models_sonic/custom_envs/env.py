import numpy as np
from .retro_sonic import make_sonic, make_sonic256


def make_env(env_name, state=None, seed=-1, render_mode=False):
    if env_name == 'sonic':
        env = make_sonic(tate=state)
    elif env_name == 'sonic256':
        env = make_sonic256(state=state)
    else:
        print("couldn't find this env")
    return env
