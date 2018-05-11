import numpy as np
from custom_envs.retro_sonic import make_sonic, make_sonic224


def make_env(env_name, seed=-1, render_mode=False):
    if env_name == 'car_racing':
        env = CarRacing()
        if (seed >= 0):
            env.seed(seed)
    elif env_name == 'sonic':
        env = make_sonic()
    elif env_name == 'sonic224':
        env = make_sonic224()
    else:
        print("couldn't find this env")
    return env
