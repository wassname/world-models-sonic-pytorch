from .retro_sonic import make_sonic


def make_env(env_name, state=None, game=None, seed=-1, render_mode=False, slow=False, to_gray=False, image_size=256, max_episode_steps=4000):
    if env_name == 'sonic':
        env = make_sonic(state=state, game=game, slow=slow, image_size=image_size, to_gray=to_gray, max_episode_steps=max_episode_steps)
    elif env_name == 'sonic256':
        env = make_sonic(state=state, game=game, slow=slow, image_size=256, max_episode_steps=max_episode_steps)
    else:
        print("couldn't find this env")
    return env
