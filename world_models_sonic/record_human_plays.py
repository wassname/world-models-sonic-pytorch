"""
Run on linux with "sudo `which python` live.py"
"""

# numeric
import numpy as np

import keyboard
import logging

from .custom_envs.wrappers import discrete_actions, sonic_buttons

# INIT
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def keyboard_to_actions():
    """Transform pressed keys to actions."""

    # set the keys you want to use for actions here
    # see http://www.gadgetreview.com/wp-content/uploads/2015/02/Sega-Genesis-Mk2-6button.jpg
    # to list all key names `import keyboard; keyboard._os_keyboard.build_tables(); keyboard._os_keyboard.from_name.keys()`
    button_names = [
        "5", #"B",
        "4", # A",
        "SHIFT", # "MODE",
        "ENTER", # "START",
        "w", # "UP",
        "s", #"DOWN",
        "a", # "LEFT",
        "d", # "RIGHT",
        "6", # "C",
        "7", #"Y",
        "8", # "X",
        "9", # "Z"
    ]
    button_names = [name.lower() for name in button_names]
    action = np.zeros((12,))
    logger.debug('keyboard._pressed_events %s', keyboard._pressed_events)
    for code, e in keyboard._pressed_events.items():
        if e.event_type == 'down':
            if e.name.lower() in button_names:
                i = button_names.index(e.name)
                action[i] = 1
    return action


def keyboard_to__discrete_actions():
    """Transform pressed keys to actions."""

    # set the keys you want to use for actions here
    # see http://www.gadgetreview.com/wp-content/uploads/2015/02/Sega-Genesis-Mk2-6button.jpg
    # to list all key names `import keyboard; keyboard._os_keyboard.build_tables(); keyboard._os_keyboard.from_name.keys()`
    action = []
    logger.debug('keyboard._pressed_events %s', keyboard._pressed_events)
    for code, e in keyboard._pressed_events.items():
        if e.event_type == 'down':
            if e.name.upper() in sonic_buttons:
                action.append(e.name.upper())


    if action in discrete_actions:
        return discrete_actions.index(action)
    elif action[::-1] in discrete_actions:
        return discrete_actions.index(action[::-1])
    # Now I want jump to overide right
    elif 'B' in action:
        return discrete_actions.index(['B'])
    else:
        return 0


    return action
