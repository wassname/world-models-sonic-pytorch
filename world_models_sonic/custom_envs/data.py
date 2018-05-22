import pandas as pd

train_states = pd.read_csv('./data/sonic_env/sonic-train.csv')
validation_states = pd.read_csv('./data/sonic_env/sonic-validation.csv')

sonic_buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
discrete_actions = [[], ['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                    ['DOWN', 'B'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B']]