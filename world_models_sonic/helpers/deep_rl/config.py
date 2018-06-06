from deep_rl.utils.config import Config as _Config
from deep_rl.utils.normalizer import RescaleNormalizer


class Config(_Config):
    """Modified from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/config.py."""

    def __init__(self):
        super().__init__(self)
        self.train_world_model = True
        self.world_model_batch_size = 1

        self.curiosity = False
        self.intrinsic_reward_normalizer = RescaleNormalizer()
        self.curiosity_only = False
        self.curiosity_weight = 0.1
        self.curiosity_boredom = 0
