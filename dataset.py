from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from rollouts import load_rollouts


class VAEDataset(Dataset):
    def __init__(self, rollout_path):
        super(VAEDataset, self).__init__()
        rollouts = load_rollouts(rollout_path)
        self.images = rollouts['observations']

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.images)

