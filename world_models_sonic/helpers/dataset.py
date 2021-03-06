import dask
from dask.callbacks import Callback
from tqdm import tqdm_notebook
import torch.utils.data
import numpy as np
import dask.array as da
import os
import zarr
import h5py

from world_models_sonic import config
from .samplers import SequenceInChunkSampler


class TQDMDaskProgressBar(Callback):
    """
    A tqdm progress bar for dask.

    Usage:
        ```
        with TQDMDaskProgressBar():
            da.compute()
        ```

    See: http://dask.pydata.org/en/latest/diagnostics-local.html?highlight=progress
    """

    def __init__(self, start=None, start_state=None, pretask=None, posttask=None, finish=None, **kwargs):
        super().__init__(start=start, start_state=start_state, pretask=pretask, posttask=posttask, finish=finish)
        self.tqdm_args = kwargs

    def _start_state(self, dsk, state):
        self._tqdm = tqdm_notebook(total=sum(len(state[k]) for k in ['ready', 'waiting', 'running', 'finished']), **self.tqdm_args)

    def _posttask(self, key, result, dsk, state, worker_id):
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        self._tqdm.close()


class NumpyDataset(torch.utils.data.Dataset):
    """Dataset wrapping arrays.

    Each sample will be retrieved by indexing array along the first dimension.

    Arguments:
        *arrays (numpy.array): arrays that have the same size of the first dimension.
    """

    def __init__(self, *arrays):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays)
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(array[index].compute() for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]


def load_cache_data(basedir=config.base_vae_data_dir, env_name='sonic256', data_cache_file=None, chunksize=None, action_dim=12, seq_len=1, batch_size=16, image_size=256, pin_memory=True):
    if data_cache_file is None:
        data_cache_file = os.path.join(config.base_vae_data_dir, 'sonic_rollout_cache.hdf5')

    # For some reason it's 20x faster to random read inside chunks from 1 big dask hdf5 file
    # Than to do it directly from zarr arrays. Maybe I should have written to hdf5 from the start
    # but it seems to be the fast the it's one long sequential array
    if not os.path.isfile(data_cache_file):

        # load zarr arrays
        z_obs = zarr.open(os.path.join(basedir, 'obs_data.zarr'), mode='r')
        z_act = zarr.open(os.path.join(basedir, 'action_data.zarr'), mode='r')
        z_don = zarr.open(os.path.join(basedir, 'done_data.zarr'), mode='r')
        z_rew = zarr.open(os.path.join(basedir, 'reward_data.zarr'), mode='r')
        print(z_obs, z_act, z_don, z_rew)

        # Load into dask
        z_obs = (da.from_array(z_obs, (chunksize,) + z_obs.chunks[1:]) / 255.).astype(np.float32)
        z_act = da.from_array(z_act, (chunksize,) + z_act.chunks[1:])
        z_don = da.from_array(z_don, (chunksize,) + z_don.chunks[1:])
        z_rew = da.from_array(z_rew, (chunksize,) + z_rew.chunks[1:])

        if chunksize is None:
            chunksize = z_obs.chunks

        with TQDMDaskProgressBar():
            da.to_hdf5(data_cache_file, {'/x': z_obs, '/actions': z_act, '/rewards': z_rew, '/dones': z_don})

    observations = da.from_array(h5py.File(data_cache_file, mode='r')['x'], chunks=(chunksize, image_size, image_size, 3))
    actions = da.from_array(h5py.File(data_cache_file, mode='r')['actions'], chunks=(chunksize, ))[:, None].astype(np.uint8)
    rewards = da.from_array(h5py.File(data_cache_file, mode='r')['rewards'], chunks=(chunksize, ))[:, None].astype(np.float32)
    dones = da.from_array(h5py.File(data_cache_file, mode='r')['dones'], chunks=(chunksize, ))[:, None].astype(np.uint8)
    print("Loaded from cache", data_cache_file)

    data_split = int(len(observations) * 0.8)
    dataset_train = NumpyDataset(observations[:data_split], actions[:data_split], rewards[:data_split], dones[:data_split])

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=SequenceInChunkSampler(dataset_train, seq_len=seq_len, chunksize=chunksize),
        pin_memory=pin_memory,
        shuffle=False,
        batch_size=batch_size * seq_len,
        drop_last=True
    )

    dataset_test = NumpyDataset(observations[data_split:], actions[data_split:], rewards[data_split:], dones[data_split:])
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=SequenceInChunkSampler(dataset_test, seq_len=seq_len, chunksize=chunksize),
        pin_memory=pin_memory,
        shuffle=False,
        batch_size=batch_size * seq_len,
        drop_last=True
    )

    return loader_train, loader_test
