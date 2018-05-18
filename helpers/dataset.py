import dask
from dask.callbacks import Callback
from tqdm import tqdm_notebook
import torch.utils.data
import numpy as np
import dask.array as da
import h5py
import os
import torch.utils.data
import glob

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


@dask.delayed
def load_npz(filename, shuffle=False):
    data_f = np.load(filename)
    data = [v for _, v in data_f.items()][0]
    data = np.concatenate(data, 0)

    # shuffle it
    if shuffle:
        inds = np.arange(len(data))
        np.random.shuffle(inds)
        data = data[inds]
    return data


def load_npzs(filenames, shuffle=False):
    lazy_values = [load_npz(url, shuffle=shuffle) for url in filenames]     # Lazily evaluate imread on each url

    sample = lazy_values[0].compute()
    arrays = [da.from_delayed(lazy_value,           # Construct a small Dask array
                              dtype=sample.dtype,   # for every lazy value
                              shape=sample.shape)
              for lazy_value in lazy_values]
    print(sample.shape)
    stack = da.concatenate(arrays, axis=0)
    del sample
    return stack


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
    
    


def load_cache_data(basedir='/data/vae', env_name='sonic256', data_cache_file='/tmp/sonic_vae.hdf5', image_size=256, chunksize=10, action_dim=12, seq_len=1, batch_size=16):

    if not os.path.isfile(data_cache_file):
        filenames = sorted(glob.glob(os.path.join(basedir, 'obs_data_' + env_name + '_*.npz')))
        filenames_actions = sorted(glob.glob(os.path.join(basedir, 'action_data_' + env_name + '_*.npz')))
        filenames_rewards = sorted(glob.glob(os.path.join(basedir, 'reward_data_' + env_name + '_*.npz')))
        filenames_dones = sorted(glob.glob(os.path.join(basedir, 'done_data_' + env_name + '_*.npz')))
        assert len(filenames)==len(filenames_actions)
        assert len(filenames)==len(filenames_rewards)
        assert len(filenames)==len(filenames_dones)
        data_train = load_npzs(filenames, shuffle=False)
        with TQDMDaskProgressBar():
            da.to_hdf5(data_cache_file, '/x', data_train)

        y_train = load_npzs(filenames_actions, shuffle=False)
        with TQDMDaskProgressBar():
            da.to_hdf5(data_cache_file, '/actions', y_train)

        r_train = load_npzs(filenames_rewards, shuffle=False)
        with TQDMDaskProgressBar():
            da.to_hdf5(data_cache_file, '/rewards', r_train)

        d_train = load_npzs(filenames_dones, shuffle=False)
        with TQDMDaskProgressBar():
            da.to_hdf5(data_cache_file, '/dones', d_train)
        print(data_train, y_train, r_train, d_train)
        
    # load
    observations = da.from_array(h5py.File(data_cache_file, mode='r')['x'], chunks=(chunksize, image_size, image_size, 3))
    actions = da.from_array(h5py.File(data_cache_file, mode='r')['actions'], chunks=(chunksize, action_dim)).astype(np.uint8)
    rewards = da.from_array(h5py.File(data_cache_file, mode='r')['rewards'], chunks=(chunksize, ))[:, None].astype(np.float32)
    dones = da.from_array(h5py.File(data_cache_file, mode='r')['dones'], chunks=(chunksize, ))[:, None].astype(np.uint8)
    data_split = int(len(observations)*0.8)
    
    # put cached data into pytorch loaders
    dataset_train = NumpyDataset(observations[:data_split], actions[:data_split], rewards[:data_split], dones[:data_split])
    loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=SequenceInChunkSampler(dataset_train, seq_len=seq_len, chunksize=chunksize),
        pin_memory=True, 
        shuffle=False, 
        batch_size=batch_size*seq_len, 
        drop_last=True
    )

    dataset_test = NumpyDataset(observations[data_split:], actions[data_split:], rewards[data_split:], dones[data_split:])
    loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        sampler=SequenceInChunkSampler(dataset_test, seq_len=seq_len, chunksize=chunksize),
        pin_memory=True, 
        shuffle=False, 
        batch_size=batch_size*seq_len, 
        drop_last=True
    )

    return loader_train, loader_test
