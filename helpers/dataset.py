import dask
from dask.callbacks import Callback
from tqdm import tqdm_notebook
import torch.utils.data
import numpy as np
import dask.array as da


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
def load_npz(filename):
    data_f = np.load(filename)
    data = [v for _, v in data_f.items()][0]
    data = np.concatenate(data, 0)

    # shuffle it
    inds = np.arange(len(data))
    np.random.shuffle(inds)
    data = data[inds]
    return data


def load_npzs(filenames):
    lazy_values = [load_npz(url) for url in filenames]     # Lazily evaluate imread on each url

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
