import torch.utils.data.sampler
import numpy as np


class SequenceInChunkSampler(torch.utils.data.sampler.Sampler):
    """
    Samples sequences of elements sequentially, but random sequences in a chunk.
    Arguments:
        data_source (Dataset): dataset to sample from
        seq_len (int): length of sequential sequences
        chunksize (int): length of cached data to take random sequences from

    url: https://gist.github.com/wassname/8ae1f64389c2aaceeb84fcd34c3651c3
    """

    def __init__(self, data_source, seq_len=6, chunksize=600):
        assert chunksize % seq_len == 0, 'chunk size should be a multiple of seq_len'
        assert len(data_source) > chunksize
        self.data_source = data_source
        self.seq_len = seq_len
        self.chunksize = chunksize

    def __iter__(self):
        chunk_idxs = np.arange(0, len(self.data_source), self.chunksize)
        np.random.shuffle(chunk_idxs)
        
        for chunk_idx in chunk_idxs:
            seqs = np.arange(chunk_idx, min(chunk_idx + self.chunksize, len(self.data_source)), self.seq_len)
            np.random.shuffle(seqs)
            for seq_i in seqs:
                for i in np.arange(seq_i, seq_i + self.seq_len):
                    yield i

    def __len__(self):
        return len(self.data_source)