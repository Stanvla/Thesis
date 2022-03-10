# %%
import math
import os
import pickle
from typing import TypeVar, Optional
import numpy as np
from icecream import ic
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from hubert.clustering.torch_mffc_extract import ParCzechDataset

T_co = TypeVar('T_co', covariant=True)


class ParCzechPretrain(ParCzechDataset):
    def __init__(self, df_path, km_labels, resample_rate=16000, clean_params=None, label_path=None, sep='\t'):
        super(ParCzechPretrain, self).__init__(df_path, resample_rate, clean_params, sep=sep, sort=False)

        self.label_path = label_path
        with open(os.path.join(self.label_path, 'mp3_to_int.pickle'), 'rb') as f:
            self.mp3_to_int = pickle.load(f)
        self.df['folder_int'] = self.df.mp3.astype(str).map(self.mp3_to_int)
        # folder names are numbers starting from 0
        self.current_folder = '-1'
        self.current_targ_df = None
        self.km_labels = [f'km{k}' for k in km_labels]
        self.labels = km_labels

    def get_labels(self, i):
        mp3 = self.df.iloc[i].mp3.astype(str)
        mp3_folder = f'{self.mp3_to_int[mp3]}'
        if mp3_folder != self.current_folder:
            self.get_df_from_folder(mp3_folder)
            self.current_folder = mp3_folder
        segment = self.df.iloc[i].segment_path
        segment = '/'.join(segment.split('/')[-2:])
        result_df = self.current_targ_df[self.current_targ_df.segm == segment].sort_values(by=['id'])
        return result_df

    def get_df_from_folder(self, folder_path):
        dfs = []
        files = os.listdir(os.path.join(self.label_path, folder_path))
        for f in sorted(files):
            if f.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(self.label_path, folder_path, f)))
        self.current_targ_df = pd.concat(dfs).drop(columns=['mp3'])
        self.current_targ_df['id'] = self.current_targ_df.path.str.split('/').str[-1].astype(int)

    def __getitem__(self, i):
        labels = self.get_labels(i)
        return dict(
            wave=self.get_wav(self.extract_path(i)),
            target={k: torch.from_numpy(labels[km].values) for km, k in zip(self.km_labels, self.labels)},
            idx=i,
        )


class DistributedSampler(torch.utils.data.Sampler[T_co]):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with `torch.nn.parallel.DistributedDataParallel`.
    In such a case, each process can pass a `torch.utils.data.DistributedSampler` instance
    as a `torch.utils.data.DataLoader` sampler, and load a subset of the original dataset that is exclusive to it.

    Args:
        dataset: Dataset used for sampling.

        num_replicas (int, optional): Number of processes participating in distributed training.
            By default, `world_size` is retrieved from the current distributed group.

        rank (int, optional): Rank of the current process within `num_replicas`.
            By default, `rank` is retrieved from the current distributed group.

        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the indices.

        seed (int, optional): random seed used to shuffle the sampler if `shuffle=True`.
            This number should be identical across all processes in the distributed group. Default: ``0``.

        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of replicas.
            If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the `set_epoch` method
        at the beginning of each epoch **before** creating the `DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs.
        Otherwise, the same ordering will be always used.

    """

    def __init__(
            self,
            dataset: ParCzechDataset,
            batch_size: int,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = False,
    ) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.batch_size = batch_size

        self.num_replicas = num_replicas
        self.rank = rank

        self.epoch = 0
        self.drop_last = drop_last

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def _get_indices(self):
        # todo:
        # 	1. indices should be ordered by the dict
        # 	2. sort by length inside the bins
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = []
            for lst in self.dataset.mp3_to_int.values():
                indices.extend(lst)

        return indices

    def __iter__(self):
        indices = self._get_indices()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                # now some random elements will be represented twice
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible,
            # here it is assumed that len(indices) < self.total_size
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When `shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        """
        self.epoch = epoch


class Collator:
    def __init__(self, classes, padding):
        self.classes = classes
        self.pad_value = padding

    def pad_list(self, tensor_list):
        lens = [t.shape[-1] for t in tensor_list]
        max_len = max(lens)
        result = torch.stack([
            F.pad(t, (0, max_len - t.shape[-1]), value=self.pad_value) for t in tensor_list
        ])
        return result, torch.tensor(lens), max_len

    def __call__(self, batch):
        # batch is a list of dicts, with keys [wave, target]
        # batch[i]['target'] is also a dict with keys given by different k from k-means models

        indices = [x['idx'] for x in batch]
        wavs = [x['wave'] for x in batch]
        padded_waves, wav_lens, max_len = self.pad_list(wavs)
        padded_waves = padded_waves.view(len(batch), max_len)

        # targets_by_k[k] is a list of tensors, it can be viewed as unpadded batch
        targets_by_k = {}
        for k in self.classes:
            lst_k = [x['target'][k] for x in batch]
            targets_by_k[k] = lst_k

        # use only first element from pad_list() since it returns multiple things
        padded_targets = {k: self.pad_list(lst)[0] for k, lst in targets_by_k.items()}

        return dict(
            waves=padded_waves,
            lens=wav_lens,
            targets=padded_targets,
            idx=torch.tensor(indices)
        )


class ParCzechPretrainPL(pl.LightningDataModule):
    def __init__(self, clean_params, data_path, km_labels, labels_path, num_workers, num_gpus, pin_mem, shuffle, batch_size, ignore_index):
        super(ParCzechPretrainPL, self).__init__()
        self.clean_params = clean_params
        self.data_path = data_path
        self.km_labels = km_labels
        self.labels_path = labels_path
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.pin_mem = pin_mem
        self.shuffle = shuffle
        self.bs = batch_size
        self.ignore_index = ignore_index

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_data = ParCzechPretrain(
            df_path=self.data_path,
            km_labels=self.km_labels,
            clean_params=self.clean_params,
            label_path=self.labels_path,
            sep=','
        )
        self.collate_fn = Collator(self.km_labels, self.ignore_index)

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.bs,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers / self.num_gpus if self.num_gpus != 0 else self.num_workers,
            pin_memory=self.pin_mem
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

# %%
if __name__ == '__main__':
    # %%

    # ic.configureOutput(includeContext=True)
    # ............................................... Parameters .......................................................
    df_path = '/lnet/express/work/people/stankov/alignment/subset/subset.csv'
    labels_path = '/lnet/express/work/people/stankov/alignment/clustering_all/segments'
    parczech_clean_params = dict(
        recognized_sound_coverage__segments_lb=0.45,
        recognized_sound_coverage__segments_ub=0.93,
        duration__segments_lb=0.5,
    )
    params = dict(
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        ignore_index=-1,
        epochs=50,
        mask_weight=1,
        softmax_temp=0.1,
        overfit_batches=2,
        sim=True,
        reduction='mean',
        shuffle=False,
        num_gpus=4,
        batch_scale=50,
    )

    ks = [15]

    dataset = ParCzechPretrain(
        df_path,
        ks,
        clean_params=parczech_clean_params,
        label_path=labels_path,
        sep=',',
    )
    # %%
    effective_batch = params['num_gpus'] * params['batch_size']
    large_batch_size = params['batch_scale'] * effective_batch

    folders = dataset.df.folder_int.unique()
    all_batches = []
    all_folders = []

    generator = torch.Generator()
    generator.manual_seed(1)

    for f in folders:
        folder_durations = dataset.df[dataset.df.folder_int == f].duration__segments
        folder_indices = folder_durations.index.values
        # shuffle the folder_indices
        folder_shuffled_index = torch.randperm(len(folder_indices), generator=generator).tolist()
        folder_indices = folder_indices[folder_shuffled_index]

        folder_subsets = []
        folder_batches = []
        folder_ub_large = math.ceil(len(folder_indices) / large_batch_size)

        # create subsets of the size large_batch_size
        for i in range(folder_ub_large):
            # the subset will be sorted by duration
            subset_indices = folder_indices[i * large_batch_size: min(len(folder_indices), (i + 1) * large_batch_size)]
            subset_durations = folder_durations[subset_indices]
            subset_durations = subset_durations.sort_values()
            subset_indices = subset_durations.index

            ub_batch = math.ceil(len(subset_durations) / effective_batch)
            # now from subset_indices get effective batch and shuffle the batch
            for j in range(ub_batch):
                batch_indices = subset_indices[j * effective_batch: min(len(subset_durations), (j+1) * effective_batch)]
                # shuffle batch indices, so the batch is not sorted by the length
                batch_indices = batch_indices[torch.randperm(len(batch_indices), generator=generator).tolist()]

                folder_batches.append(batch_indices)

        all_batches.append(folder_batches)
        all_folders.append(folder_indices)
    # %%
    # %%










    # %%