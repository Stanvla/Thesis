# %%
import math
import os
import pickle
from typing import Optional, Iterable
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from icecream import ic
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

# depending on how the script is executed, interactively or not, use different path for importing
if not sys.__stdin__.isatty():
    # interactive shell
    from hubert.clustering.torch_mffc_extract import ParCzechDataset
else:
    from clustering.torch_mffc_extract import ParCzechDataset


class ParCzechPretrain(ParCzechDataset):
    def __init__(self, df_path, km_labels, resample_rate=16000, filters=None, label_path=None, sep='\t'):
        super(ParCzechPretrain, self).__init__(df_path, resample_rate, df_filters=filters, sep=sep, sort=False)

        self.label_path = label_path
        with open(os.path.join(self.label_path, 'mp3_to_int.pickle'), 'rb') as f:
            self.mp3_to_int = pickle.load(f)
        self.df['folder_int'] = self.df.mp3.astype(str).map(self.mp3_to_int)
        self.df.reset_index(drop=True, inplace=True)
        # folder names are numbers starting from 0
        self.current_folder = '-1'
        self.current_targ_df = None
        self.km_labels = [f'km{k}' for k in km_labels]
        self.labels = km_labels

    def set_folders_in_list(self, folder_list):
        print(f'using folders {folder_list}')
        self.df = self.df[self.df['folder_int'].isin(folder_list)]
        self.df.reset_index(drop=True, inplace=True)

    def set_folders_not_in_list(self, folder_list):
        print(f'excluding folders {folder_list}')
        self.df = self.df[~self.df['folder_int'].isin(folder_list)]
        self.df.reset_index(drop=True, inplace=True)

    def get_folders(self):
        return sorted(list(set(self.mp3_to_int.values())))

    def get_labels(self, i):
        mp3 = self.df.iloc[i].mp3.astype(str)
        mp3_folder = f'{self.mp3_to_int[mp3]}'
        if mp3_folder != self.current_folder:
            ic(mp3_folder)
            self.get_df_from_folder(mp3_folder)
            self.current_folder = mp3_folder
        # get segment name in form `mp3/segm_index`
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
        self.current_targ_df.drop(columns=['path'], inplace=True)

    def __getitem__(self, i):
        labels = self.get_labels(i)
        wave = self.get_wav(self.extract_path(i))
        return dict(
            wave=wave,
            target={k: torch.from_numpy(labels[km].values) for km, k in zip(self.km_labels, self.labels)},
            # idx=i,
            # duration=self.df.iloc[i].duration__segments
        )


class DurationBucketedFolderAwareDistributedSampler(Iterable):
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
            batch_scale: int,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = False,
    ) -> None:
        super(DurationBucketedFolderAwareDistributedSampler, self).__init__()
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
        self.effective_batch_size = batch_size * num_replicas
        self.scaled_batch_size = self.effective_batch_size * batch_scale

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

        ic(self.num_replicas)
        ic(self.rank)
        ic(self.num_samples)
        ic(shuffle)

    def _get_indices(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        folders = self.dataset.df.folder_int.unique()
        all_batch_indices = []
        folder_idx_enlarge = 0
        padding_size = 0

        if not self.drop_last:
            # get folder index that will be enlarged
            if self.shuffle:
                folder_idx_enlarge = torch.randint(len(folders), (1,), generator=generator).item()
            padding_size = self.total_size - len(self.dataset)

        for idx, f in enumerate(folders):
            folder_durations = self.dataset.df[self.dataset.df.folder_int == f].duration__segments
            folder_indices = folder_durations.index.values

            # shuffle the folder_indices
            if self.shuffle:
                folder_shuffled_index = torch.randperm(len(folder_indices), generator=generator).tolist()
                folder_indices = folder_indices[folder_shuffled_index]

            # add extra samples to make dataset evenly divisible
            if idx == folder_idx_enlarge and not self.drop_last and padding_size != 0:
                ic(len(folder_indices), padding_size, folder_indices[:padding_size])
                if padding_size <= len(folder_indices):
                    folder_indices = np.concatenate([folder_indices, folder_indices[:padding_size]])
                else:
                    # np.tile repeats the array n times
                    folder_indices = np.tile(folder_indices, math.ceil(padding_size / len(folder_indices)))[:padding_size]
                ic(len(folder_indices), padding_size, folder_indices[:padding_size])

            folder_ub_large = math.ceil(len(folder_indices) / self.scaled_batch_size)
            # create subsets of the size self.scaled_batch_size, inside these subsets sort audio segments by length
            for i in range(folder_ub_large):
                # the subset will be sorted by duration
                subset_indices = folder_indices[i * self.scaled_batch_size: min(len(folder_indices), (i + 1) * self.scaled_batch_size)]
                subset_durations = folder_durations[subset_indices]
                subset_durations = subset_durations.sort_values()
                subset_indices = subset_durations.index

                ub_batch = math.ceil(len(subset_indices) / self.effective_batch_size)
                # divide the subset into batches and shuffle audio segments inside the batch
                for j in range(ub_batch):
                    batch_indices = subset_indices[j * self.effective_batch_size: min(len(subset_indices), (j + 1) * self.effective_batch_size)]
                    # shuffle batch indices, so the batch is not sorted by the length
                    if self.shuffle:
                        batch_shuffled_index = torch.randperm(len(batch_indices), generator=generator).tolist()
                        batch_indices = batch_indices[batch_shuffled_index]

                    all_batch_indices.extend(batch_indices)

        if self.drop_last:
            # remove tail of data to make it evenly divisible,
            # here it is assumed that len(indices) < self.total_size
            all_batch_indices = all_batch_indices[:self.total_size]

        assert len(all_batch_indices) == self.total_size
        return all_batch_indices

    def __iter__(self):
        indices = self._get_indices()
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
        # indices = [x['idx'] for x in batch]
        return dict(
            waves=padded_waves,
            lens=wav_lens,
            targets=padded_targets,
            # idx=torch.tensor(indices)
        )


class ParCzechPretrainPL(pl.LightningDataModule):
    def __init__(
            self,
            parczech_filters,
            data_path,
            km_labels,
            labels_path,
            num_workers,
            num_gpus,
            pin_mem,
            shuffle,
            batch_size,
            batch_scale,
            ignore_index,
            seed,
            drop_last,
            val_frac,
    ):
        super(ParCzechPretrainPL, self).__init__()
        self.parczech_filters = parczech_filters
        self.data_path = data_path
        self.km_labels = km_labels
        self.labels_path = labels_path
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.pin_mem = pin_mem
        self.shuffle = shuffle
        self.bs = batch_size
        self.batch_scale = batch_scale
        self.ignore_index = ignore_index
        self.seed = seed
        self.drop_last = drop_last

        self.train_data = ParCzechPretrain(
            df_path=self.data_path,
            km_labels=self.km_labels,
            filters=self.parczech_filters,
            label_path=self.labels_path,
            sep=','
        )
        self.val_data = ParCzechPretrain(
            df_path=self.data_path,
            km_labels=self.km_labels,
            filters=self.parczech_filters,
            label_path=self.labels_path,
            sep=','
        )
        folders = self.train_data.get_folders()
        val_folders_cnt = min(1, math.floor(len(folders) * val_frac))
        val_folders = folders[:val_folders_cnt]

        self.val_data.set_folders_in_list(val_folders)
        self.train_data.set_folders_not_in_list(val_folders)
        self.collate_fn = Collator(self.km_labels, self.ignore_index)

    def _get_loader(self, data, sampler):
        return DataLoader(
            data,
            batch_size=self.bs,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers // self.num_gpus if self.num_gpus != 0 else self.num_workers,
            pin_memory=self.pin_mem,
            sampler=sampler,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_sampler = DurationBucketedFolderAwareDistributedSampler(
            dataset=self.train_data,
            batch_size=self.bs,
            batch_scale=self.batch_scale,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        return self._get_loader(self.train_data, train_sampler)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_sampler = DurationBucketedFolderAwareDistributedSampler(
            dataset=self.val_data,
            batch_size=self.bs,
            batch_scale=self.batch_scale,
            shuffle=False,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        val_loader = self._get_loader(self.val_data, val_sampler)
        return val_loader


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
    # %%
