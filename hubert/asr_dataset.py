# %%
import os
import sys

import pandas as pd
import torch
import torchtext
from collections import OrderedDict
from abc import ABC, abstractmethod
from tqdm import tqdm

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from unidecode import unidecode
from Levenshtein import distance
from num2words import num2words
import pickle
from icecream import ic
import re

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
import numpy as np

# depending on how the script is executed, interactively or not, use different path for importing
try:
    from clustering.torch_mffc_extract import ParCzechDataset
    from clustering.filter_dataframe import FilterLB, FilterUB
except ModuleNotFoundError:
    from hubert.clustering.filter_dataframe import FilterLB, FilterUB
    from hubert.clustering.torch_mffc_extract import ParCzechDataset


class ParCzechTranscript:
    def __init__(self, segment_name):
        self.segment_name = segment_name
        self.recognized_transcript = None
        self.asr_transcript = None
        self.gold_transcript = None
        self.alignment = None

    @staticmethod
    def similarity(trans1, trans2):
        return max(len(trans1), len(trans2)) - distance(trans1, trans2)

    def _find_floats(self,):
        # will find ['13.1', '13. 1', '13 . 1', '13 .1' ]
        return re.findall(r"\d+\s*\.\s*\d+", self.gold_transcript)

    def _expand_floats(self):
        floats = self._find_floats()
        if floats == []:
            return self.asr_transcript
        ic(floats)
        ic(self.gold_transcript)
        ic(self.asr_transcript)
        ic(self.recognized_transcript)
        raise NotImplementedError()

    def normalize_trans(self, abbrev_dict):
        # todo check for floating point numbers
        tmp = self._expand_floats()
        for k, v in abbrev_dict.items():
            tmp = tmp.replace(k, v)

        # floating point can be a number or time
        pass

    def align(self, abbrev_dict):
        pass



class ParCzechSpeechRecDataset(ParCzechDataset):
    def __init__(self, df_path, df_filters, resample_rate=16000, sep='\t'):
        super(ParCzechSpeechRecDataset, self).__init__(df_path, resample_rate, df_filters=df_filters, sep=sep, sort=False)

    def __getitem__(self, i):
        path = self.extract_path(i)
        return [self.index_df(i).segment_path, self.get_recognized_transcript(path, i)]


class ParCzechSpeechRecogPL(pl.LightningDataModule):
    def __init__(self, df_filters, df_path, num_workers, batch_size):
        super(ParCzechSpeechRecogPL, self).__init__()
        self.dataset = ParCzechSpeechRecDataset(df_path, df_filters=df_filters)
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def __len__(self):
        return len(self.dataset)


def display_alignment(align_obj):
    s1, s2, score, start, end = align_obj
    print(f'Score = {score}')
    print(f'{"true":20}', f'{"recongnized":20}')
    print('.'*40)
    for a, b in zip(s1, s2):
        print(f'{a:20}', f'{b:20}')


def normalize(seq):
    # todo normalization for floating point numbers
    new_seq = []
    for x in seq.split():
        new_x = x.lower()
        if new_x in abbreviation_dict:
            new_x = abbreviation_dict[new_x]
        if x.isdigit():
            new_x = num2words(new_x, lang='cz')
        for y in new_x.split():
            new_seq.append(unidecode(y))
    return new_seq



# %%
if __name__ == '__main__':
    # %%
    torch.multiprocessing.set_sharing_strategy('file_system')
    df_path = '/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path_large.csv'
    filters = [
        # FilterUB(value=0.15, name='char_norm_word_dist_with_gaps_90__segments'),
        # FilterUB(value=0.025, name='avg_norm_word_dist_with_gaps__segments'),
        FilterUB(value=0.15, name='avg_norm_word_dist_with_gaps__segments'),
    ]

    dataset = ParCzechSpeechRecogPL(
        df_filters=filters,
        df_path=df_path,
        batch_size=100,
        num_workers=8
    )
    data = {}
    loader = dataset.train_dataloader()

    ic(dataset.dataset.duration_hours())
    ic(os.getcwd())
    for batch in tqdm(loader, total=len(loader)):
        keys, values = batch
        for k, v in zip(keys, values):
            data[k] = v

    ic(len(data) == len(dataset))
    with open('recognized_transcripts.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('saved')
    # with open()

    # %%
    # try:
    #     with open('filtered_text.pkl', 'rb') as f:
    #         text = pickle.load(f)
    # except FileNotFoundError:
    #     with open('../filtered_text.pkl', 'rb') as f:
    #         text = pickle.load(f)
    #
    # df_path = '/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path_large.csv'
    # dataset = ParCzechDataset(df_path)
    #
    # filters = [
    #     FilterUB(value=0.1, name='char_norm_word_dist_with_gaps_90__segments'),
    #     # FilterUB(value=0.025, name='avg_norm_word_dist_with_gaps__segments'),
    #     # FilterUB(value=0, name='avg_norm_word_dist_with_gaps__segments'),
    # ]
    # dataset.duration_hours(filters)
    #
    # # %%
    # dataset.filter_df(filters)
    #
    # # %%
    # abbreviation_dict = {
    #     '§': 'paragraf',
    #     'č': 'číslo',
    #     'mld': 'miliarda',
    #     '%': 'procent',
    #     'tj': 'tojest',
    #     'sb': 'sbírka',
    #     'cca': 'cirka',
    #     'odst': 'odstavec',
    #     'tzv': 'takzvané',
    #     'resp': 'respektive',
    #     'atd': 'a tak dále',
    #     'hod': 'hodin',
    #     'tzn': 'to znamená',
    #     'apod': 'a podobně',
    #     'kč': 'korun',
    #     '/': 'lomeno'
    # }
    # letters = set([l for i in dataset.df.index for w in text[i] for l in w])
    # ic(letters)
    #
    # filtered_indices = []
    # similarity = lambda a, b: max(len(a), len(b)) - distance(a, b)
    #
    # for i in tqdm(dataset.df.index, total=dataset.df.index.max()):
    #     asr_transcript = text[i]
    #     path = dataset.extract_path(i)
    #     recog_transcript = dataset.get_recognized_transcript(path, i)
    #
    #     # alignments = pairwise2.align.globalcx(normalize(asr_transcript), normalize(recog_transcript), similarity, gap_char=['-'])
    #     # aligned_asr, aligned_recog, score, _, _ = alignments[0]
    #     # if '-' not in aligned_asr:
    #     #     filtered_indices.append(i)
    #
    # # %%
    # letters = set([l for s in text for w in s for l in w])

    # %%
    # %%
    # %%



















    # %%


