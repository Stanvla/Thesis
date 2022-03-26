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


class TranscriptAlignment:
    def __init__(self, gold_trans, asr_trans, recog_trans, path):
        self.path = path
        self.recognized_transcript = recog_trans
        self.asr_transcript = asr_trans.lower()
        self.gold_transcript = gold_trans
        self.alignment = None

        self.abbreviation_dict = {
            '§': 'paragraf',
            'č': 'číslo',
            'mld': 'miliarda',
            '%': 'procent',
            'tj': 'tojest',
            'sb': 'sbírka',
            'cca': 'cirka',
            'odst': 'odstavec',
            'tzv': 'takzvané',
            'resp': 'respektive',
            'atd': 'a tak dále',
            'hod': 'hodin',
            'tzn': 'to znamená',
            'apod': 'a podobně',
            'kč': 'korun',
            '/': 'lomeno',
            '+': 'plus',
        }

    def expand_abbr(self, word):
        if word in self.abbreviation_dict:
            return self.abbreviation_dict[word]
        return word

    @staticmethod
    def _remove_accents(trans):
        return unidecode(trans)

    @staticmethod
    def similarity(trans1, trans2):
        return max(len(trans1), len(trans2)) - distance(trans1, trans2)

    @staticmethod
    def _find_floats(trans):
        # \d is number \s is whitespace
        # will find ['13.1', '13. 1', '13 . 1', '13 .1' ]
        return re.findall(r"\d+\.\d+", trans)

    def _float_to_words(self, float_str):
        whole, decimal = float_str.replace(' ', '').split('.')
        result_dict = dict(
            whole=[num2words(whole, lang='cz')],
            decimal=[num2words(decimal, lang='cz')]
        )
        # the decimal number can be pronounced with zeros like
        #   '1.001' = 'jedna nula nula jedna'
        #
        #   '1.001' = 'jedna cela nula nula jedna'
        #   '1.001' = 'jedna cela jedna tisicina'
        #
        #   '1.001' = 'jedna tecka nula nula jedna'
        #   '1.001' = 'jedna tecka jedna tisicina'
        #
        #   '1.001' = 'jedna carka nula nula jedna'
        #   '1.001' = 'jedna carka jedna tisicina'

        if decimal.startswith('0'):
            # _starts_with_zero returns multiple variants, only first variant does not ignore number of zeros
            result_dict['decimal'].append(self._starts_with_zero(decimal)[0])
        whole_words = ['cela', 'tecka', 'carka', '']
        decimal_words = []

        if len(decimal) == 1:
            decimal_words.extend(['desetina', 'desitiny', 'desetin'])
        elif len(decimal) == 2:
            decimal_words.extend(['setina', 'setiny', 'setin'])
            pass
        elif len(decimal) == 3:
            decimal_words.extend(['tisicina', 'tisiciny', 'tisicin'])
        else:
            NotImplementedError('Handling decimal parts smaller than .001 is not implemented yet.')
        results = []
        for wh_num in result_dict['whole']:
            for wh in whole_words:
                for dec_num in result_dict['decimal']:
                    for dec in decimal_words:
                        results.append(f'{wh_num} {wh} {dec_num} {dec}'.replace('  ', ' ').rstrip())
        return results

    @staticmethod
    def _starts_with_zero(str_num):
        zeros_cnt = 0
        for char in str_num:
            if char != '0':
                break
            zeros_cnt += 1

        zeros = ' '.join(num2words('0', lang='cz') for _ in range(zeros_cnt))
        numbers = []
        # check if the number in form that starts with zeros like 09
        if zeros_cnt < len(str_num):
            number_without_zeros = num2words(str_num[zeros_cnt: len(str_num)], lang='cz')
            numbers.append(zeros + ' ' + number_without_zeros)
            # add variant with only one zero
            if zeros_cnt > 1:
                numbers.append(num2words('0', lang='cz') + ' ' + number_without_zeros)
        else:
            numbers.append(zeros)
            if zeros_cnt > 1:
                numbers.append(num2words('0', lang='cz'))
        return numbers

    def _time_to_words(self, time_str):
        hours, minutes = time_str.replace(' ', '').split('.')
        results_dict = dict(
            hours=[num2words(hours, lang='cz')],
            # sometimes minutes are not pronounced
            minutes=['', num2words(minutes, lang='cz')]
        )
        if hours.startswith('0'):
            results_dict['hours'].extend(self._starts_with_zero(hours))

        if minutes.startswith('0'):
            results_dict['minutes'].extend(self._starts_with_zero(minutes))

        results_list = []
        for h in ['hodin', '']:
            for m in ['minut', '']:
                for h_num in results_dict['hours']:
                    for m_num in results_dict['minutes']:
                        # need to handle double white-space and white-space at the end
                        results_list.append(f'{h_num} {h} {m_num} {m}'.replace('  ', ' ').rstrip())
        return results_list

    @staticmethod
    def _dict_replacement(transcript, replace_dict):
        transcripts = [transcript]
        for k in replace_dict:
            new_transcripts = []
            for v in replace_dict[k]:
                for tr in transcripts:
                    new_transcripts.append(tr.replace(k, v))
            transcripts = new_transcripts
        return transcripts

    def _transform_floats(self, trans, floats):
        replace_dict_floats = {f: self._float_to_words(f) for f in floats}
        replace_dict_times = {f: self._time_to_words(f) for f in floats}
        replace_dict = {}
        for f in floats:
            replace_dict[f] = replace_dict_floats[f] + replace_dict_times[f]
        return self._dict_replacement(trans, replace_dict)

    @staticmethod
    def _transform_natural_num(trans):
        results = []
        for t in trans:
            new_t = []
            for w in t.split():
                if w.isdigit():
                    new_t.append(num2words(w, lang='cz'))
                else:
                    new_t.append(w)
            results.append(' '.join(new_t))
        return results

    @staticmethod
    def _has_numbers(trans):
        return bool(re.search(r'\d', trans))

    def _expand_digits(self) -> list:
        if not self._has_numbers(self.asr_transcript):
            return [self.asr_transcript]

        # handles floating point numbers, natural numbers, times
        floats = self._find_floats(self.asr_transcript)
        if floats == []:
            return self._transform_natural_num([self.asr_transcript])
        # some floats found, can be either real numbers or time
        results = self._transform_floats(self.asr_transcript, floats)
        results = [r for r in results if abs(len(r.split()) - len(self.recognized_transcript.split())) < 3]
        ic(floats)
        ic(self.gold_transcript)
        ic(self.asr_transcript)
        ic(self.recognized_transcript)
        for r in results:
            ic(r)
        ic('--' * 40)
        # todo for each result need to transform remaining natural numbers
        return results

    def _normalize_trans(self):

        # floating point can be a number or time
        new_transcripts = self._expand_digits()
        results = []

        for new_transcript in new_transcripts:
            norm_trans = []
            for w in new_transcript.split():
                norm_w = self.expand_abbr(w.lower())
                norm_trans.append(self._remove_accents(norm_w))
            results.append(norm_trans)
        return results

    def align(self, abbrev_dict):
        pass


class ParCzechSpeechRecDataset(ParCzechDataset):
    def __init__(self, df_path, df_filters, resample_rate=16000, sep='\t'):
        super(ParCzechSpeechRecDataset, self).__init__(df_path, resample_rate, df_filters=df_filters, sep=sep, sort=False)
        try:
            with open('recognized_transcripts.pkl', 'rb') as f:
                self.recognized_transcripts = pickle.load(f)
        except:
            with open('../recognized_transcripts.pkl', 'rb') as f:
                self.recognized_transcripts = pickle.load(f)

    def get_recognized(self, i):
        segm_path = self.index_df(i, column_name='segment_path')
        return self.recognized_transcripts[segm_path]

    def df_update_max_edit_distance(self):
        pass

    def __getitem__(self, i):
        path = self.extract_path(i)
        return dict(
            gold_trans=self.get_gold_transcript(path),
            asr_trans=self.get_asr_transcript(path),
            recog_trans=self.get_recognized(i),
            path=self.index_df(i, 'segment_path'),
        )


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
    cnt = 0
    threshold = 7204
    alignments = []
    for batch in tqdm(loader, total=len(loader)):
        if cnt > threshold:
            break

        asr_trans, gold_trans, recog_trans, paths = batch['asr_trans'], batch['gold_trans'], batch['recog_trans'], batch['path']
        for asr, gold, recog, path in zip(asr_trans, gold_trans, recog_trans, paths):
            alignments.append(TranscriptAlignment(gold, asr, recog, path))

        cnt += 1

    # %%
    cnt = 0
    for a in alignments:
        if re.findall(r"\d+\.\d+ hodin", a.asr_transcript) != [] or re.findall(r"\d+\.\d+ hodin", a.gold_transcript) != []:
            cnt += 1
            ic(a.gold_transcript)
            ic(a.asr_transcript)
            ic(a.recognized_transcript)
            ic('--'*40)

    ic(cnt)
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


