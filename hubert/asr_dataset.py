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
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

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
            'tj': 'to jest',
            'sb': 'sbírka',
            'cca': 'cirka',
            'odst': 'odstavec',
            'tzv': 'takzvaně',
            'resp': 'respektivě',
            'atd': 'a tak dále',
            'hod': 'hodin',
            'tzn': 'to znamena',
            'apod': 'a podobně',
            'kč': 'korun',
            '/': 'lomeno',
            '+': 'plus',
            '09': 'nula devět',
        }

    def expand_abbr(self, word):
        if word in self.abbreviation_dict:
            return self.abbreviation_dict[word]
        return word

    @staticmethod
    def _remove_accents(trans):
        return unidecode(trans)

    def custom_num2words(self, string, remove_accents=False):
        if remove_accents:
            return self._remove_accents(num2words(string, lang='cz'))
        return num2words(string, lang='cz')

    @staticmethod
    def similarity(trans1, trans2):
        return max(len(trans1), len(trans2)) - distance(trans1, trans2)

    @staticmethod
    def _find_floats(trans, time=False):
        # \d is digit
        if time:
            return re.findall(r"\d+\.\d+ hodin", trans)
        return re.findall(r"\d+\.\d+", trans)

    @staticmethod
    def _find_digits(trans):
        return re.findall(r"\d+", trans)

    def _float_to_words(self, float_str):
        whole, decimal = float_str.replace(' ', '').split('.')
        result_dict = dict(
            whole=[self.custom_num2words(whole)],
            decimal=[self.custom_num2words(decimal)]
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

    def _starts_with_zero(self, str_num):
        zeros_cnt = 0
        for char in str_num:
            if char != '0':
                break
            zeros_cnt += 1

        zeros = ' '.join(self.custom_num2words('0') for _ in range(zeros_cnt))
        numbers = []
        # check if the number in form that starts with zeros like 09
        if zeros_cnt < len(str_num):
            number_without_zeros = self.custom_num2words(str_num[zeros_cnt: len(str_num)])
            numbers.append(zeros + ' ' + number_without_zeros)
            # add variant with only one zero
            if zeros_cnt > 1:
                numbers.append(self.custom_num2words('0') + ' ' + number_without_zeros)
        else:
            numbers.append(zeros)
            if zeros_cnt > 1:
                numbers.append(self.custom_num2words('0'))
        return numbers

    def _time_to_words(self, time_str, time_suffix=False):
        if time_suffix:
            time_str = time_str.replace('hodin', '')

        hours, minutes = time_str.replace(' ', '').split('.')
        results_dict = dict(
            hours=[self.custom_num2words(hours)],
            # sometimes minutes are not pronounced
            minutes=['', self.custom_num2words(minutes)]
        )
        if hours.startswith('0'):
            results_dict['hours'].extend(self._starts_with_zero(hours))

        if minutes.startswith('0'):
            results_dict['minutes'].extend(self._starts_with_zero(minutes))

        results_list = []
        for h_num in results_dict['hours']:
            for h in ['hodin', '']:
                for m_num in results_dict['minutes']:
                    for m in ['minut', '']:

                        if m_num == '':
                            result = f'{h_num} {h}'
                        else:
                            if h == '':
                                result = f'{h_num} {m_num}'
                            else:
                                if h == 'hodin' and m == '':
                                    result = f'{h_num} {m_num}'
                                else:
                                    result = f'{h_num} {h} {m_num} {m}'

                        # need to handle double white-space and white-space at the end
                        results_list.append(result.replace('  ', ' ').rstrip())

        if time_suffix:
            new_results_list = []
            for r in results_list:
                new_results_list.append(r)
                if 'hodin' not in r and 'minut' not in r:
                    new_results_list.append(r + ' ' + 'hodin')
            results_list = new_results_list
        return list(set(results_list))

    def _dict_replacement(self, transcript, replace_dict, recognized):
        # replace_dict[orig_float_string1] = [verb_var_1_1, verb_var_1_2, ...]
        # len(replace_dict) = N

        combinations = [[]]
        for k in replace_dict:
            new_combinations = []
            for v in replace_dict[k]:
                for com in combinations:
                    new_combinations.append(com + [v])
            combinations = new_combinations
        # combinations[i] = [verb_var_1_x1, verb_var_2_x2, ... verb_var_N_xN]
        from operator import mul
        from functools import reduce
        ic(len(combinations))
        ic(reduce(mul, [len(v) for v in replace_dict.values()], 1))

        alignments = []
        recognized_split = recognized.split()
        gap_char = '-'
        for combination in combinations:
            # align combination to the recognized transcript
            combination_split = [w for words in combination for w in words.split()]
            tmp_alignments = pairwise2.align.globalcs(
                combination_split,
                recognized_split,
                match_fn=self.similarity,
                open=-3,
                extend=0,
                gap_char=[gap_char],
            )
            aligned_comb, aligned_recog, orig_score, _, _ = tmp_alignments[0]

            # manually score the alignment by computing edit distance between aligned words
            acc = 0
            score = 0
            alignment = []
            aligned_gap = False
            cnt_aligned = 0
            for ac, ar in zip(aligned_comb, aligned_recog):
                if ac == gap_char:
                    continue
                acc += max(len(ac), len(ar))
                score += self.similarity(ac, ar)
                alignment.append([ac, ar])
                cnt_aligned += 1
                if ar == gap_char:
                    aligned_gap = True
                    break

            if aligned_gap:
                continue

            # score += np.log2(aligned_cnt)
            # for each align word extract its aligned variant and then join them according to the combination
            alignment_idx = 0
            replacement = []
            for words in combination:
                comb_len = len(words.split())
                aligned_comb = alignment[alignment_idx: alignment_idx + comb_len]
                replacement.append([words, ' '.join([w for _, w in aligned_comb])])
                alignment_idx += comb_len

            alignments.append([replacement, score/acc, cnt_aligned, score/acc + 0.02*cnt_aligned, score/acc + np.log2(cnt_aligned), orig_score])
        score_index = 3

        best_alignments = sorted(alignments, key=lambda x: x[score_index], reverse=True)
        ic(best_alignments[:5])
        best_score = best_alignments[0][score_index]
        filtered_best_alignments = []
        for alignment in best_alignments:
            score = alignment[score_index]
            if score < best_score:
                break
            filtered_best_alignments.append(alignment)

        # filtered_best_alignments = sorted(filtered_best_alignments, key=lambda x: x[2], reverse=True)

        ic(filtered_best_alignments)
        new_transcripts = []
        for best_alignment in filtered_best_alignments:
            new_transcript = transcript
            for key, value in zip(replace_dict, best_alignment[0]):
                new_transcript = new_transcript.replace(key, value[1].upper())
            new_transcripts.append(new_transcript)
        return new_transcripts

    def _transform_floats(self, trans, floats):
        replace_dict_floats = OrderedDict((f, self._float_to_words(f)) for f in floats)
        replace_dict_times = OrderedDict((f, self._time_to_words(f)) for f in floats)
        replace_dict = {}
        for f in floats:
            replace_dict[f] = replace_dict_floats[f] + replace_dict_times[f]
        return self._dict_replacement(trans, replace_dict)

    def _transform_time(self, trans, times, recognized):
        replace_dict_times = OrderedDict((f, self._time_to_words(f, time_suffix=True)) for f in times)
        return self._dict_replacement(trans, replace_dict_times, recognized)

    def _transform_natural_num(self, trans):
        results = []
        for t in trans:
            new_t = []
            for w in t.split():
                if w.isdigit():
                    new_t.append(self.custom_num2words(w))
                else:
                    new_t.append(w)
            results.append(' '.join(new_t))
        return results

    @staticmethod
    def _has_numbers(trans):
        return bool(re.search(r'\d', trans))

    @staticmethod
    def _replace_with_dummy(trans, replacement_dict):
        new_trans = trans
        for k in replacement_dict:
            xs = 'X' * len(k)
            new_trans = new_trans.replace(k, xs)
        return new_trans

    @staticmethod
    def _find_order(trans, patterns):
        return [[p, trans.find(p)] for p in patterns]

    def _expand_digits(self, asr_trans, recognized_trans) -> list:
        if not self._has_numbers(asr_trans):
            return [asr_trans]

        # handles floating point numbers, natural numbers, times
        floats = self._find_floats(asr_trans, time=False)
        if floats == []:
            return self._transform_natural_num([asr_trans])
        # some floats found, can be either real numbers or time
        # now need to deal with time and floats
        # the problem is that time can be written as "XX.XX hodin" and also as "XX.XX"
        # so first find replacement for patterns like "XX.XX hodin"
        # and then deal with pattern "XX.XX", in this case consider both floats and times
        # also there can be the same float multiple time in the transcript

        patterns_order = []
        replacement_dicts = []
        tmp_asr_trans = asr_trans

        times = self._find_floats(asr_trans, time=True)
        if times != []:
            ic(times)
            patterns_order.extend(self._find_order(tmp_asr_trans, times))
            replace_dict_times = OrderedDict((f, self._time_to_words(f, time_suffix=True)) for f in times)
            # before finding floats need to replace time stamps with dummy strings of the same lengths
            tmp_asr_trans = self._replace_with_dummy(tmp_asr_trans, replace_dict_times)
            replacement_dicts.append(replace_dict_times)

        floats = self._find_floats(tmp_asr_trans, time=False)
        if floats != []:
            ic(floats)
            # floats can also be time
            patterns_order.extend(self._find_order(tmp_asr_trans, floats))
            replace_dict_floats = OrderedDict((f, self._time_to_words(f) + self._float_to_words(f)) for f in times)
            tmp_asr_trans = self._replace_with_dummy(tmp_asr_trans, replace_dict_floats)
            replacement_dicts.append(replace_dict_floats)

        nums = self._find_digits(tmp_asr_trans)
        if nums != []:
            ic(nums)
            patterns_order.extend(self._find_order(tmp_asr_trans, nums))
            replace_dict_num = OrderedDict((n, [self.custom_num2words(n)]) for n in nums)
            replacement_dicts.append(replace_dict_num)

        ic(floats)
        results = self._transform_time(asr_trans, times, recognized_trans)
        # results = self._transform_floats(trans, floats)
        # results = [r for r in results if len(r.split()) == len(self.recognized_transcript.split())]
        ic(self.gold_transcript)
        ic(self.asr_transcript)
        ic(asr_trans)
        ic(self.recognized_transcript)
        # ic(result)
        for r in results:
            ic(r)
        ic('--' * 40)
        # todo for each result need to transform remaining natural numbers
        return results

    def _normalize_trans(self, remove_accents=False):
        normalized_asr = []
        for w in self.asr_transcript.split():
            norm_w = self.expand_abbr(w.lower())
            if remove_accents:
                normalized_asr.append(self._remove_accents(norm_w))
            else:
                normalized_asr.append(norm_w)
        # floating point can be a number or time
        return self._expand_digits(' '.join(normalized_asr), self.recognized_transcript)

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
    transcripts_pickle_file = '/lnet/express/work/people/stankov/alignment/transcripts.pkl'
    ic(os.getcwd())
    transcripts = []
    if not os.path.isfile(transcripts_pickle_file):
        print('reading through dataloder')
        for batch in tqdm(loader, total=len(loader)):
            asr_trans, gold_trans, recog_trans, paths = batch['asr_trans'], batch['gold_trans'], batch['recog_trans'], batch['path']
            for asr, gold, recog, path in zip(asr_trans, gold_trans, recog_trans, paths):
                transcripts.append(dict(gold=gold, asr=asr, recog=recog, path=path))
        with open(transcripts_pickle_file, 'wb') as f:
            pickle.dump(transcripts, f)
    else:
        print('unpickling')
        with open(transcripts_pickle_file, 'rb') as f:
            transcripts = pickle.load(f)

    alignments = []
    for t in tqdm(transcripts):
        alignments.append(TranscriptAlignment(t['gold'], t['asr'], t['recog'], t['path']))
    # %%
    cnt = 0
    threshold = 10
    for a in alignments:
        if cnt == threshold:
            break
        if len(re.findall(r"\d+\.\d+ hodin", a.asr_transcript)) > 1:
            ic(cnt)
            a._normalize_trans()
            cnt += 1
            # ic('--'*40)

    print(cnt)
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


