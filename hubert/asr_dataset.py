# %%
import os
import sys

import pandas as pd
import torch
import torchtext
from collections import OrderedDict, Counter
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


class TransNorm:
    def __init__(self, gold_trans, asr_trans, recog_trans, path, max_alignment_cnt):
        self.path = path
        self.recognized_transcript = recog_trans
        self.asr_transcript = asr_trans.lower()
        self.gold_transcript = gold_trans
        self.alignment = None
        self.normalized = []
        self.gap_char = '-'
        self.max_alignment_cnt = max_alignment_cnt
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
    @staticmethod
    def _custom_distance(s1, s2):
        replacement_dict = {
            'á': 'a',

            'é': 'e',

            'í': 'i',
            'y': 'i',
            'ý': 'i',

            'ú': 'u',
            'ů': 'u',

            'ó': 'o',
        }
        pass

    def expand_abbr(self, word):
        if word in self.abbreviation_dict:
            return self.abbreviation_dict[word]
        return word

    @staticmethod
    def _remove_accents(trans):
        return unidecode(trans)

    @staticmethod
    def custom_num2words(number, remove_accents=False):
        # number can be either int or string

        if remove_accents:
            return TransNorm._remove_accents(num2words(number, lang='cz'))
        return num2words(number, lang='cz')

    @staticmethod
    def _num2words_21to99_special(num_int, remove_accents=False):
        if num_int % 10 == 0:
            return []

        ones_num = num_int % 10
        ones_str = TransNorm.custom_num2words(ones_num, remove_accents)

        tens_num = (num_int % 100) - ones_num
        tens_str = TransNorm.custom_num2words(tens_num, remove_accents)

        return [
            f'{ones_str}a{tens_str}',
            f'{ones_str} a {tens_str}',
        ]

    @staticmethod
    def _extended_custom_num2words(string, remove_accents=False):
        results = [
            TransNorm.custom_num2words(string, remove_accents)
        ]
        num = int(string)
        if 20 < num:
            residual_tens = num % 100
            residual_high = num - residual_tens

            # can be an empty list, this means that the number was a multiple of 10
            alt_forms_tens = TransNorm._num2words_21to99_special(residual_tens, remove_accents)

            for alt_f in alt_forms_tens:
                # num > 100
                if residual_high != 0:
                    results.append(TransNorm.custom_num2words(residual_high) + ' ' + alt_f)
                else:
                    results.append(alt_f)

            # 1100 can be jedenáct set
            if 1100 < num < 1999:
                alt_form_high = TransNorm.custom_num2words(residual_high // 100) + ' ' + 'set'
                if residual_tens == 0:
                    results.append(alt_form_high)
                else:
                    for alt_f in alt_forms_tens:
                        results.extend([
                            alt_form_high + ' ' + alt_f,
                            alt_form_high + ' ' + TransNorm.custom_num2words(residual_tens, remove_accents)
                        ])

            if 1000 < num < 10000:
                results.append(TransNorm.custom_num2words(residual_high // 100) + ' ' + TransNorm.custom_num2words(residual_tens))
        return results

    @staticmethod
    def similarity(trans1, trans2):
        return max(len(trans1), len(trans2)) - distance(trans1, trans2)

    @staticmethod
    def _find_dates(gold):
        tmp_results = [x.rstrip() for x in re.findall(r'\d+\s\.\s\d+\s\.\s*\d*', gold)]
        # replace dots and double ws
        tmp_results = [x.replace('.', '').replace('  ', ' ').rstrip() for x in tmp_results]
        results = []
        for x in tmp_results:
            if len(x.split()) == 2:
                day, month = x.split()
            else:
                day, month, year = x.split()

            if 0 < int(day) < 32 and 0 < int(month) < 13:
                results.append(x)
        return results


    @staticmethod
    def _find_times(trans, time=False):
        # \d is digit
        if time:
            return re.findall(r"\d+\.\d+ hodin", trans)
        return re.findall(r"\d+\.\d+", trans)

    @staticmethod
    def _find_floats(gold):
        def clean_floats_and_str(floats_lst, source_str, extra_str=''):
            clean_results = []
            for f in floats_lst:
                dummy = 'X'*len(f)
                source_str = source_str.replace(f, dummy, 1)
                if extra_str != '':
                    f = f.replace(extra_str, '')
                clean_results.append(f.replace(', ', ''))
            return clean_results, source_str

        float_base_regex = r'\d+\s,\s\d+'
        if re.findall(float_base_regex, gold) == []:
            return [], []

        regex_prefix = [
            'deficit', 'těch', 'minus', 'kolem', 'výši', 'nebo', 'bude', 'než',
            'ani', 'plat', 'nějaké', 'úvazek', 'nadýchal', 'plnění', 'hodnoty',
            'hranici', 'mezi', 'tam', 'nad', 'měli', ' ze', 'nejsou', 'PM',
            'asi', 'na', 'být', 'je', ' na', ' z', ' od', ' do', ' mezi',
            'částkou', 'současných', 'tolerance', 'činit', 'růst', 'krát',
            'koeficientu', 'koeficientem', 'koeficient', 'koeficient na',
            'tempo', 'průměrně', 'plocha', 'plus', 'hodnotu', 'násobku',
            'přebytek', 'čísle', 'komise', 'průmysl', 'Proč', 'prostě',
            'rozpočtováno', 'utraceno', 'vybírám',
        ]

        regex_suffix = [
            'řádku', 'dny', 'exekuce', 'průměrné', 'gramu', 'GHz', 'decibelu',
            'mld', 'miliardy', 'milionu', 'mil .', 'mil ', '%', 'procentní',
            'tisíce',  'stupně', 'krát', 'prac', 'Kč', 'promile', 'metru',
            'tis ', 'bili', 'náso', 'hod', 'letech', 'do', 'desetinásobku',
            'bil ', 'hrubé', 'procenta', 'roku', 'HDP', 'dne ', 'miliony',
            'km', 'miliony', 'kilometru', 'minuty', 'měsíce', 'měsících',
            'tuny', 'miliarda', 'koruny', 'dítěte', 'eura', 'megawattu',
            'týdne', 'za ', 'až',
        ]

        gold = gold.lower()
        raw_results = re.findall(r"\s(0\s,\s\d+)", gold)
        results, gold = clean_floats_and_str(re.findall(r"\s(0\s,\s\d+)", gold), gold)

        for suffix in regex_suffix:
            suffix = ' ' + suffix
            suffix_regex = float_base_regex + suffix
            tmp_results = re.findall(suffix_regex, gold)
            raw_results.extend(tmp_results)
            tmp_results, gold = clean_floats_and_str(tmp_results, gold, suffix)
            results.extend(tmp_results)

        for prefix in regex_prefix:
            prefix = prefix + ' '
            prefix_regex = prefix + float_base_regex
            tmp_results = re.findall(prefix_regex, gold)
            raw_results.extend(tmp_results)
            tmp_results, gold = clean_floats_and_str(tmp_results, gold, prefix)
            results.extend(tmp_results)

        # handle if the string starts with float
        tmp_results = [x for x in re.findall(float_base_regex, gold) if gold.startswith(x)]
        raw_results.extend(tmp_results)
        tmp_results, _ = clean_floats_and_str(tmp_results, gold)
        results.extend(tmp_results)

        return results

    @staticmethod
    def _find_digits(trans, time=False):
        if time:
            return re.findall(r"\d+ hodin[y]*", trans)
        return re.findall(r"\d+", trans)

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

    def _float_to_words(self, float_str):
        whole, decimal = float_str.lstrip().rstrip().replace(' ', '.').split('.')
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

        recognized_key_words = [
            'půl', 'celé', 'celá', 'celý' 'celých', 'celém',
        ]
        if int(whole) == 0:
            result_dict['whole'].append('žádná')


        if decimal.startswith('0'):
            # _starts_with_zero returns multiple variants, only first variant does not ignore number of zeros
            result_dict['decimal'].append(self._starts_with_zero(decimal)[0])
        whole_words = ['celé', 'celá', 'celý', 'tečka', 'čárka', '']
        decimal_words = ['']

        if len(decimal) == 1:
            # decimal_words.extend(['desetina', 'desitiny', 'desetin'])
            decimal_words.append('desetin')
        elif len(decimal) == 2:
            # decimal_words.extend(['setina', 'setiny', 'setin'])
            decimal_words.append('setin')
            pass
        elif len(decimal) == 3:
            # decimal_words.extend(['tisicina', 'tisiciny', 'tisicin'])
            decimal_words.append('tisícin')
        else:
            NotImplementedError('Handling decimal parts smaller than .001 is not implemented yet.')
        results = []
        for wh_num in result_dict['whole']:
            for wh in whole_words:
                for dec_num in result_dict['decimal']:
                    for dec in decimal_words:
                        results.append(f'{wh_num} {wh} {dec_num} {dec}'.replace('  ', ' ').rstrip())
        return results

    def _float_time_to_words(self, time_str, time_suffix=False):
        if time_suffix:
            time_str = time_str.replace('hodin', '')

        hours, minutes = time_str.replace(' ', '').split('.')
        hours_int = int(hours)
        results_dict = dict(
            hours=[self.custom_num2words(hours)],
            # sometimes minutes are not pronounced
            minutes=['', self.custom_num2words(minutes)]
        )

        if hours_int > 12:
            results_dict['hours'].append(self.custom_num2words(f'{hours_int % 12}'))

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

        # handle the case for "XX.30", this can be transcribed as "půl XX"
        if minutes == '30':
            if hours_int >= 12:
                hours_int_short = (hours_int % 12) + 1
                results_list.append(f'půl {self.custom_num2words(f"{hours_int_short}")}')
            else:
                results_list.append(f'půl {self.custom_num2words(f"{hours_int + 1}")}')

        if time_suffix:
            new_results_list = []
            for r in results_list:
                new_results_list.append(r)
                if 'hodin' not in r and 'minut' not in r:
                    new_results_list.append(r + ' ' + 'hodin')
            results_list = new_results_list
        return list(set(results_list))

    def _digit_time_to_words(self, time_str):
        hours = re.sub(r"hodin[y]*", "", time_str)
        results = self._extended_custom_num2words(hours)
        if int(hours) > 12:
            results.append(self.custom_num2words(f'{int(hours)%12}'))
        new_results = []
        for r in results:
            new_results.append(r)
            new_results.append(f'{r} hodin')
        return new_results

    def _eval_alignment(self, replacement, recogized):
        total_len = 0
        score = 0
        aligned_gap = False
        cnt_aligned = 0
        alignment = []
        for repl, recog in zip(replacement, recogized):
            if repl == self.gap_char:
                continue
            total_len += max(len(repl), len(recog))
            score += self.similarity(repl, recog)
            alignment.append([repl, recog])
            cnt_aligned += 1
            # do not want to have gaps in recognized transcript
            if recog == self.gap_char:
                aligned_gap = True
                break
        return total_len, score, aligned_gap, cnt_aligned, alignment

    def _filter_replacements(self, repl_lst, recognized_split, repl_cnt):
        alignments = []
        for repl in repl_lst:
            repl_split = repl.split()
            tmp_alignments = pairwise2.align.localcs(
                repl_split,
                recognized_split,
                match_fn=self.similarity,
                open=-0.3,
                extend=-0.1,
                gap_char=[self.gap_char],
            )
            aligned_repl, aligned_recog, orig_score, _, _ = tmp_alignments[0]
            total_len, score, aligned_gap, cnt_aligned, alignment = self._eval_alignment(aligned_repl, aligned_recog)
            if aligned_gap:
                continue

            alignments.append([repl, score / total_len + 0.02*cnt_aligned])

        best_replacements = sorted(alignments, key=lambda x: x[1], reverse=True)
        cnt_candidates = min(self.max_alignment_cnt + repl_cnt, len(best_replacements))
        return [repl for repl, _ in best_replacements[:cnt_candidates]]

    def _apply_replacements(self, transcript, replacements, recognized):
        # replace_dict[orig_float_string1] = [verb_var_1_1, verb_var_1_2, ...]
        # len(replace_dict) = N
        recognized_split = recognized.split()
        combinations = [[]]
        replacements_dict = {}
        repl_counter = Counter([r for r, _ in replacements])
        ic(replacements)
        for rep_key, rep_lst in replacements:
            new_combinations = []
            # it is not possible to evaluate all combinations
            # so leave only combinations that somehow occur in the recognized transcript
            if rep_key in replacements_dict:
                filtered_rep_lst = replacements_dict[rep_key]
            else:
                filtered_rep_lst = self._filter_replacements(rep_lst, recognized_split, repl_counter[rep_key])
                replacements_dict[rep_key] = filtered_rep_lst

            for v in filtered_rep_lst:
                for com in combinations:
                    new_combinations.append(com + [v])
            combinations = new_combinations
        # combinations[i] = [verb_var_1_x1, verb_var_2_x2, ... verb_var_N_xN]
        ic(len(combinations))
        ic(replacements_dict)

        alignments = []
        for combination in combinations:
            # align combination to the recognized transcript
            combination_split = [w for words in combination for w in words.split()]
            tmp_alignments = pairwise2.align.globalcs(
                combination_split,
                recognized_split,
                match_fn=self.similarity,
                open=-0.9,
                extend=-0.1,
                gap_char=[self.gap_char],
            )
            aligned_comb, aligned_recog, orig_score, _, _ = tmp_alignments[0]

            # manually score the alignment by computing edit distance between aligned words
            total_len, score, aligned_gap, cnt_aligned, alignment = self._eval_alignment(aligned_comb, aligned_recog)

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

            alignments.append([replacement, score/total_len, cnt_aligned, score/total_len + 0.02*cnt_aligned, orig_score])
        score_index = 3

        best_alignments = sorted(alignments, key=lambda x: x[score_index], reverse=True)
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
            for (key, _), value in zip(replacements, best_alignment[0]):
                new_transcript = new_transcript.replace(key, value[1].upper(), 1)
            new_transcripts.append(new_transcript)
        return new_transcripts

    @staticmethod
    def _has_numbers(trans):
        return bool(re.search(r'\d', trans))

    @staticmethod
    def _find_order(trans, patterns):
        new_trans = trans
        result = []
        for p in patterns:
            order = new_trans.find(p)
            result.append([p, order])
            dummy = 'X' * len(p)
            # replace the first occurrence
            new_trans = new_trans.replace(p, dummy, 1)
        return result, new_trans

    @staticmethod
    def _merge_replacements(replacements, patterns):
        ordered_patterns = sorted(patterns, key=lambda x: x[1])
        ordered_replacements = []
        for p, _ in ordered_patterns:
            for replacement in replacements:
                for rep_key, rep_lst in replacement:
                    if rep_key == p:
                        ordered_replacements.append((rep_key, rep_lst))
                        break
        return ordered_replacements, ordered_patterns

    def _expand_digits(self, asr_trans, recognized_trans) -> list:
        if not self._has_numbers(asr_trans):
            return [asr_trans]

        # some floats found, can be either real numbers or time
        # now need to deal with time and floats
        # the problem is that time can be written as "XX.XX hodin" and also as "XX.XX"
        # so first find replacement for patterns like "XX.XX hodin"
        # and then deal with pattern "XX.XX", in this case consider both floats and times
        # also there can be the same float multiple time in the transcript

        patterns_order = []
        replacements = []
        tmp_asr_trans = asr_trans

        # `XX.XX hodin[y]`
        explicit_times_float = self._find_times(asr_trans, time=True)
        if explicit_times_float != []:
            times_order, tmp_asr_trans = self._find_order(tmp_asr_trans, explicit_times_float)
            patterns_order.extend(times_order)
            time_replacement = [(f, self._float_time_to_words(f, time_suffix=True)) for f in explicit_times_float]
            replacements.append(time_replacement)

        # words of the form `XX.XX` are times
        hidden_times_float = self._find_times(tmp_asr_trans, time=False)
        if hidden_times_float != []:
            floats_order, tmp_asr_trans = self._find_order(tmp_asr_trans, hidden_times_float)
            patterns_order.extend(floats_order)
            float_replacement = [(f, self._float_time_to_words(f)) for f in hidden_times_float]
            replacements.append(float_replacement)

        # todo
        #  words of the form `XX , XX` are possibly floats

        # even natural numbers can be time
        time_num = self._find_digits(tmp_asr_trans, time=True)
        if time_num != []:
            nums_time_order, tmp_asr_trans = self._find_order(tmp_asr_trans, time_num)
            patterns_order.extend(nums_time_order)
            num_time_replacement = [(n, self._digit_time_to_words(n)) for n in time_num]
            replacements.append(num_time_replacement)

        # todo
        #  large numbers of order millions/milliards are written in the form `12 495 071`
        #  need to remove spaces

        nums = self._find_digits(tmp_asr_trans)
        if nums != []:
            nums_order, _ = self._find_order(tmp_asr_trans, nums)
            patterns_order.extend(nums_order)
            num_replacement = [(n, self._extended_custom_num2words(n)) for n in nums]
            replacements.append(num_replacement)

        replacements, ordered_patterns = self._merge_replacements(replacements, patterns_order)
        ic(ordered_patterns)

        ic(self.gold_transcript)
        ic(asr_trans)
        ic(self.recognized_transcript)

        results = self._apply_replacements(asr_trans, replacements, recognized_trans)

        # ic(result)
        for r in results:
            ic(r)
        ic('--' * 40)
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
        alignments.append(TransNorm(t['gold'], t['asr'], t['recog'], t['path'], 2))
    # %%

    def transform_date(date_str):
        # 1 první
        # 2 druhý
        # 3 třetí
        # 4 čtvrtý
        # 5 pátý
        # 6 šestý
        # 7 sedmý
        # 8	osmý
        # 9 devátý
        # 10 desátý
        # 11 jedenáctý
        # 12 dvanáctý
        # 13 třináctý
        # 14 čtrnáctý
        # 15 patnáctý
        # 16 šestnáctý
        # 17 sedmnáctý
        # 18 osmnáctý
        # 19 devatenáctý
        # 20 dvacátý
        #
        # 21	jednadvacet, jedenadvacet, dvacet jeden, dvacet jedna	jednadvacátý, jedenadvacátý, dvacátý první, dvacátý prvý
        # 22	dvaadvacet, dvacet dva	dvaadvacátý, dvacátý druhý
        # 23	třiadvacet, dvacet tři	třiadvacátý, dvacátý třetí
        # 24	čtyřiadvacet, dvacet čtyři	čtyřiadvacátý, dvacátý čtvrtý
        # 25	pětadvacet, dvacet pět	pětadvacátý, dvacátý pátý
        # 26	šestadvacet, dvacet šest	šestadvacátý, dvacátý šestý
        # 27	sedmadvacet, dvacet sedm	sedmadvacátý, dvacátý sedmý
        # 28	osmadvacet, dvacet osm	osmadvacátý, dvacátý osmý
        # 29	devětadvacet, dvacet devět	devětadvacátý, dvacátý devátý
        # 30	třicet	třicátý

        replacement_dict_day = {
            '1': [num2words('1', lang='cz'), ' první',],
            '2': [num2words('2', lang='cz'), ' druhý',],
            '3': [num2words('3', lang='cz'), ' třetí',],
            '4': [num2words('4', lang='cz'), ' čtvrtý', ],
            '5': [num2words('5', lang='cz'), ' pátý', ],
            '6': [num2words('6', lang='cz'), ' šestý', ],
            '7': [num2words('7', lang='cz'), ' sedmý', ],
            '8': [num2words('8', lang='cz'), '	osmý', ],
            '9': [num2words('9', lang='cz'), ' devátý', ],
            '10': [num2words('10', lang='cz'), ' desátý', ],
            '11': [num2words('11', lang='cz'), ' jedenáctý', ],
            '12': [num2words('12', lang='cz'), ' dvanáctý', ],
            '13': [num2words('13', lang='cz'), ' třináctý', ],
            '14': [num2words('14', lang='cz'), ' čtrnáctý', ],
            '15': [num2words('15', lang='cz'), ' patnáctý', ],
            '16': [num2words('16', lang='cz'), ' šestnáctý', ],
            '17': [num2words('17', lang='cz'), ' sedmnáctý', ],
            '18': [num2words('18', lang='cz'), ' osmnáctý', ],
            '19': [num2words('19', lang='cz'), ' devatenáctý', ],
            '20': [num2words('20', lang='cz'), ' dvacátý', ],
            '21': [num2words('21', lang='cz'), ],
            '22': [num2words('22', lang='cz'), ],
            '23': [num2words('23', lang='cz'), ],
            '24': [num2words('24', lang='cz'), ],
            '25': [num2words('25', lang='cz'), ],
            '26': [num2words('26', lang='cz'), ],
            '27': [num2words('27', lang='cz'), ],
            '28': [num2words('28', lang='cz'), ],
            '29': [num2words('29', lang='cz'), ],
            '30': [num2words('30', lang='cz'), ],
            '31': [num2words('31', lang='cz'), ],
        }
        return None

    # 0 , 5 promile ... nula pět promile
    # původních nějakých 6 , 50 Kč na až nějakých 12 , 50 Kč ... způvodních nějakých šest korun padesáti na až nějakých dvanáct korunu padesát
    # 54 , 58 Kč ... padesát čtyři korun padesát osm ale čtú
    # 0 , 3 násobku ... nula tři násobku
    # 0 , 66 %  ... nula šedesát šest procent
    # o 0 , 1 % ... ojednu desetinu procenta
    # na 0 , 1 % z tržeb ...  na jednu desetinu procento
    # a 23 , 58 % ... dvacet třicet i padesát osm procent ..289
    #  0 , 2 % ... dvě desetiny procenta
    # 0 , 75 % nula sedmdesát pět procenta

    # for % need to filter
    #   \d0\s\d0, ... 50,30 procent
    #   also first number should be smaller than 100

    # 0,1 – nula/žádná celá jedna (desetina)

    # jedenapůl
    # 0 , 0001 procenta
    # 72 202 hektarů in gold and asr
    # 7 600 korunami
    # cm
    # : ku
    cnt = 0
    times_cnt = 0
    float_cnt = 0
    date_cnt = 0
    nums_cnt = 0
    for a in alignments:
        if not a._has_numbers(a.asr_transcript):
            continue
        nums_cnt += 1
        if a._find_dates(a.gold_transcript):
            date_cnt +=1
        if a._find_floats(a.gold_transcript):
            float_cnt += 1
        if a._find_times(a.asr_transcript):
            times_cnt += 1




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


