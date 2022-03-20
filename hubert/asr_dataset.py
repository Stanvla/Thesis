# %%
import sys
import torchtext
from collections import OrderedDict
from abc import ABC, abstractmethod
from tqdm import tqdm

# depending on how the script is executed, interactively or not, use different path for importing
try:
    from clustering.torch_mffc_extract import ParCzechDataset
    from clustering.filter_dataframe import FilterLB, FilterUB
except ModuleNotFoundError:
    from hubert.clustering.filter_dataframe import FilterLB, FilterUB
    from hubert.clustering.torch_mffc_extract import ParCzechDataset


class ParCzechSpeechRecDataset(ParCzechDataset):
    def __init__(self, df_path, resample_rate=16000, clean_params=None, sep='\t'):
        super(ParCzechDataset, self).__init__(df_path, resample_rate, clean_params, sep=sep, sort=False)

# %%
if __name__ == '__main__':
    # %%
    df_path = '/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path_large.csv'
    dataset = ParCzechDataset(df_path)
    dataset.filter_df([
        FilterUB(value=0, name='avg_norm_word_dist__segments'),
        FilterUB(value=20, name='duration__segments'),
    ])

    # %%
    text = []
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        p = dataset.extract_path(i)
        text.append(dataset.get_asr_transcript(p))

    # %%
    letters = set([l for s in text for w in s for l in w])

    # %%
    letters
    # %%
    bad_letters = [
        # '\\',
        # '§',
        # '%',
        # '+',
        # '.',
        '/',
        # '0',
        # '1',
        # '2',
        # '3',
        # '4',
        # '5',
        # '6',
        # '7',
        # '8',
        # '9',
    ]
    for i, s in enumerate(text):
        for l in bad_letters:
            if l in s:
                print(f'{l} :: {i:04} {s}')
    # %%
    import os
    os.listdir()
    # %%

    import pickle
    with open('filtered_text.pkl', 'wb') as f:
        pickle.dump(text, f)


