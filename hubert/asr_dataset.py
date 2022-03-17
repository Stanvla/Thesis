# %%
import sys
from abc import ABC, abstractmethod

# depending on how the script is executed, interactively or not, use different path for importing
if not sys.stdout.isatty():
    from clustering.torch_mffc_extract import ParCzechDataset
else:
    from hubert.clustering.torch_mffc_extract import ParCzechDataset


class ParCzechSpeechRecDataset(ParCzechDataset):
    def __init__(self, df_path, resample_rate=16000, clean_params=None, sep='\t'):
        super(ParCzechDataset, self).__init__(df_path, resample_rate, clean_params, sep=sep, sort=False)



