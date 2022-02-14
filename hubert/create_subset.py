# %%
import os
import pandas as pd
import random
from hubert.clustering.torch_mffc_extract import clean_data
from icecream import ic
import pickle
# %%
with open('/lnet/express/work/people/stankov/alignment/clustering_all/segments/int2mp3.pickle', 'rb') as f:
    int2mp3 = pickle.load(f)

with open('/lnet/express/work/people/stankov/alignment/clustering_all/segments/mp3_to_int.pickle', 'rb') as f:
    mp3_to_int = pickle.load(f)

mp3s = [mp3 for lst in int2mp3[:5] for mp3 in lst]

# %%

df = pd.read_csv('clean_with_path_large.csv', sep='\t')

parczech_clean_params = dict(
    recognized_sound_coverage__segments_lb=0.45,
    recognized_sound_coverage__segments_ub=0.93,
    duration__segments_lb=0.5,
)

df = clean_data(df, parczech_clean_params).reset_index(drop=True)
# %%
df['mp3'] = df.mp3_name.str.split('_').str[1]
subset = df[df['mp3'].isin(mp3s)]
# %%
subset['local_path'] = subset.segment_path.str.split('/').str[-3:].str.join('/')
# %%
# %%
index = [i for i in df.index]
random.seed(0xDEAD)
random.shuffle(index)
hours_limit = 150

for i in range(2, len(index), 500):
    # select first i rows based on shuffled index
    df_subset = df.iloc[index[:i]]
    duration = df_subset.duration__segments.sum() / 3600
    print(f'{i:3} {duration:.2f}')
    if duration >=hours_limit:
        break


























# %%
