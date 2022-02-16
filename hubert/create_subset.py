# %%
import os
import pandas as pd
import random
from clustering.torch_mffc_extract import clean_data
from icecream import ic
import pickle
import shutil
from tqdm import tqdm
import logging
# %%

def get_subset(df_path, clean_params, int2mp3, idx):
    df = pd.read_csv(df_path, sep='\t')
    df = clean_data(df, clean_params).reset_index(drop=True)
    df['mp3'] = df.mp3_name.str.split('_').str[1]
    mp3s = [mp3 for lst in int2mp3[:idx] for mp3 in lst]
    return df[df['mp3'].isin(mp3s)].reset_index(drop=True)

logging.basicConfig(
    filename="/lnet/express/work/people/stankov/alignment/Thesis/subset.log",
    filemode='w',
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S %d.%m.%Y'
)

subset_path = '/lnet/express/work/people/stankov/alignment/subset'

with open('/lnet/express/work/people/stankov/alignment/clustering_all/segments/int2mp3.pickle', 'rb') as f:
    int2mp3 = pickle.load(f)

with open('/lnet/express/work/people/stankov/alignment/clustering_all/segments/mp3_to_int.pickle', 'rb') as f:
    mp3_to_int = pickle.load(f)


# %%

parczech_clean_params = dict(
    recognized_sound_coverage__segments_lb=0.45,
    recognized_sound_coverage__segments_ub=0.93,
    duration__segments_lb=0.5,
)
subset = get_subset(
    '/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path_large.csv',
    parczech_clean_params,
    int2mp3,
    5
)
logging.debug(f'contains {subset.duration__segments.sum() / 3600:.3f} hours')
# %%
subset['output_path'] = subset_path + '/' + subset.segment_path.str.split('/').str[-3:].str.join('/')
logging.debug(f'Nan values {subset.isnull().values.any()}')


# %%
for i, row in enumerate(subset.itertuples()):
    if i % 500 == 0:
        logging.debug(f'{i:5}/{len(subset)}, ({100 * i / len(subset):.3f}%)')

    if not os.path.isdir(row.output_path):
        os.makedirs(row.output_path)
    # the only files needed
    files = [
        f'{row.mp3}.prt',
        f'{row.mp3}.asr',
        f'{row.mp3}.wav',
    ]

    for f in files:
        src_path = os.path.join(row.segment_path, f)
        trg_path = os.path.join(row.output_path, f)

        if not os.path.isfile(src_path):
            logging.debug(f'file does not exist {trg_path}')

        if not os.path.isfile(trg_path):
            shutil.copyfile(src_path, trg_path)


subset.to_csv(os.path.join(subset_path, 'subset.csv'), index=False)
logging.debug('zipping subset')
shutil.make_archive('subset_arch', 'zip', subset_path)
logging.debug('done, exiting')


























# %%
