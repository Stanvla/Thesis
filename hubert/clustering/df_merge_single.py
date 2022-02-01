# %%
import argparse
import numpy as np
import os
import pickle
import pandas as pd
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--id", default='197', type=str, help="job id")
parser.add_argument("--file", default='/lnet/express/work/people/stankov/alignment/Thesis/HuBERT/clustering/jobs/197.pkl', type=str, help="file list")
parser.add_argument("--success", default='/lnet/express/work/people/stankov/alignment/Thesis/HuBERT/clustering/jobs/197.pkl', type=str, help="file list")

args = parser.parse_args([] if "__file__" not in globals() else None)
logging.basicConfig(filename=f"{args.id}.log", filemode='w', level=logging.DEBUG, format='[%(levelname)s] %(asctime)s - %(message)s', datefmt='%H:%M:%S '
                                                                                                                                              '%d.%m.%Y')
# script will receive its id and a file with directories

# read all directory names
mfcc_dir = '/lnet/express/work/people/stankov/alignment/mfccs'
output_dir = '/lnet/express/work/people/stankov/alignment/mfccs/merged'

id = args.id

with open(args.file, 'rb') as f:
    dirs = pickle.load(f)

logging.debug('removing pickle')
os.remove(args.file)

logging.debug(f'start merging {len(dirs)} dirs')
# iterate over directories
mp3_buffer = []
for i, d in enumerate(dirs):
    dir_path = os.path.join(mfcc_dir, d)
    # read new_segments
    segments = []
    processed = 0
    for segm in os.listdir(dir_path):
        if not segm.endswith('.csv'):
            continue
        processed += 1
        df = pd.read_csv(os.path.join(dir_path, segm))

        # bad dataframe
        if len(df.columns) != 40:
            columns = '[' + ",".join(f'{c}' for c in df.columns) + ']'
            logging.error(f'Bad dataframe :: mp3={d}, segment={segm},  columns={columns} shape={df.shape}')
            continue

        df = df.astype({f'{i}': np.float16 for i in range(39)})
        segments.append(df)

    # store new_segments
    if segments == [] or processed == 0:
        logging.error(f'No valid segments from dir {d}')
    else:
        mp3_buffer.append(pd.concat(segments, axis=0, ignore_index=True))

logging.debug('finished merging')
logging.debug('save merged')
merged = pd.concat(mp3_buffer, axis=0, ignore_index=True)
merged.to_csv(
    os.path.join(output_dir, f'{id}.csv'),
    index=False,
)
logging.debug('done')
with open(os.path.join(args.success, id), 'w') as f:
    f.write('ok')
