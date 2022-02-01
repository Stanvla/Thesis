# %%
import ast
import os
from pprint import pprint
import numpy as np
import pandas as pd
# base_dir = os.path.dirname(os.getcwd())
base_dir = '/lnet/express/work/people/stankov/alignment'

# load the df with metadata
splitted_df = pd.read_csv(os.path.join(base_dir, 'split_filelist.csv'), sep='\t')
splitted_df['speakers__segments'] = splitted_df['speakers__segments'].apply(lambda x: ast.literal_eval(x))

rename = dict(
    context_test='parczech-3.0-asr-context.test/',
    context_dev='parczech-3.0-asr-context.dev/',

    segments_test='parczech-3.0-asr-segments.test/',
    segments_dev='parczech-3.0-asr-segments.dev/',

    speakers_test='parczech-3.0-asr-speakers.test/',
    speakers_dev='parczech-3.0-asr-speakers.dev/',

    train='parczech-3.0-asr-train/',
    other='parczech-3.0-asr-other/'
)
segment_base_dir = '/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/'
# splitted_df.path is path but in another folder, so it is a correct local path (last two folders) if we change the base folder
#   for example '/lnet/express/work/people/stankov/alignment/results/full/segments-aligned/jan/sentences_2017060712481302/061'
# also each mp3 is stored in  'sentence_{}' folder, remove 'sentence_' prefix from the folder name
# finally, map the dataset type to its path using dict
splitted_df['segment_path'] = segment_base_dir + splitted_df.type.map(rename) + splitted_df.path.str.split('/').str[-2:].str.join('/').str.split('_').str[-1]
# now the segment path is
#  '/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/parczech-3.0-asr-other/2017060712481302/061'

# %%
df = splitted_df.drop(columns=['path', 'name'])
df.to_csv(os.path.join(base_dir, 'Thesis', 'clean_with_path.csv'), sep='\t', index=False)
# df = pd.read_csv(os.path.join(base_dir, 'clean_with_path.csv'), sep='\t')
