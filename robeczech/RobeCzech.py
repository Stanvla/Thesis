# %%
import ast
import os
from pprint import pprint
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def get_robeczech():
    checkpoint = 'ufal/robeczech-base'
    robeCzech = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return robeCzech, tokenizer


# %%

base_dir = os.path.dirname(os.getcwd())

# load the df with metadata
splitted_df = pd.read_csv(os.path.join(base_dir, 'split_filelist.csv'), sep='\t')
splitted_df['speakers__segments'] = splitted_df['speakers__segments'].apply(lambda x: ast.literal_eval(x))
# %%

rename = dict(
    context_dev='parczech-3.0-asr-context.dev',
    context_test='parczech-3.0-asr-context.test',
    train='parczech-3.0-asr-train',
    other='parczech-3.0-asr-other'
)


# %%
# replace all values in the "type" column using dict
splitted_df = splitted_df.replace({'type':  rename})
# rename column "segment_index__segments" to "seg_ind"
splitted_df = splitted_df.rename(columns=dict(segment_index__segments="seg_ind"))
# get only the mp3 name without prefix
splitted_df.mp3_name = splitted_df.mp3_name.str.split('_').str[-1]

# add path to the segment transcript
segment_base_dir = '/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT'

# purpose of the tricks below is to pad 'seg_ind' with correct number of zeros
#   (each mp3 has different number of segments, so the segments will be padded  differently, depending on the number of segments for given mp3)

# first get "length" of the segment index, computed as length of the string
splitted_df['segment_str_len'] = splitted_df.seg_ind.map(str).str.len()
# here comes the magic,
#   * for all mp3s compute the max str_length of the segment
#     (the magic is that groupby will return only as many rows, as the number of mp3s, but transform function will copy the max value for each row
#      given the mp3 name. As a result each row depending on the mp3 name will have its own max value, obviously for same mp3s the max value will be the same)
#   * Then subtract 'segment_str_len' from max_len, so the "longest" segment will not be padded but the "shortest" will be padded with max_length-1 zeros
splitted_df['zeros_pref_size'] = splitted_df.groupby('mp3_name').segment_str_len.transform('max') - splitted_df.segment_str_len
# add dummy column with all zeros,
#   unfortunately we can not simply write '0'*df.zeros_pref_size
# so this column will be further repeated "df.zeros_pref_size" times
splitted_df['zero'] = '0'
# construct the path to the text file
splitted_df['text_path'] = segment_base_dir + \
                           '/' + splitted_df.type + \
                           '/' + splitted_df.mp3_name + \
                           '/' + splitted_df.zero.str.repeat(splitted_df.zeros_pref_size) + splitted_df.seg_ind.map(str)+ \
                           '/' + splitted_df.mp3_name + '.prt'

# get splits of the dataset
train_df = splitted_df[splitted_df.type == 'parczech-3.0-asr-train']
other_df = splitted_df[splitted_df.type == 'parczech-3.0-asr-other']
context_dev = splitted_df[splitted_df.type == 'parczech-3.0-asr-context.dev']
context_test = splitted_df[splitted_df.type == 'parczech-3.0-asr-context.test']

# %%

# ['speakers__segments',
#  'words_cnt__segments',
#  'chars_cnt__segments',
#  'duration__segments',
#  'speakers_cnt__segments',
#  'missed_words__segments',
#  'missed_words_percentage__segments',
#  'missed_chars__segments',
#  'missed_chars_percentage__segments',
#  'recognized_sound_coverage__segments',
#  'correct_end__segments',
#  'avg_char_duration__segments',
#  'std_char_duration__segments',
#  'median_char_duration__segments',
#  'char_duration_60__segments',
#  'char_duration_70__segments',
#  'char_duration_75__segments',
#  'char_duration_80__segments',
#  'char_duration_90__segments',
#  'avg_norm_word_dist__segments',
#  'std_norm_word_dist__segments',
#  'median_norm_word_dist__segments',
#  'char_norm_word_dist_60__segments',
#  'char_norm_word_dist_70__segments',
#  'char_norm_word_dist_75__segments',
#  'char_norm_word_dist_80__segments',
#  'char_norm_word_dist_90__segments',
#  'avg_norm_word_dist_with_gaps__segments',
#  'std_norm_word_dist_with_gaps__segments',
#  'median_norm_word_dist_with_gaps__segments',
#  'char_norm_word_dist_with_gaps_60__segments',
#  'char_norm_word_dist_with_gaps_70__segments',
#  'char_norm_word_dist_with_gaps_75__segments',
#  'char_norm_word_dist_with_gaps_80__segments',
#  'char_norm_word_dist_with_gaps_90__segments',
#  'mp3_name',
#  'segment_index__segments',
#  'continuous_gaps_cnt_normalized1__file',
#  'normalized_dist_with_gaps_80__file',
#  'name',
#  'path',
#  'type']

# pprint(splitted_df.columns.tolist())

# %%
# create RobeCzech fine-tune datasets : extended_train, extended_dev, extended_test with no constraint on the recognition quality,

# add missing segments
def extend_df(source_df, other_df, columns):
    source_mp3s = set(source_df.mp3_name.unique().tolist())

    # the resulting df will have columns [mp3_name, segment_index, path]
    result_df = source_df[columns].copy()

    # extract segments that belong to source mp3s, select columns, copy
    recover_df = other_df[other_df.mp3_name.isin(source_mp3s)]
    recover_df = recover_df[columns].copy()

    # remove selected segments from other_df
    modified_other_df = other_df.drop(
        other_df[other_df.mp3_name.isin(source_mp3s)].index,
        axis=0
    ).copy()

    return result_df.append(recover_df).sort_values(['mp3_name', 'seg_ind']).reset_index(drop=True), modified_other_df

columns = ['mp3_name', 'seg_ind', 'type', 'text_path']

# start with dev and test and extend them with missing segments
extended_test, reduced_other = extend_df(context_test, other_df, columns)
extended_dev, reduced_other = extend_df(context_dev, reduced_other, columns)
# then extend train with rest from other_df
extended_train = train_df.append(reduced_other)[columns].sort_values(['mp3_name', 'seg_ind']).reset_index(drop=True)

# %%
def parse_mp3_name(name):
    year = name[:4]
    month = name[4: 6]
    day = name[6: 8]
    start_hours = name[8:10]
    start_mins = name[10: 12]
    end_hours = name[12: 14]
    end_mins = name[14: ]
    return f'{day}.{month}.{year} {start_hours}:{start_mins}-{end_hours}:{end_mins}'

for i, mp3 in enumerate(extended_dev.mp3_name.unique().tolist()):
    if i>2:break
    print(mp3)
    print(parse_mp3_name(mp3))
    text = []

    for segment_path in extended_dev[extended_dev.mp3_name == mp3].text_path.tolist():
        with open(segment_path, 'r') as f:
            text.append(f.read())
    print(''.join(text))
# %%
extended_train.text_path[0]
# %%

context_dev_mp3s = extended_dev.mp3_name.unique()




























# %%
