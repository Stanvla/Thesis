# %%
# ## Segment statistics
#   Statistics for each segment (for example `2018011711181132/01/stats.tsv`):
#        * `words_cnt` - number of words
#        * `chars_cnt` - sum of the lengths of words
#        * `duration` -  duration of the segment in seconds
#        * `speakers_cnt` - number of speakers
#        * `missed_words` - number of missed words
#        * `missed_words_percentage` - `missed_words` normalized by the number of words
#        * `missed_chars` - sum of lengths of missed words
#        * `missed_chars_percentage` - `missed_chars` normalized by the sum of lengths of the words
#        * `recognized_sound_coverage` - percentage of audio length that is covered by some original words aligned to non empty strings
#        * `correct_end` - if segment has correct ending time (only the case for the last segment in the audio)
#        * `avg_char_duration` - this is average of some Xi, where Xi = duration(word_i)/len(word_i)
#        * `std_char_duration` - standard deviation for the above statistic
#        * `median_char_duration` - instead of computing average in avg_char_duration, we compute the median
#        * `char_duration_{60, 70, 75, 80, 90}` - Nth percentile variant for the median_char_duration
#        * `avg_norm_word_dist` - average of the normalized edit distances (without gaps, i.e. when original word is aligned to an empty string) for each
#        word in the segment, as  discussed in the global statistics. However here we take into account words of any length.
#        * `std_norm_word_dist` - standard deviation of the normalized edit distance (without gaps)
#        * `median_norm_word_dist` - median of the normalized edit distance (without gaps)
#        * `char_norm_word_dist_{60, 70, 75, 80, 90}` - Nth percentile of the normalized edit distance (without gaps), normailization is done by the maximum
#        of the lengths
#        * `avg_norm_word_dist_with_gaps` - equivalent to `avg_norm_word_dist`, but now we are not ignoring gaps
#        * `std_norm_word_dist_with_gaps` - equivalent to `std_norm_word_dist`, but now we are not ignoring gaps
#        * `median_norm_word_dist_with_gaps` -  equivalent to `median_norm_word_dist`, but now we are not ignoring gaps
#        * `char_norm_word_dist_with_gaps_{60, 70, 75, 80, 90}` - equivalent to `char_norm_word_dist_N`, but now we are not ignoring gaps
# %%
import ast
import os
import pickle
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

cwd = os.getcwd()
# get parent directory of current working directory with python scripts
base_dir = os.path.dirname(cwd)

pprint(sorted(os.listdir(base_dir)))
print(os.path.dirname(cwd))

# %%

# load the df with metadata
splitted_df = pd.read_csv(os.path.join(base_dir, 'split_filelist.csv'), sep='\t')
splitted_df['speakers__segments'] = splitted_df['speakers__segments'].apply(lambda x: ast.literal_eval(x))

# %%
# show columns
pprint(splitted_df.columns.tolist())

# show all types of datasets ... train, other, context_dev ....
pprint(splitted_df.type.unique().tolist())

# %%

# get splits of the dataset
train_df = splitted_df[splitted_df.type == 'train']
other_df = splitted_df[splitted_df.type == 'other']
context_dev = splitted_df[splitted_df.type == 'context_dev']
context_test = splitted_df[splitted_df.type == 'context_test']

# %%
train_df.head()
# %%

# extract mp3 names
train_mp3_names = train_df.mp3_name.unique().tolist()
other_mp3_names = other_df.mp3_name.unique().tolist()
context_dev_mp3_names = context_dev.mp3_name.unique().tolist()
context_test_mp3_names = context_test.mp3_name.unique().tolist()

# %%

pprint(len(context_test_mp3_names))


# %%

# display continuity of specific part of the dataframe
# need to pass this specific part [train, other, ...] as df
# also pass names of all mp3 files in this df

def display_continuity(mp3_names, df, title, img_name=None, show=True):
    plt.rcParams["figure.figsize"] = (50, 12)
    plt.grid(True, axis='y', linestyle=':')

    # M is for plotting (y axis)
    M = 0

    for i, mp3 in enumerate(mp3_names):
        # get all segments in the current mp3 and sort them
        segments = sorted(df[df.mp3_name == mp3].segment_index__segments.tolist())
        # y_axis
        ys = [i] * len(segments)
        # segments is x_axis
        plt.scatter(ys, segments, marker='|')

        M = max(max(segments), M)

    plt.title(title)
    plt.xlabel('Mp3')
    plt.ylabel('Segments')
    plt.yticks(np.arange(0, M + 1, 2))
    plt.xlim(-5, len(mp3_names) + 5)

    if show:
        plt.show()
    else:
        if img_name is None:
            raise ValueError('No name provided for the image.')
        plt.savefig(f'{img_name}.png')


# %%
# train_mp3_subset = train_mp3_names[:500]
# title = Training set continuity [500/20000]
display_continuity(
    context_dev_mp3_names,
    context_dev,
    'Continuity of context dev',
    'context_dev_cont',
    show=True
)


# %%
# we can ideally fix `context_dev/test` and `train` dataframes by inserting all missing segments from other df
# since it is an ideal case, only "good enough" segments will be inserted

# first for each dataframe create dictionary ... {mp3_name_in_df: [segments_in_df]}
def get_segments_by_mp3(df):
    all_mp3 = df.mp3_name.unique().tolist()
    result = {}
    for mp3 in tqdm(all_mp3):
        result[mp3] = df[df.mp3_name == mp3].segment_index__segments.tolist()
    return result


# create dictionaries for `context_test/dev` and `other` dataframes
cont_dev_segments_by_mp3 = get_segments_by_mp3(context_dev)
cont_test_segments_by_mp3 = get_segments_by_mp3(context_test)

# dictionary for `other` df
# here it takes about 11 minutes, so rather save/load the dict
if not os.path.isfile('other_segments_by_mp3.pickle'):
    other_segments_by_mp3 = get_segments_by_mp3(other_df)
    with open('other_segments_by_mp3.pickle', 'wb') as handle:
        pickle.dump(other_segments_by_mp3, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('other_segments_by_mp3.pickle', 'rb') as handle:
        other_segments_by_mp3 = pickle.load(handle)

# dictionary for `train` df
if not os.path.isfile('train_segments_by_mp3.pickle'):
    train_segments_by_mp3 = get_segments_by_mp3(train_df)
    with open('train_segments_by_mp3.pickle', 'wb') as handle:
        pickle.dump(train_segments_by_mp3, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('train_segments_by_mp3.pickle', 'rb') as handle:
        train_segments_by_mp3 = pickle.load(handle)


# %%

# *for each mp3* create a sorted list of lists [[start_time, end_time, origin], ...] for each segment,
# where origin is 1 original and 0 if  fall_back
# some segment_ids may be missing,  for this reason insert dummy segments with fixed some fixed duration (2 seconds) and origin 2
def extract_mp3_segments(mp3, source_segments, other_segments, df_source, df_fall_back):
    # find max segment index
    M = max(max(source_segments), max(other_segments) if other_segments != {} else 0)

    all_segments = []
    time_elapsed = 0
    for segment_id in range(0, M + 1):
        # these are default values, meaning that segment is missing
        origin = 0
        duration_sec = 5

        if segment_id in source_segments:
            origin = 2
            duration_sec = df_source[(df_source.mp3_name == mp3) & (df_source.segment_index__segments == segment_id)].duration__segments.item()

        if segment_id in other_segments:
            origin = 1
            duration_sec = df_fall_back[(df_fall_back.mp3_name == mp3) & (df_fall_back.segment_index__segments == segment_id)].duration__segments.item()

        all_segments.append([1, time_elapsed, time_elapsed + duration_sec, origin])
        time_elapsed += duration_sec
    return all_segments


# given list of segments merge together segments with the same origin
def get_merged_segments_by_origin(all_segments_unmerged, duration_constraint=0):
    # a list of lists [[start_time, end_time, origin], ...] where inner list represent one segment
    # depending on the caller origin can be a number from different range, the number itself gives the origin of the segment

    # merge together segments with same origin
    merged = []

    origin_durations = [0, 0, 0]
    # count how many segments were merged together
    merged_cnt = 0
    _, current_start, current_end, current_origin = all_segments_unmerged[0]
    for i, (cnt, start_time, end_time, origin) in enumerate(all_segments_unmerged):
        # segment from another origin appeared or it is the last segment
        if (origin != current_origin):
            # add only segments that satisfy the duration constraint
            if (current_end - current_start >= duration_constraint) and current_origin != 0:
                merged.append([merged_cnt, current_start, current_end, current_origin])
                origin_durations[current_origin] += current_end - current_start
            else:
                # here we have segments that are either shorter or are missing
                merged.append([merged_cnt, current_start, current_end, 0])
                origin_durations[0] += current_end - current_start

            current_start = start_time
            current_end = end_time
            current_origin = origin
            merged_cnt = cnt
        else:
            # segment continues, so update end and increment counter of merged_segments
            current_end = end_time
            merged_cnt += cnt

    # check if last segment was added correctly using last end time
    if current_end != merged[-1][2]:
        if (current_end - current_start >= duration_constraint) and current_origin != 0:
            merged.append([merged_cnt, current_start, current_end, current_origin])
            origin_durations[current_origin] += current_end - current_start
        else:
            # here we have segments that are either shorter or are missing
            merged.append([merged_cnt, current_start, current_end, 0])
            origin_durations[0] += current_end - current_start

    return merged, origin_durations


def add_mp3_to_plot(axis, mp3_id, segments, colors, labels, line_widths):
    # iterate over different origins
    for current_origin, (color, label, line_width) in enumerate(zip(colors, labels, line_widths)):
        label_shown = False
        for cnt, start, end, origin in segments:
            if current_origin != origin:
                continue
            else:
                # add label if was not previously shown
                plot_label = label if not label_shown and mp3_id == 0 else None
                # now treat time_start and time_end as y_coordinate, x_coordinate will be mp3 id
                axis.plot([mp3_id, mp3_id], [start, end], color=color, linewidth=line_width, label=plot_label)
                label_shown = True
    return axis


def get_time_duration(duration):
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    return f'{minutes}:{seconds}'


# above function on the y axis has segment ID, here y axis will have time on the y axis
def display_time_continuity(segments_by_mp3, segments_by_mp3_fall_back, df, df_fall_back, duration_constraint, title, img_name=None, show=True):
    # plotting settings
    # clear previous plot
    plt.clf()
    plt.close('all')
    # figure width and height
    plt.rcParams["figure.figsize"] = (40, 26)
    plotting_params = dict(
        fall_back=dict(
            colors=['black', 'red', 'blue'],
            labels=['missing', 'fall_back', 'original'],
            line_widths=[0.5, 1.5, 1.5]
        ),
        new_dataset=dict(
            colors=['black', 'blue'],
            labels=['missing', 'using'],
            line_widths=[0.5, 1.5]
        )
    )
    fig, (ax1, ax2) = plt.subplots(2)

    # duration of original segments in the dataset (no constraint)
    original_duration = 0
    # duration of fall_back segments (no constraint)
    fall_back_duration = 0
    # duration of original segments WITH constraint
    passed_original_duration = 0
    # duration of the new dataset = original + fall_back with time constraint
    improved_passed_original_duration = 0
    max_duration = 0

    # iterate over the mp3 and plot segments,
    #   * first plot the case when we have NO time constraint and use fall_back segments
    #   * second plot the case when we put a duration constraint and use fall_back segments as normal segments
    #   (do not distinguish original and fall_back segments)
    for i, (mp3, orig_segments) in enumerate(tqdm(segments_by_mp3.items())):
        segments = sorted(df[df.mp3_name == mp3].segment_index__segments.tolist())

        # create sets for better lookup
        source_segments = set(segments)
        other_segments = set(segments_by_mp3_fall_back[mp3]) if mp3 in segments_by_mp3_fall_back else {}

        # create a list of lists [[1, start_time, end_time, origin], ...] where inner list represent one segment
        # origin can be 0=missing, 1=fall_back, 2=original_segment
        all_segments_unmerged = extract_mp3_segments(mp3, source_segments, other_segments, df, df_fall_back)

        # this is a list of lists, where each inner list is [merged_cnt, start_time, end_time, origin]
        # origin = {original_df=2, fall_back_df=1, missing=0}
        all_segments_merged, origin_durations1 = get_merged_segments_by_origin(all_segments_unmerged=all_segments_unmerged)
        ax1 = add_mp3_to_plot(ax1, i, all_segments_merged, **plotting_params['fall_back'])

        # what if we use original data but with time constraint
        _, origin_durations2 = get_merged_segments_by_origin(all_segments_unmerged, duration_constraint)
        passed_original_duration += origin_durations2[2]


        # now we suppose that fall_back segments will be used together with source segments as one dataset
        # create new origin = {missing=0, used=1} and put a duration constraint on the segments and leave only long enough segments
        all_segments_unmerged_origin = [[cnt, start_time, end_time, 0 if origin == 0 else 1] for cnt, start_time, end_time, origin in all_segments_merged]
        all_segments_merged_origin, origin_durations3 = get_merged_segments_by_origin(all_segments_unmerged_origin, duration_constraint)

        ax2 = add_mp3_to_plot(ax2, i, all_segments_merged_origin, **plotting_params['new_dataset'])
        original_duration += origin_durations1[2]
        fall_back_duration += origin_durations1[1]
        improved_passed_original_duration += origin_durations3[1]

        max_duration = max(max_duration, sum(origin_durations1))

    # add title for the whole figure
    fig.suptitle(title)

    ax1.grid(True, axis='y', linestyle=':')
    ax1.set_title(f'Adding good enough segments. Original duration {get_time_duration(original_duration)}. Good enough duration {get_time_duration(fall_back_duration)}. (no '
                  f'time constraint)')
    ax1.set_yticks(np.arange(0, max_duration + 20, 20))
    ax1.set_xlim(-2, i + 2)
    ax1.legend()

    ax2.grid(True, axis='y', linestyle=':')
    ax2.set_title(f'Using all good enough segments. Original duration {get_time_duration(passed_original_duration)}. Improved duration '
                  f'{get_time_duration(improved_passed_original_duration)}. (>= {duration_constraint})')
    ax2.set_yticks(np.arange(0, max_duration + 20, 20))
    ax2.set_xlim(-2, i + 2)
    ax2.legend()

    plt.xlabel('Mp3')
    plt.ylabel('Time in seconds')

    if show:
        plt.show()
    else:
        if img_name is None:
            raise ValueError('No name provided for the image.')

        if os.path.exists(f'{img_name}.png'):
            os.remove(f'{img_name}.png')
        plt.savefig(f'{img_name}.png')


# %%

# filter other df to remove discontinuity in the

def explore_statistic(base_directory, statistic, point_list, df, N=30, delta=1.0):
    zipped = False
    print()
    print(statistic)
    percentiles = [i / len(df) for i in range(len(df))]
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(18, 12))
    sorted_df = df.sort_values(statistic)
    sorted_df['percentiles'] = percentiles
    plt.plot(sorted_df[statistic], percentiles)
    deltas = [delta] * len(point_list)

    _min = df[statistic].min()
    _Max = df[statistic].max()
    length = _Max - _min
    for p, delta in zip(point_list, deltas):
        # directory = shared.create_dir_if_not_exist([base_directory, statistic, f'{p - delta:.1f}--{p:.1f}--{p + delta:.1f}'])
        directory = None
        l, m, u = p / 100 - delta / 100, p / 100, p / 100 + delta / 100
        accepted = sorted_df[(sorted_df['percentiles'] > l) & (sorted_df['percentiles'] <= m)].df(N)
        rejected = sorted_df[(sorted_df['percentiles'] > m) & (sorted_df['percentiles'] <= u)].df(N)
        threshold = sorted_df[(sorted_df['percentiles'] - m).abs() < 0.001][statistic]
        threshold = threshold.mean()
        print(f'{m} perc of {statistic} = {threshold:.3f} ')
        paths_accepted = accepted['path']
        paths_rejected = rejected['path']

        if any([os.path.isdir(f) for f in paths_accepted]):
            pass
            # zipit(paths_accepted, os.path.join(directory, 'accepted.zip'))
            # zipit(paths_rejected, os.path.join(directory, 'rejected.zip'))

        plt.axhline(m, linestyle=':', color='red')
        plt.axvline(threshold, linestyle=':', color='green')
        plt.text(threshold - length * 0.013, 1, f'{threshold:.3f}', rotation=90, color='green')
        plt.text(threshold - length * 0.013, 0.01, f'{m:.3f}', rotation=90, color='green')

        plt.text(_min + length * 0.01, m + 0.005, f'{m:.3f}', color='red')
        plt.text(_Max - length * 0.01, m + 0.005, f'{threshold:.3f}', color='red')

    # plt.grid(linestyle='-.')
    plt.ylabel('Percentiles')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.linspace(_min, _Max, 11))
    plt.title(f'{statistic}')
    fig.savefig(os.path.join(base_directory, statistic, f'{statistic}.png'))


def filter_df(df,
              missed_chars_percentage__segments1=0.065,
              char_norm_word_dist_80__segments=0.3,
              std_norm_word_dist_with_gaps__segments=0.334,
              recognized_sound_coverage__segments=0.625,
              std_char_duration__segments=0.175,
              avg_char_duration__segments=0.175,
              char_norm_word_dist_with_gaps_90__segments=0.795,
              avg_norm_word_dist__segments=0.165,
              duration__segments_ub=54,
              duration__segments_lb=0.81,
              missed_words_percentage__segments=0.118,
              missed_chars_percentage__segments2=0.052,
              char_norm_word_dist_with_gaps_60__segments=0.11,
              median_norm_word_dist__segments=0.01,
              ):
    # default filtering scheme

    # missed_chars_percentage__segments             <=  0.065
    # char_norm_word_dist_80__segments              <=  0.3
    # std_norm_word_dist_with_gaps__segments        <=  0.334
    # recognized_sound_coverage__segments           >=  0.625
    # std_char_duration__segments                   <=  0.175
    # avg_char_duration__segments                   <=  0.175
    # char_norm_word_dist_with_gaps_90__segments    <   0.795
    # avg_norm_word_dist__segments                  <   0.165
    # duration__segments                            <   54
    # duration__segments                            >   0.81
    # missed_words_percentage__segments             <   0.118
    # missed_chars_percentage__segments             <   0.052
    # char_norm_word_dist_with_gaps_60__segments    <   0.11
    # median_norm_word_dist__segments               <   0.01

    # filtered = df[df.missed_chars_percentage__segments <= 0.08]  # changed
    # filtered = filtered[filtered.char_norm_word_dist_80__segments <= 0.35]  # changed
    # filtered = filtered[filtered.std_norm_word_dist_with_gaps__segments <= 0.35]  # changed
    # filtered = filtered[filtered.recognized_sound_coverage__segments >= 0.6]  # changed

    filtered = df[df.missed_chars_percentage__segments <= missed_chars_percentage__segments1]  # changed
    filtered = filtered[filtered.char_norm_word_dist_80__segments <= char_norm_word_dist_80__segments]  # changed
    filtered = filtered[filtered.std_norm_word_dist_with_gaps__segments <= std_norm_word_dist_with_gaps__segments]  # changed
    filtered = filtered[filtered.recognized_sound_coverage__segments >= recognized_sound_coverage__segments]  # changed
    filtered = filtered[filtered.std_char_duration__segments <= std_char_duration__segments]
    filtered = filtered[filtered.avg_char_duration__segments <= avg_char_duration__segments]
    filtered = filtered[filtered.char_norm_word_dist_with_gaps_90__segments < char_norm_word_dist_with_gaps_90__segments]
    filtered = filtered[filtered.avg_norm_word_dist__segments < avg_norm_word_dist__segments]
    filtered = filtered[filtered.duration__segments < duration__segments_ub]
    filtered = filtered[filtered.duration__segments > duration__segments_lb]
    filtered = filtered[filtered.missed_words_percentage__segments < missed_words_percentage__segments]
    filtered = filtered[filtered.missed_chars_percentage__segments < missed_chars_percentage__segments2]
    filtered = filtered[filtered.char_norm_word_dist_with_gaps_60__segments < char_norm_word_dist_with_gaps_60__segments]
    filtered = filtered[filtered.median_norm_word_dist__segments < median_norm_word_dist__segments]

    # explore_statistic(directory_filtered, 'recognized_sound_coverage__segments', np.linspace(12.5, 20, 4), filtered)
    return filtered


# %%
filtered_df = filter_df(other_df)
filtered_segments_by_mp3 = get_segments_by_mp3(filtered_df)
# %%

display_time_continuity(
    cont_dev_segments_by_mp3,
    filtered_segments_by_mp3,
    context_dev,
    filtered_df,
    duration_constraint=7,
    title='Time continuity of context dev (missed_chars_percentage <= 0.08, char_norm_word_dist_80 <= 0.35, std_norm_word_dist_with_gaps__segments <= 0.35, '
    'recognized_sound_coverage >= 0.6)',
    img_name='time_context_dev_cont_real_01',
    show=False
)

# %%
display_time_continuity(
    cont_test_segments_by_mp3,
    filtered_segments_by_mp3,
    context_test,
    filtered_df,
    duration_constraint=7,
    title='Time continuity of context test (missed_chars_percentage <= 0.08, char_norm_word_dist_80 <= 0.35, std_norm_word_dist_with_gaps__segments <= 0.35, '
    'recognized_sound_coverage >= 0.6)',
    img_name='time_context_test_cont_real_01',
    show=False
)


# %%

# visualize ideal recovery (when all missing segments recovered from other df)
def recover(segments_by_mp3, fall_back_segments_by_mp3):
    # todo can apply filters for fall_back_segments
    #  ... say `missed_chars_percentage`,  `recognized_sound_coverage`
    for mp3, segments in segments_by_mp3.items():
        # create sets for better lookup
        source_segments = set(segments)
        other_segments = set(fall_back_segments_by_mp3[mp3])

        # find min and max segment index
        m = min(min(source_segments), min(other_segments))
        M = max(max(source_segments), max(other_segments))

        # result will be list of segments, where each segment is represented by an index converted to string
        # for segments from other their indices will be in brackets
        result = []
        discontinuous = False
        for i in range(m, M + 1):
            found = False
            if i in source_segments:
                result.append(f'{i}')
                found = True
            if i in other_segments:
                result.append(f'[{i}]')
                found = True
            # if for some reason the segment is not in either of the sets append the place-holder
            if not found:
                result.append('@')
                discontinuous = True

        result = '--'.join(result)
        print(f'{"cont" if not discontinuous else "disc"}:{mp3}::{result}')

    # todo :: after computing results for each mp3, can visualize recovery using display continuity
    # todo :: y axis can be time not just segments so y in range(0, 60 * 14)


recover(cont_test_segments_by_mp3, other_segments_by_mp3)
print()
recover(cont_dev_segments_by_mp3, other_segments_by_mp3)

# the strategy is the following fist try to recover using existing segments
# advanced will be to use even bad segments, extracting good words

# %%
os.listdir(os.getcwd())
# %%


cont_df = []
for mp3 in tqdm(train_mp3_names):
    sorted_segment_indices = sorted(train_df[train_df.mp3_name == mp3].segment_index__segments)
    cont_segments = []

    prev = min(sorted_segment_indices)
    start = prev
    segment_duration = 0
    segment_words = 0
    segment_chars = 0

    segments = []
    segment = []
    for segment_ind in sorted_segment_indices:
        if segment_ind - prev > 1:
            segments.append([start, prev])
            # segment = []
            # cont_segments.append(dict(mp3=mp3, start=start, end=prev, words=segment_words, duration=segment_duration, chars=segment_chars))
            # segment_words = 0
            # segment_duration = 0
            # segment_chars = 0
            start = segment_ind
        # segment.append(segment_ind)
        # segment_duration += train_df[(train_df.mp3_name == mp3) & (train_df.segment_index__segments == segment_ind)].duration__segments.item()
        # segment_words += train_df[(train_df.mp3_name == mp3) & (train_df.segment_index__segments == segment_ind)].words_cnt__segments.item()
        # segment_chars += train_df[(train_df.mp3_name == mp3) & (train_df.segment_index__segments == segment_ind)].chars_cnt__segments.item()
        prev = segment_ind
    cont_df.append(dict(mp3=mp3, segments=segments))
# %%
# pd.DataFrame(cont_df)
x = [dict(mp3=element['mp3'], start=s, end=e) for element in cont_df for s, e in element['segments']]
pd.DataFrame(x).to_csv('segments.csv', index=False)
# %%
cont_segments_df = pd.read_csv('segments.csv')
# %%
cont_segments_df['duration'] = 0
cont_segments_df['chars'] = 0
cont_segments_df['words'] = 0
cont_segments_df['segments'] = 0
# %%
n = 1000
train_df_mp3_subset = train_mp3_names[:n]
cont_segments_df_subset = cont_segments_df[cont_segments_df.mp3.isin(train_df_mp3_subset)]
# pprint(splitted_df.columns.tolist())

# %%
new = []
for mp3 in tqdm(train_df_mp3_subset):
    mp3_df = train_df[train_df.mp3_name == mp3]
    # mp3_segments = cont_segments_df[cont_segments_df.mp3 == mp3]
    for i, row in cont_segments_df[cont_segments_df.mp3 == mp3].iterrows():
        start = row.start
        end = row.end
        x = mp3_df[mp3_df.segment_index__segments.between(start, end)][['words_cnt__segments', 'chars_cnt__segments', 'duration__segments']].sum()
        duration = x['duration__segments'].item()
        words = x['words_cnt__segments'].item()
        chars = x['chars_cnt__segments'].item()
        new.append(dict(mp3=mp3, start=start, end=end, duration=duration, words=words, chars=chars))
        # print(start, end, x)

# %%
pd.DataFrame(new).head()
