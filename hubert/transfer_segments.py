# %%
import os
import pandas as pd
import datetime


df = pd.read_csv('/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path_large.csv', sep='\t')
df['tmp_path'] = df.segment_path.str.split('/', ).str[:-2].str.join('/')
df['id'] = df.segment_path.str.split('/', ).str[-2:].str.join('/')
index_df = df[['id', 'tmp_path', 'type']].set_index('id').rename(columns={'tmp_path':'path'})

path_mapping = index_df.to_dict('index')
# %%

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, split, concat, lit
import pickle

def do_my_logging(log_msg):
    # logger = logging.getLogger('__FILE__')
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('{} :: {}'.format(now, log_msg))


spark = SparkSession.builder.appName('name').getOrCreate()
clusering_dir = '/lnet/express/work/people/stankov/alignment/clustering_all'
output_dir = '/lnet/express/work/people/stankov/alignment/clustering_all/segments'

do_my_logging(f'clustering_dir = {clusering_dir}')
do_my_logging(f'output_dir = {output_dir}')

for f in os.listdir(clusering_dir):
    p = os.path.join(clusering_dir, f)
    if not f.endswith('.csv') and os.path.isfile(p):
        do_my_logging(f'removing {p}')
        os.remove(p)

do_my_logging('reading')
cl_df = spark.read.csv(clusering_dir, header=True, inferSchema=True)

do_my_logging('getting unique segments')
split_col = split(cl_df['path'], '/')
cl_df = cl_df.withColumn('mp3', split_col.getItem(0))
cl_df = cl_df.withColumn('segm', split_col.getItem(1))
cl_df = cl_df.withColumn('segm', concat(col('mp3'), lit('/'), col('segm')))
mp3s = cl_df.select('mp3').distinct().collect()
mp3s = [s.mp3 for s in mp3s]

do_my_logging('splitting all mp3s into folders')
buffer = []
buffer_limit = len(mp3s) // 100
mp3s_divided = []
mp3_to_int = {}
for mp3 in mp3s:
    if len(buffer) >= buffer_limit:
        mp3s_divided.append(buffer)
        buffer = []
    buffer.append(mp3)
    mp3_to_int[mp3] = len(mp3s_divided)

if buffer != []:
    mp3s_divided.append(buffer)

do_my_logging(f'total folders = {len(mp3s_divided)}')
do_my_logging(f'total_mp3s {len(mp3s):,}, mp3s divided to folders {sum(len(x) for x in mp3s_divided):,}')

do_my_logging('writing segments')
for i, lst in enumerate(mp3s_divided):
    tmp_df = cl_df.where(col('mp3').isin(lst))
    tmp_df.write.csv(os.path.join(output_dir, f'{i}'), header=True, mode='overwrite')

    ref_set = set(lst)
    my_set = set([x.mp3 for x in tmp_df.select('mp3').distinct().collect()])
    if ref_set != my_set:
        do_my_logging(f'{i}, ref is not equal to my')
        do_my_logging(f'ref = {ref_set}')
        do_my_logging(f' my = {my_set}')

    if i % 5 == 0:
        do_my_logging(f'{i:4}/{len(mp3s_divided)} ({i/len(mp3s_divided) * 100:.2f}%)')


do_my_logging(f'pickling {os.path.join(output_dir, "mp3_to_int.pickle")}')
with open(os.path.join(output_dir, "mp3_to_int.pickle"), 'wb') as handle:
    pickle.dump(mp3_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

do_my_logging(f'pickling {os.path.join(output_dir, "int2mp3.pickle")}')
with open(os.path.join(output_dir, "int2mp3.pickle"), 'wb') as handle:
    pickle.dump(mp3s_divided, handle, protocol=pickle.HIGHEST_PROTOCOL)

do_my_logging('done, exiting')
# %%
