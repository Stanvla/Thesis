# %%
import argparse
import os
from icecream import ic
import shutil
parser = argparse.ArgumentParser()

parser.add_argument("--parallel_jobs_cnt",
                    default=100,
                    type=int,
                    help="Number of jobs that will run parallel.")

parser.add_argument("--verticals_src_dir",
                    default='/lnet/express/work/people/stankov/alignment/results/full/merged/',  # lnet
                    type=str,
                    help="Directory where verticals are stored")


args = parser.parse_args([] if "__file__" not in globals() else None)

# clean output directory
output_dir = '/lnet/express/work/people/stankov/alignment/mfccs/merged'
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

mfcc_dir = '/lnet/express/work/people/stankov/alignment/mfccs'
working_dir = '/lnet/express/work/people/stankov/alignment/Thesis/HuBERT/clustering'
jobs_dir = os.path.join(working_dir, 'jobs')
scripts_dir = os.path.join(jobs_dir, 'scripts')
success_dir = os.path.join(scripts_dir, 'success')
config_file = os.path.join(working_dir, 'config_merge')

# first clean the directory
if os.path.exists(jobs_dir):
    shutil.rmtree(jobs_dir)

# then create new empty directory
if not os.path.exists(success_dir):
    os.makedirs(success_dir)

# read all segments for each mp3
mp3_sizes = {}
for mp3 in os.listdir(mfcc_dir):
    mp3_size = 0
    mp3_path = os.path.join(mfcc_dir, mp3)
    if not os.path.isdir(mp3_path):
        continue

    if len(os.listdir(mp3_path)) == 0:
        ic(mp3_path)
    else:
        mp3_sizes[mp3] = len(os.listdir(mp3_path))

# %%
import pickle

total_size = sum(mp3_sizes.values())
job_max_size = total_size / args.parallel_jobs_cnt

jobs = []
job = []
acc = 0
job_id = 0

for mp3, size in mp3_sizes.items():
    job.append(mp3)
    acc += size

    if acc >= job_max_size:
        job_file = os.path.join(jobs_dir, f'{job_id}.pkl')
        with open(job_file, 'wb') as f:
            pickle.dump(job, f)
        jobs.append(dict(file=job_file, job_id=job_id))

        job = []
        acc = 0
        job_id += 1
# %%
import subprocess
from time import sleep

# read configs for qsub
with open(config_file, 'r') as f:
    header = f.read()

scripts = []
for i, job in enumerate(jobs):
    command = f'python {os.path.join(working_dir, "df_merge_single.py")} --id={job["job_id"]} --file={job["file"]} --success={success_dir}'

    script_path = os.path.join(scripts_dir, f'{i}.sh')
    # print('\n'.join([header.format(i), command]))
    with open(script_path, 'w') as f:
        f.write('\n'.join([header.format(i, f'{0.5 * i + 0.1}'), command]))

    scripts.append(script_path)

for script in scripts:
    subprocess.run(f'qsub {script}'.split())