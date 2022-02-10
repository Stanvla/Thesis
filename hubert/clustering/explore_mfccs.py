# %%
import pandas as pd
import os

mfccs_dir = '/lnet/express/work/people/stankov/alignment/mfcc'

# %%
df_list = []
for i, fn in enumerate(os.listdir(mfccs_dir)):
    if i == 10:
        break
    if not fn.endswith('.csv'):
        pass
    df_list.append(pd.read_csv(os.path.join(mfccs_dir, fn)))
df = df_list[0]
# %%
df[['sentence', 'segment', 'id']] = df.path.str.split('/', expand=True)
