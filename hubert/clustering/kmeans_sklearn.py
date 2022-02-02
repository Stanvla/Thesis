# %%
import os
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import numpy as np

# %%
if __name__ == '__main__':
    # %%
    source_dir = '/root/common_voice_data/cv-corpus-7.0-2021-07-21/cs/mffcs'
    dataframes = []
    for f in os.listdir(source_dir):
        if not f.endswith('.csv'):
            continue
        dataframes.append(pd.read_csv(os.path.join(source_dir, f)))
    df = pd.concat(dataframes)
    # %%
    clusters = [10, 25, 50]
    X = df[[f'{i}' for i in range(39)]]
    models = []
    for k in tqdm(clusters):
        model = MiniBatchKMeans(n_clusters=k, batch_size=50*(10**3), random_state=0xDEAD)
        model.fit(X)
        models.append(model)

    # %%
    output = pd.DataFrame({f'{k}': m.labels_ for k, m in zip(clusters, models)})
    output['path'] = df['path'].to_numpy()
    # %%
    output.to_csv(os.path.join(source_dir, 'labels.csv'), index=False)
