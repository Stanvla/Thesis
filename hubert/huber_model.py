# %%
import os
import pandas
import numpy as np

import torch
from torch import nn
import torchaudio
# import pytorch_lightning as pl


# %%
if __name__ == '__main__':
    # %%
    # aux_num_out ... when provided, attach an extra linear layer on top of encoder, which can be used for fine-tuning.
    hubert_base_model = torchaudio.models.hubert_base()
    # %%
    similarity = nn.CosineSimilarity(dim=1)
    # cosine similarity will take two vectors
    # 1. embedding of the target class
    # 2. projection of the hubert_model output

    x = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    y = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
    out = similarity(x, y)
    out
    # %%
    # input is expected to contain raw, unnormalized scores for each class
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # %%
    # %%
    # %%
    pass

