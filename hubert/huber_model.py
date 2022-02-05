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
    similarity = nn.CosineSimilarity(dim=-1)
    # cosine similarity will take two vectors
    # 1. embedding of the target class
    # 2. projection of the hubert_model output

    # how to use cosine similarity correctly https://github.com/pytorch/pytorch/issues/11202#issuecomment-997138697
    # Suppose x1 has shape (m,d) and x2 has shape (n,d), then we can use
    #   pairwise_sim = F.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1)
    # [m, 1, d], [1, n, d] --> [m , n]

    x = torch.tensor(
        [
            [[1, 0],
            [0, 1],
            [1, 1]],

            [[1, 0],
             [0, 1],
             [1, 1]],
        ], dtype=torch.float).unsqueeze(2)
    # x.shape = [batch=2, max_seq_len=3, 1, proj_dim=2]

    y = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).view(1, 1, 4, 2)
    # y.shape = [1, 1, num_classes=4, proj_dim=2]

    out = similarity(x, y)
    # [batch=2, max_seq_len=3, 1, proj_dim=2]  [1, 1, num_classes=4, proj_dim=2] --> [batch, max_seq_len, num_classes]
    out
    # %%
    # input is expected to contain raw, unnormalized scores for each class
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # %%
    # %%
    # %%
    pass

