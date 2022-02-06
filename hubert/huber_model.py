# %%
import pytorch_lightning as pl
import torch
import torchaudio
from icecream import ic
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn


class HubertPretrainPL(pl.LightningModule):
    def __init__(self, hubert_model, cluster_sizes, proj_dim, hubert_features=768, p=0.08, l=10):
        super().__init__()
        # can access Conv feature extractor using hubert_base_model.feature_extractor and BERT using hubert_base_model.encoder
        # extractor takes audios and lengths and returns features and new lengths
        # encoder takes features and lengths and returns only new features
        # https://github.com/pytorch/audio/tree/main/torchaudio/models/wav2vec2
        self.hubert_model = hubert_model
        self.p = p
        self.l = l
        self.mask = 0
        self.cluster_sizes = cluster_sizes
        self.proj_dim = proj_dim
        self.proj_layer = nn.Linear(in_features=hubert_features, out_features=proj_dim)
        self.cluster_proj_layers = nn.ModuleDict({f'Cluster_{k}_projection': nn.Embedding(num_embeddings=k, embedding_dim=proj_dim) for k in cluster_sizes})
        self.batch_mask_indices = []

    def _mask_span(self, feature_batch, frames_cnt):
        masked_batch = []
        self.batch_mask_indices = []
        for i, (features, length) in enumerate(zip(feature_batch, frames_cnt)):
            masked_cnt = int(length * self.p)
            # generate starting points for masking
            mask_starts = torch.randperm(length - self.l, device=self.device)[:masked_cnt]
            # span masks of lengths self.l starting from indices in mask_starts
            # say self.l = 2 and mask_start[i] = 5, then need to create mask at indices 5,6,7
            index_mask = torch.stack([mask_starts + i for i in range(self.l)], dim=1).view(-1)
            self.batch_mask_indices.append(index_mask)
            # create span mask
            mask = torch.ones_like(features, device=self.device)
            mask[index_mask, :] = self.mask
            masked_batch.append(features * mask)
        return torch.stack(masked_batch)

    def forward(self, inputs, wave_lens, inference=True):

        features_batch, frames_cnt = hubert_base_model.feature_extractor()
        if not inference:
            features_batch = self._mask_span(features_batch, frames_cnt)
        encoder_features = self.hubert_model.encdoer(features_batch, wave_lens)
        projected_features = self.proj_layer(encoder_features)
        return projected_features, wave_lens

    def _loss(self):
        pass

    def training_step(self, batch, batch_index):
        inputs, wave_lens = batch['waves'], batch['lens']


# %%
if __name__ == '__main__':
    # %%
    # aux_num_out ... when provided, attach an extra linear layer on top of encoder, which can be used for fine-tuning.
    hubert_base_model = torchaudio.models.hubert_base()

    inputs = torch.stack([
        torch.rand(16000 * 5),
        torch.concat([torch.rand(16000 * 4), torch.zeros(16000)])
    ])
    # inputs.shape = [batch=2, n_samples]
    features_batch, lengths_batch = hubert_base_model.feature_extractor(inputs, torch.tensor([16000 * 5, 16000 * 4]))

    features_np = features_batch.detach().numpy()
    lengths_np = lengths_batch.detach().numpy()

    # %%
    similarity = nn.CosineSimilarity(dim=-1)
    # cosine similarity will take two vectors
    # 1. embedding of the target class
    # 2. projection of the hubert_model output

    # how to use cosine similarity correctly https://github.com/pytorch/pytorch/issues/11202#issuecomment-997138697
    # Suppose x1 has shape (m,d) and x2 has shape (n,d), then we can use
    #   pairwise_sim = F.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1)
    # [m, 1, d], [1, n, d] --> [m , n]

    outputs = torch.tensor([
        [[1, 0],
         [0, 1],
         [1, 1]],

        [[1, 0],
         [0, 1],
         [1, 1]],
    ], dtype=torch.float).unsqueeze(2)
    # outputs.shape = [batch=2, max_seq_len=3, 1, proj_dim=2]

    embeddings = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).view(1, 1, 4, 2)
    # embeddings.shape = [1, 1, num_classes=4, proj_dim=2]

    similarity_scores = similarity(outputs, embeddings)
    # [batch=2, max_seq_len=3, 1, proj_dim=2]  [1, 1, num_classes=4, proj_dim=2] --> [batch, max_seq_len, num_classes]
    similarity_scores
    # %%
    # To address the second decision, we denote the cross-entropy loss computed over masked time steps as Lm.
    # Lm(f; X, M, Z) = \sum_{t∈M} log p_f (z_t | \tilde X, t )

    # The input is expected to contain raw, unnormalized scores for each class.
    # input has to be a Tensor of size either (minibatch, C) or (minibatch, C, d_1, d_2, ..., d_K) with K ≥ 1 for the K-dimensional case.
    # The latter is useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images.
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # %%
    batch = torch.arange(112).view(2, 4, 14)
    lengths = torch.tensor([
        [13, 10, 5, 10],
        [13, 5, 10, 10],
    ])
    ic(batch, batch.shape)
    ic(lengths, lengths.shape)

    index = torch.tensor([
        [[0, 1],
         [1, 2]],

        [[0, 5],
         [1, 4]]
    ])
    ic(index)
    torch.gather(batch, dim=2, index=lengths.unsqueeze(-1))

    # how to use torch.gather
    #
    # if dim == 0  ::  result_{i,j,k} = input_{index_{i,j,k}, j,              k}
    # if dim == 1  ::  result_{i,j,k} = input_{j,             index_{i,j,k},  k}
    # if dim == 2  ::  result_{i,j,k} = input_{i,             j,              index_{i,j,k}}
    #
    # indices_for_index = torch.tensor([
    #         [[{0,0,0}, {0,0,1}],
    #          [{0,1,0}, {0,1,1}]],
    #
    #         [[{1,0,0}, {1,0,1}],
    #          [{1,1,0}, {1,1,1}]]
    #     ])
    # docs:
    #   * https://pytorch.org/docs/stable/generated/torch.gather.html
    #   * https://stackoverflow.com/a/54706716
    # %%
    # MASK BY LENGTH
    # will use broadcasting
    # index each row
    batch_row_indices = torch.arange(batch.shape[-1])
    # batch_row_indices = torch.arange(batch.shape[-1]).repeat(batch.shape[0] * batch.shape[1]).view(*batch.shape)
    # mask by length
    # shapes ::
    #   batch_row_indices.shape = [14]
    #   lengths.unsqueeze(-1).shape = [2, 4, 1]
    lengths_mask = batch_row_indices < lengths.unsqueeze(-1)
    # [2, 4, 14]
    lengths_mask
    # masked_batch = batch * lengths_mask
    # masked_batch

    # %%
    x = torch.arange(12).reshape(3, 4) * 100
    ic(x)
    indices = torch.tensor([0, 2])
    ic(indices)
    ic(torch.index_select(x, 0, indices))
    ic(torch.index_select(x, 1, indices))
    # %%
    p = 0.3
    pop_size = 14
    device = 'cpu'
    num_samples = int(pop_size * p)
    # %%

    pass
