# %%
import pytorch_lightning as pl
import torch
import torchaudio
from icecream import ic
import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch import optim
from clustering.torch_mffc_extract import compute_frames, ParCzechDataset
from torch.utils.data import DataLoader
import pandas as pd
import pickle


class ParCzechPretrain(ParCzechDataset):
    def __init__(self, df_path, km_labels, resample_rate=16000, clean_params=None, label_path=None, sep='\t'):
        super(ParCzechPretrain, self).__init__(df_path, resample_rate, clean_params, sep=sep, sort=False)
        self.label_path = label_path
        with open(os.path.join(self.label_path, 'mp3_to_int.pickle'), 'rb') as f:
            self.mp3_to_int = pickle.load(f)
        # folder names are numbers starting from 0
        self.current_folder = '-1'
        self.current_targ_df = None
        self.km_labels = [f'km{k}' for k in km_labels]

    def get_labels(self, i):
        mp3 = self.df.iloc[i].mp3.astype(str)
        mp3_folder = f'{self.mp3_to_int[mp3]}'
        if mp3_folder != self.current_folder:
            self.get_df_from_folder(mp3_folder)
            self.current_folder = mp3_folder
        segment = self.df.iloc[i].segment_path
        segment = '/'.join(segment.split('/')[-2:])
        result_df = self.current_targ_df[self.current_targ_df.segm == segment].sort_values(by=['id'])
        return result_df

    def get_df_from_folder(self, folder_path):
        dfs = []
        files = os.listdir(os.path.join(self.label_path, folder_path))
        for f in sorted(files):
            if f.endswith('.csv'):
                dfs.append(pd.read_csv(os.path.join(self.label_path, folder_path, f)))
        self.current_targ_df = pd.concat(dfs).drop(columns=['mp3'])
        self.current_targ_df['id'] = self.current_targ_df.path.str.split('/').str[-1].astype(int)

    def __getitem__(self, i):
        labels = self.get_labels(i)
        batch = dict(
            wave=self.get_wav(self.extract_path(i)),
            target=[torch.from_numpy(labels[km].values) for km in self.km_labels],
        )
        return batch

class HubertPretrainPL(pl.LightningModule):
    def __init__(self,
                 hubert_model,
                 cluster_sizes,
                 proj_dim,
                 mask_weight,
                 softmax_temp,
                 betas,
                 warm_up_steps,
                 total_steps,
                 hubert_features=768,
                 peak_lr=5e-4,
                 p=0.08,
                 l=10,
                 ignore_index=-1
                 ):
        super().__init__()
        # can access Conv feature extractor using hubert_base_model.feature_extractor and BERT using hubert_base_model.encoder
        # extractor takes audios and lengths and returns features and new lengths
        # encoder takes features and lengths and returns only new features
        # https://github.com/pytorch/audio/tree/main/torchaudio/models/wav2vec2
        self.hubert_model = hubert_model
        self.ignore_index = ignore_index

        self.p = p
        self.l = l

        self.betas = betas
        self.warm_up_steps = warm_up_steps
        self.lr_inc = peak_lr / warm_up_steps
        self.lr_dec = peak_lr / (total_steps - warm_up_steps)

        self.mask = 0
        self.mask_loss_weight = mask_weight

        self.softmax_temp = softmax_temp

        self.proj_dim = proj_dim
        self.cluster_sizes = cluster_sizes
        self.proj_layer = nn.Linear(in_features=hubert_features, out_features=proj_dim)
        self.cluster_proj_layers = nn.ModuleDict(
            OrderedDict(
                ((f'{k}', nn.Embedding(num_embeddings=k, embedding_dim=proj_dim)) for k in cluster_sizes)
            )
        )

    def _mask_span(self, feature_batch, frames_cnt):
        masked_batch = []
        batch_mask_indices = []
        for i, (features, length) in enumerate(zip(feature_batch, frames_cnt)):
            masked_cnt = int(length * self.p)
            # generate starting points for masking
            mask_starts = torch.randperm(length - self.l, device=self.device)[:masked_cnt]
            # span masks of lengths self.l starting from indices in mask_starts
            # say self.l = 2 and mask_start[i] = 5, then need to create mask at indices 5,6,7
            index_mask = torch.stack([mask_starts + i for i in range(self.l)], dim=1).view(-1)
            batch_mask_indices.append(index_mask)
            # create span mask
            mask = torch.ones_like(features, device=self.device)
            print(mask.shape)
            mask[index_mask, :] = self.mask
            masked_batch.append(features * mask)
        return torch.stack(masked_batch), batch_mask_indices

    def forward(self, inputs, wave_lens, inference=True):
        features_batch, frames_cnt = self.hubert_model.feature_extractor(inputs, wave_lens)
        batch_mask_indices = None
        if not inference:
            features_batch, batch_mask_indices = self._mask_span(features_batch, frames_cnt)

        encoder_features = self.hubert_model.encoder(features_batch, wave_lens)
        projected_features = self.proj_layer(encoder_features)
        return projected_features, frames_cnt, batch_mask_indices

    def _compute_cos_sim(self, projected_features):
        # projected_features.shape = [batch, n_frames, proj_dim]
        embedded_targets = {k: layer(torch.arange(k)) for k, layer in zip(self.cluster_sizes, self.cluster_proj_layers.values())}

        # compute cosine similarity for each k-means model
        # how to use cosine similarity correctly https://github.com/pytorch/pytorch/issues/11202#issuecomment-997138697
        # Suppose x1 has shape (m,d) and x2 has shape (n,d), then we can use
        #   pairwise_sim = F.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1)
        # [m, 1, d], [1, n, d] --> [m , n]
        similarity_scores = {}
        for k, embedded_target in embedded_targets.items():
            # embedded_targets[k].shape = [k, self.proj_dim]
            similarity_scores[k] = F.cosine_similarity(
                projected_features[:, :, None, :],  # [batch, n_frames, 1, proj_dim]
                embedded_target[None, None, :, :],  # [1,     1,        k, proj_dim]
                dim=-1  # common dimension
            ) * self.softmax_temp
            # similarity_scores[k].shape = [batch, n_frames, k]
        return similarity_scores

    def _compute_loss_acc(self, similarity_scores, frames_cnt, targets, batch_mask_indices):
        # targets.shape = [n_clusterings, batch, n_frames]
        # n_frames.shape = [batch, ]
        # batch_mask_indices is a list, batch_mask_indices[i] = torch tensor with indices where span mask was applied

        # cross_entropy recap
        #   The input is expected to contain raw, unnormalized scores for each class.
        #   input has to be a Tensor of size either (minibatch, C).
        total_mask_loss, total_unmask_loss = 0, 0
        total_mask_acc, total_unmask_acc = 0, 0
        # iterate over different clustering models
        for (k, scores), k_target in zip(similarity_scores.items(), targets):
            # scores.shape = [batch, n_frames, k]
            clustering_mask_loss, clustering_unmask_loss = 0, 0
            clustering_mask_acc, clustering_unmask_acc = 0, 0

            # iterate over sequences in the batch
            for seq_score, target, seq_len, index_mask in zip(scores, k_target, frames_cnt, batch_mask_indices):
                # seq_score.shape = [n_frames, k]
                # target.shape = [n_frames]
                # index_mask.shape = [n_masked_frames] ... differs for each sequence, that is why processing each seq separately
                # seq_len is an int

                valid_seq_score = seq_score[:seq_len]
                target = target[:seq_len]
                # valid_seq_score.shape = [seq_len, k]
                # cross entropy loss over masked frames
                mask_loss = F.cross_entropy(valid_seq_score[index_mask], target[index_mask], reduction='sum')
                mask_acc = accuracy(valid_seq_score[index_mask], target[index_mask], )
                clustering_mask_loss += mask_loss
                clustering_mask_acc += mask_acc

                # cross entropy loss over frames without mask
                index_unmask = torch.ones(valid_seq_score.shape[0], device=self.device, dtype=torch.bool)
                index_unmask[index_mask] = False
                unmask_loss = F.cross_entropy(valid_seq_score[index_unmask], target[index_unmask], reduction='sum')
                unmask_acc = accuracy(valid_seq_score[index_unmask], target[index_unmask], )
                clustering_unmask_acc += unmask_acc
                clustering_unmask_loss += unmask_loss
            # average across batch
            total_mask_loss += clustering_mask_loss / scores.shape[0]
            total_unmask_loss += clustering_unmask_loss / scores.shape[0]

            total_mask_acc += clustering_mask_acc / scores.shape[0]
            total_unmask_acc += clustering_unmask_acc / scores.shape[0]

        return total_mask_loss, total_unmask_loss, total_mask_acc, total_unmask_acc

    def training_step(self, batch, batch_index, inference=False):
        inputs, wave_lens, targets = batch['waves'], batch['lens'], batch['targets']
        # inputs.shape = [batch, max_wave_len]
        # targets.shape = [batch, n_frames] ... here n_frames << max_wave_len

        projected_features, frames_cnt, batch_mask_indices = self(inputs, wave_lens, inference=inference)
        # projected_features.shape = [batch, n_frames, proj_dim]
        ic(frames_cnt)
        ic(targets[0].shape)
        similarity_scores = self._compute_cos_sim(projected_features)

        mask_loss, unmask_loss, mask_acc, unmask_acc = self._compute_loss_acc(similarity_scores, frames_cnt, targets, batch_mask_indices)
        total_loss = self.mask_loss_weight * mask_loss + (1 - self.mask_loss_weight) * unmask_loss
        return dict(loss=total_loss, mask_loss=mask_loss, unmask_loss=unmask_loss, mask_acc=mask_acc, unmask_acc=unmask_acc, batch_size=inputs.shape[0])

    def validation_step(self, batch, batch_index):
        return self.training_step(batch, batch_index, inference=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0, betas=self.betas)
        return dict(optimizer=optimizer)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if self.trainer.global_step < self.warm_up_steps:
            lr_scale = self.lr_inc
        else:
            lr_scale = - self.lr_dec

        for pg in optimizer.param_groups:
            pg['lr'] += lr_scale

        optimizer.step(closure=optimizer_closure)


def gen_wav_targ(wav_sec, classes, sr, frame_len):
    samples = int(sr * wav_sec)
    n_frames = int(wav_sec // frame_len)
    return dict(
        wave=torch.rand(samples),
        target=torch.randint(classes, size=(n_frames, ))
    )


def pad_list(tensor_list):
    lens = [t.shape[-1] for t in tensor_list]
    max_len = max(lens)
    result = torch.stack([
        F.pad(t, (0, max_len - t.shape[-1])) for t in tensor_list
    ])
    return result, torch.tensor(lens), max_len

class Collator:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, batch):
        wavs = [x['wave'] for x in batch]
        padded_waves, wav_lens, max_len = pad_list(wavs)
        padded_waves = padded_waves.view(len(batch), max_len)
        # each target in this list is a list of target_tensors for different k
        targets = [x['target'] for x in batch]
        targets_by_k = []
        for i in range(self.n_classes):
            lst = [t[i] for t in targets]
            targets_by_k.append(lst)

        padded_targets = [pad_list(t)[0] for t in targets_by_k]
        return dict(
            waves=padded_waves,
            lens=wav_lens,
            targets=padded_targets,
        )

# %%
if __name__ == '__main__':
    # %%
    # ............................................... Parameters .......................................................
    df_path = '/lnet/express/work/people/stankov/alignment/subset/subset.csv'
    labels_path = '/lnet/express/work/people/stankov/alignment/clustering_all/segments'
    parczech_clean_params = dict(
        recognized_sound_coverage__segments_lb=0.45,
        recognized_sound_coverage__segments_ub=0.93,
        duration__segments_lb=0.5,
    )
    params = dict(
        batch_size=2,
        num_workers=os.cpu_count(),
        # pin_memory=False,
        # num_workers=4,
        pin_memory=False,
        ignore_index=-1,
        epochs=5,
    )

    ks = [15, 25, 100]

    # ............................................... Dataset .......................................................
    dataset = ParCzechPretrain(
        df_path=df_path,
        km_labels=ks,
        clean_params=parczech_clean_params,
        label_path=labels_path,
        sep=','
    )
    collate_fn = Collator(len(ks))
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 4, pin_memory=True)

    # ............................................... Model .......................................................
    hubert_base_model = torchaudio.models.hubert_base()
    hubert_pretrain = HubertPretrainPL(
        hubert_base_model,
        cluster_sizes=ks,
        proj_dim=256,
        mask_weight=0.5,
        softmax_temp=0.1,
        betas=(0.9, 0.98),
        warm_up_steps=100,
        total_steps=500,
        hubert_features=768,
        peak_lr=5e-4,
        p=0.08,
        l=10,
        ignore_index=params['ignore_index']
    )

    # ............................................... Training .......................................................
    # trainer = pl.Trainer(gpus=2, strategy='ddp', logger=logger, profiler=profiler, num_sanity_val_steps=0, max_epochs=params['epochs'], callbacks=cb,
    #                      deterministic=True,
    #                      precision=16)

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=params['epochs'],
        deterministic=True,
        check_val_every_n_epoch=0,
        fast_dev_run=10,
        gpus=1,
    )

    trainer.fit(hubert_pretrain, dataloader)
    # %%
    # aux_num_out ... when provided, attach an extra linear layer on top of encoder, which can be used for fine-tuning.
    hubert_base_model = torchaudio.models.hubert_base()
    secs = torch.rand(150) + 1.5
    secs = [s.item() for s in secs] + [2, 2.05, 2.11, 2.01, 2.2]
    batch = collate_fn([gen_wav_targ(s, 10, 16000, 0.02) for s in secs], nfft=640, sr=16000, frame_len=0.02)

    features_batch, frames_cnt = hubert_base_model.feature_extractor(batch['waves'], batch['lens'])
    total_dif = 0
    for my, ref, l, features in zip(batch['n_frames'], frames_cnt, secs, features_batch):
        total_dif += ref.item() - my
        print(f'ms={l:.4f}, my={my:3}, ref={ref.item():3}, diff={ref.item() - my:2}')

    print(total_dif)
    # %%

    inputs = torch.stack([
        torch.rand(16000 * 5),
        torch.concat([torch.rand(16000 * 4), torch.zeros(16000)])
    ])
    # inputs.shape = [batch=2, n_samples]

    ignore_index = -1
    targets = torch.stack([
        torch.stack([
            torch.randint(0, 5, size=(250,)),
            F.pad(torch.randint(0, 5, size=(200,)), (0, 50), value=ignore_index)
        ]),
        torch.stack([
            torch.randint(0, 10, size=(250,)),
            F.pad(torch.randint(0, 10, size=(200,)), (0, 50), value=ignore_index)
        ])
    ])

    batch = dict(
        waves=inputs,
        lens=torch.tensor([16000 * 5, 16000 * 4]),
        targets=targets,
    )

    hubert_pretrain = HubertPretrainPL(
        hubert_base_model,
        cluster_sizes=[5, 10],
        proj_dim=256,
        mask_weight=0.5,
        softmax_temp=0.1,
        betas=(0.9, 0.98),
        warm_up_steps=100,
        total_steps=500,
        hubert_features=768,
        peak_lr=5e-4,
        p=0.08,
        l=10,
        ignore_index=ignore_index
    )

    l = hubert_pretrain.training_step(batch, 1)

