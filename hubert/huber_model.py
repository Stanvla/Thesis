# %%
import os
import pickle
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from icecream import ic
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from hubert.clustering.torch_mffc_extract import ParCzechDataset


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
        self.labels = km_labels

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
            target={k: torch.from_numpy(labels[km].values) for km, k in zip(self.km_labels, self.labels)},
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
                 ignore_index=-1,
                 sim=True,
                 reduction='sum',
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

        self.sim = sim
        self.reduction = reduction

        self.betas = betas
        self.warm_up_steps = warm_up_steps
        self.lr_inc = peak_lr / warm_up_steps
        self.lr_dec = peak_lr / (total_steps - warm_up_steps)

        self.mask = torch.tensor(0, device=self.device)
        self.mask_loss_weight = mask_weight

        self.softmax_temp = softmax_temp

        self.proj_dim = proj_dim
        self.cluster_sizes = cluster_sizes
        self.cluster_sizes_str = [f'{k}' for k in cluster_sizes]

        self.proj_layer = nn.Linear(in_features=hubert_features, out_features=proj_dim)
        self.encoder_proj_layers_sim = nn.ModuleDict(
            OrderedDict(
                ((k, nn.Linear(in_features=hubert_features, out_features=proj_dim)) for k in self.cluster_sizes_str)
            )
        )

        self.encoder_proj_layers_cls = nn.ModuleDict(
            OrderedDict(
                ((f'{k}', nn.Linear(in_features=hubert_features, out_features=k)) for k in self.cluster_sizes)
            )
        )

        self.target_proj_layers = nn.ModuleDict(
            OrderedDict(
                ((f'{k}', nn.Embedding(num_embeddings=k, embedding_dim=proj_dim)) for k in cluster_sizes)
            )
        )

    # generate spans of masks for the batch
    def _mask_span(self, feature_batch, frames_cnt):
        # feature_batch.shape = [batch, max_seq_len, cnn_features]
        # frames_cnt.shape = [batch]

        batch_mask_indices = []
        batch_masks = []
        # iterate over each sequence in the batch and generate mask
        for i, (features, length) in enumerate(zip(feature_batch, frames_cnt)):
            masked_cnt = int(length * self.p)
            # generate starting points for masking
            mask_starts = torch.randperm(length - self.l, device=self.device)[:masked_cnt]
            # span masks of lengths self.l starting from indices in mask_starts
            # say self.l = 2 and mask_start[i] = 5, then need to create mask at indices 5,6,7
            index_mask = torch.stack([mask_starts + i for i in range(self.l)], dim=1).view(-1)
            batch_mask_indices.append(index_mask)

            # create span mask, here we use only row indices
            mask = torch.zeros(features.shape[0], device=self.device, dtype=torch.bool)
            # for mask indices set True
            mask.index_fill_(0, index_mask.to(self.device), True)
            # so where the mask is True we will replace the values
            batch_masks.append(mask)

        batch_masks = torch.stack(batch_masks)
        masked_batch = feature_batch.masked_fill(batch_masks.view(-1), self.mask)
        ic(masked_batch[0])
        return masked_batch, batch_mask_indices, batch_masks

    def forward(self, inputs, wave_lens, inference=True):
        # inputs.shape = [batch, max_wave_len]
        # wave_lens.shape = [batch]

        features_batch, frames_cnt = self.hubert_model.feature_extractor(inputs, wave_lens)
        batch_mask_indices, batch_masks = None, None
        if not inference:
            features_batch, batch_mask_indices, batch_masks = self._mask_span(features_batch, frames_cnt)

        encoder_features = self.hubert_model.encoder(features_batch, frames_cnt)
        return encoder_features, frames_cnt, batch_mask_indices, batch_masks

    def _compute_cos_sim(self, encoder_features):
        if not self.sim:
            projected_features_dict = {k: self.encoder_proj_layers_cls[f'{k}'](encoder_features) for k in self.cluster_sizes}
            return projected_features_dict

        # projected_features[k].shape = [batch, n_frames, proj_dim]
        # k is an int that gives the number of clusters for some K-means model
        projected_features_dict = {k: self.encoder_proj_layers_sim[f'{k}'](encoder_features) / self.softmax_temp for k in self.cluster_sizes}

        embedded_targets = {k: layer(torch.arange(k, device=self.device)) for k, layer in zip(self.cluster_sizes, self.target_proj_layers.values())}

        # compute cosine similarity for each k-means model
        # how to use cosine similarity correctly https://github.com/pytorch/pytorch/issues/11202#issuecomment-997138697
        # Suppose x1 has shape (m,d) and x2 has shape (n,d), then we can use
        #   pairwise_sim = F.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1)
        # [m, 1, d], [1, n, d] --> [m , n]
        similarity_scores = {}
        # similarity_scores[k].shape = [batch, n_frames, k]

        for k, embedded_target in embedded_targets.items():
            # embedded_targets[k].shape = [k, self.proj_dim]
            projected_features = projected_features_dict[k]

            similarity_scores[k] = F.cosine_similarity(
                projected_features[:, :, None, :],  # [batch, n_frames, 1, proj_dim]
                embedded_target[None, None, :, :],  # [1,     1,        k, proj_dim]
                dim=-1  # common dimension
            ) / self.softmax_temp
            # similarity_scores[k].shape = [batch, n_frames, k]
        return similarity_scores

    def _get_loss_acc(self, seq, seq_len, mask, trg):
        # seq.shape = [max_seq_len, k]
        # mask.shape = [max_seq_len]
        # trg.shape = [max_seq_len]

        # at the end extract only seq_len elements
        new_mask = mask[:seq_len]
        new_seq = seq[:seq_len]
        new_trg = trg[:seq_len]
        # to do masked_select need to add new dimension to mask
        # then need to reshape the result of masked_select, since it flattens the tensor

        new_seq = new_seq.masked_select(new_mask.unsqueeze(1)).view(-1, seq.shape[1])
        new_trg = new_trg.masked_select(new_mask)

        # cross_entropy recap
        #   The input is expected to contain raw, unnormalized scores for each class.
        #   input has to be a Tensor of size either (minibatch, C).
        loss = F.cross_entropy(new_seq, new_trg, reduction=self.reduction)
        # multiply acc by the number of elements in the sequence
        acc = accuracy(new_seq, new_trg) * new_seq.shape[0]
        return loss, acc, new_seq.shape[0]

    def _compute_loss_acc(self, similarity_scores, frames_cnt, targets, batch_mask_indices, batch_masks):
        # similarity_scores[k].shape = [batch, n_frames, k]  (similarity_scores is a dict)
        # frames_cnt.shape = [batch]
        # targets[i].shape = [batch, n_frames]  (targets is a list)
        # batch_mask_indices[i] = tensor with indices with masked frames (batch_mask_indices is a list)
        # batch_masks[i].shape = [n_frames]  (batch_masks is a list)


        total_mask_loss, total_unmask_loss = 0, 0
        total_mask_acc, total_unmask_acc, total_acc = 0, 0, 0

        # iterate over different clustering models
        for (k, scores), k_target in zip(similarity_scores.items(), targets):
            # scores.shape = [batch, n_frames, k]

            # metrics for a specific clustering model k
            clustering_mask_loss, clustering_unmask_loss = 0, 0
            clustering_mask_acc, clustering_unmask_acc, clustering_total_acc = 0, 0, 0
            cnt_mask, cnt_unmask, cnt_total = 0, 0, 0

            # iterate over sequences in the batch
            for seq_score, target, seq_len, mask in zip(scores, k_target, frames_cnt, batch_masks):
                # seq_score.shape = [n_frames, k]
                # target.shape = [n_frames]
                # index_mask.shape = [n_masked_frames] ... differs for each sequence, that is why processing each seq separately
                # seq_len is an int

                # cross entropy loss and acc over frames without mask
                unmask_loss, unmask_acc, unmask_size = self._get_loss_acc(seq_score, seq_len, mask, target)
                clustering_unmask_loss += unmask_loss
                clustering_unmask_acc += unmask_acc
                cnt_unmask += unmask_size

                # cross entropy loss and acc over frames with mask
                mask_loss, mask_acc, mask_size = self._get_loss_acc(seq_score, seq_len, ~mask, target)
                clustering_mask_loss += mask_loss
                clustering_mask_acc += mask_acc
                cnt_mask += mask_size

                # total accuracy
                clustering_total_acc += accuracy(seq_score[:seq_len], target[:seq_len]) * seq_len
                cnt_total += seq_len

            # average across batch
            total_mask_loss += clustering_mask_loss / scores.shape[0]
            total_unmask_loss += clustering_unmask_loss / scores.shape[0]

            total_mask_acc += clustering_mask_acc / cnt_mask
            total_unmask_acc += clustering_unmask_acc / cnt_unmask
            total_acc += clustering_total_acc / cnt_total
        # total_{mask,unmask}_{loss,acc} are sums of losses for different clustering models
        return total_mask_loss, total_unmask_loss, total_mask_acc / len(similarity_scores), total_unmask_acc / len(similarity_scores), total_acc / len(
            similarity_scores)

    def training_step(self, batch, batch_index, inference=False):
        inputs, wave_lens, targets = batch['waves'], batch['lens'], batch['targets']
        # inputs.shape = [batch, max_wave_len]
        # wave_lens.shape = [batch]
        # targets[k].shape = [batch, n_frames] ... targets is a dict and n_frames << max_wave_len

        encoder_features, frames_cnt, batch_mask_indices, batch_masks = self(inputs, wave_lens, inference=inference)
        # encoder_features.shape = [batch, max_seq_len, hubert_features]
        scores = self._compute_cos_sim(encoder_features)
        # scores[k].shape = [batch, n_frames, k]

        mask_loss, unmask_loss, mask_acc, unmask_acc, total_acc = self._compute_loss_acc(scores, frames_cnt, targets, batch_mask_indices,
                                                                                         batch_masks)
        total_loss = self.mask_loss_weight * mask_loss + (1 - self.mask_loss_weight) * unmask_loss
        return dict(
            loss=total_loss,
            mask_loss=mask_loss,
            unmask_loss=unmask_loss,
            mask_acc=mask_acc,
            unmask_acc=unmask_acc,
            total_acc=total_acc,
            batch_size=inputs.shape[0]
        )

    def validation_step(self, batch, batch_index):
        return self.training_step(batch, batch_index, inference=True)

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=0.001, betas=self.betas)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return dict(optimizer=optimizer)

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #     if self.trainer.global_step < self.warm_up_steps:
    #         lr_scale = self.lr_inc
    #     else:
    #         lr_scale = - self.lr_dec
    #
    #     for pg in optimizer.param_groups:
    #         pg['lr'] += lr_scale
    #
    #     optimizer.step(closure=optimizer_closure)

    def training_epoch_end(self, outputs):
        total_loss = sum(out['loss'] for out in outputs) / len(outputs)
        masked_loss = sum(out['mask_loss'] for out in outputs) / len(outputs)
        unmasked_loss = sum(out['unmask_loss'] for out in outputs) / len(outputs)

        total_acc = sum(out['total_acc'] for out in outputs) / len(outputs)
        masked_acc = sum(out['mask_acc'] for out in outputs) / len(outputs)
        unmasked_acc = sum(out['unmask_acc'] for out in outputs) / len(outputs)

        self.logger.experiment.add_scalar(f'Loss/Total', total_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f'Loss/Masked', masked_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f'Loss/Unmasked', unmasked_loss, self.current_epoch)

        self.logger.experiment.add_scalar(f'Acc/Total', total_acc, self.current_epoch)
        self.logger.experiment.add_scalar(f'Acc/Masked', masked_acc, self.current_epoch)
        self.logger.experiment.add_scalar(f'Acc/Unmasked', unmasked_acc, self.current_epoch)


class Collator:
    def __init__(self, classes, padding):
        self.classes = classes
        self.pad_value = padding

    def pad_list(self, tensor_list):
        lens = [t.shape[-1] for t in tensor_list]
        max_len = max(lens)
        result = torch.stack([
            F.pad(t, (0, max_len - t.shape[-1]), value=self.pad_value) for t in tensor_list
        ])
        return result, torch.tensor(lens), max_len

    def __call__(self, batch):
        # batch is a list of dicts, with keys [wave, target]
        # batch[i]['target'] is also a dict with keys given by different k from k-means models

        wavs = [x['wave'] for x in batch]
        padded_waves, wav_lens, max_len = self.pad_list(wavs)
        padded_waves = padded_waves.view(len(batch), max_len)

        # targets_by_k[k] is a list of tensors, it can be viewed as unpadded batch
        targets_by_k = {}
        for k in self.classes:
            lst_k = [x['target'][k] for x in batch]
            targets_by_k[k] = lst_k

        # use only first element from pad_list() since it returns multiple things
        padded_targets = {k: self.pad_list(lst)[0] for k, lst in targets_by_k.items()}

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
        pin_memory=False,
        ignore_index=-1,
        epochs=25,
        mask_weight=0.5,
        softmax_temp=0.1,
        overfit_batches=1,
        sim=False,
        reduction='mean',
    )

    ks = [15, 25]
    # ............................................... Dataset .......................................................

    dataset = ParCzechPretrain(
        df_path=df_path,
        km_labels=ks,
        clean_params=parczech_clean_params,
        label_path=labels_path,
        sep=','
    )
    collate_fn = Collator(ks, padding=params['ignore_index'])
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 4, pin_memory=True)

    # %%
    iterator = next(iter(dataloader))

    # %%

    # ............................................... Model .......................................................
    hubert_base_model = torchaudio.models.hubert_base()
    hubert_pretrain = HubertPretrainPL(
        hubert_base_model,
        cluster_sizes=ks,
        proj_dim=256,
        mask_weight=params['mask_weight'],
        softmax_temp=params['softmax_temp'],
        betas=(0.9, 0.98),
        warm_up_steps=100,
        total_steps=500,
        hubert_features=768,
        peak_lr=5e-4,
        p=0.08,
        l=10,
        ignore_index=params['ignore_index'],
        sim=params['sim'],
        reduction=params['reduction']
    )

    # ............................................... Training .......................................................
    # Logs are saved to os.path.join(save_dir, name, version)
    # save_dir/name/sub_dir/version
    logger = TensorBoardLogger(
        save_dir='logs',
        name=f'clusters={"_".join(list(map(str, ks)))}_'
             f'mw={params["mask_weight"]:.2f}_'
             f'st={params["softmax_temp"]:.2f}_'
             f'overfit={params["overfit_batches"]}_'
             f'sim=f{params["sim"]}'
             f'_red={params["reduction"]}_'
             + '_' + datetime.now().strftime('%d.%m__%H.%M'),
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=params['epochs'],
        deterministic=True,
        # check_val_every_n_epoch=0,
        # fast_dev_run=10,
        overfit_batches=params['overfit_batches'],
        gpus=1,
        checkpoint_callback=False,
        logger=logger
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
