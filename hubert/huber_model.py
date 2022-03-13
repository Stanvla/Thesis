# %%
from collections import OrderedDict
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from icecream import ic
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch import optim
from torchmetrics.functional import accuracy
from pretrain_dataset import ParCzechPretrainPL


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
        torch.manual_seed(0)
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
        masked_batch = feature_batch.masked_fill(batch_masks.unsqueeze(-1), self.mask)
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

    def _accuracy(self, seq, trg):
        predicted = seq.argmax(1)
        return torch.sum(predicted == trg) / seq.shape[0]

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
        # targets[k].shape = [batch, n_frames]  (targets is a dict)
        # batch_mask_indices[i] = tensor with indices with masked frames (batch_mask_indices is a list)
        # batch_masks[i].shape = [n_frames]  (batch_masks is a list)
        #   batch_mask[i] is a bool tensor with True values corresponding to masked frames

        total_mask_loss, total_unmask_loss = 0, 0
        total_mask_acc, total_unmask_acc, total_acc = 0, 0, 0

        # iterate over different clustering models
        for k, scores in similarity_scores.items():
            # scores.shape = [batch, n_frames, k]

            # extract specific target
            k_targets = targets[k]

            batch_size = scores.shape[0]

            # metrics for a specific clustering model k
            clustering_mask_loss, clustering_unmask_loss = 0, 0
            clustering_mask_acc, clustering_unmask_acc, clustering_total_acc = 0, 0, 0
            cnt_mask, cnt_unmask, cnt_total = 0, 0, 0

            # iterate over sequences in the batch
            for seq_score, k_target, seq_len, mask in zip(scores, k_targets, frames_cnt, batch_masks):
                # seq_score.shape = [n_frames, k]
                # target.shape = [n_frames]
                # seq_len is an int
                # mask = [n_frames]

                # cross entropy loss and acc over frames without mask
                unmask_loss, unmask_acc, unmask_size = self._get_loss_acc(seq_score, seq_len, mask, k_target)
                clustering_unmask_loss += unmask_loss
                clustering_unmask_acc += unmask_acc
                cnt_unmask += unmask_size

                # cross entropy loss and acc over frames with mask
                mask_loss, mask_acc, mask_size = self._get_loss_acc(seq_score, seq_len, ~mask, k_target)
                clustering_mask_loss += mask_loss
                clustering_mask_acc += mask_acc
                cnt_mask += mask_size

                # total accuracy
                clustering_total_acc += accuracy(seq_score[:seq_len], k_target[:seq_len]) * seq_len
                cnt_total += seq_len

            # average across batch
            total_mask_loss += clustering_mask_loss / batch_size
            total_unmask_loss += clustering_unmask_loss / batch_size

            total_mask_acc += clustering_mask_acc / cnt_mask
            total_unmask_acc += clustering_unmask_acc / cnt_unmask
            total_acc += clustering_total_acc / cnt_total
        # total_{mask,unmask}_{loss,acc} are sums of losses for different clustering models
        return total_mask_loss, total_unmask_loss, total_mask_acc / len(similarity_scores), total_unmask_acc / len(similarity_scores), total_acc / len(
            similarity_scores)

    def training_step(self, batch, batch_index, inference=False):
        inputs, wave_lens, targets, indices = batch['waves'], batch['lens'], batch['targets'], batch['idx']
        # ic(inputs.shape)
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
        # ic(batch_index)
        # ic(total_loss)
        return dict(
            loss=total_loss,
            mask_loss=mask_loss.detach(),
            unmask_loss=unmask_loss.detach(),
            mask_acc=mask_acc,
            unmask_acc=unmask_acc,
            total_acc=total_acc,
            batch_size=inputs.shape[0]
        )

    # def validation_step(self, batch, batch_index):
    #     return self.training_step(batch, batch_index, inference=True)

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

    # def training_epoch_end(self, outputs):
    #     total_loss = sum(out['loss'] for out in outputs) / len(outputs)
    #     masked_loss = sum(out['mask_loss'] for out in outputs) / len(outputs)
    #     unmasked_loss = sum(out['unmask_loss'] for out in outputs) / len(outputs)
    #
    #     total_acc = sum(out['total_acc'] for out in outputs) / len(outputs)
    #     masked_acc = sum(out['mask_acc'] for out in outputs) / len(outputs)
    #     unmasked_acc = sum(out['unmask_acc'] for out in outputs) / len(outputs)
    #
    #     self.logger.experiment.add_scalar(f'Loss/Total', total_loss, self.current_epoch)
    #     self.logger.experiment.add_scalar(f'Loss/Masked', masked_loss, self.current_epoch)
    #     self.logger.experiment.add_scalar(f'Loss/Unmasked', unmasked_loss, self.current_epoch)
    #
    #     self.logger.experiment.add_scalar(f'Acc/Total', total_acc, self.current_epoch)
    #     self.logger.experiment.add_scalar(f'Acc/Masked', masked_acc, self.current_epoch)
    #     self.logger.experiment.add_scalar(f'Acc/Unmasked', unmasked_acc, self.current_epoch)


# %%
if __name__ == '__main__':
    # %%
    # ic.configureOutput(includeContext=True)
    # ............................................... Parameters .......................................................
    df_path = '/lnet/express/work/people/stankov/alignment/subset/subset.csv'
    labels_path = '/lnet/express/work/people/stankov/alignment/clustering_all/segments'
    parczech_clean_params = dict(
        recognized_sound_coverage__segments_lb=0.45,
        recognized_sound_coverage__segments_ub=0.93,
        duration__segments_lb=0.5,
        duration__segments_ub=40,
    )
    params = dict(
        seed=0xDEAD,
        ignore_index=-1,
        deterministic=True,
        # ------------ model params --------------
        sim=True,
        mask_weight=1,
        reduction='mean',
        softmax_temp=0.1,
        # ------------ dataset params ------------
        batch_size=2,
        num_workers=0,
        drop_last=True,
        batch_scale=10,
        pin_memory=True,
        # ------------ trainer params ------------
        n_gpus=2,
        epochs=50,
        stragegy='ddp',
        accelerator='ddp',
        fast_dev_run=100,
        overfit_batches=100,
        num_processes=None,
    )

    ks = [15]
    # ............................................... Dataset .......................................................
    dataset = ParCzechPretrainPL(
        clean_params=parczech_clean_params,
        data_path=df_path,
        km_labels=ks,
        labels_path=labels_path,
        num_workers=params['num_workers'],
        num_gpus=torch.cuda.device_count(),
        pin_mem=params['pin_memory'],
        shuffle=not params['deterministic'],
        batch_size=params['batch_size'],
        batch_scale=params['batch_scale'],
        ignore_index=params['ignore_index'],
        seed=params['seed'],
        drop_last=params['drop_last'],
    )
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
        # num_sanity_val_steps=0,
        max_epochs=params['epochs'],
        deterministic=params['deterministic'],
        # check_val_every_n_epoch=0,
        # fast_dev_run=10,
        fast_dev_run=params['fast_dev_run'],
        gpus=params['n_gpus'],
        checkpoint_callback=False,
        accelerator=params['accelerator'],
        replace_sampler_ddp=False,
        # strategy=params['stragegy'],
        # logger=logger
    )

    trainer.fit(hubert_pretrain, dataset)
    # %%