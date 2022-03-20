# %%
import os
import sys
from collections import OrderedDict
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from icecream import ic
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch import optim
from torchmetrics.functional import accuracy
import math

if not sys.__stdin__.isatty():
    # running in interactive shell
    from hubert.pretrain_dataset import ParCzechPretrainPL
    from hubert.clustering.filter_dataframe import FilterLB, FilterUB
else:
    from pretrain_dataset import ParCzechPretrainPL
    from clustering.filter_dataframe import FilterLB, FilterUB



class HubertPretrainPL(pl.LightningModule):
    def __init__(self,
                 hubert_model,
                 cluster_sizes,
                 proj_dim,
                 mask_weight,
                 softmax_temp,
                 betas,
                 warm_up_perc,
                 epochs,
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
        self.epochs = epochs
        self.lr = peak_lr
        self.warm_up_epochs = math.ceil(epochs * warm_up_perc)

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
            # for masked indices set True
            mask.index_fill_(0, index_mask.to(self.device), True)
            # so where the mask is True we will replace the values
            batch_masks.append(mask)

        batch_masks = torch.stack(batch_masks)
        masked_batch = feature_batch.masked_fill(batch_masks.unsqueeze(-1), self.mask)
        return masked_batch, batch_masks

    def forward(self, inputs, wave_lens):
        # inputs.shape = [batch, max_wave_len]
        # wave_lens.shape = [batch]

        features_batch, frames_cnt = self.hubert_model.feature_extractor(inputs, wave_lens)
        features_batch, batch_masks = self._mask_span(features_batch, frames_cnt)

        encoder_features = self.hubert_model.encoder(features_batch, frames_cnt)
        return encoder_features, frames_cnt, batch_masks

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
        return torch.sum(predicted == trg)

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
        loss = F.cross_entropy(new_seq, new_trg, reduction='sum')
        # multiply acc by the number of elements in the sequence
        acc = self._accuracy(new_seq, new_trg)
        return loss, acc, new_seq.shape[0]

    def _compute_loss_acc(self, similarity_scores, frames_cnt, targets, batch_masks):
        # similarity_scores[k].shape = [batch, n_frames, k]  (similarity_scores is a dict)
        # frames_cnt.shape = [batch]
        # targets[k].shape = [batch, n_frames]  (targets is a dict)
        # batch_mask_indices[i] = tensor with indices with masked frames (batch_mask_indices is a list)
        # batch_masks[i].shape = [n_frames]  (batch_masks is a list)
        #   batch_mask[i] is a bool tensor with True values corresponding to masked frames

        total_mask_loss, total_unmask_loss = 0, 0
        total_mask_acc, total_unmask_acc, total_acc = 0, 0, 0
        cnt_mask, cnt_unmask, cnt_total = 0, 0, 0

        # iterate over different clustering models
        for k, scores in similarity_scores.items():
            # scores.shape = [batch, n_frames, k]

            # extract specific target
            k_targets = targets[k]

            # iterate over sequences in the batch
            for seq_score, k_target, seq_len, mask in zip(scores, k_targets, frames_cnt, batch_masks):
                # seq_score.shape = [n_frames, k]
                # target.shape = [n_frames]
                # seq_len is an int
                # mask = [n_frames]

                # cross entropy loss and acc over frames without mask
                unmask_loss, unmask_acc, unmask_size = self._get_loss_acc(seq_score, seq_len, mask, k_target)
                total_unmask_loss += unmask_loss
                total_unmask_acc += unmask_acc
                cnt_unmask += unmask_size

                # cross entropy loss and acc over frames with mask
                mask_loss, mask_acc, mask_size = self._get_loss_acc(seq_score, seq_len, ~mask, k_target)
                total_mask_loss += mask_loss
                total_mask_acc += mask_acc
                cnt_mask += mask_size

                # total accuracy
                total_acc += accuracy(seq_score[:seq_len], k_target[:seq_len]) * seq_len
                cnt_total += seq_len

        total_mask_acc = total_mask_acc / cnt_mask
        total_unmask_acc = total_unmask_acc / cnt_unmask
        total_acc = total_acc / (cnt_mask + cnt_unmask)

        if self.reduction == 'mean':
            total_mask_loss = total_mask_loss / cnt_mask
            total_unmask_loss = total_unmask_loss / cnt_unmask
        elif self.reduction == 'sum':
            batch_size = frames_cnt.shape[0]
            total_mask_loss = total_mask_loss / batch_size
            total_unmask_loss = total_unmask_loss / batch_size
        else:
            raise RuntimeError(f'{self.reduction} can be ["sum", "mean"]')

        return total_mask_loss, total_unmask_loss, total_mask_acc, total_unmask_acc, total_acc

    def _perform_step(self, batch):
        inputs, wave_lens, targets = batch['waves'], batch['lens'], batch['targets']
        # inputs.shape = [batch, max_wave_len]
        # wave_lens.shape = [batch]
        # targets[k].shape = [batch, n_frames] ... targets is a dict and n_frames << max_wave_len

        encoder_features, frames_cnt, batch_masks = self(inputs, wave_lens)
        # encoder_features.shape = [batch, max_seq_len, hubert_features]
        scores = self._compute_cos_sim(encoder_features)
        # scores[k].shape = [batch, n_frames, k]

        mask_loss, unmask_loss, mask_acc, unmask_acc, total_acc = self._compute_loss_acc(scores, frames_cnt, targets, batch_masks)
        total_loss = self.mask_loss_weight * mask_loss + (1 - self.mask_loss_weight) * unmask_loss
        return total_loss, mask_loss.detach(), unmask_loss.detach(), total_acc.detach(), mask_acc.detach(), unmask_acc.detach()

    def training_step(self, batch, batch_index):
        total_loss, mask_loss, unmask_loss, total_acc, mask_acc, unmask_acc = self._perform_step(batch)
        return dict(
            loss=total_loss,
            mask_loss=mask_loss,
            unmask_loss=unmask_loss,
            mask_acc=mask_acc,
            unmask_acc=unmask_acc,
            total_acc=total_acc,
        )

    def validation_step(self, batch, batch_index):
        total_loss, mask_loss, unmask_loss, total_acc, mask_acc, unmask_acc = self._perform_step(batch)
        return dict(
            loss=total_loss,
            mask_loss=mask_loss,
            unmask_loss=unmask_loss,
            mask_acc=mask_acc,
            unmask_acc=unmask_acc,
            total_acc=total_acc,
        )

    def configure_optimizers(self):
        def lr_scheduler(cur_epoch):
            if cur_epoch < self.warm_up_epochs:
                scalar = cur_epoch / self.warm_up_epochs
            else:
                scalar = (self.epochs - cur_epoch)/(self.epochs - self.warm_up_epochs)
            assert scalar >= 0
            return float(scalar)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def _epoch_end(self, outputs, name):
        total_loss = sum(out['loss'] for out in outputs) / len(outputs)
        masked_loss = sum(out['mask_loss'] for out in outputs) / len(outputs)
        unmasked_loss = sum(out['unmask_loss'] for out in outputs) / len(outputs)
        total_loss = self.all_gather(total_loss).mean()
        masked_loss = self.all_gather(masked_loss).mean()
        unmasked_loss = self.all_gather(unmasked_loss).mean()

        total_acc = sum(out['total_acc'] for out in outputs) / len(outputs)
        masked_acc = sum(out['mask_acc'] for out in outputs) / len(outputs)
        unmasked_acc = sum(out['unmask_acc'] for out in outputs) / len(outputs)
        total_acc = self.all_gather(total_acc).mean()
        masked_acc = self.all_gather(masked_acc).mean()
        unmasked_acc = self.all_gather(unmasked_acc).mean()

        # log only at rank 0
        if self.global_rank == 0:
            # ic(total_loss, masked_loss, unmasked_loss, total_acc, masked_acc, unmasked_acc)
            self.logger.experiment.add_scalar(f'Loss Total/{name}', total_loss, self.current_epoch)
            self.logger.experiment.add_scalar(f'Loss Masked/{name}', masked_loss, self.current_epoch)
            self.logger.experiment.add_scalar(f'Loss Unmasked/{name}', unmasked_loss, self.current_epoch)

            self.logger.experiment.add_scalar(f'Acc Total/{name}', total_acc, self.current_epoch)
            self.logger.experiment.add_scalar(f'Acc Masked/{name}', masked_acc, self.current_epoch)
            self.logger.experiment.add_scalar(f'Acc Unmasked/{name}', unmasked_acc, self.current_epoch)
        return total_loss, masked_loss, unmasked_loss, total_acc, masked_acc, unmasked_acc

    def training_epoch_end(self, outputs):
        _ = self._epoch_end(outputs, 'Train')

    def validation_epoch_end(self, outputs):
        total_loss, _, _, total_acc, masked_acc, _ = self._epoch_end(outputs, 'Val')
        self.log('val_total_loss', total_loss, logger=False, sync_dist=True)
        self.log('val_total_acc', total_acc, logger=False, sync_dist=True)
        self.log('val_masked_acc', masked_acc, logger=False, sync_dist=True)


def get_logging_dir_name(parameters, data, clusters):
    naming = dict(
        deterministic='det-{}',
        # ------------ model params --------------
        sim='sim-{}',
        mask_weight='mw-{:.2f}',
        reduction='red-{}',
        softmax_temp='st{:.2f}',
        peak_lr='lr-{:.5f}',
        warm_up='wu-{:.2f}',
        # ------------ trainer params ------------
        epochs='ep{:03}',
    )
    effective_batch_size = parameters["batch_size"] * parameters["n_gpus"]
    result = [
        f'ks-' + '.'.join(list(map(str, clusters))),
        f'ebs{effective_batch_size}'
    ]

    if parameters['limit_train_batches'] == 1.0 and not parameters['fast_dev_run']:
        cnt = len(data.train_data) // effective_batch_size
        ic('Using all train data')
        result.append(f'cnt{cnt}')

    if parameters['limit_train_batches'] != 1.0:
        result.append(f'cnt-{parameters["limit_train_batches"]}')

    if parameters['fast_dev_run']:
        result.append(f'cnt-{parameters["fast_dev_run"]}')

    for key, name in naming.items():
        result.append(name.format(parameters[key]))

    time = datetime.now().strftime('%d.%m__%H.%M')
    return '_'.join(result), time


# %%
if __name__ == '__main__':
    # %%
    # ic.configureOutput(includeContext=True)
    ckp_path_pretrain = ''

    # ............................................... Parameters .......................................................
    df_path = '/lnet/express/work/people/stankov/alignment/subset/subset.csv'
    labels_path = '/lnet/express/work/people/stankov/alignment/clustering_all/segments'
    parczech_filters = [
        FilterLB(value=0.45, name='recognized_sound_coverage__segments'),
        FilterUB(value=0.93, name='recognized_sound_coverage__segments'),
        FilterLB(value=0.5, name='duration__segments'),
        FilterUB(value=20, name='duration__segments'),
    ]

    params = dict(
        seed=0xDEAD,
        ignore_index=-1,
        deterministic=False,
        # ------------ model params --------------
        sim=True,
        mask_weight=1,
        reduction='mean',
        softmax_temp=0.01,
        p=0.08,
        l=10,
        peak_lr=1e-4,
        hubert_features=768,
        betas=(0.9, 0.98),
        proj_dim=256,
        warm_up=0.1,
        # ------------ dataset params ------------
        batch_size=4,
        num_workers=8,
        drop_last=False,
        batch_scale=10,
        pin_memory=torch.cuda.is_available(),
        val_fraction=0.1,
        # ------------ trainer params ------------
        n_gpus=torch.cuda.device_count(),
        epochs=50,
        strategy='ddp',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        fast_dev_run=False,
        overfit_batches=None,
        num_processes=1 if torch.cuda.is_available() else 2,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )
    params['num_workers'] = params['num_workers'] * params['n_gpus']

    ks = [15, 25, 100]
    # ............................................... Dataset .......................................................

    dataset = ParCzechPretrainPL(
        parczech_filters=parczech_filters,
        data_path=df_path,
        km_labels=ks,
        labels_path=labels_path,
        num_workers=params['num_workers'],
        num_gpus=params['n_gpus'],
        pin_mem=params['pin_memory'],
        shuffle=not params['deterministic'],
        batch_size=params['batch_size'],
        batch_scale=params['batch_scale'],
        ignore_index=params['ignore_index'],
        seed=params['seed'],
        drop_last=params['drop_last'],
        val_frac=params['val_fraction'],
    )
    # ............................................... Model .......................................................
    hubert_base_model = torchaudio.models.hubert_base()

    hubert_pretrain = HubertPretrainPL(
        hubert_base_model,
        cluster_sizes=ks,
        sim=params['sim'],
        proj_dim=params['proj_dim'],
        reduction=params['reduction'],
        hubert_features=params['hubert_features'],

        warm_up_perc=params['warm_up'],
        epochs=params['epochs'],
        betas=params['betas'],
        peak_lr=params['peak_lr'],

        p=params['p'],
        l=params['l'],
        mask_weight=params['mask_weight'],
        softmax_temp=params['softmax_temp'],

        ignore_index=params['ignore_index'],
    )

    # ............................................... Training .......................................................
    # Logs are saved to os.path.join(save_dir, name, version)
    # save_dir/name/sub_dir/version
    logging_dir, cur_time = get_logging_dir_name(params, dataset, ks)
    logger = TensorBoardLogger(save_dir='logs', name=logging_dir, version=cur_time)

    checkpoint_dir = os.path.join('logs', logging_dir, f'{cur_time}', 'checkpoints')
    checkpoint_fn = '.ep{epoch:03d}__val_tot_loss-{val_total_loss:.3f}__val_tot_acc-{val_total_acc:.3f}__val_mask_acc-{val_masked_acc:.3f}'
    # save based on valid loss and valid acc
    loss_checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        mode='min',
        dirpath=checkpoint_dir,
        auto_insert_metric_name=False,
        filename='tot_loss' + checkpoint_fn,
        save_weights_only=True,
    )

    masked_acc_checkpoint_callback = ModelCheckpoint(
        monitor="val_masked_acc",
        mode='max',
        dirpath=checkpoint_dir,
        auto_insert_metric_name=False,
        filename='mask_acc' + checkpoint_fn,
        save_weights_only=True,
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        auto_insert_metric_name=False,
        filename='last_ckp' + checkpoint_fn,
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=params['epochs'],
        deterministic=params['deterministic'],
        # check_val_every_n_epoch=0,
        # fast_dev_run=10,
        gpus=params['n_gpus'],
        fast_dev_run=params['fast_dev_run'],
        # enable_checkpointing=False,
        replace_sampler_ddp=False,
        accelerator=params['accelerator'],
        strategy=params['strategy'],
        num_processes=params['num_processes'],
        limit_train_batches=params['limit_train_batches'],
        limit_val_batches=params['limit_val_batches'],
        callbacks=[loss_checkpoint_callback, masked_acc_checkpoint_callback, last_checkpoint_callback],
        logger=logger,
        precision=16,
    )

    if ckp_path_pretrain == '':
        trainer.fit(hubert_pretrain, dataset)
    else:
        raise NotImplementedError('Implement uploading the checkpoints')
    # %%