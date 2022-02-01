# %%
import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
from icecream import ic

class ParCzechDataset(Dataset):
    def __init__(self, df_path, resample_rate=16000, clean_params=None):
        super(ParCzechDataset, self).__init__()
        df = pd.read_csv(df_path, sep='\t')
        df = self.clean_data(df, clean_params)
        self.df = df.sort_values(by=['duration__segments'], ascending=False).copy().reset_index(drop=True)
        self.new_sr = resample_rate
        self.resample_transform = None

    def clean_data(self, df, params):
        # thresholds were selected based on the plot
        df = df[df.recognized_sound_coverage__segments > params['recognized_sound_coverage__segments_lb']]
        df = df[df.recognized_sound_coverage__segments < params['recognized_sound_coverage__segments_ub']]
        # removed 404.5 hours
        # use only long enough segments
        df = df[df.duration__segments > params['duration__segments_lb']]
        return df

    def extract_path(self, i):
        row = self.df.iloc[i]
        # need to remove prefix  'sentences_'
        mp3_name = row.mp3_name.split('_')[-1]
        return os.path.join(row.segment_path, mp3_name)

    def get_gold_transcript(self, path):
        with open(f'{path}.prt', 'r') as f:
            return f.read().rstrip()

    def get_asr_transcript(self, path):
        with open(f'{path}.asr', 'r') as f:
            return f.read().rstrip()

    def resample(self, sr, wav):
        if self.resample_transform is None:
            self.resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.new_sr)
        return self.resample_transform(wav)

    def get_wav(self, path):
        wav, sr = torchaudio.load(f'{path}.wav', normalize=True)
        # stereo to mono if needed
        if wav.size(0) == 2:
            wav = torch.mean(wav, dim=0).unsqueeze(0)
        return self.resample(sr, wav)

    def __getitem__(self, i):
        path = self.extract_path(i)
        return dict(
            gold_transcript=self.get_gold_transcript(path),
            asr_transcript=self.get_asr_transcript(path),
            wav=self.get_wav(path),
            path=os.path.dirname(path)
        )

    def __len__(self):
        return len(self.df)


class CommonVoiceDataset(Dataset):
    def __init__(self, base_dir, type, resample_rate=16000):
        self.data_path = os.path.join(base_dir, 'clips')
        self.df = pd.read_csv(os.path.join(base_dir, f'{type}.tsv'), sep='\t')
        self.resample_rate = resample_rate
        self.resample_transform = None

    def resample(self, waveform, sr):
        if self.resample_transform is None:
            self.resample_transform = torchaudio.transforms.Resample(sr, self.resample_rate)
        return self.resample_transform(waveform)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.item()
        waveform, sample_rate = torchaudio.load(os.path.join(self.data_path, self.df.path[i]))
        return dict(
            wav=self.resample(waveform, sample_rate),
            path=self.df.path[i]
        )

    def __len__(self):
        return len(self.df)


class MFCCExtractorPL(pl.LightningModule):
    def __init__(self, n_mffcs, n_mels, f_max, resample_rate, output_dir, n_fft=400):
        super(MFCCExtractorPL, self).__init__()
        self.output_dir = output_dir
        self.n_fft = n_fft
        self.MFCC_transform = torchaudio.transforms.MFCC(
            resample_rate,
            n_mfcc=n_mffcs,
            melkwargs=dict(
                n_mels=n_mels,
                n_fft=n_fft,  # default
                hop_length=n_fft // 2,  # default
                f_max=f_max,
            )
        )
        self.delta_transform = torchaudio.transforms.ComputeDeltas()

    def prepare_data(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def forward(self, batch):
        wavs = batch[0]

        mfccs_batch = self.MFCC_transform(wavs)
        deltas_batch = self.delta_transform(mfccs_batch)
        deltas2_batch = self.delta_transform(deltas_batch)
        # all shapes [batch_size, 1, 13, max_n_frames]

        # stacking features
        output = torch.cat([mfccs_batch, deltas_batch, deltas2_batch], dim=2).squeeze().permute(0, 2, 1)
        # [batch_size, max_n_frames, 13 * 3]
        return output


class SaveResultsCB(pl.Callback):
    def __init__(self, target_path, n_fft, buffer_size, df_type):
        self.df_type = df_type
        self.output_dir = target_path
        self.n_fft = n_fft
        # number of frames to store at one csv
        self.buffer_size = buffer_size

        self.dataframes = []
        self.current_buffer = 0
        # count how many df written to disk
        self.cnt = 0

    def extract_name(self, path):
        if self.df_type == 'common_voice':
            return path
        elif self.df_type == 'parczech':
            return os.path.join(path.split('/')[-2:])
        else:
            raise NotImplementedError(f'{self.df_type} is not supported')

    def write_df(self, trainer):
        output_path = os.path.join(self.output_dir, f'{trainer.global_rank:02}-{self.cnt:03}.csv')
        result = pd.concat(self.dataframes).reset_index()
        result['path'] = result['path'] + '/' + result['index'].astype(str)
        result.drop('index', axis=1).to_csv(output_path, index=False)

        self.current_buffer = 0
        self.dataframes = []
        self.cnt += 1

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, paths, lengths = batch
        outputs = outputs.cpu().numpy()
        for wav_len, features, path in zip(lengths, outputs, paths):
            n_frames = compute_frames(wav_len, self.n_fft)
            self.current_buffer += n_frames

            # select only useful frames without padding
            features = features[:n_frames + 1]

            features_df = pd.DataFrame(data=features)
            features_df['path'] = self.extract_name(path)
            self.dataframes.append(features_df)

            if self.current_buffer >= self.buffer_size:
                self.write_df(trainer)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        if self.dataframes != []:
            self.write_df(trainer)


def compute_frames(wave_len, n_fft):
    return 2 * wave_len // n_fft


def collate_fn(batch):
    M = max([x['wav'].size(-1) for x in batch])
    wavs = []
    paths = []
    for x in batch:
        padded = F.pad(x['wav'], (0, M - x['wav'].size(-1)))
        wavs.append(padded)
        paths.append(x['path'])

    # save lengths of waveforms, will be used to cut the padding from spectrogram
    lengths = [x['wav'].size(-1) for x in batch]
    return torch.stack(wavs, dim=0), paths, lengths


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, lim=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    if lim is not None:
        plt.axvline(x=lim, color='red')
    plt.show(block=False)


# %%
if __name__ == '__main__':
    # %%

    # logging_file = '/lnet/express/work/people/stankov/alignment/Thesis/mfcc.log'
    parczech_df_path = '/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path.csv'

    # under base dir there are tsv file and clips/ folder
    common_voice_base_dir = '/root/common_voice_data/cv-corpus-7.0-2021-07-21/cs'

    # directory where mfccs will be stored
    output_dir = os.path.join(common_voice_base_dir, 'mffcs')

    # logging.basicConfig(filename=logging_file, filemode='a', level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S %d.%m.%Y')

    params = dict(
        resample_rate=16000,
        batch_size=60,
        n_mffcs=13,
        n_mels=40,
        n_fft=400,
        buffer_size=50000,
        df_type='common_voice',
    )

    parczech_clean_params = dict(
        recognized_sound_coverage__segments_lb=0.45,
        recognized_sound_coverage__segments_ub=0.93,
        duration__segments_lb=0.5,
    )
    # dataset = ParCzechDataset(parczech_df_path, resample_rate=params['resample_rate'], clean_params=parczech_clean_params)
    dataset = CommonVoiceDataset(common_voice_base_dir, 'train', params['resample_rate'])
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 4, pin_memory=True)
    extractor = MFCCExtractorPL(n_mffcs=params['n_mffcs'], n_mels=params['n_mels'], n_fft=params['n_fft'], f_max=params['resample_rate']// 2,
                                output_dir=output_dir, resample_rate=params['resample_rate'])


    cb = SaveResultsCB(output_dir, params['n_fft'], buffer_size=params['buffer_size'], df_type=params['df_type'])
    trainer = pl.Trainer(gpus=-1, strategy='ddp', num_sanity_val_steps=0, callbacks=cb, precision=16, deterministic=True)
    trainer.predict(extractor, dataloader)

