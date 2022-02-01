# %%
import os.path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ParCzechDataset(Dataset):
    def __init__(self, df, resample_rate=16000):
        super(ParCzechDataset, self).__init__()
        self.df = df.copy().reset_index(drop=True)
        self.new_sr = resample_rate
        self.resample_transform = None

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

    logging_file = '/lnet/express/work/people/stankov/alignment/Thesis/mfcc.log'
    if os.path.exists(logging_file):
        os.remove(logging_file)

    logging.basicConfig(filename=logging_file, filemode='a', level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S %d.%m.%Y')

    resample_rate = 16000
    df = pd.read_csv('/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path.csv', sep='\t')

    # clean df a bit

    # thresholds were selected based on the plot
    df = df[(df.recognized_sound_coverage__segments > 0.45) & (df.recognized_sound_coverage__segments < 0.93)]
    # removed 404.5 hours

    # use only long enough segments
    df = df[df.duration__segments > 0.1]

    dataset = ParCzechDataset(df.sort_values(by=['duration__segments'], ascending=False), resample_rate=resample_rate)
    dataloader = DataLoader(dataset, batch_size=105, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MFCC_transform = torchaudio.transforms.MFCC(
        resample_rate,
        n_mfcc=13,
        melkwargs=dict(
            n_mels=40,
            n_fft=400,  # default
            hop_length=400//2,  # default
            f_max=16000//2,
        )
    ).to(device)

    delta_transform = torchaudio.transforms.ComputeDeltas().to(device)

    # %%
    # directory where mfccs will be stored
    logging.info('Start')
    target_dir = '/lnet/express/work/people/stankov/alignment/mfccs'
    processed = 0
    for i, (wavs, paths, lengths) in enumerate(dataloader):

        wavs = wavs.to(device)
        mfccs_batch = MFCC_transform(wavs)
        deltas_batch = delta_transform(mfccs_batch)
        deltas2_batch = delta_transform(deltas_batch)
        # mfccs_batch, deltas_batch, deltas2_batch have shape [batch_size, 1, 13, max_n_frames]

        # now for each x in batch stack mfccs, deltas, deltas2
        # resulting tensor will have shape [batch_size, 13*3, max_n_frames]
        # then transpose each x in batch,
        batch = torch.cat([mfccs_batch, deltas_batch, deltas2_batch], dim=2).squeeze().permute(0, 2, 1).cpu().numpy()
        # the resulting batch will have shape [batch_size, max_n_frames, 13*3]

        for wav_len, features, path in zip(lengths, batch, paths):
            n_frames = compute_frames(wav_len, 400)
            # select only useful frames without padding
            features = features[:n_frames+1]
            # plot_spectrogram(features.T, title=f'{n_frames}', lim=n_frames)

            mp3_name, segment_name = path.split('/')[-2:]

            features_df = pd.DataFrame(data=features)
            features_df['path'] = os.path.join(mp3_name, segment_name)

            output_path = os.path.join(target_dir, mp3_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            features_df.to_csv(os.path.join(output_path, f'{segment_name}.csv'), index=False)
            # print(os.path.join(output_path, f'{segment_name}.csv'))

        processed += wavs.size(0)
        if i % 4 == 0 and i > 0:
            logging.info(f'Processed {i+1} batches, or {processed} examples ({processed / len(dataset) * 100:.3f}%).')

        # print(os.path.join(target_dir, mp3_name, f'{segment_name}.csv'))

