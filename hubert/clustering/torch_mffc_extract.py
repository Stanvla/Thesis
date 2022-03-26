# %%
import logging
import os.path
import shutil
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from icecream import ic
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from io import StringIO

try:
    from hubert.clustering.filter_dataframe import clean_data_parczech
except ModuleNotFoundError:
    from filter_dataframe import clean_data_parczech


class ParCzechDataset(Dataset):
    def __init__(self, df_path, resample_rate=16000, df_filters=None, sep='\t', sort=True, train_flag=True, iloc=True, *args, **kwargs):
        super(ParCzechDataset, self).__init__()
        self.df = pd.read_csv(df_path, sep=sep)
        self.filter_df(df_filters)

        if train_flag:
            self.df = self.df[(self.df.type == 'train') | (self.df.type == 'other')]

        if sort:
            self.df = self.df.sort_values(by=['duration__segments'], ascending=False).copy().reset_index(drop=True)

        self.new_sr = resample_rate
        self.resample_transform = None
        # this configures the __getitem__ method, when self.iloc is true index in __getitem__ is interpreted as integer location in dataframe
        # when self.iloc is False index in __getitem__ is interpreted as an element in self.df.index
        self.iloc = iloc

    def index_df(self, i, column_name=None):
        if self.iloc:
            row = self.df.iloc[i]
        else:
            row = self.df.loc[i]

        if column_name is not None:
            return row[column_name]
        return row

    def get_mp3_name(self, i):
        row = self.index_df(i)
        try:
            # need to remove prefix  'sentences_'
            mp3_name = row.mp3_name.split('_')[-1]
        except:
            ic(row)
            raise ValueError(f'can not find row by index {i}')
        return mp3_name

    def extract_path(self, i):
        row = self.index_df(i)
        mp3_name = self.get_mp3_name(i)
        return os.path.join(row.segment_path, mp3_name)

    def get_gold_transcript(self, path):
        with open(f'{path}.prt', 'r') as f:
            return f.read().rstrip()

    def get_asr_transcript(self, path):
        with open(f'{path}.asr', 'r') as f:
            return f.read().rstrip()

    def _safe_read_df(self, path, names, header, sep, dtypes, na_values):
        if not os.path.isfile(path):
            print(f'{path} does not exist')
        replace_dict = {
            '"': "__double_quotes__",
        }
        with open(path, 'r') as f:
            src = ''.join(f.readlines())

        for k, v in replace_dict.items():
            src = src.replace(k, v)

        df = pd.read_csv(StringIO(src), names=names, header=header, sep=sep, dtype=dtypes, na_values=na_values)

        return df

    def get_recognized_df(self, path, i):
        # will extract recognized based on word ids
        header = ['word', 'word_id', 'start_time', 'end_time', 'XXX', 'avg_char_duration', 'speaker']

        try:
            words_df = self._safe_read_df(
                f'{path}.words',
                names=header,
                header=None,
                sep='\t',
                # replace_col=['word'],
                dtypes=None,
                na_values=None,
            )
        except Exception as e:
            ic(path)
            ic(e)
            raise ValueError(f'Can not read file {path}')

        word_ids = words_df['word_id'].values.tolist()

        # read aligned file
        path_aligned = f"/lnet/express/work/people/stankov/alignment/results/full/words-aligned/jan/words_{self.get_mp3_name(i)}.tsv"
        # normalization is done by the length of the "true_word"
        header_aligned = ['true_w', 'trans_w', 'joined', 'id', 'recognized', 'dist', 'dist_norm', 'start', 'end', 'time_len_ms', 'time_len_norm']
        dtypes = dict(
            true_w=str,
            trans_w=str,
            joined=bool,
            id=str,
            recognized=bool
        )
        for name in header_aligned:
            if name not in dtypes:
                dtypes[name] = float

        aligned_df = self._safe_read_df(
            path_aligned,
            header_aligned,
            sep='\t',
            dtypes=dtypes,
            na_values='-',
            header=0
        )
        aligned_df.trans_w = aligned_df.trans_w.fillna('-')
        aligned_df = aligned_df[aligned_df['id'].isin(word_ids)]
        return aligned_df

    def get_recognized_transcript(self, path, i):
        aligned_df = self.get_recognized_df(path, i)
        # from miliseconds to seconds
        start_time = aligned_df['start'].min() / 1000
        end_time = aligned_df['end'].max() / 1000
        mp3_name = self.get_mp3_name(i)

        path_transcribed = f'/lnet/express/work/people/stankov/alignment/results/full/scrapping/jan/{mp3_name}/{mp3_name}.tsv'
        if not os.path.isfile(path_transcribed):
            # print(f'{mp3_name} is not in scraping')
            path_transcribed = f'/lnet/express/work/people/stankov/alignment/results/full/time-extracted/jan/{mp3_name}.tsv'

        header_transcribed = ['start', 'end', 'recognized', 'true_word', 'cnt', 'dist']
        transcribed_df = pd.read_csv(path_transcribed, names=header_transcribed, header=None, sep='\t')
        # ic(start_time, end_time, transcribed_df.head())
        transcript = transcribed_df[(transcribed_df.start >= start_time) & (transcribed_df.end <= end_time)].recognized.values.tolist()
        return ' '.join(transcript)

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

    def duration_hours(self, filters=None):
        if filters is not None:
            df = clean_data_parczech(self.df, filters)
        else:
            df = self.df
        return df.duration__segments.sum() / 3600

    def plot_stat(self, col_name, filters=None):
        if filters is not None:
            df = clean_data_parczech(self.df, filters)
        else:
            df = self.df
        # plt.boxplot(dataset.df.avg_char_duration__segments, vert=False)

        print(df.sort_values(by=[col_name])[col_name])
        plt.plot(range(len(df)), df.sort_values(by=[col_name])[col_name])
        plt.title(f'{col_name} sorted, {self.duration_hours(filters):.2f}h')
        plt.xlabel('segments')
        plt.ylabel(col_name)
        plt.show()

    def filter_df(self, filters, reset_index=False):
        if filters is None:
            return

        self.df = clean_data_parczech(self.df, filters)
        if reset_index:
            self.df.reset_index(drop=True, inplace=True)

    def get_columns(self):
        return self.df.columns.values

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


def clean_data(df, params):
    # thresholds were selected based on the plot
    df = df[(df.type == 'train') | (df.type == 'other')]
    df = df[df.recognized_sound_coverage__segments > params['recognized_sound_coverage__segments_lb']]
    df = df[df.recognized_sound_coverage__segments < params['recognized_sound_coverage__segments_ub']]
    # removed 404.5 hours
    # use only long enough segments
    ic(df.duration__segments.sum() / 3600)
    if 'duration__segments_lb' in params:
        df = df[df.duration__segments > params['duration__segments_lb']]

    if 'duration__segments_ub' in params:
        df = df[df.duration__segments < params['duration__segments_ub']]
        ic(df.duration__segments.sum() / 3600)
    return df


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
        self.sr = resample_rate

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
        wavs, _, lens = batch

        mfccs_batch = self.MFCC_transform(wavs)
        deltas_batch = self.delta_transform(mfccs_batch)
        deltas2_batch = self.delta_transform(deltas_batch)
        # all shapes [batch_size, 1, 13, max_n_frames]

        # stacking features
        output = torch.cat([mfccs_batch, deltas_batch, deltas2_batch], dim=2).squeeze().permute(0, 2, 1)
        # [batch_size, max_n_frames, 13 * 3]
        n_frames = torch.tensor([compute_frames(l, self.sr) for l in lens], device=self.device)
        return output, n_frames


def compute_frames(wave_len, sample_rate):
    ms_int = int(wave_len / sample_rate * 1000)
    # these "random" operations mimic how hubert.feature extractor counts frames in the audio
    new_ms = (ms_int - (ms_int % 5) - 1) // 20
    return new_ms


class SaveResultsCB(pl.Callback):
    def __init__(self, target_path, n_fft, buffer_size, df_type, total_batches, resample_rate=16000, frame_length=20):
        self.df_type = df_type
        self.output_dir = target_path
        self.n_fft = n_fft
        # number of frames to store at one csv
        self.buffer_size = buffer_size
        self.frame_length = frame_length

        self.dataframes = []
        self.current_buffer = 0
        # count how many df written to disk
        self.cnt = 0
        self.resample_rate = resample_rate
        self.total_duration_sec = 0
        self.loggers = {}
        self.total_batches = total_batches

    def extract_name(self, path):
        if self.df_type == 'common_voice':
            return path
        elif self.df_type == 'parczech':
            return '/'.join(path.split('/')[-2:])
        else:
            raise NotImplementedError(f'{self.df_type} is not supported')

    def write_df(self, trainer):
        output_path = os.path.join(self.output_dir, f'{trainer.global_rank:02}-{self.cnt:04}.csv')
        result = pd.concat(self.dataframes).reset_index()
        result['path'] = result['path'] + '/' + result['index'].astype(str)
        result.drop('index', axis=1).to_csv(output_path, index=False)

        self.current_buffer = 0
        self.dataframes = []
        self.cnt += 1

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, paths, wave_lens = batch
        self.total_duration_sec += sum(w_len / self.resample_rate for w_len in wave_lens)

        mfcc_features, frames_cnt = outputs[0].cpu().numpy(), outputs[1].cpu().numpy()
        for n_frames, features, path in zip(frames_cnt, mfcc_features, paths):
            self.current_buffer += n_frames

            # select only useful frames without padding
            features = features[:n_frames]

            features_df = pd.DataFrame(data=features)
            features_df['path'] = self.extract_name(path)
            self.dataframes.append(features_df)

            if self.current_buffer >= self.buffer_size:
                self.write_df(trainer)

        if batch_idx % 50 == 0:
            logger = self.loggers[pl_module.global_rank]
            logger.debug(f'gpu={pl_module.global_rank:2} batches processed {batch_idx:4}/{self.total_batches} ... {batch_idx / self.total_batches:.4f}')

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        # setup loggers for each gpu
        # logging.basicConfig(filename=logging_file, filemode='a', level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S %d.%m.%Y')
        handler = logging.FileHandler(f'gpu-{pl_module.global_rank}.log')
        formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%H:%M:%S %d.%m.%Y')
        handler.setFormatter(formatter)

        logger = logging.getLogger(f'{pl_module.global_rank}')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        self.loggers[pl_module.global_rank] = logger

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        if self.dataframes != []:
            self.write_df(trainer)

        logger = self.loggers[pl_module.global_rank]
        total_duration_hours = int(self.total_duration_sec // 3600)
        remaining_seconds = int(self.total_duration_sec % 3600)
        total_duration_mins = int(remaining_seconds // 60)
        total_duration_secs = int(remaining_seconds % 60)
        logger.debug(f'gpu={pl_module.global_rank:2} finished, {total_duration_hours:3}:{total_duration_mins:2}:{total_duration_secs:.3f} or'
                     f' {self.total_duration_sec:.3f} seconds')


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
    # logging.basicConfig(filename=logging_file, filemode='a', level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S %d.%m.%Y')
    params = dict(
        resample_rate=16000,
        batch_size=70,
        n_mffcs=13,
        n_mels=40,
        n_fft=640,
        buffer_size=130000,
        df_type='parczech',
        frame_length_ms=20,
        data_type='validated'
    )
    parczech_clean_params = dict(
        recognized_sound_coverage__segments_lb=0.45,
        recognized_sound_coverage__segments_ub=0.93,
        duration__segments_lb=0.5,
    )

    if 'lnet' in os.getcwd():
        df_path = '/lnet/express/work/people/stankov/alignment/Thesis/clean_with_path_large.csv'
        # df = pd.read_csv(df_path, sep='\t')
        # directory where mfccs will be stored
        output_dir = '/lnet/express/work/people/stankov/alignment/mfcc'
        dataset = ParCzechDataset(df_path, resample_rate=params['resample_rate'], clean_params=parczech_clean_params)
    else:
        # under base dir there are tsv file and clips/ folder
        base_dir = '/root/common_voice_data/cv-corpus-7.0-2021-07-21/cs'
        # directory where mfccs will be stored
        output_dir = os.path.join(base_dir, 'mffcs')
        dataset = CommonVoiceDataset(base_dir, params['data_type'], params['resample_rate'])

    # %%
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 4, pin_memory=True)
    extractor = MFCCExtractorPL(n_mffcs=params['n_mffcs'], n_mels=params['n_mels'], n_fft=params['n_fft'], f_max=params['resample_rate'] // 2,
                                output_dir=output_dir, resample_rate=params['resample_rate'])

    cb = SaveResultsCB(output_dir, params['n_fft'], buffer_size=params['buffer_size'], df_type=params['df_type'], frame_length=params['frame_length_ms'],
                       total_batches=len(dataloader))

    trainer = pl.Trainer(gpus=-1, strategy='ddp', num_sanity_val_steps=0, callbacks=cb, deterministic=True, progress_bar_refresh_rate=0)
    # trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0, callbacks=cb, precision=16, deterministic=True, limit_predict_batches=10)
    trainer.predict(extractor, dataloader)

    ic('done')
