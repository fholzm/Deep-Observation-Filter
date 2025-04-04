import glob
import torch, torchaudio
import pandas as pd


class DirectionalNoiseDatasetPrerendered(torch.utils.data.Dataset):

    def __init__(self, config, directory):
        self.directory = directory
        self.n_remotemics = len(config["remote_mic"]["position"])
        self.n_virtualmics = 1
        self.delay = config["model"]["delay"]
        self.nfft = config["nfft"]
        self.hopsize = config["model"]["hopsize"]
        self.window = torch.hann_window(self.nfft)

        sample_data, _ = torchaudio.load(self.directory + f"scene_0.wav")
        self.samplelength = sample_data.shape[-1] - self.delay
        self.metadata = pd.read_pickle(self.directory + "metadata.pkl")

    def __len__(self):
        return len(glob.glob1(self.directory, "*.wav"))

    def __getitem__(self, index):
        signal, _ = torchaudio.load(self.directory + f"scene_{index}.wav")

        rm_signal = signal[: self.n_remotemics, self.delay : self.samplelength]
        vm_signal = signal[-self.n_virtualmics :, 0 : self.samplelength - self.delay]

        return rm_signal, vm_signal, self.metadata.loc[index].to_dict()
