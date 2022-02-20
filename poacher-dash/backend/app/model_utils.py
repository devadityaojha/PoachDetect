import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy import signal
import torch.nn as nn
import torch.nn.functional as F


class AudioData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_name):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_name = file_name

    def __len__(self):
        return 1

    def transform(self, idx, filepath):

        samplerate, data = wavfile.read(filepath)
        data = data[:50000]
        frequencies, times, spectogram = signal.spectrogram(data, samplerate, nfft=1000, mode='angle')

        return frequencies[:100]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filepath = self.file_name

        return {"sample": self.transform(idx, filepath)}


class Model(nn.Module):

    def __init__(self, n_input=100):
        super(Model, self).__init__()
        n_channel = 1

        self.encoder = nn.Sequential(
            nn.Conv1d(n_input, n_channel, kernel_size=1, stride=2),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(1),
            nn.Conv1d(n_channel, n_channel, kernel_size=1),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(1),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=1),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(1)
        )

        self.fc1 = nn.Linear(2 * n_channel, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        return torch.sigmoid(x).squeeze()


def infer(model, audio_data):
    model.eval()
    a, b = 1, 100
    audio_data = torch.Tensor(audio_data).view(1, 100, 1)
    print(audio_data.size())
    predictions = model(audio_data.float().view(1, 100, 1))
    print(predictions)

    return predictions >= 0.5
