import numpy as np
import torch
import fnmatch
from sklearn.model_selection import train_test_split
import bisect
import sys
import os

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
sys.path.insert(0, ROOT_DIR)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_files, chunk_length=200, train=True, mean=None, std=None, random_state=42):
        self.train = train
        file_paths_train, file_paths_test = train_test_split(
            path_to_files, test_size=0.2, random_state=random_state,
        )

        self.files = file_paths_train if train else file_paths_test

        self.chunk_length = chunk_length

        if mean == None or std == None:
            self.mean, self.std = self.compute_mean_and_std()

        self.audio_lengths = [np.load(file).shape[0] for file in self.files]
        self.cumulative_lengths = np.cumsum(self.audio_lengths)
        self.n_chunks = sum(self.audio_lengths) // chunk_length

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        chunk_idx = idx
        file_idx = bisect.bisect_left(self.cumulative_lengths, chunk_idx * self.chunk_length)
        start = chunk_idx * self.chunk_length - (self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0)
        end = start + self.chunk_length
        sample = torch.from_numpy(np.load(self.files[file_idx])[start:end]).double()
        sample = (sample - self.mean) / self.std
        return sample

    def compute_mean_and_std(self):
        samples = np.concatenate([np.load(file) for file in self.files])
        mean = samples.mean()
        std = samples.std()
        return torch.tensor(mean), torch.tensor(std)



def fetch_npy_file_paths(data_dir):
    data_matches = []
    for root, dir_names, file_names in os.walk(data_dir):
        for file_name in fnmatch.filter(file_names, "*.npy"):
            data_matches.append((root, file_name))
    file_paths = []
    for root, file_name in data_matches:
        file_paths.append(os.path.join(root, file_name))
    return file_paths


if __name__ == "__main__":
    data_dir = os.path.join(ROOT_DIR, "data/blizzard/adventure_and_science_fiction")

    file_paths = fetch_npy_file_paths(data_dir=data_dir)
    dataset = AudioDataset(file_paths, train=True, random_state=42)
    train_loader = torch.utils.data.DataLoader(AudioDataset(file_paths, train=True, random_state=42), batch_size=128)
    for b in train_loader:
        print(b)
    test_loader = torch.utils.data.DataLoader(AudioDataset(file_paths, train=False, random_state=42), batch_size=128)
    print(train_loader)
    print(test_loader)
