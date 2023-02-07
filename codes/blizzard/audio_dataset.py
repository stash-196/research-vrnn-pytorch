import numpy as np
import torch
import fnmatch
from sklearn.model_selection import train_test_split

import sys
import os

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
sys.path.insert(0, ROOT_DIR)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_files, train=True, mean=None, std=None, random_state=42):
        self.train = train
        file_paths_train, file_paths_test = train_test_split(
            path_to_files, test_size=0.2, random_state=random_state,
        )
        self.files = file_paths_train if train else file_paths_test

        if mean==None or std==None:
            self.mean, self.std = self.compute_mean_and_std()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = np.load(self.files[idx])
        sample = torch.from_numpy(sample)
        sample = (sample - self.mean) / self.std
        return sample

    def compute_mean_and_std(self):
        first_sample = np.load(self.files[0])
        self.mean = np.mean(first_sample)
        self.std = np.std(first_sample)
        for i in range(1, len(self.files)):
            sample = np.load(self.files[i])
            self.mean = ((self.mean * i) + np.mean(sample)) / (i + 1)
            self.std = ((self.std * i) + np.std(sample)) / (i + 1)
        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)
        return mean, std


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

    train_loader = torch.utils.data.DataLoader(AudioDataset(file_paths, train=True, random_state=42))
    test_loader = torch.utils.data.DataLoader(AudioDataset(file_paths, train=False, random_state=42))
    print(train_loader)
    print(test_loader)
