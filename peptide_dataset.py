import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')

class LatentDataset(Dataset):
    '''
        Dataset with (latent code (256 dim), label)
    '''

    def __init__(self, data_path='data/latent_dataset.pkl'):
        # open pickle and grab data
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.latents = data_dict['latent']
        self.labels = data_dict['labels']


    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return {
            'latent': torch.tensor(self.latents[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx])
        }

def peptide_collate_fn(batch):
    latent = torch.stack([item['latent'] for item in batch])
    label = torch.stack([item['label'] for item in batch])

    return {
        'latent': latent,
        'label': label
    }

class LatentDataModule:
    def __init__(self, data_path='data/latent_dataset.pkl', batch_size=32, num_workers=4, train_val_split=0.9, shuffle=True, seed=10478):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.shuffle = shuffle
        self.seed = seed

        self.setup()

    def setup(self):
        full_dataset = LatentDataset(self.data_path)

        dataset_size = len(full_dataset)
        train_size = int(self.train_val_split *  dataset_size)
        val_size = dataset_size - train_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)

        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=peptide_collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=peptide_collate_fn,
            pin_memory=True
        )
    
    def full_dataloader(self):
        full_dataset = LatentDataset(self.data_path)
        return DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=peptide_collate_fn,
            pin_memory=True
        )
    
    