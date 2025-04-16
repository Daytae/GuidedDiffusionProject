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

    def __init__(self, latents, labels):
        # open pickle and grab data
      
        self.latents = latents
        self.labels = labels

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
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            latent = data['latent']
            label = data['labels']
            full_dataset = LatentDataset(latent, label)

            dataset_size = len(full_dataset)
            print(f"Loaded full dataset of {dataset_size} examples")

            self.full_dataloader = DataLoader(
                full_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0, # to possibly fix serializastion bug
                collate_fn=peptide_collate_fn,
                pin_memory=False
            )
