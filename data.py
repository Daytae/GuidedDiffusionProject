import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from peptidevae.load_vae import load_vae, vae_forward, vae_decode


# Dataset of PeptideLatentCode
class PeptideLatentDataset(Dataset):
    def __init__(self, dataset_csv_path, vae, dataobj, max_peptide_sequence_length : int = 50):
        super().__init__()
        df = pd.read_csv(dataset_csv_path)
        self.vae = vae
        self.dataobj = dataobj
        self.max_peptide_sequence_length = max_peptide_sequence_length

        # get the peptide sequence / labels
        self.sequences = df['sequence'].values
        self.labels = torch.tensor(df['extinct'].values, dtype=torch.int8)

        # turns the peptide seqs into latents
        self.latents = self._encode_all_sequences()
    
    def _encode_all_sequences(self):
        '''Encodes all of the peptide sequences's into latent representations'''
        encoding_batch_size = 1024
        all_latents = []

        for i in tqdm(range(0, len(self.sequences), encoding_batch_size), desc="Encoding peptides"):
            batched_peptide_sequences = self.sequences[i:i+encoding_batch_size]
            with torch.no_grad():
                latents = vae_forward(batched_peptide_sequences, self.dataobj, self.vae)
            _latents, _loss = latents
            all_latents.append(_latents)
        
        return torch.cat(all_latents, dim=0)
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        label = self.labels[idx]
        return latent, label # x, y pair

