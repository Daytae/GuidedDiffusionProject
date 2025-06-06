import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
from torch.utils.data import Dataset, DataLoader

from load_and_sample import *
from guided_diffusion import guided_diffusion_1d
torch.set_float32_matmul_precision("high")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

# Read from the latent data file and put it into a dataloader

class LatentDataset(Dataset):
    def __init__(self, latents):
        self.latents = torch.from_numpy(latents).float().unsqueeze(1)

    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx]

# Initialize the diffusion model

def create_diffusion_model(unet_dim=128, latent_dim=128, num_timesteps=1000):
    torch.cuda.empty_cache()

    unet_model = guided_diffusion_1d.Unet1D(
        dim = unet_dim,
        channels=1,
        dim_mults=(1, 2, 4, 8)
    ).to(device)

    diffusion_model = guided_diffusion_1d.GaussianDiffusion1D(
        unet_model,
        seq_length=latent_dim,
        timesteps=num_timesteps,
        objective='pred_v'
    ).to(device)

    return diffusion_model

def sample_diffusion(diffusion_model, sample_batch_size=4, latent_dim=128):
    diffusion_model.eval()
    with torch.no_grad():
        latents = diffusion_model.sample(batch_size=sample_batch_size)
        latents = latents.reshape(sample_batch_size, latent_dim)
        return latents
    
# load vae
vae = load_vae_selfies("./saved_models/epoch=447-step=139328.ckpt")

# dataset
latents = np.load("latents_final.npy")
latents_dataset = LatentDataset(latents=latents)
# latents_dataloader = DataLoader(latents_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# load and train model
model = create_diffusion_model(unet_dim=128, latent_dim=128, num_timesteps=1000)

print("Created model & loaded data")

trainer = guided_diffusion_1d.Trainer1D(
    diffusion_model=model,
    dataset = latents_dataset,
    train_batch_size=32,
    save_and_sample_every=100000,
    num_samples=16,
    results_folder='./diffusion_results',
    num_workers=0
)

trainer.train()
    