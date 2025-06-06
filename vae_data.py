import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import pickle

import lightning.pytorch as pl
import torch

from load_and_sample import *
import gzip

torch.set_float32_matmul_precision("high")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

vae = load_vae_selfies("./saved_models/epoch=447-step=139328.ckpt")
vae.to(device).eval()

selfies_latents = []
failed_selfies = []

save_every = 50000 # save every 50k

with gzip.open('data/train_selfie.gz', 'rt', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f, desc='Processing Selfies strings...')):
        selfie = line.strip()

        # get latent code and add it
        try:
            latent = selfies_to_latent([selfie], vae=vae)
            selfies_latents.append(latent[0].cpu().numpy())

        # theres like two buggy ones
        except Exception as e:
            print(f"Failed SELFIES at index {i}: {selfie[:50]}...")
            print(f"Error: {e}")
            failed_selfies.append((i, selfie, str(e)))

         # save every 50k samples or so
        if (i + 1) % save_every == 0:
            np.save(f"latents_chunk.npy", np.array(selfies_latents, dtype=np.float32))
            print(f"Saving iter {i}")
            
np.save(f"latents_final.npy", np.array(selfies_latents, dtype=np.float32))