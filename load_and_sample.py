import json
import lightning.pytorch as pl
import torch
from torch.nn.utils.rnn import pad_sequence
import selfies as sf
import time

from molformers.models.BaseTrainer import VAEModule
from molformers.models.BaseVAESwiGLURope import BaseVAE
from typing import List, Union

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")


def load_vae_selfies(path_to_vae_statedict, vocab_path="data/vocab.json"):
    """Load a VAE model for SELFIES representation"""
    
    # Load vocabulary
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    # Initialize model with same architecture as your training script
    model = BaseVAE(
        vocab,
        d_bnk=16,
        n_acc=8,
        
        d_dec=64,
        decoder_num_layers=3,
        decoder_dim_ff=256,
        
        d_enc=256,
        encoder_dim_ff=512,
        encoder_num_layers=3,
    )
    
    # Load state dict
    state_dict = torch.load(path_to_vae_statedict, map_location=device)["state_dict"]
    
    print(f"loading model from {path_to_vae_statedict}")
    
    # Remove 'model.' prefix if present (from nn.DataParallel)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("model."):
            new_key = key[6:]  # remove the 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    # Wrap in VAEModule
    vae = VAEModule(model).eval().to(device)
    
    return vae


def collate_selfies_fn(batch: List[torch.Tensor], vocab) -> torch.Tensor:
    """Collate function for SELFIES tokens"""
    return pad_sequence(batch, batch_first=True, padding_value=vocab['[pad]'])


def forward_selfies(selfies: Union[str, List[str]], vae):
    """Convert SELFIES string(s) to latent vector(s)
    Also returns the loss
    """
    # Ensure input is a list
    if isinstance(selfies, str):
        selfies = [selfies]
    
    # Convert SELFIES to tokens using VAEModule's method
    tokens = []
    for s in selfies:
        token_tensor = vae.selfie_to_tokens(s).to(device)
        tokens.append(token_tensor)
    
    # Collate tokens
    tokens_batch = collate_selfies_fn(tokens, vae.vocab)
    
    with torch.no_grad():
        out = vae.model(tokens_batch)
        z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
        loss = out["loss"]
    
    # Reshape to match expected output format
    return z.reshape(-1, vae.model.n_acc * vae.model.d_bnk), loss


def latent_to_selfies_batch(z: torch.Tensor, vae, argmax=True, max_len=256):
    """Convert batch of latent vectors to SELFIES strings"""
    z = z.to(device)
    
    with torch.no_grad():
        # Use VAEModule's sample method to generate tokens
        tokens = vae.sample(
            z.view(-1, vae.model.n_acc * vae.model.d_bnk),
            argmax=argmax,
            max_len=max_len
        )
    
    # Convert tokens to SELFIES strings
    selfies_list = []
    for token_seq in tokens:
        selfie = vae.tokens_to_selfie(token_seq, drop_after_stop=True)
        selfies_list.append(selfie)
    
    return selfies_list


def latent_to_selfies(z: torch.Tensor, vae, argmax=True, max_len=256):
    """Convert latent vector(s) to SELFIES string(s)
    Wrapper around latent_to_selfies_batch for consistency
    """
    z = z.to(device)
    results = latent_to_selfies_batch(z, vae, argmax=argmax, max_len=max_len)
    return results


# Helper function to convert between single SELFIES and latent (without loss calculation)
def selfies_to_latent_helper(selfie: str, vae):
    """Convert a single SELFIES string to latent vector without calculating loss"""
    with torch.no_grad():
        tokens = vae.selfie_to_tokens(selfie).unsqueeze(0).to(device)
        out = vae.model(tokens)
        z = out["mu_ign"] + out["sigma_ign"] * torch.randn_like(out["sigma_ign"])
    
    return z.reshape(-1, vae.model.n_acc * vae.model.d_bnk)


def selfies_to_latent(selfies: Union[str, List[str]], vae):
    """Convert SELFIES string(s) to latent vector(s) without loss calculation"""
    if isinstance(selfies, str):
        return selfies_to_latent_helper(selfies, vae)
    elif isinstance(selfies, list):
        return torch.cat([selfies_to_latent_helper(s, vae) for s in selfies], dim=0)


# Example usage function
def example_usage():
    """Example of how to use these functions"""
    # Load the VAE
    vae = load_vae_selfies("./saved_models/epoch=447-step=139328.ckpt")
    
    # Convert SELFIES to latent
    selfies = "[C][C][C][O]"  # Example SELFIES string
    z, loss = forward_selfies(selfies, vae)
    print(f"Latent shape: {z.shape}, Loss: {loss.item()}")
    
    # Convert latent back to SELFIES
    reconstructed_selfies = latent_to_selfies(z, vae)
    print(f"Reconstructed: {reconstructed_selfies}")
    
    # Batch processing
    selfies_batch = [
        "[C][C][C][C][=Branch1][C][=O][N][N][C][=Branch1][C][=O][N][C][=C][C][=C][C][=C][Ring1][=Branch1]",
        "[C][C][=Branch1][C][=O][N][C][C][C][C][Branch1][C][C][C][Branch2][Ring2][#Branch2][C][C][C][Branch1][C][C][C][Ring1][Branch2][C][=Branch1][C][=O][C][=C][C][C][Branch1][C][C][C][Branch1][C][C][C][C][C][Ring1][Branch2][Branch1][C][C][C][C][C][Ring1][=N][Ring2][Ring1][Ring1][C][C][Ring2][Ring1][=N][Branch1][C][C][C][=Branch1][C][=O][O]",
        "[C][C][=Branch1][C][=O][N][C][Branch1][C][C][C][C][=C][C][=C][Branch2][Ring1][Branch2][C][#C][C][=C][C][=N][C][Branch1][N][N][C][C][C][C][Branch1][C][F][C][Ring1][#Branch1][=N][Ring1][=N][C][=C][Ring2][Ring1][Branch1]",
        "[C][C][=C][C][=C][C][Branch2][Ring1][P][C][C][N][C][=Branch1][C][=O][C][C][C][C][=Branch1][C][=O][N][Branch1][=N][C][C][=C][C][=C][Branch1][C][Cl][C][=C][Ring1][#Branch1][C][Ring1][#C][=N][Ring2][Ring1][#Branch2]",
        "[C][C][C][=C][N][Branch1][=Branch1][N][Branch1][C][C][C][C][=C][Ring1][=Branch2][C][=Branch1][C][=O][C][=C][N][=C][C][=C][Ring1][=Branch1][C][Ring1][O][=O]",
    ]
    z_batch, loss_batch = forward_selfies(selfies_batch, vae)
    print(f"Batch latent shape: {z_batch.shape}")
    
    reconstructed_batch = latent_to_selfies(z_batch, vae)
    print(f"Reconstructed batch: {reconstructed_batch}")

    # calculate reconstruction accuracy
    num_correct = sum(
        1 for original, reconstructed in zip(selfies_batch, reconstructed_batch)
        if original == reconstructed
    )
    accuracy = num_correct / len(selfies_batch)
    print(f"Reconstruction accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    example_usage()