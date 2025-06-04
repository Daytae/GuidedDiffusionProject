from math import log
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import selfies as sf

DEFAULT_TOKENS = ['<start>', '<stop>', '[#Branch1]', '[#Branch2]', 
    '[#C-1]', '[#C]', '[#N+1]', '[#N]', '[#O+1]', '[=B]', '[=Branch1]', 
    '[=Branch2]', '[=C-1]', '[=C]', '[=N+1]', '[=N-1]', '[=NH1+1]', 
    '[=NH2+1]', '[=N]', '[=O+1]', '[=OH1+1]', '[=O]', '[=PH1]', '[=P]', 
    '[=Ring1]', '[=Ring2]', '[=S+1]', '[=SH1]', '[=S]', '[=Se+1]', '[=Se]', 
    '[=Si]', '[B-1]', '[BH0]', '[BH1-1]', '[BH2-1]', '[BH3-1]', '[B]', '[Br+2]', 
    '[Br-1]', '[Br]', '[Branch1]', '[Branch2]', '[C+1]', '[C-1]', '[CH1+1]', 
    '[CH1-1]', '[CH1]', '[CH2+1]', '[CH2]', '[C]', '[Cl+1]', '[Cl+2]', '[Cl+3]', 
    '[Cl-1]', '[Cl]', '[F+1]', '[F-1]', '[F]', '[H]', '[I+1]', '[I+2]', '[I+3]', 
    '[I]', '[N+1]', '[N-1]', '[NH0]', '[NH1+1]', '[NH1-1]', '[NH1]', '[NH2+1]', 
    '[NH3+1]', '[N]', '[O+1]', '[O-1]', '[OH0]', '[O]', '[P+1]', '[PH1]', '[PH2+1]', 
    '[P]', '[Ring1]', '[Ring2]', '[S+1]', '[S-1]', '[SH1]', '[S]', '[Se+1]', '[Se-1]', 
    '[SeH1]', '[SeH2]', '[Se]', '[Si-1]', '[SiH1-1]', '[SiH1]', '[SiH2]', '[Si]'
]

VOCAB = {token: i for i, token in enumerate(DEFAULT_TOKENS)}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class InfoTransformerVAE(pl.LightningModule):
    def __init__(self,
        bottleneck_size: int = 2,
        d_model: int = 128,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
    ):
        super().__init__()

        assert bottleneck_size != None, "Dont set bottleneck_size to None. Unbounded sequences dont support this yet"

        self.max_string_length = 256

        self.vocab = VOCAB
        self.vocab_size = len(self.vocab)

        self.bottleneck_size = bottleneck_size
        self.d_model         = d_model
        self.is_autoencoder  = is_autoencoder

        # TODO
        self.kl_factor    = kl_factor

        self.min_posterior_std = min_posterior_std
        encoder_embedding_dim  = 2 * d_model

        self.encoder_token_embedding   = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=encoder_embedding_dim)
        self.encoder_position_encoding = PositionalEncoding(encoder_embedding_dim, dropout=encoder_dropout, max_len=5_000)
        self.decoder_token_embedding   = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=d_model)
        self.decoder_position_encoding = PositionalEncoding(d_model, dropout=decoder_dropout, max_len=5_000)
        self.decoder_token_unembedding = nn.Parameter(torch.randn(d_model, self.vocab_size))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=encoder_embedding_dim,
            nhead=encoder_nhead,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            activation='relu',
            batch_first=True
        ), num_layers=encoder_num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=decoder_nhead,
            dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout,
            activation='relu',
            batch_first=True
        ), num_layers=decoder_num_layers)

    def encode_tokens(self, selfie):
        tokens = list(sf.split_selfies(selfie)) + ['<stop>']
        tokens = torch.tensor([self.vocab[token] for token in tokens])
        return tokens

    def sample_prior(self, n):
        if self.bottleneck_size is None:
            # TODO: idk what to do there lol, seq len doesn't exist anymore 
            sequence_length = self.sequence_length
        else:
            sequence_length = self.bottleneck_size

        return torch.randn(n, sequence_length, self.d_model).to(self.device)

    def sample_posterior(self, mu, sigma, n=None):
        if n is not None:
            mu = mu.unsqueeze(0).expand(n, -1, -1, -1)

        return mu + torch.randn_like(mu) * sigma

    def generate_pad_mask(self, tokens):
        """ Generate mask that tells encoder to ignore all but first stop token """
        mask = tokens == 1
        inds = mask.float().argmax(dim=-1) # Returns first index along axis when multiple present
        mask[torch.arange(0, tokens.shape[0]), inds] = False
        return mask 

    def encode(self, tokens):
        embed = self.encoder_token_embedding(tokens)
        embed = self.encoder_position_encoding(embed)

        pad_mask = self.generate_pad_mask(tokens)
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)
        mu = encoding[..., :self.d_model]
        sigma = F.softplus(encoding[..., self.d_model:]) + self.min_posterior_std

        if self.bottleneck_size is not None:
            mu = mu[:, :self.bottleneck_size, :]
            sigma = sigma[:, :self.bottleneck_size, :]

        return mu, sigma

    def decode(self, z, tokens):
        embed = self.decoder_token_embedding(tokens[:, :-1])
        embed = torch.cat([
            # Zero is the start token
            torch.zeros(embed.shape[0], 1, embed.shape[-1], device=self.device),
            embed
        ], dim=1)
        embed = self.decoder_position_encoding(embed)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embed.shape[1]).to(self.device)
        decoding = self.decoder(tgt=embed, memory=z, tgt_mask=tgt_mask)
        logits = decoding @ self.decoder_token_unembedding

        return logits
