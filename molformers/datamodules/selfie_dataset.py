import gzip
from pathlib import Path
from typing import List

import lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from molformers.datamodules.batch_sampler import SequenceLengthBatchSampler


class SELFIESDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, vocab: dict[str, int], batch_size: int, num_workers: int, len_sample: bool = False) -> None:
        super().__init__()

        self.data_root = data_root
        self.vocab = vocab

        self.batch_size = batch_size
        self.num_workers = num_workers

        if '[start]' not in self.vocab:
            raise ValueError("Vocab must contain '[start]' token")
        if '[stop]' not in self.vocab:
            raise ValueError("Vocab must contain '[stop]' token")
        if '[pad]' not in self.vocab:
            raise ValueError("Vocab must contain '[pad]' token")
        
        self.len_sample = len_sample

    def train_dataloader(self) -> DataLoader:
        ds = SELFIESDataset(self.data_root, 'train', self.vocab)
        if self.len_sample:
            return DataLoader(
                ds,
                # batch_size=self.batch_size,
                # shuffle=True,
                num_workers=self.num_workers,
                collate_fn=ds.get_collate_fn(),
                batch_sampler=SequenceLengthBatchSampler(ds.selfies, bucket_size=25, batch_size=self.batch_size)
            )
        else:
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=ds.get_collate_fn()
            )

    def val_dataloader(self) -> DataLoader:
        ds = SELFIESDataset(self.data_root, 'val', self.vocab)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ds.get_collate_fn()
        )

    def test_dataloader(self) -> DataLoader:
        ds = SELFIESDataset(self.data_root, 'test', self.vocab)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ds.get_collate_fn()
        )

class SELFIESDataset(Dataset):
    def __init__(self, data_root: str, split: str, vocab: dict[str, int]) -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.vocab = vocab

        path = Path(data_root) / f'{split}_selfie.gz'

        with gzip.open(path, 'rt') as f:
            self.selfies = [l.strip() for l in f.readlines()]

    def __len__(self) -> int:
        return len(self.selfies)

    def __getitem__(self, index: int) -> torch.Tensor:
        selfie = f"[start]{self.selfies[index]}[stop]"

        tokens = fast_split(selfie)
        tokens = torch.tensor([self.vocab[tok] for tok in tokens])

        return tokens

    def get_collate_fn(self):
        def collate(batch: List[torch.Tensor]) -> torch.Tensor:
            return pad_sequence(batch, batch_first=True, padding_value=self.vocab['[pad]'])
        return collate

# Faster than sf.split_selfies because it doesn't check for invalid selfies
def fast_split(selfie: str) -> list[str]:
    return [f"[{tok}" for tok in selfie.split("[") if tok]