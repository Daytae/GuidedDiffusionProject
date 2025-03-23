import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
import pandas as pd
import itertools
from tqdm import tqdm

# Organizes train/test/val into one lightning data module
# So, it deals in units of DataSets (train, val, test)
class DataModuleKmers(pl.LightningDataModule):
    def __init__(self, batch_size, k, version=1, load_data=True): 
        # k is the k-mer of the data (i.e. nucleotides are 3-mers)
        super().__init__() 
        self.batch_size = batch_size 
        if version == 1: DatasetClass = DatasetKmers # defined below
        else: raise RuntimeError('Invalid data version') 
        self.train  = DatasetClass(dataset='train', k=k, load_data=load_data) 
        self.val    = DatasetClass(dataset='val', k=k, vocab=self.train.vocab, vocab2idx=self.train.vocab2idx, load_data=load_data )
        self.test   = DatasetClass(dataset='test', k=k, vocab=self.train.vocab, vocab2idx=self.train.vocab2idx, load_data=load_data )
    
    # the DataLoader serves the trainer batches of data during training:
    # Basically implements batching, shuffling, parallelizing, etc.
    def train_dataloader(self):
        return DataLoader(
            self.train, # takes in a dataset and will turn it into batches
            batch_size=self.batch_size, # turns into batch_size batches
            pin_memory=True, # pinned memory --> faster CPU to GPU
            shuffle=True, # avoids order of data effecting training
            collate_fn=collate_fn, # defines how we turn sequences into actual batches, in our case this means padding
            num_workers=10
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size, 
            pin_memory=True,
            shuffle=False, # dont need to shuffle for val obviously
            collate_fn=collate_fn, 
            num_workers=10
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test, 
            batch_size=self.batch_size,
            pin_memory=True, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=10
        )


# Handles lower level details of reading from the CSV data:
# 1. How to read from CSV
# 2. How to split into kmer tokens
# 3. How to map a token into an integer
# deals in units of actual individual tokens

class DatasetKmers(Dataset): # asssuming train data 
    def __init__(self, dataset='train', data_path=None, k=3, vocab=None, vocab2idx=None, load_data=False):
        if data_path is None: 
            path_to_data = 'data/uniref-cropped.csv' 
        df = pd.read_csv(path_to_data)
        self.dataset = dataset
        train_seqs = df['sequence'].values  # 4_500_000  sequences 
        # SEQUENCE LENGTHS ANALYSIS:  Max = 299, Min = 100, Mean = 183.03 

        # breaks all 4.5 million seqs into lists of chars in regular_data
        self.k = k
        regular_data = [] 
        for seq in train_seqs: 
            regular_data.append([token for token in seq]) # list of tokens
        
        # first get initial vocab set 
        # Gets all 3-wide tokens in the whole dataset (i.e. AQF, WLM, W--, etc.)
        if vocab is None:
            # really odd way to get the 21 (no idea what the 21st is) amino acids, I guess '.' is the 21st? and thats why we discard it
            self.regular_vocab = set((token for seq in regular_data for token in seq))  # 21 tokens  
            self.regular_vocab.discard(".") 

            # add padding token
            if '-' not in self.regular_vocab: 
                self.regular_vocab.add('-')  # '-' used as pad token when length of sequence is not a multiple of k

            # make it into list of all possible 3-mers
            # add start/stop token
            self.vocab = ["".join(kmer) for kmer in itertools.product(self.regular_vocab, repeat=k)] # 21**k tokens 
            self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))] # 21**k + 2 tokens 
        else: 
            self.vocab = vocab 

        # creats map from the vocab (i.e. AQL or some triplet of amino acids) to the number that that represents
        if vocab2idx is None:
            self.vocab2idx = { v:i for i, v in enumerate(self.vocab) }
        else:
            self.vocab2idx = vocab2idx
        
        # creates the actual data by converting the sequences to (possibly padded) triplets
        self.data = []
        if load_data:
            # tqdm shows progress bar
            for seq in tqdm(regular_data):
                token_num = 0
                kmer_tokens = []
                # take tokens in groups of 3, and pad ones that don't work
                while token_num < len(seq):
                    kmer = seq[token_num:token_num+k]
                    while len(kmer) < k:
                        kmer += '-' # padd so we always have length k 
                    kmer_tokens.append("".join(kmer)) 
                    token_num += k 
                self.data.append(kmer_tokens) 
        
        # take the first 90% if train, the next 5% for val and the next 5% for test
        num_data = len(self.data) 
        ten_percent = int(num_data/10) 
        five_percent = int(num_data/20) 
        if self.dataset == 'train': # 90 %
            self.data = self.data[0:-ten_percent] 
        elif self.dataset == 'val': # 5 %
            self.data = self.data[-ten_percent:-five_percent] 
        elif self.dataset == 'test': # 5 %
            self.data = self.data[-five_percent:] 
        else: 
            raise RuntimeError("dataset must be one of train, val, test")

    # converts from AGYTVRSGCMGAQ --> [AGY, TVR, SGC, MGA, Q--] for example
    def tokenize_sequence(self, list_of_sequences):   
        ''' 
        Input: list of sequences in standard form (ie 'AGYTVRSGCMGA...')
        Output: List of tokenized sequences where each tokenied sequence is a list of kmers
        '''
        tokenized_sequences = []
        for seq in list_of_sequences:
            token_num = 0
            kmer_tokens = []
            while token_num < len(seq):
                kmer = seq[token_num:token_num + self.k]
                while len(kmer) < self.k:
                    kmer += '-' # padd so we always have length k  
                if type(kmer) == list: kmer = "".join(kmer)
                kmer_tokens.append(kmer) 
                token_num += self.k 
            tokenized_sequences.append(kmer_tokens) 
        return tokenized_sequences 

    # takes [AGY, TVR, SGC, MGA, Q--], adds stop token [AGY, TVR, SGC, MGA, Q--, <stop>],
    # then converts to index: [43, 594, 245, 303, 1024, 1]
    
    def encode(self, tokenized_sequence):
        return torch.tensor([self.vocab2idx[s] for s in [*tokenized_sequence, '<stop>']])

    def decode(self, tokens):
        '''
        Inpput: Iterable of tokens specifying each kmer in a given protien (ie [3085, 8271, 2701, 2686, ...] )
        Output: decoded protien string (ie GYTVRSGCMGA...)
        '''
        dec = [self.vocab[t] for t in tokens]
        # Chop out start token and everything past (and including) first stop token
        stop = dec.index("<stop>") if "<stop>" in dec else None # want first stop token
        protien = dec[0:stop] # cut off stop tokens
        while "<start>" in protien: # start at last start token (I've seen one case where it started w/ 2 start tokens)
            start = (1+dec.index("<start>")) 
            protien = protien[start:]
        protien = "".join(protien) # combine into single string 
        return protien

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx]) 

    # @property allows to access dataset.vocab_size, but vocab_size is actually a function, pretty neat
    @property
    def vocab_size(self):
        return len(self.vocab)

# get longest molecule in batch and pad the data with stop tokens so all are same size
def collate_fn(data):
    # Length of longest molecule in batch 
    max_size = max([x.shape[-1] for x in data])
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )

