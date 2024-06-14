import lightning as L
import datasets
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd


class RedditScenariosCorpus(Dataset):
    def __init__(self, input_ids, attention_mask, outputs):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.outputs = outputs

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "outputs": torch.tensor(self.outputs[idx], dtype=torch.float32)
        }

    def __len__(self):
        return len(self.outputs)


class RedditScenariosModule(L.LightningDataModule):
    def __init__(self, config=None, tokenizer='roberta-base'):
        super().__init__()
        # Default configuration
        default_config = {
            'batch_size': 32,
            'num_workers': 4,
            'val_size': 0.2,
            'output': 'Feelings'
        }
        # Update default config with any values provided by the user
        if config is not None:
            default_config.update(config)
        self.cfg = default_config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def setup(self, stage: str):
        def tokenize_function(examples):
            return self.tokenizer(
                examples['selftext'], truncation=True, padding="max_length"
            )

        if stage == "fit":
            df = pd.read_feather(self.cfg['path']).iloc[:5000].copy()
            train, val = train_test_split(
                df,
                test_size=self.cfg['val_size'],
                stratify=df[self.cfg['output']])
            train = datasets.Dataset.from_pandas(train)
            val = datasets.Dataset.from_pandas(val)
            train_tokenized = train.map(tokenize_function, batched=True)
            val_tokenized = val.map(tokenize_function, batched=True)

            self.trn_dset = RedditScenariosCorpus(
                train_tokenized['input_ids'], 
                train_tokenized['attention_mask'], 
                train_tokenized[self.cfg['output']]
            )
            self.val_dset = RedditScenariosCorpus(
                val_tokenized['input_ids'], 
                val_tokenized['attention_mask'], 
                val_tokenized[self.cfg['output']]
            )
 
    def train_dataloader(self):
        return DataLoader(
            self.trn_dset,
            num_workers=self.cfg['num_workers'],
            batch_size=self.cfg['batch_size']
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            num_workers=self.cfg['num_workers'],
            batch_size=self.cfg['batch_size']
        )

