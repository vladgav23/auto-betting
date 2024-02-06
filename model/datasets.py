import json
import os
import random

import pytorch_lightning as pl
import torch.nn
from torch.utils.data import DataLoader

class PriceLadderDataset(torch.utils.data.Dataset):
    def __init__(self, directory=None, stats_file=None, market_ids=None):
        self.directory = directory
        self.stats_file = stats_file
        self.market_ids = market_ids
        self.file_list = []
        self.cumulative_lengths = []  # Stores the cumulative number of observations per file
        self.load_file_list()

    def __len__(self):
        if not self.cumulative_lengths:
            return 0
        # The total length is the last element in the cumulative_lengths array
        return self.cumulative_lengths[-1]

    def load_file_list(self):
        # Load market lengths from stats_file
        with open(self.stats_file, 'r') as file:
            market_dict = json.load(file)

        market_set = set(self.market_ids)

        # Filter and prepare file list based on market_ids
        if self.market_ids is not None:
            market_dict = [x for x in market_dict if x['market_id'] in market_set]

        market_lengths = [x['len'] for x in market_dict]
        self.file_list = [self.directory + x['market_id'] + '.pt' for x in market_dict]

        # Calculate cumulative lengths
        self.cumulative_lengths = [sum(market_lengths[:i+1]) for i in range(len(market_lengths))]

    def find_file_and_local_idx(self, idx):
        # Find which file contains the idx-th observation
        file_idx = next(i for i, total in enumerate(self.cumulative_lengths) if total >= idx)
        # Compute local index within the file
        local_idx = idx - (self.cumulative_lengths[file_idx - 1] if file_idx > 0 else 0)
        return self.file_list[file_idx], local_idx

    def __getitem__(self, idx):
        file_path, local_idx = self.find_file_and_local_idx(idx)
        data = torch.load(file_path)  # Assuming the data can be directly loaded with torch.load
        return data[local_idx - 1]

class PriceLadderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, stats_file: str, train_split: float, batch_size: int, dilute=1.0):
        super(PriceLadderDataModule, self).__init__()
        self.data_dir = data_dir
        self.stats_file = stats_file
        self.dilute = dilute
        self.train_split = train_split
        self.batch_size = batch_size

        self.market_ids = [x.rstrip('.pt') for x in os.listdir(self.data_dir)]

        if self.dilute < 1:
            self.market_ids = random.sample(self.market_ids, int(len(self.market_ids) * self.dilute))

    def setup(self, stage="fit"):
        if stage == "fit":
            train_size = int(self.train_split * len(self.market_ids))

            self.train_ids = self.market_ids[:train_size]
            self.val_ids = self.market_ids[train_size:]

            # Initialize data
            self.train_dataset = PriceLadderDataset(directory=self.data_dir,
                                                    stats_file=self.stats_file,
                                                    market_ids=self.train_ids)

            self.val_dataset = PriceLadderDataset(directory=self.data_dir,
                                                  stats_file=self.stats_file,
                                                  market_ids=self.val_ids)

        elif stage == "test" or stage == "predict":
            self.test_dataset = PriceLadderDataset(directory=self.data_dir,
                                                   stats_file=self.stats_file
                                                   )


    @staticmethod
    def collate_batch(batch, has_target=True):
        batch_dict = {
            'metadata': {
                'market_ids': [x['market_id'] for x in batch],
                'selection_ids': [x['selection_ids'] for x in batch],
                'seconds_to_start': [x['raw_sts'] for x in batch]
            },
            'pred_tensors': {
                'track': torch.tensor([x['track'] for x in batch]),
                'race_type': torch.tensor([x['race_type'] for x in batch]),
                'norm_sts': torch.stack([x['seconds_to_start'] for x in batch]),
                'lpts': torch.stack([x['lpt'] for x in batch]),
                'mover_flags': torch.stack([x['mover_flag'] for x in batch]),
                'traded': torch.stack([x['traded'] for x in batch]),
                'last_trades': torch.stack([x['last_trades'] for x in batch]),
                'back': torch.stack([x['back'] for x in batch]),
                'lay': torch.stack([x['lay'] for x in batch])
            }
        }

        if has_target:
            batch_dict['target'] = torch.stack([x['target'] for x in batch])

        return batch_dict

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12, collate_fn=self.collate_batch, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, collate_fn=self.collate_batch, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15, collate_fn=self.collate_batch, persistent_workers=True)


if __name__ == "__main__":
    dm = PriceLadderDataModule(data_dir="E:/Data/Extracted/Processed/TrainNew_Postprocessed/",
                               stats_file="E:/Data/Extracted/Processed/TrainNew_newstats.json",
                               train_split=0.8,
                               batch_size=64,
                               dilute=0.05)

    dm.setup()

    trn_dl = dm.train_dataloader()
    sample_data = next(iter(trn_dl))

    print(sample_data)