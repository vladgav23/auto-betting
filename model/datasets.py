import json
import os
import random

import pytorch_lightning as pl
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class PriceLadderDataset(torch.utils.data.Dataset):
    def __init__(self, directory=None, stats_file=None, mtl_factor=None, num_cached_markets_factor=None, max_traded_length=None, back_lay_length=10, last_trades_len=100, max_horses=4, track_to_int=None,rt_to_int=None,market_ids=None, live=False):
        self.directory = directory
        self.market_ids = market_ids
        self.num_cached_markets_factor = num_cached_markets_factor
        self.mtl_factor = mtl_factor
        self.stats_file = stats_file
        self.last_trades_len = last_trades_len
        self.back_lay_length = back_lay_length
        self.max_horses = max_horses
        self.max_traded_length = max_traded_length

        self.track_to_int = track_to_int
        self.rt_to_int = rt_to_int

        self.file_list = []

        self.max_sts = 0
        self.min_sts = 999999
        self.traded_ladder_length = 0
        self.total_length = 0
        self.live = live

        if not self.live:
            self.load_file_list()

    def __len__(self):
        return self.total_length

    def shuffle_file_list(self):
        local_random = random.Random()
        local_random.shuffle(self.file_list)
        return

    def load_market_data_json(self, market_ids_in_folder=None):
        self.market_data = []
        with open(self.stats_file, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    obj = json.loads(line)
                    race_name_split = obj['race_name'].split()
                    obj['race_type'] = race_name_split[2] if len(race_name_split) > 2 else "Unknown"
                    if market_ids_in_folder is not None:
                        if obj['market_id'] in market_ids_in_folder:
                            self.market_data.append(obj)
                    else:
                        self.market_data.append(obj)
                except json.JSONDecodeError:
                    continue

        self.max_sts = max([x['max_seconds_to_start'] for x in self.market_data])
        self.min_sts = min([x['min_seconds_to_start'] for x in self.market_data])

        if self.max_traded_length is None:
            self.max_traded_length = int(max([x['max_traded_length'] for x in self.market_data]) * self.mtl_factor)

        track_list = set([x['track'] for x in self.market_data])
        race_type_list = set([x['race_type'] for x in self.market_data])

        if self.track_to_int is None:
            self.track_to_int = {track: idx for idx, track in enumerate(track_list)}
        else:
            if self.market_ids is not None:
                market_id_set = self.market_ids
                tracks_in_set = list(set([x['track'] for x in self.market_data if x['market_id'] in market_id_set]))
            else:
                tracks_in_set = list(set([x['track'] for x in self.market_data]))
            new_tracks = set(x for x in tracks_in_set if x not in self.track_to_int)
            if new_tracks:
                print("New track found in val data - removing from file list: " + str(new_tracks))
                self.market_data = [x for x in self.market_data if x['track'] not in new_tracks]

        if self.rt_to_int is None:
            self.rt_to_int = {rt: idx for idx, rt in enumerate(race_type_list)}
        else:
            if self.market_ids is not None:
                market_id_set = self.market_ids
                rt_in_set = list(set([x['race_type'] for x in self.market_data if x['market_id'] in market_id_set]))
            else:
                rt_in_set = list(set([x['race_type'] for x in self.market_data]))
            new_rt = set(x for x in rt_in_set if x not in self.rt_to_int)
            if new_rt:
                print("New race type found in val data - removing from file list: " + str(new_rt))
                self.market_data = [x for x in self.market_data if x['race_type'] not in new_rt]

    def load_file_list(self):
        market_ids_in_folder = set(x.rstrip('.pt') for x in os.listdir(self.directory))

        self.load_market_data_json(market_ids_in_folder)

        # Find market IDs that aren't in the mappings set up in train
        market_ids_in_stats = [x['market_id'] for x in self.market_data]

        if self.market_ids is not None:
            self.file_list = list(set(market_ids_in_folder).intersection(self.market_ids).intersection(market_ids_in_stats))
            self.file_list = [self.directory + x + '.pt' for x in self.file_list]
        else:
            self.file_list = list(set(market_ids_in_folder).intersection(market_ids_in_stats))
            self.file_list = [self.directory + x + '.pt' for x in self.file_list]

        self.shuffle_file_list()

        for market in self.file_list:
            pt = torch.load(market)
            self.total_length += len(pt)

        self.num_cached_markets = len(self.file_list) // self.num_cached_markets_factor

        self.processed_markets = 0
        self.cached_markets = []

    def load_cache(self):
        # Load the rest last batch of caching
        market_lists = []
        if int(self.processed_markets / self.num_cached_markets) == (self.num_cached_markets_factor - 1):
            list_to_load = self.file_list[self.processed_markets:]
        else:
            list_to_load = self.file_list[self.processed_markets:(self.processed_markets + self.num_cached_markets)]

        for file in list_to_load:
            market_lists.append(torch.load(file))
            self.processed_markets += 1

        self.cached_markets = [x for y in market_lists for x in y]

    def process_dict(self, dict, extra_info):
        # Normalize sts
        input_sts = torch.tensor((dict['seconds_to_start'] - self.min_sts) / (self.max_sts - self.min_sts))
        price_tensor_list = dict['price_tensor_list']

        # Pad and stack back ladders
        raw_back_ladders = [sel['back_ladder'] for sel in price_tensor_list]

        padded_back_ladders = []
        for ld in raw_back_ladders:
            seq_length = ld.shape[0]
            padding_length = self.back_lay_length - seq_length

            # Check if padding is needed
            if padding_length > 0:
                # Padding size (left, right, top, bottom)
                pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                padded_seq = F.pad(ld, pad_size, "constant", 0)  # Padding with zeros
            else:
                padded_seq = ld

            padded_back_ladders.append(padded_seq)

        input_back_ladders = torch.stack(padded_back_ladders)

        # Pad and stack lay ladders
        raw_lay_ladders = [sel['lay_ladder'] for sel in price_tensor_list]

        padded_lay_ladders = []
        for ld in raw_lay_ladders:
            seq_length = ld.shape[0]
            padding_length = self.back_lay_length - seq_length

            # Check if padding is needed
            if padding_length > 0:
                # Padding size (left, right, top, bottom)
                pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                padded_seq = F.pad(ld, pad_size, "constant", 0)  # Padding with zeros
            else:
                padded_seq = ld

            padded_lay_ladders.append(padded_seq)

        input_lay_ladders = torch.stack(padded_lay_ladders)

        # Pad and stack traded ladders
        raw_traded_ladders = [sel['traded_ladder'] for sel in price_tensor_list]

        traded_ladders = []
        for ld in raw_traded_ladders:
            if ld.nelement() == 0:
                padded_seq = torch.zeros([self.max_traded_length, 2])
            else:
                seq_length = ld.shape[0]
                padding_length = self.max_traded_length - seq_length

                # Check if padding is needed
                if padding_length > 0:
                    pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                    padded_seq = F.pad(ld, pad_size, "constant", 0)  # Padding with zeros
                else:
                    padded_seq = ld

            traded_ladders.append(padded_seq)

        input_traded_ladders = torch.stack(traded_ladders)

        # Pad and stack trade sequences
        raw_last_trade_seq = [sel['last_trades'] for sel in price_tensor_list]

        last_trade_sq = []
        for lt_seq in raw_last_trade_seq:
            if lt_seq.nelement() == 0:
                # Handle empty tensor
                padded_seq = torch.zeros((self.last_trades_len, 3))
            else:
                seq_length = lt_seq.shape[0]
                padding_length = self.last_trades_len - seq_length

                # Check if padding is needed
                if padding_length > 0:
                    # Padding size (left, right, top, bottom)
                    pad_size = (0, 0, 0, padding_length)  # Assuming you want to pad at the bottom
                    padded_seq = F.pad(lt_seq, pad_size, "constant", 0)  # Padding with zeros
                else:
                    padded_seq = lt_seq

            last_trade_sq.append(padded_seq[-100:])

        input_last_trade_seq = torch.stack(last_trade_sq)

        if extra_info is None:
            track_name = next(x['track'] for x in self.market_data if x['market_id'] == dict['market_id'])
            race_type = next(x['race_type'] for x in self.market_data if x['market_id'] == dict['market_id'])
        else:
            track_name = extra_info['track']
            race_type = extra_info['race_type']

        # Gather metadata
        try:
            tensor_dict = {
                'market_id': dict['market_id'],
                'track': self.track_to_int[track_name],
                'race_type': self.rt_to_int[race_type],
                'selection_ids': dict['selection_ids'],
                'lpt': torch.tensor([p['lpt'] if p['lpt'] is not None else 0 for p in dict['price_tensor_list']]),
                'mover_flag': torch.tensor([p['mover_flag'] for p in dict['price_tensor_list']]),
                'raw_sts': dict['seconds_to_start'],
                'seconds_to_start': input_sts,
                'back': input_back_ladders,
                'lay': input_lay_ladders,
                'traded': input_traded_ladders,
                'last_trades': input_last_trade_seq
            }
        except Exception as e:
            print("market_id " + dict['market_id'])
            raise(e)

        if 'target' in dict:
            tensor_dict['target'] = torch.stack(dict['target'])

        self.transform_tensor_dict(tensor_dict)

        return tensor_dict

    def transform_tensor_dict(self, tensor_dict):
        back = tensor_dict['back']
        lay = tensor_dict['lay']
        traded = tensor_dict['traded']
        lpts = tensor_dict['lpt']
        last_trades = tensor_dict['last_trades']

        if 'target' in tensor_dict:
            target_tensor = tensor_dict['target']

            # Transform prices into ratio to LPT
            target_tensor[:, range(0, 3)] = target_tensor[:,range(0,3)] / lpts.unsqueeze(1)

            # Log transform total volume
            target_tensor[:, 3] = torch.log(target_tensor[:, 3])

            # Concatenate target
            tensor_dict['target'] = target_tensor.view(-1)

        # Standardise prices by div to LPT
        back[:, :, 0] = torch.where(back[:, :, 0] != 0, back[:, :, 0] / lpts.unsqueeze(1), 0)
        lay[:, :, 0] = torch.where(lay[:, :, 0] != 0, lay[:, :, 0] / lpts.unsqueeze(1), 0)
        traded[:, :, 0] = torch.where(traded[:, :, 0] != 0, traded[:, :, 0] / lpts.unsqueeze(1), 0)

        # Standardise sizes by log transforming
        back[:, :, 1] = torch.where(back[:, :, 1] != 0, (torch.log(back[:, :, 1])), 0)
        lay[:, :, 1] = torch.where(lay[:, :,  1] != 0, (torch.log(lay[:, :, 1])), 0)
        traded[:, :, 1] = torch.where(traded[:, :, 1] != 0, (torch.log(traded[:, :, 1])), 0)

        # Transform last trades
        last_trades[:, :, 0] = last_trades[:, :, 0] / lpts.unsqueeze(1)
        last_trades[:, :, 1] = torch.where(last_trades[:, :, 1] != 0, torch.log(last_trades[:, :, 1]), 0)
        last_trades[:, :, 2] = last_trades[:, :, 2] / torch.max(last_trades[:, :, 2])

    def __getitem__(self, idx):
        if not self.cached_markets:
            self.load_cache()

        item = random.choice(self.cached_markets)

        # Find the index of the item
        item_index = next(i for i, x in enumerate(self.cached_markets) if x['seconds_to_start'] == item['seconds_to_start'] and x['market_id'] == item['market_id'])

        # Remove the item using the index
        del self.cached_markets[item_index]

        # Restart cache loading after each epoch
        if self.processed_markets == len(self.file_list):
            self.processed_markets = 0

        return self.process_dict(item)

class PriceLadderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, stats_file: str, mtl_factor: float, train_split: float, num_cached_markets_factor: int, batch_size: int, max_traded_length=None, track_to_int=None,rt_to_int=None, dilute=1.0):
        super(PriceLadderDataModule, self).__init__()
        self.data_dir = data_dir
        self.stats_file = stats_file
        self.dilute = dilute
        self.mtl_factor = mtl_factor
        self.train_split = train_split
        self.num_cached_markets_factor = num_cached_markets_factor
        self.batch_size = batch_size
        self.max_traded_length = max_traded_length
        self.track_to_int = track_to_int
        self.rt_to_int = rt_to_int
        self.train_dataset = None

        self.market_ids = [x.rstrip('.pt') for x in os.listdir(self.data_dir)]

        if self.dilute < 1:
            self.market_ids = random.sample(self.market_ids, int(len(self.market_ids) * self.dilute))

    def setup(self, stage="fit"):
        if stage == "fit":
            if self.train_dataset is None:
                train_size = int(self.train_split * len(self.market_ids))

                self.train_ids = self.market_ids[:train_size]
                self.val_ids = self.market_ids[train_size:]

                # Initialize data
                self.train_dataset = PriceLadderDataset(directory=self.data_dir,stats_file=self.stats_file,mtl_factor=self.mtl_factor, num_cached_markets_factor=self.num_cached_markets_factor,max_traded_length=self.max_traded_length, market_ids=self.train_ids)

                self.track_to_int = self.train_dataset.track_to_int
                self.rt_to_int = self.train_dataset.rt_to_int

                self.val_dataset = PriceLadderDataset(directory=self.data_dir,
                                                      stats_file=self.stats_file,
                                                      mtl_factor=self.mtl_factor,
                                                      num_cached_markets_factor=self.num_cached_markets_factor,
                                                      max_traded_length=self.max_traded_length,
                                                      track_to_int=self.track_to_int,
                                                      rt_to_int=self.rt_to_int,
                                                      market_ids=self.val_ids)

                if self.max_traded_length is None:
                    self.max_traded_length = self.train_dataset.max_traded_length

        elif stage == "test" or stage == "predict":
            if self.track_to_int is None or self.rt_to_int is None:
                print("WARNING: No mapping provided for tracks/races during predict - pass through the one from train instead.")

            self.test_dataset = PriceLadderDataset(directory=self.data_dir,
                                                   stats_file=self.stats_file,
                                                   mtl_factor=self.mtl_factor,
                                                   num_cached_markets_factor=self.num_cached_markets_factor,
                                                   max_traded_length=self.max_traded_length,
                                                   track_to_int=self.track_to_int,
                                                   rt_to_int=self.rt_to_int
                                                   )

            if self.max_traded_length is None:
                self.max_traded_length = self.test_dataset.max_traded_length

    @staticmethod
    def collate_batch(batch, has_target):
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_batch, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_batch, persistent_workers=False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_batch, persistent_workers=False)
    #
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_batch, persistent_workers=False)


if __name__ == "__main__":
    dm = PriceLadderDataModule("E:/Data/Extracted/Processed/Train/","E:/Data/Extracted/Processed/Train.json",
                               train_split=0.8,
                               num_cached_markets_factor=25,
                               mtl_factor=1.1,
                               batch_size=64)

    dm.setup()

    trn_dl = dm.train_dataloader()

    sample_data = next(iter(trn_dl))

    print(' ')