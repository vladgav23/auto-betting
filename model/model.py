import torch
import json
import os
torch.set_float32_matmul_precision('high')
import pytorch_lightning as pl
import torch.nn.functional as F
# import torch.nn as nn
# import math
from torcheval.metrics.functional import r2_score
from .datasets import PriceLadderDataModule
from torch.utils.data import DataLoader

# class PositionalEncoding(torch.nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return x

class PriceLadderModel(pl.LightningModule):
    def __init__(self, max_traded_length, track_to_int, rt_to_int, learning_rate: float = 1e-5, weight_decay: float = 1e-6, dropout_prob: float = 0.25):
        super(PriceLadderModel, self).__init__()

        self.dropout_prob = dropout_prob
        self.track_to_int = track_to_int
        self.rt_to_int = rt_to_int

        # Dataset properties
        self.max_traded_length = max_traded_length

        # Define parameters for the data loaders
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Project last traded price
        self.proj_lpts = torch.nn.Linear(1, 32)

        # Project mover flag
        self.proj_mover = torch.nn.Linear(1, 32)

        # Project seconds to start
        self.proj_sts = torch.nn.Linear(1, 128)

        # Embeddings
        self.embed_track = torch.nn.Embedding(len(track_to_int), 512)
        self.embed_racetype = torch.nn.Embedding(len(rt_to_int), 256)

        # Back and lay ladder projection
        self.proj_back = torch.nn.Linear(2, 128)
        self.proj_lay = torch.nn.Linear(2, 128)

        self.tf_back_lay = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
        self.proj_back_lay = torch.nn.Linear(10 * 128 * 2, 512)

        # Traded ladder projection
        self.proj_traded_price = torch.nn.Linear(2, 128)
        self.tf_traded = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True)
        self.proj_traded_ladder = torch.nn.Linear(self.max_traded_length * 128, 1024)

        # Last trades projection and transformer
        self.proj_single_last_trade = torch.nn.Linear(3, 256)
        self.tf_single_last_trade = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.proj_all_last_trades = torch.nn.Linear(100 * 256, 1024)

        # Project runner (Back/Lay + Traded + Last 100 trades + Mover + LPT
        self.proj_runner = torch.nn.Linear(512 + 1024 + 1024 + 32 + 32, 2048)

        # Project market
        self.proj_market = torch.nn.Linear((2048 * 6) + 128 + 256 + 512, 4096)

        self.reg = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(4096, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 3 * 6)
        )

    def forward(self, pred_tensor_dict):
        batch_size = pred_tensor_dict['lpts'].shape[0]

        # Project single features
        projected_seconds = self.proj_sts(pred_tensor_dict['norm_sts'].unsqueeze(1))
        projected_mover_flag = self.proj_mover(
            pred_tensor_dict['mover_flags'].to(torch.float).view(-1, 1)
        ).view(batch_size, 6, -1)
        projected_lpts = self.proj_lpts(
            pred_tensor_dict['lpts'].view(-1, 1)
        ).view(batch_size, 6, -1)
        embedded_track = self.embed_track(pred_tensor_dict['track'])
        embedded_rt = self.embed_racetype(pred_tensor_dict['race_type'])

        # Process back and lay ladders
        back_mask = (pred_tensor_dict['back'] == torch.zeros((1,2),device=self.device)).all(3)
        projected_back = self.proj_back(pred_tensor_dict['back'])

        if torch.any(back_mask):
            projected_back[back_mask] = torch.zeros(projected_back[back_mask].shape, device=self.device)

        lay_mask = (pred_tensor_dict['lay'] == torch.zeros((1, 2), device=self.device)).all(3)
        projected_lay = self.proj_lay(pred_tensor_dict['lay'])

        if torch.any(lay_mask):
            projected_lay[lay_mask] = torch.zeros(projected_lay[lay_mask].shape, device=self.device)

        cat_back_lay = torch.cat((projected_back, projected_lay), dim=2).view(batch_size, -1, 128)

        tf_back_lay_mask = (cat_back_lay == torch.zeros(128, device=self.device)).all(2)
        tf_back_lay = self.tf_back_lay(cat_back_lay, src_key_padding_mask=tf_back_lay_mask)

        tf_back_lay = tf_back_lay.view(batch_size, 6, -1)
        projected_back_lay = F.gelu(self.proj_back_lay(tf_back_lay))

        # Process traded ladder
        traded_mask = (pred_tensor_dict['traded'] == torch.zeros((1, 2), device=self.device)).all(3)

        projected_traded = self.proj_traded_price(pred_tensor_dict['traded'])

        if torch.any(traded_mask):
            projected_traded[traded_mask] = torch.zeros(projected_traded[traded_mask].shape, device=self.device)

        projected_traded = projected_traded.view(batch_size, -1, 128)

        tf_traded_mask = (projected_traded == torch.zeros(128, device=self.device)).all(2)

        tf_traded = self.tf_traded(projected_traded, src_key_padding_mask=tf_traded_mask)

        tf_traded = tf_traded.view(batch_size, 6, -1)
        projected_traded_ladder = F.gelu(self.proj_traded_ladder(tf_traded))

        # Process last trades
        trade_mask = (pred_tensor_dict['last_trades'] == torch.zeros(3, device=self.device)).all(3)

        projected_single_trade = self.proj_single_last_trade(pred_tensor_dict['last_trades'])

        if torch.any(trade_mask):
            projected_single_trade[trade_mask] = torch.zeros(projected_single_trade[trade_mask].shape, device=self.device)

        projected_single_trade = projected_single_trade.view(batch_size, -1, 256)

        tf_trade_mask = (projected_single_trade == torch.zeros(256, device=self.device)).all(2)
        tf_trade = self.tf_single_last_trade(projected_single_trade, src_key_padding_mask=tf_trade_mask)

        tf_trade = tf_trade.view(batch_size, 6, -1)
        projected_all_trades = F.gelu(self.proj_all_last_trades(tf_trade))

        # Project runner
        projected_runner = F.dropout(self.proj_runner(
            torch.cat(
                (projected_back_lay, projected_traded_ladder, projected_all_trades, projected_mover_flag, projected_lpts), dim=2
            )
        ),p=self.dropout_prob,training=self.training)

        projected_market = self.proj_market(torch.cat((projected_runner.view(batch_size, -1), projected_seconds, embedded_track, embedded_rt), dim=1))

        raw_output = self.reg(projected_market)

        return raw_output

    def custom_loss(self, outputs, targets):
        target_volumes = (torch.exp(targets.view(-1, 6, 4)[:,:,3]) / torch.exp(targets.view(-1, 6, 4)[:,:,3]).sum(dim=1).unsqueeze(1)).unsqueeze(2)

        losses = F.l1_loss(outputs.view(-1, 6, 3), targets.view(-1, 6, 4)[:, :, :3], reduction='none')

        vols_weighted_losses = torch.sum(target_volumes * losses,dim=[1,2])

        if torch.any(torch.isnan(vols_weighted_losses)):
            print('Loss is NaN')

        return torch.mean(vols_weighted_losses)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['pred_tensors'])
        loss = self.custom_loss(outputs, batch['target'])
        self.log('train_loss', loss)
        self.log('batch_r2', r2_score(outputs.reshape(-1), batch['target'].reshape(-1, 6, 4)[:, :, :3].reshape(-1)))
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['pred_tensors'])
        loss = self.custom_loss(outputs, batch['target'])
        self.log('val_loss', loss)

        # if batch_idx == 0:
        #     if not self.markets_to_track:
        #         self.markets_to_track = batch['metadata']['market_ids'][:1]
        #
        #     prices_outputs = outputs.view(-1, 4, 3) * batch['pred_tensors']['lpts'].unsqueeze(2)
        #     prices_targets = batch['target'].view(-1, 4, 4)[:, :, :3] * batch['pred_tensors']['lpts'].unsqueeze(2)
        #
        #     for market_id in self.markets_to_track:
        #         obs_idx = batch['metadata']['market_ids'].index(market_id)
        #
        #         for sel in range(0,len(batch['metadata']['selection_ids'][obs_idx])):
        #             log_name = batch['metadata']['market_ids'][obs_idx] + "_" + str(batch['metadata']['selection_ids'][obs_idx][sel]) + "_" + str(
        #                 batch['metadata']['seconds_to_start'][obs_idx])
        #             pred_max = prices_outputs[obs_idx, sel, 0].item()
        #             pred_min = prices_outputs[obs_idx, sel, 1].item()
        #             pred_wap = prices_outputs[obs_idx, sel, 2].item()
        #             tgt_max = str(round(prices_targets[obs_idx, sel, 0].item(),2))
        #             tgt_min = str(round(prices_targets[obs_idx, sel, 1].item(),2))
        #             tgt_wap = str(round(prices_targets[obs_idx, sel, 2].item(),2))
        #
        #             self.log(log_name + '_max_' + tgt_max, pred_max)
        #             self.log(log_name + '_min_' + tgt_min, pred_min)
        #             self.log(log_name + '_wap_' + tgt_wap, pred_wap)
        return loss

    def predict_step(self, batch, batch_idx):
        batch['predictions'] = self(batch['pred_tensors'])

        return batch

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),
            'monitor': 'val_loss',
        }
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'}

# Model testing
if __name__ == '__main__':
    data_dir = "E:/Data/Extracted/Processed/TrainNew_Postprocessed/"
    dm = PriceLadderDataModule(data_dir=data_dir,
                               stats_file="E:/Data/Extracted/Processed/TrainNew_newstats.json",
                               train_split=0.8,
                               batch_size=256,
                               dilute=0.01)
    dm.setup()

    sample_tensor = torch.load(data_dir + os.listdir(data_dir)[0])[0]
    track_to_int = json.load(open('E:/Data/Extracted/Processed/TrainNew_track_to_int.json'))
    rt_to_int = json.load(open('E:/Data/Extracted/Processed/TrainNew_rt_to_int.json'))

    model = PriceLadderModel(max_traded_length=sample_tensor['traded'].shape[1],
                             track_to_int=track_to_int,
                             rt_to_int=rt_to_int)

    sample_dataloader = DataLoader(
        dm.train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=dm.collate_batch)

    for i, sample_data in enumerate(iter(sample_dataloader)):
        if i > 10:
            break

        with torch.no_grad():
            loss = model.training_step(sample_data, i)
            print(loss)
