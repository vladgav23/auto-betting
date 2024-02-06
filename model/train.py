from model import PriceLadderModel
from datasets import PriceLadderDataModule
import pandas as pd
import os
import torch
import json

from datetime import datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

pd.options.display.max_columns = None
os.environ['TMPDIR'] = 'E:/temp'

if __name__ == '__main__':
    seed_everything(9090)

    logger = WandbLogger(entity="vlad-gavrilov", project="price-ladder-betting")
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint_path = "E:/checkpoints/"+datetime.now().strftime('%Y%m%d_%H%M')

    os.mkdir(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename='price-ladder-{epoch:02d}-{val_loss:.4f}'
    )

    trainer = Trainer(callbacks=[early_stop_callback,checkpoint_callback], max_epochs=100, devices=1, accelerator='gpu',
                      logger=logger, deterministic=True, gradient_clip_val=0.5,
                      num_sanity_val_steps=0)

    # Feed into model
    with trainer.init_module():
        data_dir = "E:/Data/Extracted/Processed/TrainNew_Postprocessed/"
        dm = PriceLadderDataModule(data_dir=data_dir,
                                   stats_file="E:/Data/Extracted/Processed/TrainNew_newstats.json",
                                   train_split=0.8,
                                   batch_size=256)
        dm.setup()

        sample_tensor = torch.load(data_dir + os.listdir(data_dir)[0])[0]
        track_to_int = json.load(open('E:/Data/Extracted/Processed/TrainNew_track_to_int.json'))
        rt_to_int = json.load(open('E:/Data/Extracted/Processed/TrainNew_rt_to_int.json'))

        model = PriceLadderModel(max_traded_length=sample_tensor['traded'].shape[1],
                                 track_to_int=track_to_int,
                                 rt_to_int=rt_to_int)

    trainer.fit(model, dm)