from model import PriceLadderModel
from datasets import PriceLadderDataModule
import pandas as pd
import os
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
        patience=8,
        verbose=True,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='E:/checkpoints/',
        filename='price-ladder-{epoch:02d}-{val_loss:.4f}'
    )

    trainer = Trainer(callbacks=[early_stop_callback,checkpoint_callback], max_epochs=100, devices=1, accelerator='gpu',
                      logger=logger, deterministic=True, gradient_clip_val=0.5,
                      num_sanity_val_steps=0)

    # Feed into model
    with trainer.init_module():
        dm = PriceLadderDataModule("E:/Data/Extracted/Processed/Train/",
                                   "E:/Data/Extracted/Processed/Train.json",
                                   train_split=0.8,
                                   num_cached_markets_factor=5,
                                   mtl_factor=1.1,
                                   batch_size=128)
        dm.setup()

        model = PriceLadderModel(max_traded_length=dm.max_traded_length,
                                 track_to_int=dm.track_to_int,
                                 rt_to_int=dm.rt_to_int)

    trainer.fit(model, dm)