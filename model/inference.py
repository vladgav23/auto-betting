import pandas as pd
import torch
from datasets import PriceLadderDataModule
from model import PriceLadderModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
import os

pd.options.display.max_columns = None

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def unpack_predictions(self, predictions):
        # Initialize an empty list to store the data
        data = []

        for batch in predictions:
            for i, market_id in enumerate(batch['metadata']['market_ids']):
                selection_ids = batch['metadata']['selection_ids'][i]
                seconds_to_start = [batch['metadata']['seconds_to_start'][i]] * len(selection_ids)
                market_ids = [market_id] * len(selection_ids)
                lpts = batch['pred_tensors']['lpts'][i]
                mover_flag = batch['pred_tensors']['mover_flags'][i].numpy().tolist()

                best_available_back = [round(x,2) for x in (batch['pred_tensors']['back'][i, :, 0, 0] * lpts).numpy().tolist()]
                best_available_lay = [round(x,2) for x in (batch['pred_tensors']['lay'][i, :, 0, 0] * lpts).numpy().tolist()]

                best_available_back_size = [round(x,2) for x in torch.exp(batch['pred_tensors']['back'][i, :, 0, 1]).numpy().tolist()]
                best_available_lay_size = [round(x,2) for x in torch.exp(batch['pred_tensors']['lay'][i, :, 0, 1]).numpy().tolist()]

                target = batch['target'][i].view(4, 4)
                prediction = batch['predictions'][i].view(4, 3)

                # Transform target and pred
                # Log transform total volume
                target[:, 3] = torch.exp(target[:, 3])

                # Transform prices into ratio to LPT
                target[:, :3] = target[:, :3] * lpts.unsqueeze(1)

                # Transform prices into ratio to LPT
                prediction[:, :3] = prediction[:, :3] * lpts.unsqueeze(1)

                target_vol = [round(x,2) for x in target[:, 3].numpy().tolist()]
                target_wap = [round(x,2) for x in target[:, 2].numpy().tolist()]
                predicted_wap = [round(x,2) for x in prediction[:, 2].numpy().tolist()]

                target_max_price = [round(x,2) for x in target[:, 0].numpy().tolist()]
                predicted_max_price = [round(x,2) for x in prediction[:, 0].numpy().tolist()]
                target_min_price = [round(x,2) for x in target[:, 1].numpy().tolist()]
                predicted_min_price = [round(x,2) for x in prediction[:, 1].numpy().tolist()]

                df = pd.DataFrame(
                    {
                        "market_id": market_ids,
                        "selection_id": selection_ids,
                        "seconds_to_start": seconds_to_start,
                        "last_traded_price": lpts.numpy().tolist(),
                        "mover": mover_flag,
                        "best_back": best_available_back,
                        "best_back_size": best_available_back_size,
                        "best_lay": best_available_lay,
                        "best_lay_size": best_available_lay_size,
                        "target_wap": target_wap,
                        "target_vol": target_vol,
                        "predicted_wap": predicted_wap,
                        "target_max_price": target_max_price,
                        "predicted_max_price": predicted_max_price,
                        "target_min_price": target_min_price,
                        "predicted_min_price": predicted_min_price
                    }
                )

                data.append(df)

        # Unpack each batch in the predictions

        return pd.concat(data)

    def write_on_epoch_end(self, trainer, pl_module: "LightningModule", predictions, batch_indices):
        df = self.unpack_predictions(predictions)

        # Save to CSV
        df.to_csv(os.path.join(self.output_dir, f"holdout-model-predictions.csv"), index=False)

if __name__ == '__main__':
    pred_writer_cb = PredictionWriter("predictions/")

    ckpt_to_use = "E:/checkpoints/price-ladder-epoch=06-val_loss=0.1166.ckpt"
    ckpt_file = torch.load(ckpt_to_use)
    max_traded_length_train = int(ckpt_file['state_dict']['proj_traded_ladder.weight'].shape[1] / 64)
    track_mapping = ckpt_file['track_to_int']
    race_type_mapping = ckpt_file['rt_to_int']

    dm = PriceLadderDataModule("E:/Data/Extracted/Processed/Holdout/",
                               "E:/Data/Extracted/Processed/Holdout.json",
                               train_split=None,
                               num_cached_markets_factor=1,
                               mtl_factor=1.1,
                               max_traded_length=max_traded_length_train,
                               track_to_int=track_mapping,
                               rt_to_int=race_type_mapping,
                               batch_size=1024)

    dm.setup(stage="predict")

    model = PriceLadderModel(max_traded_length=max_traded_length_train,
                                                  track_to_int=track_mapping,
                                                  rt_to_int=race_type_mapping)

    model.load_state_dict(ckpt_file['state_dict'])

    trainer = Trainer(devices=1,accelerator='gpu',callbacks=[pred_writer_cb])

    trainer.predict(model, dm)