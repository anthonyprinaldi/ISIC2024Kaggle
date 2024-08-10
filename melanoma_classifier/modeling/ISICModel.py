from pathlib import Path
from typing import List, Optional, Union

import lightning as L
import torch
import torch.nn as nn

from .EffNet import get_model as effnet_models
from .EffNet import image_sizes as effnet_image_sizes
from .metrics import AUC_20


class ISICModel(L.LightningModule):

    def __init__(self,
                 model_name:str ="EfficientNetB1",
                 lr: float=1e-4,
                 weight_decay: float = 5e-3,
                 calculate_metrics: bool = False,
                 num_meta_features: int = 0,
                 meta_network_dim: Optional[List[int]]=None,
                 weight: Optional[float]=None,
                 effnet_location: Optional[Union[str, Path]]=None,
                 ):
        super().__init__()
        self.model = effnet_models(
            model_name=model_name,
            location=effnet_location,
        )
        self.image_size = effnet_image_sizes[model_name]
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics = calculate_metrics
        self.num_meta_features = num_meta_features
        self.meta_network_dim = meta_network_dim
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0,]) if weight is None else torch.tensor(weight),
        )

        self.val_auc_20 = AUC_20()

        # change the last layer to have
        if self.num_meta_features > 0:
            assert self.meta_network_dim is not None, \
                "Meta network dimensions must be provided."

            meta_network_layers = [
                nn.Linear(self.num_meta_features, self.meta_network_dim[0]),
                nn.BatchNorm1d(self.meta_network_dim[0]),
                nn.SiLU(),
                nn.Dropout(0.5),
            ]
            for i in range(len(self.meta_network_dim) - 1):
                meta_network_layers.append(nn.Linear(self.meta_network_dim[i], self.meta_network_dim[i+1]))
                meta_network_layers.append(nn.BatchNorm1d(self.meta_network_dim[i+1]))
                meta_network_layers.append(nn.SiLU())
                meta_network_layers.append(nn.Dropout(0.5))

            self.meta_network = nn.Sequential(*meta_network_layers)

            self.model._fc = nn.Linear(self.model._fc.in_features, self.meta_network_dim[-1])

            self.final_layer = nn.Linear(self.meta_network_dim[-1]*2, 1)

    def forward(self, image, metadata=None):

        image_features = self.model(image)

        if self.num_meta_features > 0:
            meta_features = self.meta_network(metadata)
            # concatenate the image features and the meta features
            features = torch.cat([image_features, meta_features], dim=1)
            return self.final_layer(features)
        else:
            return image_features
        
    def get_metrics(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        preds = (outputs > 0.5).float()

        y = targets.cpu().detach()
        preds = preds.cpu().detach()
        
        # acc = accuracy_score(y, preds)
        acc = (preds == y).float().mean()
        # auc_score = roc_auc_score(y, outputs.cpu().detach())
        # f1 = f1_score(y, preds)
        # precision = precision_score(y, preds)
        # recall = recall_score(y, preds)

        return {
            "accuracy": acc,
            # "auc_20": auc_20,
            # "auc": auc_score,
            # "f1": f1,
            # "precision": precision,
            # "recall": recall,
        }

    def training_step(self, batch, batch_idx):
        step_dict = self._step(batch, metrics=self.metrics, return_preds=False)

        for metric, value in step_dict.items():
            self.log(f"train_{metric}", value, prog_bar=True, on_step=True, on_epoch=True)

        return step_dict["loss"]

    def validation_step(self, batch, batch_idx):
        step_dict = self._step(batch, metrics=self.metrics, return_preds=True)

        for metric, value in step_dict.items():
            if metric == "preds":
                continue
            self.log(f"val_{metric}", value, prog_bar=True, on_step=False, on_epoch=True)

        self.val_auc_20.update(step_dict["preds"], batch[1])

        return step_dict["loss"]
    
    def predict_step(self, batch):
        logits = self._step(batch, metrics=False, return_preds=True)["preds"]
        return torch.sigmoid(logits)
    
    def on_validation_epoch_end(self):
        self.log("val_auc_20", self.val_auc_20.compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.val_auc_20.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=10,
        #     eta_min=1e-6,
        # )
        # return [optimizer], [lr_scheduler]
        return optimizer

    def _step(self, batch, metrics=False, return_preds=False):
        (image, metadata), y = batch
        y_hat = self.forward(image, metadata)
        res = {}

        loss = self.criterion(y_hat.squeeze(-1), y.float())
        res["loss"] = loss

        if metrics:
            res.update(self.get_metrics(y_hat, y))
        
        if return_preds:
            res["preds"] = y_hat

        return res
