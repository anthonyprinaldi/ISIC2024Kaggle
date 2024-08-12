import torch.nn as nn

from .ISICModel import ISICModel


class XGBoostModel(ISICModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, image, metadata=None):
        image_features = self.model(image)
        return image_features

    def predict_step(self, batch, batch_idx):
        (image, metadata), labels = batch
        feats = self.forward(image, metadata)
        return feats
    
    def _post_init(self):
        self.model._fc = nn.Identity()
        self.model._swish = nn.Identity()