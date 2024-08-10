from pathlib import Path
from typing import Optional, Union

from efficientnet_pytorch import EfficientNet


def get_model(
        model_name: str,
        location: Optional[Union[str, Path]]=None,
        ) -> EfficientNet:

    if location is not None:
        if not isinstance(location, Path):
            location = Path(location)
        print(f"Loading EffNet model from {location}")
        models = {
            "EfficientNetB0": EfficientNet.from_pretrained("efficientnet-b0", num_classes=1, weights_path=location / "efficientnet-b0.pth"),
            "EfficientNetB1": EfficientNet.from_pretrained("efficientnet-b1", num_classes=1, weights_path=location / "efficientnet-b1.pth"),
            "EfficientNetB2": EfficientNet.from_pretrained("efficientnet-b2", num_classes=1, weights_path=location / "efficientnet-b2.pth"),
            "EfficientNetB3": EfficientNet.from_pretrained("efficientnet-b3", num_classes=1, weights_path=location / "efficientnet-b3.pth"),
            "EfficientNetB4": EfficientNet.from_pretrained("efficientnet-b4", num_classes=1, weights_path=location / "efficientnet-b4.pth"),
        }
    
    else:
        models = {
            "EfficientNetB0": EfficientNet.from_pretrained("efficientnet-b0", num_classes=1),
            "EfficientNetB1": EfficientNet.from_pretrained("efficientnet-b1", num_classes=1),
            "EfficientNetB2": EfficientNet.from_pretrained("efficientnet-b2", num_classes=1),
            "EfficientNetB3": EfficientNet.from_pretrained("efficientnet-b3", num_classes=1),
            "EfficientNetB4": EfficientNet.from_pretrained("efficientnet-b4", num_classes=1),
        }
    
    return models[model_name]

image_sizes = {
    "EfficientNetB0": 224,
    "EfficientNetB1": 240,
    "EfficientNetB2": 260,
    'EfficientNetB3': 300,
    'EfficientNetB4': 380,
}