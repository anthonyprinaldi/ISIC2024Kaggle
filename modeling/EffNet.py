from efficientnet_pytorch import EfficientNet

models = {
    "EfficientNetB0": EfficientNet.from_pretrained("efficientnet-b0", num_classes=1),
    "EfficientNetB1": EfficientNet.from_pretrained("efficientnet-b1", num_classes=1),
    "EfficientNetB2": EfficientNet.from_pretrained("efficientnet-b2", num_classes=1),
    "EfficientNetB3": EfficientNet.from_pretrained("efficientnet-b3", num_classes=1),
    "EfficientNetB4": EfficientNet.from_pretrained("efficientnet-b4", num_classes=1),
}

image_sizes = {
    "EfficientNetB0": 224,
    "EfficientNetB1": 240,
    "EfficientNetB2": 260,
    'EfficientNetB3': 300,
    'EfficientNetB4': 380,
}

batch_sizes = {
    "EfficientNetB0": 150,
    "EfficientNetB1": 100,
    "EfficientNetB2": 64,
    'EfficientNetB3': 50,
    'EfficientNetB4': 20
}