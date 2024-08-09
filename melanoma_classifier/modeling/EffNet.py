from efficientnet_pytorch import EfficientNet

models = {
    "EfficientNetB0": EfficientNet.from_pretrained("efficientnet-b0", num_classes=1, weights_path='efficientnet-b0.pth'),
    "EfficientNetB1": EfficientNet.from_pretrained("efficientnet-b1", num_classes=1, weights_path='efficientnet-b1.pth'),
    "EfficientNetB2": EfficientNet.from_pretrained("efficientnet-b2", num_classes=1, weights_path='efficientnet-b2.pth'),
    "EfficientNetB3": EfficientNet.from_pretrained("efficientnet-b3", num_classes=1, weights_path='efficientnet-b3.pth'),
    "EfficientNetB4": EfficientNet.from_pretrained("efficientnet-b4", num_classes=1, weights_path='efficientnet-b4.pth'),
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