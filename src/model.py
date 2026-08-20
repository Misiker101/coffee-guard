import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"]
NUM_CLASSES = len(CLASS_NAMES)


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B0 with a replaced classification head.

    We freeze the convolutional backbone and only train the final layer
    first (fast, works on CPU / free Colab GPU), then optionally unfreeze
    for fine-tuning — see src/train.py for the two-phase schedule.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def load_model_for_inference(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
