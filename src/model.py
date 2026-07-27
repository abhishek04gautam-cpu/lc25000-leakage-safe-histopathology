"""Model definitions for the LC25000 cancer image classification framework."""
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    densenet121,
    efficientnet_b0,
    resnet18,
)


def get_models(dataset):
    if dataset.data_type != "image":
        raise ValueError("The final LC25000 framework expects image datasets only.")
    if dataset.name != "LC25000":
        raise ValueError("This implementation is configured for LC25000.")
    return {
        "SimpleCNN": SimpleCNN(dataset.num_classes),
        "TransferCNN_ResNet18": TransferCNN(dataset.num_classes),
    }

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_type = "cnn"
        self.model_name = "SimpleCNN"
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class TransferCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        training_mode="staged_finetune",
        load_pretrained_weights=True,
    ):
        super().__init__()
        self.training_mode = training_mode
        self.backbone_family = "resnet"
        self.model_type = "cnn"
        if training_mode == "head_only":
            self.model_name = "TransferCNN_ResNet18_HeadOnly"
        elif training_mode == "staged_finetune":
            self.model_name = "TransferCNN_ResNet18"
        else:
            raise ValueError("training_mode must be 'staged_finetune' or 'head_only'.")
        weights = (
            ResNet18_Weights.DEFAULT
            if load_pretrained_weights
            else None
        )
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
        if training_mode == "head_only":
            self.freeze_backbone_train_head_only()
        else:
            self.freeze_for_staged_finetuning()

    def freeze_backbone_train_head_only(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.fc.parameters():
            parameter.requires_grad = True

    def freeze_for_staged_finetuning(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not name.startswith(
                ("conv1", "bn1", "layer1", "layer2")
            )
        for parameter in self.model.fc.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        return self.model(x)

class TransferDenseNet121(nn.Module):
    def __init__(self, num_classes, training_mode="staged_finetune"):
        super().__init__()
        self.training_mode = training_mode
        self.backbone_family = "densenet"
        self.model_type = "cnn"
        self.model_name = "TransferCNN_DenseNet121"
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
        if training_mode == "head_only":
            self.freeze_backbone_train_head_only()
        elif training_mode == "staged_finetune":
            self.freeze_for_staged_finetuning()
        else:
            raise ValueError("training_mode must be 'staged_finetune' or 'head_only'.")

    def freeze_backbone_train_head_only(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def freeze_for_staged_finetuning(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for name, parameter in self.model.features.named_parameters():
            if name.startswith(("denseblock4", "norm5")):
                parameter.requires_grad = True
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        return self.model(x)

class TransferEfficientNetB0(nn.Module):
    def __init__(self, num_classes, training_mode="staged_finetune"):
        super().__init__()
        self.training_mode = training_mode
        self.backbone_family = "efficientnet"
        self.model_type = "cnn"
        self.model_name = "TransferCNN_EfficientNetB0"
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
        if training_mode == "head_only":
            self.freeze_backbone_train_head_only()
        elif training_mode == "staged_finetune":
            self.freeze_for_staged_finetuning()
        else:
            raise ValueError("training_mode must be 'staged_finetune' or 'head_only'.")

    def freeze_backbone_train_head_only(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def freeze_for_staged_finetuning(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for block_index in [6, 7, 8]:
            if block_index < len(self.model.features):
                for parameter in self.model.features[block_index].parameters():
                    parameter.requires_grad = True
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        return self.model(x)
