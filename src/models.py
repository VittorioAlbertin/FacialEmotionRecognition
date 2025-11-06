"""
Model architectures for emotion recognition.

Supports multiple CNN and Vision Transformer architectures with
custom heads for 13-dimensional emotion probability prediction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm


class EmotionModel(nn.Module):
    """
    Base class for emotion recognition models.

    All models output a 13-dimensional probability distribution over emotions.
    """

    def __init__(self, num_emotions=13):
        super().__init__()
        self.num_emotions = num_emotions

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")


class ResNetEmotion(EmotionModel):
    """
    ResNet-based emotion recognition model.

    Args:
        architecture: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
        pretrained: Use ImageNet pretrained weights
        num_emotions: Number of emotion classes (default: 13)
    """

    def __init__(self, architecture='resnet18', pretrained=True, num_emotions=13):
        super().__init__(num_emotions)

        # Load pretrained ResNet
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif architecture == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif architecture == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_emotions),
            nn.Softmax(dim=1)
        )

        self.architecture = architecture

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Probability distribution of shape (batch_size, num_emotions)
        """
        return self.backbone(x)

    def __repr__(self):
        return f"ResNetEmotion(architecture={self.architecture}, num_emotions={self.num_emotions})"


class EfficientNetEmotion(EmotionModel):
    """
    EfficientNet-based emotion recognition model.

    Args:
        architecture: 'efficientnet_b0', 'efficientnet_b1', etc.
        pretrained: Use ImageNet pretrained weights
        num_emotions: Number of emotion classes (default: 13)
    """

    def __init__(self, architecture='efficientnet_b0', pretrained=True, num_emotions=13):
        super().__init__(num_emotions)

        # Load pretrained EfficientNet from timm
        self.backbone = timm.create_model(architecture, pretrained=pretrained)

        # Get the number of features from the classifier
        num_features = self.backbone.classifier.in_features

        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_emotions),
            nn.Softmax(dim=1)
        )

        self.architecture = architecture

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Probability distribution of shape (batch_size, num_emotions)
        """
        return self.backbone(x)

    def __repr__(self):
        return f"EfficientNetEmotion(architecture={self.architecture}, num_emotions={self.num_emotions})"


class ViTEmotion(EmotionModel):
    """
    Vision Transformer-based emotion recognition model.

    Args:
        architecture: 'vit_base_patch16_224', 'vit_small_patch16_224', etc.
        pretrained: Use ImageNet pretrained weights
        num_emotions: Number of emotion classes (default: 13)
    """

    def __init__(self, architecture='vit_base_patch16_224', pretrained=True, num_emotions=13):
        super().__init__(num_emotions)

        # Load pretrained ViT from timm
        self.backbone = timm.create_model(architecture, pretrained=pretrained)

        # Get the number of features from the head
        num_features = self.backbone.head.in_features

        # Replace the head
        self.backbone.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_emotions),
            nn.Softmax(dim=1)
        )

        self.architecture = architecture

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Probability distribution of shape (batch_size, num_emotions)
        """
        return self.backbone(x)

    def __repr__(self):
        return f"ViTEmotion(architecture={self.architecture}, num_emotions={self.num_emotions})"


def create_model(model_type='resnet18', pretrained=True, num_emotions=13):
    """
    Factory function to create emotion recognition models.

    Args:
        model_type: Type of model architecture
            - 'resnet18', 'resnet34', 'resnet50', 'resnet101'
            - 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'
            - 'vit_base_patch16_224', 'vit_small_patch16_224'
        pretrained: Use ImageNet pretrained weights
        num_emotions: Number of emotion classes (default: 13)

    Returns:
        EmotionModel instance

    Example:
        >>> model = create_model('resnet18', pretrained=True)
        >>> model = create_model('efficientnet_b0', pretrained=True)
    """
    model_type = model_type.lower()

    # ResNet models
    if model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        return ResNetEmotion(architecture=model_type, pretrained=pretrained, num_emotions=num_emotions)

    # EfficientNet models
    elif model_type.startswith('efficientnet'):
        return EfficientNetEmotion(architecture=model_type, pretrained=pretrained, num_emotions=num_emotions)

    # Vision Transformer models
    elif model_type.startswith('vit'):
        return ViTEmotion(architecture=model_type, pretrained=pretrained, num_emotions=num_emotions)

    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported: resnet18/34/50/101, efficientnet_b0/b1/b2, vit_base/small_patch16_224")


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Test ResNet
    model_resnet = create_model('resnet18', pretrained=False)
    print(f"\n✓ Created: {model_resnet}")

    # Test EfficientNet
    model_effnet = create_model('efficientnet_b0', pretrained=False)
    print(f"✓ Created: {model_effnet}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model_resnet(dummy_input)
    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sum: {output.sum(dim=1)} (should be ~1.0)")

    # Count parameters
    total_params = sum(p.numel() for p in model_resnet.parameters())
    trainable_params = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)
    print(f"\n✓ Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
