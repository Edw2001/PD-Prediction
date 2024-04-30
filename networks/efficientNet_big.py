"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3

# Assuming model_dict is used for creating specific model instances
model_dict = {
    'efficientnet_b0': [efficientnet_b0, 1280],
    'efficientnet_b1': [efficientnet_b1, 1280],
    'efficientnet_b2': [efficientnet_b2, 1408],
    'efficientnet_b3': [efficientnet_b3, 1536],
}

class SupConEfficientNet(nn.Module):
    def __init__(self, name='efficientnet_b0', head='mlp', feat_dim=128):
        super(SupConEfficientNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        print(f"Initializing model {name} with output dim {dim_in}")
        self.encoder = model_fun(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # Assuming you remove the classifier
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        feat = self.encoder(x)
        print(f"Shape after encoder: {feat.shape}")

        # Ensuring the output is properly shaped for the linear layers
        feat = torch.flatten(feat, 1)
        print(f"Shape after flattening: {feat.shape}")

        feat = self.head(feat)
        print(f"Shape after head: {feat.shape}")

        feat = F.normalize(feat, dim=1)
        print(f"Shape after normalization: {feat.shape}")
        return feat


class SupCEEfficientNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='efficientnet_b0', num_classes=2):
        super(SupCEEfficientNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(pretrained=True)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)  # Encode the input

        # Pool and flatten if needed
        if len(feat.shape) == 4 and feat.shape[2] > 1 and feat.shape[3] > 1:
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = torch.flatten(feat, 1)  # Flatten to [batch_size, channels]
        else:
            feat = torch.flatten(feat, 1)

        return self.fc(feat)

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='efficientnet_b0', num_classes=2):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
