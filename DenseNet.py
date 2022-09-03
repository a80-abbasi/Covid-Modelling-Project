import re

import torchvision
import torch
from torch import nn
from torchvision.models import DenseNet121_Weights

CKPT_PATH = 'model.pth.tar'


class DenseNet121(nn.Module):

    def __init__(self, num_classes, weights=True):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121()
        num_features = self.densenet121.classifier.in_features
        if weights:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_features, 14)
            )
            load_weights(self)
        self.densenet121.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.densenet121(x)
        return x


def load_weights(model, device='cpu'):
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    checkpoint = torch.load(CKPT_PATH, map_location=torch.device(device))
    state_dict = checkpoint['state_dict']
    remove_data_parallel = True  # Change if you don't want to use nn.DataParallel(model)

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]
        # Delete old key only if modified.
        if match or remove_data_parallel:
            del state_dict[key]

    model.load_state_dict(state_dict)

# def load_densenet121(num_classes, pretrained=True, device='cpu'):
#     model = DenseNet121(num_classes)
#     if pretrained:
#         load_weights(model, device)
#     model.densenet121.classifier = model.densenet121.classifier[0]
#     return model
