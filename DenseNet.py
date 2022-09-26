import re

import torchvision
import torch
from torch import nn
# from torchvision.models import DenseNet121_Weights

PRETRAINED_MODEL = 'model.pth.tar'


class DenseNet121(nn.Module):
    def __init__(self, num_classes, drop_rate=0, weights=True, keep_expected_value=True):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(drop_rate=drop_rate)
        num_features = self.densenet121.classifier.in_features
        if weights:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_features, 14)
            )
            load_weights(self)
        self.densenet121.classifier = nn.Linear(num_features, num_classes)
        if drop_rate > 0 and keep_expected_value:
            self.keep_expected_value(drop_rate)

    def forward(self, x):
        x = self.densenet121(x)
        return x

    def keep_expected_value(self, drop_rate):
        keep_ratio = 1 - drop_rate
        denselayer_finder = re.compile(r'densenet121.features.denseblock(\d+).denselayer(\d+).*')
        multiply = False
        last_block = 1
        last_layer = 1
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                result = denselayer_finder.findall(name)
                if result:
                    block, layer = map(int, result[0])
                    if block != last_block or layer != last_layer:
                        multiply = True
                        last_block = block
                        last_layer = layer
                if multiply:
                    if 'weight' in name:
                        parameter.data = parameter.data / keep_ratio
                        multiply = False


def load_weights(model, device='cpu'):
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    checkpoint = torch.load(PRETRAINED_MODEL, map_location=torch.device(device))
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
