import torch
from torch import nn


class XRaySegmentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = self._down_conv(1, 32, pool=False)         # 32 * 224
        self.conv2 = self._down_conv(32, 64, pool=True)         # 64 * 112
        self.conv3 = self._down_conv(64, 128, pool=True)        # 128 * 56
        self.conv4 = self._down_conv(128, 256, pool=True)       # 256 * 28
        self.conv5 = self._down_conv(256, 512, pool=True)       # 512 * 14
        self.conv6 = self._down_conv(512, 1024, pool=True)      # 1024 * 7

        self.upconv5 = self._up_conv(1024, 512, reduce=False)   # 512 * 14
        self.upconv4 = self._up_conv(1024, 256, reduce=True)    # 256 * 28
        self.upconv3 = self._up_conv(512, 128, reduce=True)     # 128 * 56
        self.upconv2 = self._up_conv(256, 64, reduce=True)      # 64 * 112
        self.upconv1 = self._up_conv(128, 32, reduce=True)      # 32 * 224

        self.extract = nn.Sequential(
            nn.Conv2d(32, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def _up_conv(self, _in, _out, reduce):
        layers = []
        up_in = _in
        if reduce:
            up_in = _in//2
            layers += [
                nn.Conv2d(_in, up_in, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(up_in),
                nn.Conv2d(up_in, up_in, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(up_in)
            ]

        layers += [
            nn.ConvTranspose2d(up_in, _out, kernel_size=4,
                               stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(_out)
        ]

        return nn.Sequential(*layers)

    def _down_conv(self, _in, _out, pool):
        layers = []
        if pool:
            layers += [
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Dropout(p=0.4)
            ]

        layers += [
            nn.Conv2d(_in, _out, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(_out),
            nn.Conv2d(_out, _out, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(_out)
        ]

        return nn.Sequential(*layers)

    def forward(self, image):
        out1 = self.conv1(image)        # 32 * 224
        out2 = self.conv2(out1)         # 64 * 112
        out3 = self.conv3(out2)         # 128 * 56
        out4 = self.conv4(out3)         # 256 * 28
        out5 = self.conv5(out4)         # 512 * 14
        out6 = self.conv6(out5)         # 1024 * 7

        up5 = self.upconv5(out6)                                # 512 * 14
        up4 = self.upconv4(torch.cat((out5, up5), dim=1))       # 256 * 28
        up3 = self.upconv3(torch.cat((out4, up4), dim=1))       # 128 * 56
        up2 = self.upconv2(torch.cat((out3, up3), dim=1))       # 64 * 112
        up1 = self.upconv1(torch.cat((out2, up2), dim=1))       # 32 * 224

        probs = self.extract(up1)                               # 1 * 224
        return probs
