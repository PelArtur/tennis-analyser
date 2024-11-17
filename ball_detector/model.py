import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.batch_norm(self.relu(self.conv(x)))


class TrackNet(nn.Module):
    def __init__(self, in_channels: int = 9, out_channels: int = 256, training: bool =False):
        super().__init__()
        self.in_channels = in_channels    #RGB, 1 frame = 3 in_channels. 3 frames by default
        self.out_channels = out_channels
        self.training = training

        self.vgg16 = nn.Sequential(
            Conv(in_channels=self.in_channels, out_channels=64),
            Conv(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=64, out_channels=128),
            Conv(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=128, out_channels=256),
            Conv(in_channels=256, out_channels=256),
            Conv(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=256, out_channels=512),
            Conv(in_channels=512, out_channels=512),
            Conv(in_channels=512, out_channels=512)
        )

        self.dnn = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(in_channels=512, out_channels=512),
            Conv(in_channels=512, out_channels=512),
            Conv(in_channels=512, out_channels=512),
            nn.Upsample(scale_factor=2),
            Conv(in_channels=512, out_channels=128),
            Conv(in_channels=128, out_channels=128),
            nn.Upsample(scale_factor=2),
            Conv(in_channels=128, out_channels=64),
            Conv(in_channels=64, out_channels=64),
            Conv(in_channels=64, out_channels=self.out_channels)
        )

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.dnn(x)
        if self.training:
            return x
        return self.soft_max(x)
