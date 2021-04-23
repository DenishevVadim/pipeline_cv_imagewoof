import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 pool=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.Maxpool = nn.MaxPool2d(4)
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.pool: x = self.Maxpool(x)
        return x


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.res1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64))
        self.conv2 = ConvBlock(64, 128, pool=True)  # 128 x 32 x 32
        self.res2 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 512, pool=True)  # 256 x 8 x 8
        self.res3 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.conv4 = ConvBlock(512, 1024, pool=True)  # 512 x 2 x 2
        self.res4 = nn.Sequential(ConvBlock(1024, 1024), ConvBlock(1024, 1024))
        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # 1024 x 1 x 1
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.conv4(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        return out