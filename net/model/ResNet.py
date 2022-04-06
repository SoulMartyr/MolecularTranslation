from torch import nn
from torch.nn import Module


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding=0):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, base_planes, stride=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, base_planes)
        self.bn1 = nn.BatchNorm2d(base_planes)
        self.conv2 = conv3x3(base_planes, base_planes, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(base_planes)
        self.conv3 = conv1x1(base_planes, base_planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(base_planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.generate(layer_num=3, in_plane=64, base_planes=64, down_stride=1)
        self.layer2 = self.generate(layer_num=4, in_plane=256, base_planes=128, down_stride=2)
        self.layer3 = self.generate(layer_num=6, in_plane=512, base_planes=256, down_stride=2)
        self.layer4 = self.generate(layer_num=3, in_plane=1024, base_planes=512, down_stride=2)

    def generate(self, layer_num, in_plane, base_planes, down_stride):
        downsample = nn.Sequential(
            conv1x1(in_plane, base_planes * Bottleneck.expansion, down_stride),
            nn.BatchNorm2d(base_planes * Bottleneck.expansion),
        )
        layers = []
        for i in range(layer_num):
            if i == 0:
                layers.append(
                    Bottleneck(in_planes=in_plane, base_planes=base_planes, downsample=downsample, stride=down_stride))
            else:
                layers.append(
                    Bottleneck(in_planes=base_planes * Bottleneck.expansion, base_planes=base_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def generate_resnet():
    resnet = ResNet()
    return resnet
