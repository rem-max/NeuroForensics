import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, has_bn=True, **kwargs) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride);
        self.bn1 = norm_layer(planes) if has_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=False);
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes) if has_bn else nn.Identity();
        self.downsample = downsample;
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x;
        out = self.conv1(x);
        out = self.bn1(out);
        out = self.relu(out)
        out = self.conv2(out);
        out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity;
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, has_bn=True, **kwargs) -> None:
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = planes;
        self.conv1 = conv1x1(inplanes, width);
        self.bn1 = norm_layer(width) if has_bn else nn.Identity()
        self.conv2 = conv3x3(width, width, stride);
        self.bn2 = norm_layer(width) if has_bn else nn.Identity()
        self.conv3 = conv1x1(width, planes * self.expansion);
        self.bn3 = norm_layer(planes * self.expansion) if has_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=False);
        self.downsample = downsample;
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x;
        out = self.conv1(x);
        out = self.bn1(out);
        out = self.relu(out);
        out = self.conv2(out)
        out = self.bn2(out);
        out = self.relu(out);
        out = self.conv3(out);
        out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity;
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: Any, layers: List[int], in_channels: int = 3, num_classes: int = 1000, **kwargs) -> None:
        super(ResNet, self).__init__()
       
        attack_type = kwargs.get('attack_type', None)
        defense_type = kwargs.get('defense_type', None)
        
        use_no_tracking = False
        if attack_type in ['badnets', 'blended']:
            use_no_tracking = True
            print(f"--- [ResNet] Using BatchNorm without tracking (attack={attack_type}) ---")
        elif defense_type == 'alignins':
            use_no_tracking = True
            print(f"--- [ResNet] Using BatchNorm without tracking (defense={defense_type}) ---")
        
        if use_no_tracking:
            norm_layer = lambda num_features: nn.BatchNorm2d(num_features, track_running_stats=False)
        else:
            norm_layer = nn.BatchNorm2d
            print(f"--- [ResNet] Using standard BatchNorm with tracking ---")
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        print(f"--- [ResNet] Using standard aggressive stem for in_channels = {in_channels} ---")
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Any, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer;
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = [];
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x);
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        x = self.avgpool(x);
        x = torch.flatten(x, 1);
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(in_channels: int = 3, **kwargs: Any) -> ResNet:
    print(f"--- CHECKPOINT B (resnet.py): resnet18() helper RECEIVED in_channels={in_channels} ---")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, **kwargs)


def resnet34(in_channels: int = 3, **kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, **kwargs)


def resnet50(in_channels: int = 3, **kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, **kwargs)
