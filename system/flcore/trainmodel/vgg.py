"""
VGG16 in PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np


class VGG16(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, img_size=32, **kwargs):
        super(VGG16, self).__init__()

        cfg_base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        attack_type = kwargs.get('attack_type', None)
        defense_type = kwargs.get('defense_type', None)
        
        false_cases = {
            ('robust_aggregation', 'a3fl'),
            ('robust_aggregation', 'badnets'),
            ('robust_aggregation', 'blended'),
            ('robust_aggregation', 'model_replacement'),
            ('alignins', 'a3fl'),
            ('alignins', 'badnets'),
            ('alignins', 'blended'),
            ('alignins', 'model_replacement'),
            ('none', 'model_replacement'),
            ('gradient_clipping', 'model_replacement'),
            ('flame', 'badnets'),
            ('flame', 'blended'),
            ('flame', 'model_replacement'),
            ('neuroforensics', 'badnets'),
            ('neuroforensics', 'blended'),
            ('neuroforensics', 'model_replacement'),
        }
        
        true_cases = {
            ('none', 'a3fl'),
            ('none', 'badnets'),
            ('none', 'blended'),
            ('gradient_clipping', 'a3fl'),
            ('gradient_clipping', 'badnets'),
            ('gradient_clipping', 'blended'),
            ('flame', 'a3fl'),
            ('neuroforensics', 'a3fl'),
        }
        
        current_case = (defense_type, attack_type)
        
        if current_case in false_cases:
            self.use_tracking = False
            print(f"--- [VGG16] track_running_stats=False (defense={defense_type}, attack={attack_type}) ---")
        elif current_case in true_cases:
            self.use_tracking = True
            print(f"--- [VGG16] track_running_stats=True (defense={defense_type}, attack={attack_type}) ---")
        else:
            self.use_tracking = True
            print(f"⚠️ [VGG16] Unknown combination (defense={defense_type}, attack={attack_type}), defaulting to track_running_stats=True")

        self.cfg = self._adapt_cfg_for_size(cfg_base, img_size)

        self.features = self.make_layers(self.cfg, in_channels=in_channels, batch_norm=True)

        def _get_classifier_input_features():
            with torch.no_grad():
                dummy_input = torch.randn(2, in_channels, img_size, img_size)
                self.features.eval() 
                output = self.features(dummy_input)
                self.features.train()  
                return output.shape[1] * output.shape[2] * output.shape[3]

        classifier_in_features = _get_classifier_input_features()

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _adapt_cfg_for_size(self, cfg, img_size):
       
        if img_size <= 28: 
            last_m_index = len(cfg) - 1 - cfg[::-1].index('M')
            adapted_cfg = cfg[:last_m_index] + cfg[last_m_index+1:]
            return adapted_cfg
        return cfg
   

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg, in_channels=3, batch_norm=True):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
                if batch_norm:
                    if self.use_tracking:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                    else:
                        layers += [conv2d, nn.BatchNorm2d(v, track_running_stats=False), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = VGG16()
    print('done')
    x = torch.ones(size=(1, 3, 32, 32))
    print(x.shape)
    for layer in range(0, 3):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(3, 14):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(14, 24):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(24, 34):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(34, 44):
        x = model.features[layer](x)
    print(x.shape)
    x = x.view(x.size(0), -1)  # shape=[1,512]
    print(x.shape)
    x = model.classifier(x)
    print(x.shape)
