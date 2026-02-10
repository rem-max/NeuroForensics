# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List
from flcore.trainmodel.vgg import VGG16
from flcore.trainmodel.models import FedAvgCNN
from flcore.trainmodel.models import LeNet, init_weights
# 修改后 (正确的)
from flcore.trainmodel.resnet import ResNet, BasicBlock, Bottleneck
from torch import Tensor


# ------------------------------
# 通用 FakeReLU（保持你的接口）
# ------------------------------
class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ============================================================
# ResNet（Lsep 固定在 fc 之前：avgpool→flatten 之后）
# ============================================================
class ResNetAdaptivePartialModel(ResNet):
    """
    为 ResNet18 模型添加的适配器，使其兼容 FreeEagle 防御。

    inspect_layer_position 语义（扩展为7档）：
      0: full logits（完整模型输出）
      1: 初始卷积块后 (conv1+bn1+relu+maxpool)
      2: avgpool之后展平 (512维向量) - FreeEagle 使用
      3: layer1之后 (64 channels, BasicBlock输出)
      4: layer2之后 (128 channels, BasicBlock输出)
      5: layer3之后 (256 channels, BasicBlock输出)
      6: layer4之后 (512 channels, BasicBlock输出)
    """

    def __init__(self, block: Any, layers: List[int], num_classes: int = 10, inspect_layer_position: int = 2,
                 original_input_img_shape: tuple = (1, 3, 32, 32), **kwargs):
        in_channels = original_input_img_shape[1]
        super().__init__(block=block, layers=layers, in_channels=in_channels, num_classes=num_classes, **kwargs)
        self.eval()

        # 七档位：扩展支持不同ResNet层位置
        self.inspect_layer_positions = [0, 1, 2, 3, 4, 5, 6]
        if inspect_layer_position not in self.inspect_layer_positions:
            raise ValueError(f"inspect_layer_position must be one of {self.inspect_layer_positions}")
        self.inspect_layer_position = int(inspect_layer_position)

        self.input_shapes = []
        with torch.no_grad():
            # 使用 batch_size=2 避免 BatchNorm 的限制
            batch_size = 2
            dummy_shape = (batch_size,) + original_input_img_shape[1:]
            dummy = torch.randn(*dummy_shape)

            # 临时切换到 eval 模式以安全进行形状推断
            self.eval()

            # 0: 输入形状（记录单个样本的形状）
            self.input_shapes.append(original_input_img_shape)

            # 1: 初始卷积块后 (conv1+bn1+relu+maxpool)
            stem_out = self.maxpool(self.relu(self.bn1(self.conv1(dummy))))
            # 记录单个样本的形状
            single_sample_shape = (1,) + stem_out.shape[1:]
            self.input_shapes.append(single_sample_shape)

            # 2: avgpool之后展平 (Lsep层)
            x = stem_out
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            lsep_embedding = self.avgpool(x)
            feature_dim = lsep_embedding.shape[1]
            self.input_shapes.append((1, int(feature_dim), 1, 1))

            # 3-6: 各个layer的输出形状（记录单个样本的形状）
            x = stem_out

            # 3: layer1之后
            x = self.layer1(x)
            self.input_shapes.append((1,) + x.shape[1:])

            # 4: layer2之后
            x = self.layer2(x)
            self.input_shapes.append((1,) + x.shape[1:])

            # 5: layer3之后
            x = self.layer3(x)
            self.input_shapes.append((1,) + x.shape[1:])

            # 6: layer4之后
            x = self.layer4(x)
            self.input_shapes.append((1,) + x.shape[1:])

            # 恢复训练模式
            self.train()

        self.train()
        self.use_adaptive_forward = False

    def forward(self, x, mode="auto", with_latent=False):
        """
        mode:
          - "full": 返回 logits（联邦训练/评估）
          - "partial": 返回指定 inspect_layer_position 的特征
          - "auto": 按 use_adaptive_forward 决定
        """
        if with_latent:
            return self.dftnd_latent_forward(x)

        if mode == "full":
            return super().forward(x)
        elif mode == "partial":
            return self._forward_to_layer(x, self.inspect_layer_position)
        else:
            if self.use_adaptive_forward:
                return self._forward_to_layer(x, self.inspect_layer_position)
            else:
                return super().forward(x)

    def _forward_lsep(self, x: Tensor) -> Tensor:
        # 保持兼容性：默认返回 avgpool 输出 (position=2)
        return self._forward_to_layer(x, 2)

    def _forward_to_layer(self, x: Tensor, position: int) -> Tensor:
        """根据position返回对应层的特征"""
        # 临时切换到eval模式避免BatchNorm的batch_size=1错误
        was_training = self.training
        if was_training:
            self.eval()
        
        if position == 0:
            # 完整模型输出
            result = super().forward(x)
        elif position == 1:
            # 初始卷积块后 (conv1+bn1+relu+maxpool)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            result = x
        elif position == 2:
            # avgpool之后展平 (原 Lsep)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            result = x
        elif position == 3:
            # layer1之后
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            result = x
        elif position == 4:
            # layer2之后
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            result = x
        elif position == 5:
            # layer3之后
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            result = x
        elif position == 6:
            # layer4之后
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            result = x
        else:
            raise ValueError(f"Unsupported position: {position}")
        
        # 恢复原始模式
        if was_training:
            self.train()
        return result

    def dftnd_latent_forward(self, x):
        # 返回 (final logits, pre_out embedding at fc input)
        pre_out = self._forward_lsep(x)
        final = self.fc(pre_out)
        return final, pre_out


# ============================================================
# VGG16（Lsep 固定在“最后一层 fc 之前”）
# 具体为：features → flatten → classifier[0..4]（即 4096 ReLU 后）
# ============================================================
class VGGAdaptivePartialModel(VGG16):
    """
    inspect_layer_position 语义（扩展为7档）：
      0: full logits（完整模型输出）
      1: features 第1个池化层之后 (block1: 64->64->M)
      2: Lsep = 最后一层 fc 之前（classifier[0..4] 之后），FreeEagle 使用
      3: features 第2个池化层之后 (block2: 128->128->M)
      4: features 第3个池化层之后 (block3: 256->256->256->M)
      5: features 第4个池化层之后 (block4: 512->512->512->M)
      6: features 第5个池化层之后 (block5: 512->512->512->M)
    """

    def __init__(self,
                 num_classes=10,
                 inspect_layer_position=2,
                 original_input_img_shape=(1, 3, 32, 32)):

        # 从 original_input_img_shape 提取父类构造所需信息
        in_channels = int(original_input_img_shape[1])
        img_size = int(original_input_img_shape[2])

        super().__init__(num_classes=num_classes,
                         in_channels=in_channels,
                         img_size=img_size)
        self.eval()
        # 七档位：扩展支持所有VGG层位置
        self.inspect_layer_positions = [0, 1, 2, 3, 4, 5, 6]
        if inspect_layer_position not in self.inspect_layer_positions:
            raise ValueError(f"inspect_layer_position must be one of {self.inspect_layer_positions}")
        self.inspect_layer_position = int(inspect_layer_position)

        # 记录不同截断点的 shape
        self.input_shapes = []
        with torch.no_grad():
            # 使用 batch_size=2 避免 BatchNorm 的限制
            batch_size = 2
            dummy_shape = (batch_size,) + original_input_img_shape[1:]
            dummy = torch.randn(*dummy_shape)

            # 临时切换到 eval 模式以安全进行形状推断
            self.features.eval()

            # 0: 输入形状（记录单个样本的形状）
            self.input_shapes.append(original_input_img_shape)

            # 1: features 第1个池化层之后 (block1: 64->64->M, index=6)
            x = dummy
            for i in range(7):  # 包含第一个MaxPool2d (index=6)
                x = self.features[i](x)
            # 记录单个样本的形状
            self.input_shapes.append((1,) + x.shape[1:])

            # 2: Lsep = 最后一层 fc 之前（classifier[0..4] 之后）
            feat = self.features(dummy)
            flat = feat.view(feat.size(0), -1)
            prefc = flat
            for i in range(0, 5):
                prefc = self.classifier[i](prefc)
            self.input_shapes.append((1, int(prefc.shape[1]), 1, 1))

            # 3: features 第2个池化层之后 (block2: 128->128->M, index=13)
            x = dummy
            for i in range(14):  # 包含第二个MaxPool2d (index=13)
                x = self.features[i](x)
            self.input_shapes.append((1,) + x.shape[1:])

            # 4: features 第3个池化层之后 (block3: 256->256->256->M, index=23)
            x = dummy
            for i in range(24):  # 包含第三个MaxPool2d (index=23)
                x = self.features[i](x)
            self.input_shapes.append((1,) + x.shape[1:])

            # 5: features 第4个池化层之后 (block4: 512->512->512->M, index=33)
            x = dummy
            for i in range(34):  # 包含第四个MaxPool2d (index=33)
                x = self.features[i](x)
            self.input_shapes.append((1,) + x.shape[1:])

            # 6: features 第5个池化层之后 (block5: 512->512->512->M, 完整features输出)
            x = self.features(dummy)
            self.input_shapes.append((1,) + x.shape[1:])

            # 恢复训练模式
            self.features.train()
        self.train()
        self.use_adaptive_forward = False

    def forward(self, x, mode="auto", fake_relu=False):
        """
        mode:
          - "full": 返回 logits（联邦训练/评估）
          - "partial": 返回 Lsep embedding（最后一层 fc 前 4096 向量）
          - "auto": 按 use_adaptive_forward 决定
        """
        if mode == "full":
            return super().forward(x)
        elif mode == "partial":
            return self._forward_lsep(x)
        else:
            if self.use_adaptive_forward:
                return self._forward_lsep(x)
            else:
                return super().forward(x)

    def dftnd_latent_forward(self, x, fake_relu):
        # 输出 (final logits, pre_out)，其中 pre_out 就是 Lsep 4096 向量
        x = self.features(x)
        if fake_relu:
            x = FakeReLU.apply(x)
        flat = x.view(x.size(0), -1)

        # 过到最后一层 fc 之前（0..4），得到 4096 向量
        pre_out = flat
        for i in range(0, 5):
            pre_out = self.classifier[i](pre_out)

        final = self.classifier[5](pre_out)  # Dropout
        final = self.classifier[6](final)  # 最后一层 Linear → logits
        return final, pre_out

    def _forward_lsep(self, x):
        # features → flatten → classifier[0..4]，得到 4096 embedding
        # 临时切换到eval模式避免BatchNorm的batch_size=1错误
        was_training = self.training
        if was_training:
            self.eval()
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        for i in range(0, 5):
            x = self.classifier[i](x)
        
        # 恢复原始模式
        if was_training:
            self.train()
        return x

    def _forward_to_layer(self, x, position: int):
        """根据position返回对应层的特征"""
        # 临时切换到eval模式避免BatchNorm的batch_size=1错误
        was_training = self.training
        if was_training:
            self.eval()
        
        if position == 0:
            # 完整模型输出
            result = super().forward(x)
        elif position == 1:
            # features 第1个池化层之后 (block1: 64->64->M, index=6)
            for i in range(7):
                x = self.features[i](x)
            result = x
        elif position == 2:
            # Lsep = 最后一层 fc 之前（classifier[0..4] 之后）
            # 注意：_forward_lsep内部已经处理了eval模式，这里先恢复training再调用
            if was_training:
                self.train()
            result = self._forward_lsep(x)
            # _forward_lsep会恢复原始模式，所以这里直接返回
            return result
        elif position == 3:
            # features 第2个池化层之后 (block2: 128->128->M, index=13)
            for i in range(14):
                x = self.features[i](x)
            result = x
        elif position == 4:
            # features 第3个池化层之后 (block3: 256->256->256->M, index=23)
            for i in range(24):
                x = self.features[i](x)
            result = x
        elif position == 5:
            # features 第4个池化层之后 (block4: 512->512->512->M, index=33)
            for i in range(34):
                x = self.features[i](x)
            result = x
        elif position == 6:
            # features 第5个池化层之后 (block5: 512->512->512->M, 完整features输出)
            x = self.features(x)
            result = x
        else:
            raise ValueError(f"Unsupported position: {position}")
        
        # 恢复原始模式
        if was_training:
            self.train()
        return result


class CNNAdaptivePartialModel(FedAvgCNN):
    """
    为 FedAvgCNN 模型添加的适配器，使其兼容 FreeEagle 防御。

    inspect_layer_position 语义（扩展为7档）：
      0: full logits（完整模型输出）
      1: conv1 之后 (32 channels, 28x28 for CIFAR10)
      2: Lsep = 最后一个 fc 层之前 (fc1之后), FreeEagle 使用
      3: conv1 内部: Conv2d 之后，ReLU 之前
      4: conv1 内部: ReLU 之后，MaxPool2d 之前
      5: conv2 内部: Conv2d 之后，ReLU 之前
      6: conv2 内部: ReLU 之后，MaxPool2d 之前
    """

    def __init__(self,
                 num_classes=10,
                 inspect_layer_position=2,
                 original_input_img_shape=(1, 3, 32, 32)):

        # 从输入形状推断 FedAvgCNN 构造所需的参数
        in_features = int(original_input_img_shape[1])

        # 为了通用性，我们动态计算扁平化后的维度 dim
        with torch.no_grad():
            dummy = torch.randn(*original_input_img_shape)
            # 模拟 conv1 和 conv2 的前向传播来确定输出尺寸
            conv_out = nn.MaxPool2d(kernel_size=(2, 2))(
                nn.ReLU(inplace=True)(
                    nn.Conv2d(32, 64, kernel_size=5)(
                        nn.MaxPool2d(kernel_size=(2, 2))(
                            nn.ReLU(inplace=True)(
                                nn.Conv2d(in_features, 32, kernel_size=5)(dummy)
                            )
                        )
                    )
                )
            )
            dim = conv_out.flatten(1).shape[1]

        # 调用父类构造函数
        super().__init__(in_features=in_features,
                         num_classes=num_classes,
                         dim=dim)

        # 七档位：扩展支持不同卷积层位置
        self.inspect_layer_positions = [0, 1, 2, 3, 4, 5, 6]
        if inspect_layer_position not in self.inspect_layer_positions:
            raise ValueError(f"inspect_layer_position must be one of {self.inspect_layer_positions}")
        self.inspect_layer_position = int(inspect_layer_position)

        # 记录不同截断点的 shape
        self.input_shapes = []
        with torch.no_grad():
            # 使用 batch_size=2 避免 BatchNorm 的限制（如果模型中有 BN 层）
            batch_size = 2
            dummy_shape = (batch_size,) + original_input_img_shape[1:]
            dummy = torch.randn(*dummy_shape)

            # 0: 输入形状（记录单个样本的形状）
            self.input_shapes.append(original_input_img_shape)

            # 计算各层的输出形状
            # 1: conv1 完整输出 (包含 Conv2d + ReLU + MaxPool2d)
            conv1_out = self.conv1(dummy)
            self.input_shapes.append((1,) + conv1_out.shape[1:])

            # 2: fc1 输出 (Lsep层)
            conv2_out = self.conv2(conv1_out)
            flat = torch.flatten(conv2_out, 1)
            fc1_out = self.fc1(flat)
            self.input_shapes.append((1, int(fc1_out.shape[1]), 1, 1))

            # 3-6: conv1和conv2内部各子层的形状（记录单个样本的形状）
            # 需要分解conv1和conv2的内部结构
            # conv1 内部: Conv2d -> ReLU -> MaxPool2d
            conv1_conv = dummy
            for layer in self.conv1:
                conv1_conv = layer(conv1_conv)
                if isinstance(layer, nn.Conv2d):  # 3: Conv2d之后
                    self.input_shapes.append((1,) + conv1_conv.shape[1:])
                elif isinstance(layer, nn.ReLU):  # 4: ReLU之后
                    self.input_shapes.append((1,) + conv1_conv.shape[1:])

            # conv2 内部: Conv2d -> ReLU -> MaxPool2d
            conv2_conv = conv1_out
            for layer in self.conv2:
                conv2_conv = layer(conv2_conv)
                if isinstance(layer, nn.Conv2d):  # 5: Conv2d之后
                    self.input_shapes.append((1,) + conv2_conv.shape[1:])
                elif isinstance(layer, nn.ReLU):  # 6: ReLU之后
                    self.input_shapes.append((1,) + conv2_conv.shape[1:])

        self.use_adaptive_forward = False

    def forward(self, x, mode="auto", fake_relu=False):
        """
        mode:
          - "full": 返回 logits（联邦训练/评估）
          - "partial": 返回指定 inspect_layer_position 的特征
          - "auto": 按 use_adaptive_forward 决定
        """
        if mode == "full":
            return super().forward(x)
        elif mode == "partial":
            return self._forward_to_layer(x, self.inspect_layer_position)
        else:
            if self.use_adaptive_forward:
                return self._forward_to_layer(x, self.inspect_layer_position)
            else:
                return super().forward(x)

    def _forward_lsep(self, x):
        # 保持兼容性：默认返回 fc1 输出 (position=2)
        return self._forward_to_layer(x, 2)

    def _forward_to_layer(self, x, position):
        """根据position返回对应层的特征"""
        if position == 0:
            # 完整模型输出
            return super().forward(x)
        elif position == 1:
            # conv1 完整输出
            return self.conv1(x)
        elif position == 2:
            # fc1 输出 (原 Lsep)
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            return out
        elif position == 3:
            # conv1 内部: Conv2d 之后
            out = x
            out = self.conv1[0](out)  # Conv2d
            return out
        elif position == 4:
            # conv1 内部: ReLU 之后
            out = x
            out = self.conv1[0](out)  # Conv2d
            out = self.conv1[1](out)  # ReLU
            return out
        elif position == 5:
            # conv2 内部: Conv2d 之后
            out = self.conv1(x)
            out = self.conv2[0](out)  # Conv2d
            return out
        elif position == 6:
            # conv2 内部: ReLU 之后
            out = self.conv1(x)
            out = self.conv2[0](out)  # Conv2d
            out = self.conv2[1](out)  # ReLU
            return out
        else:
            raise ValueError(f"Unsupported position: {position}")


# in security/utils/partial_models_adaptive.py

class LeNetAdaptivePartialModel(LeNet):
    """
    为 LeNet 模型添加的适配器，使其兼容 FreeEagle 防御。
    """

    # vvvvvvvvvvvvvv 【修改】 vvvvvvvvvvvvvv
    def __init__(self,
                 num_classes=10,
                 inspect_layer_position=2,
                 original_input_img_shape=(1, 1, 28, 28)):

        super().__init__(num_classes=num_classes)
        # 确保权重被正确初始化
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)

        # 临时切换到评估模式以安全地进行形状推断（处理BatchNorm）
        self.eval()

        # inspect_layer_position 语义更新
        # 0: full output
        # 1: 预留 (conv_params 之后)
        # 2: Lsep = bottleneck 层之前 (卷积展平之后), FreeEagle 使用
        self.inspect_layer_positions = [0, 1, 2]
        self.inspect_layer_position = self.inspect_layer_positions[int(inspect_layer_position)]

        # 记录不同截断点的 shape
        self.input_shapes = []
        with torch.no_grad():
            # 使用 batch_size=2 避免 BatchNorm 的限制
            batch_size = 2
            dummy_shape = (batch_size,) + original_input_img_shape[1:]
            dummy = torch.randn(*dummy_shape)

            # 0: 输入形状（记录单个样本的形状）
            self.input_shapes.append(original_input_img_shape)

            # 经过 conv_params
            conv_out = self.conv_params(dummy)
            self.input_shapes.append((1,) + conv_out.shape[1:])  # 1: 预留, conv_params 输出 shape

            # Lsep 现在是 conv_params 之后，bottleneck 之前
            flat = conv_out.view(conv_out.size(0), -1)
            # 2: Lsep: 用 (1, feature_dim, 1, 1) 形式保存
            feature_dim = flat.shape[1]
            self.input_shapes.append((1, int(feature_dim), 1, 1))

        # 将模型切换回默认的训练模式
        self.train()
        self.use_adaptive_forward = False

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def forward(self, x, mode="auto", with_latent=False):
        """
        mode:
          - "full": 返回 log_softmax（联邦训练/评估）
          - "partial": 返回 Lsep embedding (卷积展平后的向量)
          - "auto": 按 use_adaptive_forward 决定
        """
        if with_latent:
            return self.dftnd_latent_forward(x)

        if mode == "full":
            return super().forward(x)
        elif mode == "partial":
            return self._forward_lsep(x)
        else:
            if self.use_adaptive_forward:
                return self._forward_lsep(x)
            else:
                return super().forward(x)

    # vvvvvvvvvvvvvv 【修改】 vvvvvvvvvvvvvv
    def _forward_lsep(self, x):
        # Lsep 现在在 bottleneck 之前
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)  # 在展平后立即返回
        return x

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def dftnd_latent_forward(self, x):
        # 返回 (final log_softmax, pre_out embedding at fc input)
        # 注意：为保持此函数功能，我们不能调用修改后的 _forward_lsep
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        pre_out = self.dropout(x)  # pre_out 仍然是最后一个fc层之前的 embedding

        final = self.fc(pre_out)
        final = F.log_softmax(final, dim=1)
        return final, pre_out