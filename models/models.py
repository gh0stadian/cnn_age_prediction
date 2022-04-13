import pretrainedmodels
import pretrainedmodels.utils
import torchvision
import torch.nn as nn

from .custom_models.adaptive_model import AdaptiveModel
from .custom_models.baseline_model import ConvModel
from .custom_models.double_conv_model import DoubleConvModel
from .custom_models.resnet_model import ResNet, Bottleneck


def get_pretrained_model_resnet50(num_classes=1):
    model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained="imagenet")
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def get_pretrained_model_resnet18(num_classes=1):
    model = torchvision.models.resnet18(pretrained=True)
    dim_feats = model._modules['fc'].in_features
    model._modules['fc'] = nn.Linear(dim_feats, num_classes, bias=True)
    return model


def get_base_model(conv_layers, conv_kernels, fc_layers, num_classes=1):
    if len(conv_layers) != 0:
        model = ConvModel(conv_layers=conv_layers,
                          in_channels=3,
                          kernel_sizes=conv_kernels,
                          lin_layers=fc_layers,
                          num_classes=num_classes
                          )
    else:
        raise ValueError('Layers cannot be empty')
    return model


def get_double_conv(conv_layers, conv_kernels, fc_layers, num_classes=1):
    if len(conv_layers) != 0:
        model = DoubleConvModel(conv_layers=conv_layers,
                                in_channels=3,
                                kernel_sizes=conv_kernels,
                                lin_layers=fc_layers,
                                num_classes=num_classes
                                )
    else:
        raise ValueError('Layers cannot be empty')
    return model


def get_resnet():
    # return ResNet(BasicBlock, [2, 2, 2, 2])  # RESNET18
    # return ResNet(BasicBlock, [3, 4, 6, 3])  # RESNET34
    return ResNet(Bottleneck, [3, 4, 6, 3], 1, 3)


def get_adaptive_model(conv_layers, conv_kernels, fc_layers, num_classes=1):
    if len(conv_layers) != 0:
        model = AdaptiveModel(conv_layers=conv_layers,
                              in_channels=3,
                              kernel_sizes=conv_kernels,
                              lin_layers=fc_layers,
                              num_classes=num_classes
                              )
    else:
        raise ValueError('Layers cannot be empty')
    return model
