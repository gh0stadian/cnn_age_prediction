import pretrainedmodels
import pretrainedmodels.utils
import torchvision
import torch.nn as nn

from .custom_models.baseline_model import BaselineModel, ConvModel


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
