import pretrainedmodels
import pretrainedmodels.utils
import torchvision
import torch.nn as nn

from .custom_models.classification_model import ClassificationMLPModel, ClassificationZeroHModel
from .custom_models.baseline_model import BaselineModel


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


def get_baseline_model(conv_1_size=8, conv_2_size=16, conv_3_size=32, conv_4_size=64,
                       lin_1_size=64, lin_2_size=32, num_classes=1, zero_h=True
                       ):
    feature_size = ((224 / 2 / 2 / 2 / 2) ** 2) * conv_4_size
    if zero_h:
        classification_model = ClassificationZeroHModel(features=int(feature_size),
                                                        classes=num_classes
                                                        )
    else:
        classification_model = ClassificationMLPModel(features=int(feature_size),
                                                      layer_1=lin_1_size,
                                                      layer_2=lin_2_size,
                                                      classes=num_classes
                                                      )
    model = BaselineModel(classification_model, conv_1_size, conv_2_size, conv_3_size, conv_4_size, kernel_size=(3, 3))
    return model
