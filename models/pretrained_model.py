import pretrainedmodels
import pretrainedmodels.utils
import torch.nn as nn


def get_pretrained_model(model_name="se_resnext50_32x4d", num_classes=101):
    model = pretrainedmodels.__dict__[model_name](pretrained="imagenet")
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    return model
