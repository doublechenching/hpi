#encoding: utf-8
from pretrainedmodels.models import bninception, xception
from pretrainedmodels.models import inceptionresnetv2
from pretrainedmodels.models.inceptionresnetv2 import BasicConv2d
from pretrainedmodels.models.xception import SeparableConv2d
from torch import nn
from pretrainedmodels.models import dpn92, dpn68b
from pretrainedmodels.models.dpn import DPN
from config import config as cfg
import torch

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

def Flatten():
    "Flattens `x` to a single dimension, often used at the end of a model."
    return Lambda(lambda x: x.view((x.size(0), -1)))


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or (1, 1)
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def InceptionV1():
    """Inception V1 or GoogleNet + BatchNormalization"""
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(cfg.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(nn.BatchNorm1d(1024),
                                      nn.Dropout(0.5),
                                      nn.Linear(1024, cfg.num_classes))

    return model

def Xception():
    model = xception(pretrained="imagenet")
    model.conv4 = SeparableConv2d(1536, 64, 3, 1, 1)
    model.bn4 = nn.BatchNorm2d(64)
    model.fc = nn.Linear(64, cfg.num_classes)

    return model

def InceptionResNetV2():
    """"Inception ResNetV2"""
    model = inceptionresnetv2(pretrained="imagenet")
    model.conv2d_7b = BasicConv2d(2080, 128, 1, 1)
    model.last_linear = nn.Linear(128, cfg.num_classes)

    return model

def DPN92():
    model = dpn92(pretrained="imagenet+5k")
    in_chs = model.last_linear.in_channels
    model.last_linear = nn.Conv2d(in_chs, cfg.num_classes, kernel_size=1, bias=True)

    return model


def NewDPN92():
    def logits(self, features):
        return self.last_linear(features)
    model = dpn92(pretrained="imagenet+5k")
    model.logits = logits.__get__(model, DPN)
    in_chs = model.last_linear.in_channels
    layers = [nn.Conv2d(in_chs, 512, kernel_size=1, bias=True)]
    layers += [AdaptiveConcatPool2d(), Flatten()]
    layers += [nn.BatchNorm1d(512*2)]
    layers += [nn.Dropout(p=0.3)]
    layers += [nn.Linear(1024, 512)]
    layers += [nn.ReLU()]
    layers += [nn.BatchNorm1d(512)]
    layers += [nn.Dropout(p=0.3)]
    layers += [nn.Linear(512, cfg.num_classes)]
    model.last_linear = nn.Sequential(*layers)

    return model


def DPN68():
    model = dpn68b(pretrained="imagenet+5k")
    in_chs = model.last_linear.in_channels
    model.last_linear = nn.Conv2d(in_chs, cfg.num_classes, kernel_size=1, bias=True)

    return model


def get_model():
    all_model = {}
    all_model['inceptionv1'] = InceptionV1
    all_model['inceptionresnetv2'] = InceptionResNetV2
    all_model['dpn68'] = DPN68
    all_model['dpn92'] = DPN92
    all_model['xception'] = Xception

    return all_model

