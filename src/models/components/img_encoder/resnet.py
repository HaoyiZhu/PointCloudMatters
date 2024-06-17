from __future__ import annotations

"""
Reference:
- https://github.com/facebookresearch/r3m
"""
import copy
import math
import os
from collections import OrderedDict
from os.path import expanduser
from typing import Callable, Dict, List, Optional, Tuple, Union

import gdown
import hydra
import omegaconf
import timm
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import src.utils as U

log = U.RankedLogger(__name__, rank_zero_only=True)


def positional_encoding_2d(height, width, channels):
    row_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width)
    col_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1)

    encoded_pos = torch.cat((row_pos.unsqueeze(2), col_pos.unsqueeze(2)), dim=2).float()

    encoded_pos[:, :, 0] = torch.sin(
        encoded_pos[:, :, 0] / math.pow(10000, 2 * 2 // height)
    )
    encoded_pos[:, :, 1] = torch.cos(
        encoded_pos[:, :, 1] / math.pow(10000, 2 * 2 // width)
    )

    encoded_pos = encoded_pos.permute(2, 0, 1).unsqueeze(0)
    encoded_pos = encoded_pos.repeat(1, channels // 2, 1, 1)

    return encoded_pos


class ResNetTorchVision(nn.Module):
    """
    ResNet based on torchvision
    """

    def __init__(
        self,
        resnet_model="resnet50",
        pretrained=True,
        channels=3,
        avg_pool=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.resnet = eval(f"torchvision.models.{resnet_model}")(pretrained=pretrained)
        if channels == 4:  # rgbd
            rgbd_conv1 = nn.Conv2d(
                4, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            rgbd_conv1.weight.data[:, :3, :, :] = self.resnet.conv1.weight.data
            rgbd_conv1.weight.data[:, -1, :, :] = 0
            self.resnet.conv1 = rgbd_conv1
        elif channels == 6:  # pointmap
            pointmap_conv1 = nn.Conv2d(
                6, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            pointmap_conv1.weight.data[:, :3, :, :] = self.resnet.conv1.weight.data
            pointmap_conv1.weight.data[:, -3:, :, :] = 0
            self.resnet.conv1 = pointmap_conv1

        if channels == 1:  # depth
            depth_conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            depth_conv1.weight.data[:, -1, :, :] = 0
            self.resnet.conv1 = depth_conv1

        self.resnet.fc = nn.Identity()

        if "resnet18" in resnet_model:
            self.num_channels = 512
        elif "resnet34" in resnet_model:
            self.num_channels = 512
        elif "resnet50" in resnet_model:
            self.num_channels = 2048
        else:
            raise NotImplementedError(resnet_model)

        if channels == 3:
            self.normlayer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif channels == 4:
            self.normlayer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]
            )
        elif channels == 6:
            self.normlayer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
                std=[0.229, 0.224, 0.225, 0.5, 0.5, 0.5],
            )
        elif channels == 1:
            self.normlayer = transforms.Normalize(mean=[0.5], std=[0.5])

        self.avg_pool = avg_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            preprocess = nn.Sequential(
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )

        x = preprocess(x)

        if not self.avg_pool:
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
        else:
            x = self.resnet(x)

        return x


class R3MResNet(ResNetTorchVision):
    download_urls: Dict[str, str] = {
        "resnet50": "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA",
        "resnet34": "https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE",
        "resnet18": "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-",
    }

    def __init__(
        self,
        resnet_model: str = "resnet50",
        r3m_cache_root: str = os.path.join(expanduser("~"), ".r3m"),
        channels=3,
        avg_pool=False,
        **kwargs,
    ) -> None:
        super().__init__(
            resnet_model=resnet_model, channels=3, pretrained=False, avg_pool=avg_pool
        )
        self.avg_pool = avg_pool
        self.r3m_cache_root = r3m_cache_root
        modelpath = os.path.join(
            self.r3m_cache_root,
            resnet_model.lower().replace("resnet", "r3m_"),
            "model.pt",
        )
        if not os.path.exists(modelpath):
            self.download_r3m(modelpath, resnet_model)

        self.load_r3m_weights(modelpath)

        if channels == 4:  # rgbd
            rgbd_conv1 = nn.Conv2d(
                4, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            rgbd_conv1.weight.data[:, :3, :, :] = self.resnet.conv1.weight.data
            rgbd_conv1.weight.data[:, -1, :, :] = 0  # init depth channel to 0
            self.resnet.conv1 = rgbd_conv1

    def download_r3m(self, modelpath: str, resnet_model: str) -> None:
        log.info(f"Downloading R3M {modelpath} model to {self.r3m_cache_root}...")

        modelurl = self.download_urls[resnet_model]
        os.makedirs(os.path.dirname(modelpath), exist_ok=True)

        if not os.path.exists(modelpath):
            gdown.download(modelurl, modelpath, quiet=False)

    def load_r3m_weights(self, modelpath: str) -> None:
        log.info(f"Loading R3M weights from {modelpath}...")
        r3m_state_dict = torch.load(modelpath, map_location="cpu")["r3m"]

        state_dict = OrderedDict()
        for k, v in r3m_state_dict.items():
            if k.startswith("module.convnet"):
                state_dict[k[15:]] = v

        self.resnet.load_state_dict(state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            preprocess = nn.Sequential(
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )

        x = preprocess(x)

        if not self.avg_pool:
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            return x
        else:
            return self.resnet(x)


if __name__ == "__main__":
    r3m_resnet = R3MResNet(resnet_model="resnet50")
    x = torch.randn((2, 3, 224, 224))
    y = r3m_resnet(x)
    print(y.shape)
