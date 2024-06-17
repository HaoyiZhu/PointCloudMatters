"""
Reference:
- https://github.com/OpenGVLab/PonderV2/blob/main/ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py
- https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/pointnet.py
"""

from functools import partial

import spconv.pytorch as spconv
import torch
import torch.nn as nn

from src.utils.sparse_tensor_utils import offset2batch


class PointNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.embedding_table = None

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 64, kernel_size=1, bias=False),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 64, kernel_size=1, bias=False),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv3 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 64, kernel_size=1, bias=False),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv4 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 128, kernel_size=1, bias=False),
            norm_fn(128),
            nn.ReLU(),
        )
        self.conv5 = spconv.SparseSequential(
            spconv.SubMConv3d(128, 512, kernel_size=1, bias=False),
            norm_fn(512),
            nn.ReLU(),
        )

        self.final = (
            spconv.SubMConv3d(512, num_classes, kernel_size=1, padding=1, bias=True)
            if num_classes > 0
            else spconv.Identity()
        )
        self.num_channels = num_classes if num_classes > 0 else 512

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final(x)

        return x.features
