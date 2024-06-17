import copy
from typing import Dict, List, Tuple, Union

import pointops
import torch
import torch.nn as nn
import torchvision
from einops import pack, rearrange, reduce, repeat, unpack

from src.utils.diffusion_policy import ModuleAttrMixin
from src.utils.pytorch_utils import dict_apply


class PCDObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        pcd_model: Union[nn.Module, Dict[str, nn.Module]],
        share_pcd_model: bool = True,
        n_obs_step: int = 2,
        pcd_nsample: int = 16,
        pcd_npoints: int = 1024,
        use_mask: bool = False,
        bg_ratio: float = 0.0,
        pcd_hidden_dim: int = 128,
        projector_layers: int = 2,
        projector_channels: List[int] = [128, 128, 128],
        pre_sample=False,
        in_channel=6,
        **kwargs,
    ):
        super().__init__()
        pcd_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_pcd_model:
            assert isinstance(pcd_model, nn.Module)
            key_model_map["pcd"] = pcd_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "pcd":
                pcd_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_pcd_model:
                    if isinstance(pcd_model, dict):
                        # have provided model for each key
                        this_model = pcd_model[key]
                    else:
                        assert isinstance(pcd_model, nn.Module)
                        # have a copy of the pcd model
                        this_model = copy.deepcopy(pcd_model)

                if this_model is not None:
                    key_model_map[key] = this_model

            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        pcd_keys = sorted(pcd_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.share_pcd_model = share_pcd_model
        self.pcd_keys = pcd_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.n_obs_step = n_obs_step
        self.use_mask = use_mask

        self.pre_sample = pre_sample

        # build fps sampler
        self.pcd_nsample = pcd_nsample
        self.pcd_npoints = pcd_npoints
        if not self.pre_sample:
            self.linear = nn.Linear(
                3 + pcd_model.num_channels, pcd_hidden_dim, bias=False
            )
            self.bn = nn.BatchNorm1d(pcd_hidden_dim)
        else:
            self.linear = nn.Linear(3 + in_channel, in_channel, bias=False)
            self.bn = nn.BatchNorm1d(in_channel)

        self.pool = nn.MaxPool1d(pcd_nsample)
        self.relu = nn.ReLU(inplace=True)
        self.use_mask = use_mask
        self.bg_ratio = bg_ratio

        # build projector
        projector = []
        for i in range(projector_layers):
            if i > 0 or (not self.pre_sample):
                projector.append(
                    nn.Conv1d(pcd_hidden_dim, projector_channels[i], kernel_size=1)
                )
            else:
                projector.append(
                    nn.Conv1d(
                        pcd_model.num_channels, projector_channels[i], kernel_size=1
                    )
                )
            projector.append(nn.BatchNorm1d(projector_channels[i]))
            projector.append(nn.ReLU(inplace=True))
        projector.append(nn.MaxPool1d(pcd_npoints))
        projector.append(
            nn.Conv1d(projector_channels[i], projector_channels[i + 1], kernel_size=1)
        )
        projector.append(nn.BatchNorm1d(projector_channels[i + 1]))
        self.projector = nn.Sequential(*projector)
        self.projector_channels = projector_channels

    def pcd_sampling(self, pxo, mask=None, return_index=False):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        n_o, count = [self.pcd_npoints], self.pcd_npoints
        for i in range(1, o.shape[0]):
            count += self.pcd_npoints
            n_o.append(count)
        n_o = torch.tensor(n_o, dtype=torch.int32, device=o.device)
        if not self.use_mask or mask is None:
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
        else:
            if self.bg_ratio > 0.0:
                fg_n_o, fg_count = (
                    [self.pcd_npoints - int(self.pcd_npoints * self.bg_ratio)],
                    self.pcd_npoints - int(self.pcd_npoints * self.bg_ratio),
                )
                for i in range(1, o.shape[0]):
                    fg_count += self.pcd_npoints - int(self.pcd_npoints * self.bg_ratio)
                    fg_n_o.append(fg_count)
                fg_n_o = torch.tensor(fg_n_o, dtype=torch.int32, device=o.device)
                bg_n_o, bg_count = (
                    [int(self.pcd_npoints * self.bg_ratio)],
                    int(self.pcd_npoints * self.bg_ratio),
                )
                for i in range(1, o.shape[0]):
                    bg_count += int(self.pcd_npoints * self.bg_ratio)
                    bg_n_o.append(bg_count)
                bg_n_o = torch.tensor(bg_n_o, dtype=torch.int32, device=o.device)
            else:
                fg_n_o = n_o

            fg_p = p[mask]
            fg_o = []
            count = 0
            curr = 0
            for i in range(o.shape[0]):
                count += mask[curr : o[i]].sum().item()
                curr = o[i]
                fg_o.append(count)
            fg_o = torch.tensor(fg_o, dtype=torch.int32, device=o.device)
            fg_idx = pointops.farthest_point_sampling(fg_p, fg_o, fg_n_o)  # (m)
            if self.bg_ratio > 0.0:
                bg_p = p[~mask]
                bg_o = []
                count = 0
                curr = 0
                for i in range(o.shape[0]):
                    count += (~mask[curr : o[i]]).sum().item()
                    curr = o[i]
                    bg_o.append(count)
                bg_o = torch.tensor(bg_o, dtype=torch.int32, device=o.device)
                bg_idx = pointops.farthest_point_sampling(bg_p, bg_o, bg_n_o)
                idx = torch.cat([fg_idx, bg_idx], dim=0)
            else:
                idx = fg_idx

        n_p = p[idx.long(), :]  # (m, 3)
        x, _ = pointops.knn_query_and_group(
            x,
            p,
            offset=o,
            new_xyz=n_p,
            new_offset=n_o,
            nsample=self.pcd_nsample,
            with_xyz=True,
        )

        x = self.relu(
            self.bn(self.linear(x).transpose(1, 2).contiguous())
        )  # (m, c, nsample)
        x = self.pool(x).squeeze(-1)  # (m, c)

        if return_index:
            return n_p, x, n_o, idx

        return x

    def encode_pcd(self, pcd_model, pcd_dict):
        if self.pre_sample:
            features = pcd_dict["feat"]
            if self.use_mask:
                coord, features, offset, idx = self.pcd_sampling(
                    (pcd_dict["coord"], features, pcd_dict["offset"]),
                    pcd_dict["mask"],
                    return_index=True,
                )
            else:
                coord, features, offset, idx = self.pcd_sampling(
                    (pcd_dict["coord"], features, pcd_dict["offset"]),
                    return_index=True,
                )
            pcd_dict["coord"] = coord
            pcd_dict["feat"] = features
            pcd_dict["offset"] = offset
            pcd_dict["grid_coord"] = pcd_dict["grid_coord"][idx.long()]
            x = pcd_model(pcd_dict)
        else:
            features = pcd_model(pcd_dict)

            if self.use_mask:
                x = self.pcd_sampling(
                    (pcd_dict["coord"], features, pcd_dict["offset"]), pcd_dict["mask"]
                )
            else:
                x = self.pcd_sampling((pcd_dict["coord"], features, pcd_dict["offset"]))

        x = rearrange(
            x,
            "(b n) c -> b c n",
            n=self.pcd_npoints,
        )
        x = self.projector(x)
        x = rearrange(
            x,
            "b c 1-> b c",
        )
        return x

    def forward(self, obs_dict):
        batch_size = None
        features = list()

        # pass all pcd obs to pcd model
        for key in self.pcd_keys:
            pcd = obs_dict[key]
            if batch_size is None:
                assert (len(pcd["offset"]) % self.n_obs_step) == 0, (
                    len(pcd["offset"]),
                    self.n_obs_step,
                )
                batch_size = len(pcd["offset"])
            assert pcd["feat"].shape[1:] == self.key_shape_map[key]

            feature = self.encode_pcd(
                (
                    self.key_model_map["pcd"]
                    if self.share_pcd_model
                    else self.key_model_map[key]
                ),
                pcd,
            )
            assert feature.shape[0] == batch_size, (feature.shape, batch_size)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0], (key, batch_size, data.shape)
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self):
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        features = [
            torch.zeros((batch_size, (self.projector_channels[-1])), device=self.device)
        ]
        for key in self.low_dim_keys:
            shape = tuple(obs_shape_meta[key]["shape"])
            this_obs = torch.zeros(
                (batch_size,) + shape, dtype=self.dtype, device=self.device
            )
            features.append(this_obs)
        result = torch.cat(features, dim=-1)
        return result.shape[1:]
