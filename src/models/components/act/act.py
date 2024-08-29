"""
Reference:
- https://github.com/tonyzhaozh/act
"""

import pointops
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_softmax
except:
    print("[Warning] torch_scatter not installed")

import torchvision.transforms as T
from einops import pack, rearrange, reduce, repeat, unpack

from src.utils import offset2batch
from src.utils.rotation_conversions import matrix_to_quaternion, rotation_6d_to_matrix

from .utils import get_sinusoid_encoding_table, reparametrize


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


class ToTensorIfNot(T.ToTensor):
    def __call__(self, pic):
        if not torch.is_tensor(pic):
            return super().__call__(pic)
        return pic


class ACT(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        hidden_dim,
        num_queries,
        num_cameras,
        action_dim=8,
        qpos_dim=9,
        env_state_dim=0,
        latent_dim=32,
        action_loss=None,
        klloss=None,
        kl_weight=20.0,
        goal_cond_dim=0,
        obs_feature_pos_embedding=None,
        freeze_backbone=False,
        ignore_vae=False,
        pretrained_weight=None,
        feature_mode="cls",
    ):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.encoder = encoder

        self.num_queries = num_queries
        self.num_cameras = num_cameras
        self.action_dim = action_dim
        self.qpos_dim = qpos_dim
        self.env_state_dim = env_state_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim  # final size of latent z
        self.goal_cond_dim = goal_cond_dim
        self.obs_feature_pos_embedding = obs_feature_pos_embedding
        self.freeze_backbone = freeze_backbone
        self.ignore_vae = ignore_vae
        self.feature_mode = feature_mode

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.action_loss = action_loss
        self.klloss = klloss

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        if self.backbone is not None:
            self.input_proj = nn.Conv2d(
                self.backbone.num_channels, self.hidden_dim, kernel_size=1
            )
            self.input_proj_robot_state = nn.Linear(self.qpos_dim, self.hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(self.qpos_dim, self.hidden_dim)
            self.input_proj_env_state = nn.Linear(self.env_state_dim, self.hidden_dim)
            self.pos = nn.Embedding(2, self.hidden_dim)

        self.cls_embed = nn.Embedding(1, self.hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            self.action_dim, self.hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(
            self.qpos_dim, self.hidden_dim
        )  # project qpos to embedding

        self.latent_proj = nn.Linear(
            self.hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var

        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + self.num_queries, self.hidden_dim),
        )  # [CLS], obs_actions, a_seq

        if self.goal_cond_dim > 0:
            self.proj_goal_cond_emb = nn.Linear(self.goal_cond_dim, self.hidden_dim)

    def build_decoder(self):
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, self.hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2 + int(self.goal_cond_dim > 0), self.hidden_dim
        )  # learned position embedding for goal cond (optional), proprio and latent

    def forward_encoder(self, data_dict):
        qpos = data_dict["qpos"]
        actions = data_dict.get("actions", None)
        is_pad = data_dict.get("is_pad", None)

        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        data_dict["is_training"] = is_training

        if is_training and not self.ignore_vae:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:  # test, no tgt actions
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        data_dict["mu"] = mu
        data_dict["logvar"] = logvar
        data_dict["latent_input"] = latent_input

        return data_dict

    def forward_obs_embed(self, data_dict):
        qpos = data_dict["qpos"]
        image = data_dict["image"]
        env_state = data_dict.get("env_state", None)
        actions = data_dict.get("actions", None)
        latent_input = data_dict["latent_input"]

        if self.goal_cond_dim > 0:
            if data_dict["goal_cond"].dim() > 2:
                data_dict["goal_cond"] = data_dict["goal_cond"].reshape(
                    data_dict["goal_cond"].shape[0], -1
                )
            goal_cond = self.proj_goal_cond_emb(data_dict["goal_cond"])

        if self.backbone is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(self.num_cameras):
                if hasattr(self.backbone, "forward_feature_extractor"):
                    features = self.backbone.forward_feature_extractor(
                        self.image_transform(image[:, cam_id]),
                        mode=self.feature_mode,
                    )
                else:
                    features = self.backbone(
                        image[:, cam_id]
                    )  # (b, c, h, w) for resnet or (b, c) for vit
                # print(features.shape)
                if features.dim() == 2:  # vit
                    features = features.unsqueeze(-1).unsqueeze(-1)
                pos = self.obs_feature_pos_embedding(features).to(features.dtype)
                # print(self.input_proj(features))
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            latent_input = latent_input.unsqueeze(0)
            if self.goal_cond_dim > 0:
                goal_cond = goal_cond.unsqueeze(0)

            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos).unsqueeze(0)
            if self.goal_cond_dim > 0:
                proprio_input = torch.cat([proprio_input, goal_cond], axis=0)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            if self.goal_cond_dim <= 0:
                src = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            else:
                src = torch.cat([qpos, env_state, goal_cond], axis=1)
            pos = self.pos.weight
            latent_input = None
            proprio_input = None

        data_dict["src"] = src
        data_dict["pos"] = pos
        data_dict["latent_input"] = latent_input
        data_dict["proprio_input"] = proprio_input

        return data_dict

    def forward_decoder(self, data_dict):
        src = data_dict["src"]
        pos = data_dict["pos"]
        latent_input = data_dict["latent_input"]
        proprio_input = data_dict["proprio_input"]

        # (bs, num_queries, hidden_dim)
        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight if latent_input is not None else None,
        )[0]

        a_hat = self.action_head(hs)  # (bs, num_queries, action_dim)

        is_pad_hat = self.is_pad_head(hs)  # (bs, num_queries, 1)

        data_dict["a_hat"] = a_hat
        data_dict["is_pad_hat"] = is_pad_hat

        return data_dict

    def forward_loss(self, data_dict):
        total_kld = self.klloss(data_dict["mu"], data_dict["logvar"])

        action_loss = self.action_loss(data_dict["a_hat"], data_dict["actions"])
        action_loss = (action_loss * ~data_dict["is_pad"].unsqueeze(-1)).mean()

        data_dict["action_loss"] = action_loss
        data_dict["kl_loss"] = total_kld
        data_dict["loss"] = action_loss + total_kld * self.kl_weight

        return data_dict

    def forward(self, data_dict):
        # obtain latent z from action sequence
        data_dict = self.forward_encoder(data_dict)

        # obtain proprioception and image features
        data_dict = self.forward_obs_embed(data_dict)

        # decode action sequence from proprioception and image features
        data_dict = self.forward_decoder(data_dict)

        if not data_dict["is_training"]:
            return data_dict

        # compute loss
        data_dict = self.forward_loss(data_dict)

        return data_dict


class ACTPCD(ACT):
    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        hidden_dim,
        num_queries,
        num_cameras,
        action_dim=8,
        qpos_dim=9,
        env_state_dim=0,
        latent_dim=32,
        action_loss=None,
        klloss=None,
        kl_weight=20.0,
        goal_cond_dim=0,
        obs_feature_pos_embedding=None,
        freeze_backbone=False,
        pcd_nsample=16,
        pcd_npoints=1024,
        sampling="fps",
        heatmap_th=0.1,
        ignore_vae=False,
        use_mask=False,
        bg_ratio=0.0,
        pre_sample=False,
        in_channels=6,
    ):
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_cameras=0,
            action_dim=action_dim,
            qpos_dim=qpos_dim,
            env_state_dim=env_state_dim,
            latent_dim=latent_dim,
            action_loss=action_loss,
            klloss=klloss,
            kl_weight=kl_weight,
            goal_cond_dim=goal_cond_dim,
            obs_feature_pos_embedding=None,
            freeze_backbone=freeze_backbone,
            ignore_vae=ignore_vae,
        )

        self.input_proj = None

        # build fps sampler
        self.pcd_nsample = pcd_nsample
        self.pcd_npoints = pcd_npoints
        self.pre_sample = pre_sample
        if not pre_sample:
            self.linear = nn.Linear(
                3 + self.backbone.num_channels, hidden_dim, bias=False
            )
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.linear = nn.Linear(
                3 + self.backbone.in_channels, self.backbone.in_channels, bias=False
            )
            self.bn = nn.BatchNorm1d(self.backbone.in_channels)
        self.pool = nn.MaxPool1d(pcd_nsample)

        self.relu = nn.ReLU(inplace=True)
        self.sampling = sampling
        self.use_mask = use_mask
        self.bg_ratio = bg_ratio

    def pcd_sampling(self, pxo, mask=None, return_index=False):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        n_o, count = [self.pcd_npoints], self.pcd_npoints
        for i in range(1, o.shape[0]):
            count += self.pcd_npoints
            n_o.append(count)
        n_o = torch.tensor(n_o, dtype=torch.int32, device=o.device)

        if "fps" in self.sampling:
            if not self.use_mask or mask is None:
                idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            else:
                if self.bg_ratio > 0.0:
                    fg_n_o, fg_count = (
                        [self.pcd_npoints - int(self.pcd_npoints * self.bg_ratio)],
                        self.pcd_npoints - int(self.pcd_npoints * self.bg_ratio),
                    )
                    for i in range(1, o.shape[0]):
                        fg_count += self.pcd_npoints - int(
                            self.pcd_npoints * self.bg_ratio
                        )
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
        else:
            raise NotImplementedError

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
        p, o = n_p, n_o

        if return_index:
            return [p, x, o, idx]
        return [p, x, o]

    def coord_embedding_sine(
        self, coord, temperature=10000, normalize=False, scale=None
    ):
        num_pos_feats = self.hidden_dim // 3
        num_pad_feats = self.hidden_dim - num_pos_feats * 3

        x_embed = coord[:, 0:1]
        y_embed = coord[:, 1:2]
        z_embed = coord[:, 2:3]

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi

        if normalize:
            eps = 1e-6
            x_embed = coord[:, 0] / (coord[:, 0].max() + eps) * scale
            y_embed = coord[:, 1] / (coord[:, 1].max() + eps) * scale
            z_embed = coord[:, 2] / (coord[:, 2].max() + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=coord.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=2
        ).flatten(1)
        pos_z = torch.stack(
            (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=2
        ).flatten(1)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=1)

        pos = torch.cat((pos, torch.zeros_like(pos)[:, :num_pad_feats]), dim=1)
        return pos

    def forward_pcd_embed(self, pcd_dict):
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
            features = self.backbone(pcd_dict)
        else:
            features = self.backbone(pcd_dict)

            if self.use_mask:
                coord, features, offset = self.pcd_sampling(
                    (pcd_dict["coord"], features, pcd_dict["offset"]), pcd_dict["mask"]
                )
            else:
                coord, features, offset = self.pcd_sampling(
                    (pcd_dict["coord"], features, pcd_dict["offset"])
                )

        pcd_pos = self.coord_embedding_sine(coord)
        features = rearrange(
            features,
            "(b n) c -> b c 1 n",
            n=self.pcd_npoints,
        )
        pcd_pos = repeat(
            pcd_pos,
            "(b n) c -> b c 1 n",
            b=features.shape[0],
        )
        return features, pcd_pos

    def forward_obs_embed(self, data_dict):
        qpos = data_dict["qpos"]
        pcd_dict = data_dict["pcds"]
        env_state = data_dict.get("env_state", None)
        actions = data_dict.get("actions", None)
        latent_input = data_dict["latent_input"]

        if self.goal_cond_dim > 0:
            if data_dict["goal_cond"].dim() > 2:
                data_dict["goal_cond"] = data_dict["goal_cond"].reshape(
                    data_dict["goal_cond"].shape[0], -1
                )
            goal_cond = self.proj_goal_cond_emb(data_dict["goal_cond"])

        if self.backbone is not None:
            # Image observation features and position embeddings
            pcd_tokens, pcd_pos = self.forward_pcd_embed(pcd_dict)

            latent_input = latent_input.unsqueeze(0)
            if self.goal_cond_dim > 0:
                goal_cond = goal_cond.unsqueeze(0)

            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos).unsqueeze(0)
            if self.goal_cond_dim > 0:
                proprio_input = torch.cat([proprio_input, goal_cond], axis=0)
            # fold camera dimension into width dimension
            src = pcd_tokens
            pos = pcd_pos
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            if self.goal_cond_dim <= 0:
                src = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            else:
                src = torch.cat([qpos, env_state, goal_cond], axis=1)
            pos = self.pos.weight
            latent_input = None
            proprio_input = None

        data_dict["src"] = src
        data_dict["pos"] = pos
        data_dict["latent_input"] = latent_input
        data_dict["proprio_input"] = proprio_input

        return data_dict


class ACTRLBench(ACT):
    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        hidden_dim,
        num_queries,
        num_cameras,
        action_dim=8,
        qpos_dim=9,
        env_state_dim=0,
        latent_dim=32,
        action_loss=None,
        klloss=None,
        kl_weight=20.0,
        goal_cond_dim=0,
        obs_feature_pos_embedding=None,
        freeze_backbone=False,
        ignore_vae=False,
        rot_type="6d",
        collision=False,
        position_loss_weight=1.0,
    ):
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_cameras=num_cameras,
            action_dim=action_dim,
            qpos_dim=qpos_dim,
            env_state_dim=env_state_dim,
            latent_dim=latent_dim,
            action_loss=action_loss,
            klloss=klloss,
            kl_weight=kl_weight,
            goal_cond_dim=goal_cond_dim,
            obs_feature_pos_embedding=obs_feature_pos_embedding,
            freeze_backbone=freeze_backbone,
            ignore_vae=ignore_vae,
        )

        self.rot_type = rot_type
        self.collision = collision
        self.position_loss_weight = position_loss_weight

    def forward_decoder(self, data_dict):
        src = data_dict["src"]
        pos = data_dict["pos"]
        latent_input = data_dict["latent_input"]
        proprio_input = data_dict["proprio_input"]

        # (bs, num_queries, hidden_dim)
        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight if latent_input is not None else None,
        )[0]

        a_hat = self.action_head(hs)  # (bs, num_queries, action_dim)
        position = a_hat[..., :3]
        if self.collision:
            collision = torch.sigmoid(a_hat[..., -1:])
            gripper = torch.sigmoid(a_hat[..., -2:-1])
            gripper = torch.cat([gripper, collision], dim=-1)
            rot = a_hat[..., 3:-2]
        else:
            gripper = torch.sigmoid(a_hat[..., -1:])
            rot = a_hat[..., 3:-1]

        if not data_dict["is_training"]:
            if self.rot_type == "6d":
                rot = rotation_6d_to_matrix(rot)
                rot = matrix_to_quaternion(rot)
            else:
                raise NotImplementedError

        a_hat = torch.cat([position, rot, gripper], dim=-1)

        is_pad_hat = self.is_pad_head(hs)  # (bs, num_queries, 1)

        data_dict["a_hat"] = a_hat
        data_dict["is_pad_hat"] = is_pad_hat

        return data_dict

    def forward_loss(self, data_dict):
        total_kld = self.klloss(data_dict["mu"], data_dict["logvar"])

        action_loss = self.action_loss(data_dict["a_hat"], data_dict["actions"])
        action_loss[..., :3] = action_loss[..., :3] * self.position_loss_weight
        action_loss = (action_loss * ~data_dict["is_pad"].unsqueeze(-1)).mean()

        data_dict["action_loss"] = action_loss
        data_dict["kl_loss"] = total_kld
        data_dict["loss"] = action_loss + total_kld * self.kl_weight

        return data_dict


class ACTRLBenchPCD(ACTPCD):
    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        hidden_dim,
        num_queries,
        num_cameras,
        action_dim=8,
        qpos_dim=9,
        env_state_dim=0,
        latent_dim=32,
        action_loss=None,
        klloss=None,
        kl_weight=20.0,
        goal_cond_dim=0,
        obs_feature_pos_embedding=None,
        freeze_backbone=False,
        pcd_nsample=16,
        pcd_npoints=1024,
        sampling="fps",
        heatmap_th=0.1,
        ignore_vae=False,
        rot_type="6d",
        collision=False,
        position_loss_weight=1.0,
        use_mask=False,
        bg_ratio=0.0,
    ):
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_cameras=0,
            action_dim=action_dim,
            qpos_dim=qpos_dim,
            env_state_dim=env_state_dim,
            latent_dim=latent_dim,
            action_loss=action_loss,
            klloss=klloss,
            kl_weight=kl_weight,
            goal_cond_dim=goal_cond_dim,
            obs_feature_pos_embedding=None,
            freeze_backbone=freeze_backbone,
            pcd_nsample=pcd_nsample,
            pcd_npoints=pcd_npoints,
            sampling=sampling,
            heatmap_th=heatmap_th,
            ignore_vae=ignore_vae,
            use_mask=use_mask,
            bg_ratio=bg_ratio,
        )

        self.rot_type = rot_type
        self.collision = collision
        self.position_loss_weight = position_loss_weight

        self.use_mask = use_mask
        self.bg_ratio = bg_ratio

    def forward_decoder(self, data_dict):
        src = data_dict["src"]
        pos = data_dict["pos"]
        latent_input = data_dict["latent_input"]
        proprio_input = data_dict["proprio_input"]

        # (bs, num_queries, hidden_dim)
        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight if latent_input is not None else None,
        )[0]

        a_hat = self.action_head(hs)  # (bs, num_queries, action_dim)
        position = a_hat[..., :3]
        if self.collision:
            collision = torch.sigmoid(a_hat[..., -1:])
            gripper = torch.sigmoid(a_hat[..., -2:-1])
            gripper = torch.cat([gripper, collision], dim=-1)
            rot = a_hat[..., 3:-2]
        else:
            gripper = torch.sigmoid(a_hat[..., -1:])
            rot = a_hat[..., 3:-1]

        if not data_dict["is_training"]:
            if self.rot_type == "6d":
                rot = rotation_6d_to_matrix(rot)
                rot = matrix_to_quaternion(rot)
            else:
                raise NotImplementedError

        a_hat = torch.cat([position, rot, gripper], dim=-1)

        is_pad_hat = self.is_pad_head(hs)  # (bs, num_queries, 1)

        data_dict["a_hat"] = a_hat
        data_dict["is_pad_hat"] = is_pad_hat

        return data_dict

    def forward_loss(self, data_dict):
        total_kld = self.klloss(data_dict["mu"], data_dict["logvar"])

        action_loss = self.action_loss(data_dict["a_hat"], data_dict["actions"])
        action_loss[..., :3] = action_loss[..., :3] * self.position_loss_weight
        action_loss = (action_loss * ~data_dict["is_pad"].unsqueeze(-1)).mean()

        data_dict["action_loss"] = action_loss
        data_dict["kl_loss"] = total_kld
        data_dict["loss"] = action_loss + total_kld * self.kl_weight

        return data_dict
