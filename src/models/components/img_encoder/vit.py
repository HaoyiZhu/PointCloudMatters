# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# adapted from:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os
import sys
import urllib
from functools import partial
from os.path import expanduser

import hydra
import numpy as np
import omegaconf
import six
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torchvision.transforms as T
from timm.models.vision_transformer import Block, PatchEmbed, resize_pos_embed

VC1_BASE_NAME = "vc1_vitb"
VC1_LARGE_NAME = "vc1_vitl"
_EAI_VC1_BASE_URL = "https://dl.fbaipublicfiles.com/eai-vc/"


# progress_bar and download_url from
# https://github.com/facebookresearch/Detectron/blob/1809dd41c1ffc881c0d6b1c16ea38d08894f8b6d/detectron/utils/io.py
def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write(
        "  [{}] {}% of {:.1f}MB file  \r".format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write("\n")


def _download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(f"Error downloading model from {_EAI_VC1_BASE_URL}:\n{e}")
        raise
    if six.PY2:
        total_size = response.info().getheader("Content-Length").strip()
    else:
        total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, "wb") as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)

    return bytes_so_far


def download_model_if_needed(ckpt_file):
    # model_base_dir = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
    # )
    # ckpt_file = os.path.join(model_base_dir, ckpt_file)
    if not os.path.isfile(ckpt_file):
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

        model_name = ckpt_file.split("/")[-1]
        model_url = _EAI_VC1_BASE_URL + model_name
        _download_url(model_url, ckpt_file)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self, global_pool=False, use_cls=True, mask_ratio=None, del_head=True, **kwargs
    ):
        super(VisionTransformer, self).__init__(**kwargs)
        if global_pool:
            self.classifier_feature = "global_pool"
        elif use_cls:
            self.classifier_feature = "use_cls_token"
        else:
            self.classifier_feature = "reshape_embedding"

        if del_head:
            del self.head  # don't use prediction head

        if self.classifier_feature == "global_pool":
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        if self.classifier_feature == "reshape_embedding":
            self.final_spatial = int(self.patch_embed.num_patches**0.5)
            self.embed_dim = (
                self.patch_embed.grid_size[0],
                self.patch_embed.grid_size[1],
                kwargs["embed_dim"],
            )

        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def handle_outcome(self, x):
        if self.classifier_feature == "global_pool":
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.classifier_feature == "use_cls_token":
            x = self.norm(x)
            outcome = x[:, 0]  # use cls token
        elif self.classifier_feature == "reshape_embedding":
            x = self.norm(x)
            outcome = reshape_embedding(
                x[:, 1:]
            )  # remove cls token and reshape embedding
        else:
            raise NotImplementedError

        return outcome

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.mask_ratio is not None:
            x, _, _ = self.random_masking(x, mask_ratio=self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(B, -1, -1), x), dim=1)

        x = self.blocks(x)
        return self.handle_outcome(x)

    def forward(self, x):
        return self.forward_features(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedAutoencoderViT(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.in_chans = in_chans
        num_patches = self.patch_embed.num_patches

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss_mask = torch.ones_like(imgs)
        if imgs.shape[1] == 4:
            loss_mask[:, 3:4, :, :] = (imgs[:, 3:4, :, :] > 0).float()

        target = self.patchify(imgs)
        loss_mask = self.patchify(loss_mask)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2 * loss_mask
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        if self.training:
            return dict(loss=loss)
        else:
            pred = self.unpatchify(pred)
            mask = self.unpatchify(
                mask.unsqueeze(-1).repeat(
                    1, 1, self.patch_embed.patch_size[0] ** 2 * self.in_chans
                )
            )
            return dict(loss=loss, pred=pred, mask=mask, imgs=imgs)


class ClipVisionTransformer(VisionTransformer):
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat(
            [
                self.cls_token.squeeze()
                + torch.zeros(B, 1, x.shape[-1], device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed.squeeze().to(x.dtype)
        x = self.norm_pre(x)

        x = self.blocks(x)
        return self.handle_outcome(x)


def reshape_embedding(x):
    N, L, D = x.shape
    H = W = int(L**0.5)
    x = x.reshape(N, H, W, D)
    x = torch.einsum("nhwd->ndhw", x)
    return x


def vit_small_patch16(**kwargs):
    """ViT small as defined in the DeiT paper."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qkv_bias=True,
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qkv_bias=True,
        **kwargs,
    )
    return model


def clip_vit_base_patch16(**kwargs):
    model = ClipVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # CLIP-specific:
        pre_norm=True,
        num_classes=512,
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        qkv_bias=True,
        **kwargs,
    )
    return model


def load_mae_encoder(model, checkpoint_path=None):
    if checkpoint_path is None:
        return model
    else:
        download_model_if_needed(checkpoint_path)

    if not os.path.isabs(checkpoint_path):
        model_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."
        )
        checkpoint_path = os.path.join(model_base_dir, checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(
            state_dict["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

    # filter out keys with name decoder or mask_token
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if "decoder" not in k and "mask_token" not in k
    }

    if model.classifier_feature == "global_pool":
        # remove layer that start with norm
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("norm")}
        # add fc_norm in the state dict from the model
        state_dict["fc_norm.weight"] = model.fc_norm.weight
        state_dict["fc_norm.bias"] = model.fc_norm.bias

    model.load_state_dict(state_dict)
    return model


def load_contrastive_vit(model, checkpoint_path=None, state_dict_key="state_dict"):
    if checkpoint_path is None:
        return model

    old_state_dict = torch.load(checkpoint_path, map_location="cpu")[state_dict_key]
    state_dict = {}
    for k in list(old_state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder") and not k.startswith(
            "module.base_encoder.head"
        ):
            # remove prefix
            state_dict[k[len("module.base_encoder.") :]] = old_state_dict[k]
        # delete renamed or unused k
        del old_state_dict[k]

    if model.classifier_feature == "global_pool":
        # remove layer that start with norm
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("norm")}
        # add fc_norm in the state dict from the model
        state_dict["fc_norm.weight"] = model.fc_norm.weight
        state_dict["fc_norm.bias"] = model.fc_norm.bias

    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(
            state_dict["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )

    model.load_state_dict(state_dict)
    return model


class ToTensorIfNot(T.ToTensor):
    def __call__(self, pic):
        if not torch.is_tensor(pic):
            return super().__call__(pic)
        return pic


class ViT(nn.Module):
    def __init__(self, model_name="vit_base_patch16", channels=3, **kwargs) -> None:
        super().__init__()

        if model_name == "vit_large_patch16":
            self.model = vit_large_patch16(
                img_size=224,
                use_cls=True,
                drop_path_rate=0.0,
            )
            self.num_channels = 1024
        elif model_name == "vit_base_patch16":
            self.model = vit_base_patch16(
                img_size=224,
                use_cls=True,
                drop_path_rate=0.0,
            )
            self.num_channels = 768
        else:
            raise NotImplementedError(f"{model_name} not supported")

        if channels == 1:
            normalization = T.Normalize([0.5], [0.5])
        elif channels == 3:
            normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif channels == 4:
            normalization = T.Normalize(
                [0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5]
            )
        elif channels == 6:
            normalization = T.Normalize(
                [0.485, 0.456, 0.406, 0.5, 0.5, 0.5],
                [0.229, 0.224, 0.225, 0.5, 0.5, 0.5],
            )
        else:
            raise ValueError("Unsupported channel size: {}".format(channels))

        self.image_transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                ToTensorIfNot(),
                normalization,
            ]
        )

        if channels == 4:
            self.model.patch_embed.proj.weight.data = torch.cat(
                [
                    self.model.patch_embed.proj.weight.data,
                    torch.zeros(
                        self.model.patch_embed.proj.weight.data.shape[0],
                        1,
                        *self.model.patch_embed.proj.weight.data.shape[2:],
                    ),  # init depth channel to 0
                ],
                dim=1,
            )
        elif channels == 6:
            self.model.patch_embed.proj.weight.data = torch.cat(
                [
                    self.model.patch_embed.proj.weight.data,
                    torch.zeros(
                        self.model.patch_embed.proj.weight.data.shape[0],
                        3,
                        *self.model.patch_embed.proj.weight.data.shape[2:],
                    ),  # init pointmap channel to 0
                ],
                dim=1,
            )
        elif channels == 1:
            self.model.patch_embed.proj.weight.data = torch.zeros(
                self.model.patch_embed.proj.weight.data.shape[0],
                1,
                *self.model.patch_embed.proj.weight.data.shape[2:],
            )

    def forward(self, x):
        return self.model(self.image_transform(x))


class MAEViT(nn.Module):
    def __init__(
        self,
        model_name="mae_vit_base_patch16",
        channels=3,
        mask_ratio=0.75,
        checkpoint_root: str = os.path.join(expanduser("~"), ".vc1"),
        **kwargs,
    ) -> None:
        super().__init__()

        if model_name == "mae_vit_large_patch16":
            self.model = mae_vit_large_patch16(
                img_size=224,
                use_cls=True,
                drop_path_rate=0.0,
                in_chans=channels,
            )
            self.num_channels = 1024
        elif model_name == "mae_vit_base_patch16":
            self.model = mae_vit_base_patch16(
                img_size=224,
                use_cls=True,
                drop_path_rate=0.0,
                in_chans=channels,
            )
            self.num_channels = 768
        else:
            raise NotImplementedError(f"{model_name} not supported")

        self.image_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.RandomHorizontalFlip(),
                ToTensorIfNot(),
                (
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    if channels == 3
                    else T.Normalize(
                        [0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5]
                    )
                ),
            ]
        )

        self.mask_ratio = mask_ratio

        checkpoint_path = os.path.join(
            checkpoint_root,
            "vc1_vitl.pth" if model_name == "mae_vit_large_patch16" else "vc1_vitb.pth",
        )
        download_model_if_needed(checkpoint_path)

        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        if state_dict["pos_embed"].shape != self.model.pos_embed.shape:
            state_dict["pos_embed"] = resize_pos_embed(
                state_dict["pos_embed"],
                self.model.pos_embed,
                getattr(self.model, "num_tokens", 1),
                self.model.patch_embed.grid_size,
            )

        # # filter out keys with name decoder or mask_token
        # state_dict = {
        #     k: v
        #     for k, v in state_dict.items()
        #     if "decoder" not in k and "mask_token" not in k
        # }

        if self.model.classifier_feature == "global_pool":
            # remove layer that start with norm
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("norm")
            }
            # add fc_norm in the state dict from the model
            state_dict["fc_norm.weight"] = self.model.fc_norm.weight
            state_dict["fc_norm.bias"] = self.model.fc_norm.bias

        if channels == 4:
            state_dict["patch_embed.proj.weight"] = torch.cat(
                [
                    state_dict["patch_embed.proj.weight"],
                    torch.zeros(
                        state_dict["patch_embed.proj.weight"].shape[0],
                        1,
                        *state_dict["patch_embed.proj.weight"].shape[2:],
                    ),
                ],
                dim=1,
            )

        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = x["images"]
        return self.model(self.image_transform(x), mask_ratio=self.mask_ratio)


class VC1ViT(ViT):
    def __init__(
        self,
        checkpoint_root: str = os.path.join(expanduser("~"), ".vc1"),
        model_name="vit_base_patch16",
        channels=3,
        **kwargs,
    ) -> None:
        super().__init__(model_name=model_name, channels=channels, **kwargs)

        checkpoint_path = os.path.join(
            checkpoint_root,
            "vc1_vitl.pth" if model_name == "vit_large_patch16" else "vc1_vitb.pth",
        )
        download_model_if_needed(checkpoint_path)

        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        if state_dict["pos_embed"].shape != self.model.pos_embed.shape:
            state_dict["pos_embed"] = resize_pos_embed(
                state_dict["pos_embed"],
                self.model.pos_embed,
                getattr(self.model, "num_tokens", 1),
                self.model.patch_embed.grid_size,
            )

        # filter out keys with name decoder or mask_token
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "decoder" not in k and "mask_token" not in k
        }

        if self.model.classifier_feature == "global_pool":
            # remove layer that start with norm
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("norm")
            }
            # add fc_norm in the state dict from the model
            state_dict["fc_norm.weight"] = self.model.fc_norm.weight
            state_dict["fc_norm.bias"] = self.model.fc_norm.bias

        self.model.load_state_dict(state_dict)

        if channels == 4:
            self.model.patch_embed.proj.weight.data = torch.cat(
                [
                    self.model.patch_embed.proj.weight.data,
                    torch.zeros(
                        self.model.patch_embed.proj.weight.data.shape[0],
                        1,
                        *self.model.patch_embed.proj.weight.data.shape[2:],
                    ),
                ],
                dim=1,
            )


if __name__ == "__main__":
    model = VC1ViT(
        # model_name="vit_large_patch16"
        model_name="vit_base_patch16"
    )
    print(model)
    print(model(torch.randn(1, 3, 345, 456)).shape)
