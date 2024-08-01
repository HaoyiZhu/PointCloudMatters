import math
import os
import random
from collections import defaultdict
from pickle import UnpicklingError
from time import time
from typing import Any, Dict, List, Tuple

import einops
import numpy as np
import torch
from torch.utils.data import Dataset

import src.utils as U
from src.data.components.rlbench.constants import SCENE_BOUNDS, loc_bounds
from src.data.components.transformpcd import ComposePCD
from src.utils.diffusion_policy import LinearNormalizer, SingleFieldLinearNormalizer
from src.utils.rotation_conversions import matrix_to_rotation_6d, quaternion_to_matrix

from .rlbench_single_task_act import (
    RLBenchSingleTaskACTPCDDataset,
    RLBenchSingleTaskACTRGBDDataset,
)

log = U.RankedLogger(__name__, rank_zero_only=True)


class RLBenchSingleTaskDiffusionPolicyRGBDDataset(RLBenchSingleTaskACTRGBDDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.obs_keys = ["qpos"]
        for cam_name in self.cameras:
            self.obs_keys.append(f"{cam_name}_rgb")
            if self.include_depth:
                self.obs_keys.append(f"{cam_name}_depth")

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        for k in self.obs_keys:
            normalizer[k] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __getitem__(self, idx):
        idx %= len(self.episodes)
        if not self.cache_episode:
            task, episode = self.episodes[idx]
            episode = U.io_utils.load_numpy_pickle(episode, backend=self.backend)
        else:
            task, episode = self.episodes[idx]

        demo, goal_cond = episode["demo"], episode["task_goal"]
        episode_len = len(demo)
        start_ts = np.random.choice(episode_len - 1)

        return_dict = dict()
        obs_dict = dict()

        obs = demo[start_ts : start_ts + self.chunk_size]
        # breakpoint()
        if not self.collision:
            qpos = np.stack(
                [
                    np.concatenate([obs_["gripper_pose"], [obs_["gripper_open"]]])
                    for obs_ in obs
                ]
            )
        else:
            qpos = np.stack(
                [
                    np.concatenate(
                        [
                            obs_["gripper_pose"],
                            [obs_["gripper_open"]],
                            [obs_["ignore_collisions"]],
                        ]
                    )
                    for obs_ in obs
                ]
            )

        if len(qpos) < self.chunk_size:
            qpos = np.pad(
                qpos,
                [[0, self.chunk_size - len(qpos)]]
                + [[0, 0] for _ in range(qpos.ndim - 1)],
                mode="edge",
            )
        qpos = torch.from_numpy(qpos).float()
        for camera_name in self.cameras:
            rgb = np.stack(
                [obs_[f"{camera_name}_rgb"].astype(float) / 255.0 for obs_ in obs]
            )
            if len(rgb) < self.chunk_size:
                rgb = np.pad(
                    rgb,
                    [[0, self.chunk_size - len(rgb)]]
                    + [[0, 0] for _ in range(rgb.ndim - 1)],
                    mode="edge",
                )
            obs_dict[f"{camera_name}_rgb"] = torch.einsum(
                "k h w c -> k c h w", torch.from_numpy(rgb).float()
            )
            if self.include_depth:
                depth = np.stack(
                    [
                        obs_[f"{camera_name}_depth"].astype(float)[:, :, None]
                        for obs_ in obs
                    ]
                )
                if len(depth) < self.chunk_size:
                    depth = np.pad(
                        depth,
                        [[0, self.chunk_size - len(depth)]]
                        + [[0, 0] for _ in range(depth.ndim - 1)],
                        mode="edge",
                    )
                obs_dict[f"{camera_name}_depth"] = torch.einsum(
                    "k h w c -> k c h w", torch.from_numpy(depth).float()
                )

        if not self.collision:
            action = np.stack(
                [
                    np.concatenate([d["gripper_pose"], [d["gripper_open"]]])
                    for d in demo[start_ts : start_ts + self.chunk_size]
                ]
            )
        else:
            action = np.stack(
                [
                    np.concatenate(
                        [
                            d["gripper_pose"],
                            [d["gripper_open"]],
                            [d["ignore_collisions"]],
                        ]
                    )
                    for d in demo[start_ts : start_ts + self.chunk_size]
                ]
            )
        if len(action) < self.chunk_size:
            action = np.pad(
                action,
                [[0, self.chunk_size - len(action)]]
                + [[0, 0] for _ in range(action.ndim - 1)],
                mode="edge",
            )

        action = torch.from_numpy(action).float()

        pos_min = torch.FloatTensor(loc_bounds[task][0])
        pos_max = torch.FloatTensor(loc_bounds[task][1])
        qpos[..., :3] = (qpos[..., :3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
        action[..., :3] = (action[..., :3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

        qpos[..., 3:7] = torch.nn.functional.normalize(qpos[..., 3:7], dim=-1)
        action[..., 3:7] = torch.nn.functional.normalize(action[..., 3:7], dim=-1)

        if self.rot_type == "6d":
            qpos_rot = matrix_to_rotation_6d(quaternion_to_matrix(qpos[..., 3:7]))
            action_rot = matrix_to_rotation_6d(quaternion_to_matrix(action[..., 3:7]))
        else:
            raise NotImplementedError

        qpos = torch.cat([qpos[..., :3], qpos_rot, qpos[..., 7:]], dim=-1)
        obs_dict["qpos"] = qpos.float()
        action = torch.cat([action[..., :3], action_rot, action[..., 7:]], dim=-1)
        return_dict["obs"] = obs_dict
        return_dict["action"] = action.float()
        return_dict["goal"] = dict(task_emb=goal_cond.reshape(-1))

        return return_dict


class RLBenchSingleTaskDiffusionPolicyPCDDataset(RLBenchSingleTaskACTPCDDataset):
    def __init__(
        self,
        n_obs_steps=2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_obs_steps = n_obs_steps
        self.obs_keys = ["qpos", "pcds"]

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        for k in self.obs_keys:
            normalizer[k] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __getitem__(self, idx):
        idx %= len(self.episodes)
        try:
            if not self.cache_episode:
                task, episode = self.episodes[idx]
                episode = U.io_utils.load_numpy_pickle(episode, backend=self.backend)
            else:
                task, episode = self.episodes[idx]
        except:
            print(os.path.join(self.root, self.episodes[idx]))
            raise

        demo, goal_cond = episode["demo"], episode["task_goal"]
        episode_len = len(demo)
        start_ts = np.random.choice(episode_len - 1)

        return_dict = dict()
        obs_dict = dict()

        # get observation at start_ts only
        obs = demo[start_ts : start_ts + self.chunk_size]
        if not self.collision:
            qpos = np.stack(
                [
                    np.concatenate([obs_["gripper_pose"], [obs_["gripper_open"]]])
                    for obs_ in obs
                ]
            )
        else:
            qpos = np.stack(
                [
                    np.concatenate(
                        [
                            obs_["gripper_pose"],
                            [obs_["gripper_open"]],
                            [obs_["ignore_collisions"]],
                        ]
                    )
                    for obs_ in obs
                ]
            )
        if len(qpos) < self.chunk_size:
            qpos = np.pad(
                qpos,
                [[0, self.chunk_size - len(qpos)]]
                + [[0, 0] for _ in range(qpos.ndim - 1)],
                mode="edge",
            )
        qpos = torch.from_numpy(qpos).float()

        obs_pcds = []
        for obs_step_idx in range(self.n_obs_steps):
            if obs_step_idx >= len(obs):
                assert len(obs_pcds), (
                    obs_step_idx,
                    start_ts,
                    len(obs),
                )
                obs_pcds.append(obs_pcds[-1])
            else:
                obs_ = obs[obs_step_idx]
                colors, coords = [], []
                if self.use_mask:
                    masks = []
                for camera_name in self.cameras:
                    colors.append(obs_[f"{camera_name}_rgb"].astype(float))
                    coords.append(obs_[f"{camera_name}_point_cloud"].astype(float))
                    if self.use_mask:
                        masks.append(obs_[f"{camera_name}_mask"].astype(float))

                colors = np.stack(colors)
                coords = np.stack(coords)

                colors = einops.rearrange(colors, "n h w c -> (n h w) c", c=3)
                coords = einops.rearrange(coords, "n h w c -> (n h w) c", c=3)
                scene_mask = (
                    (coords[:, 0] > SCENE_BOUNDS[0])
                    & (coords[:, 0] < SCENE_BOUNDS[3])
                    & (coords[:, 1] > SCENE_BOUNDS[1])
                    & (coords[:, 1] < SCENE_BOUNDS[4])
                    & (coords[:, 2] > SCENE_BOUNDS[2])
                    & (coords[:, 2] < SCENE_BOUNDS[5])
                )
                coords = coords[scene_mask]
                colors = colors[scene_mask]
                if self.use_mask:
                    masks = np.stack(masks)
                    masks = einops.rearrange(masks, "n h w -> (n h w)")
                    masks = masks[scene_mask]
                    for v in self.invalid_mask_values:
                        masks[masks == v] = 0
                    masks[masks > 0] = 1
                    masks = masks.astype(float)
                    pcd = self.transform_pcd(
                        dict(
                            coord=coords,
                            color=np.concatenate([colors, masks[:, None]], axis=-1),
                        )
                    )
                    pcd["mask"] = pcd["feat"][:, -1].bool()
                    pcd["feat"] = pcd["feat"][:, :-1]
                else:
                    pcd = self.transform_pcd(dict(coord=coords, color=colors))
                obs_pcds.append(pcd)

        obs_dict["pcds"] = obs_pcds

        if not self.collision:
            action = np.stack(
                [
                    np.concatenate([d["gripper_pose"], [d["gripper_open"]]])
                    for d in demo[start_ts : start_ts + self.chunk_size]
                ]
            )
        else:
            action = np.stack(
                [
                    np.concatenate(
                        [
                            d["gripper_pose"],
                            [d["gripper_open"]],
                            [d["ignore_collisions"]],
                        ]
                    )
                    for d in demo[start_ts : start_ts + self.chunk_size]
                ]
            )
        if len(action) < self.chunk_size:
            action = np.pad(
                action,
                [[0, self.chunk_size - len(action)]]
                + [[0, 0] for _ in range(action.ndim - 1)],
                mode="edge",
            )

        action = torch.from_numpy(action).float()

        pos_min = torch.FloatTensor(loc_bounds[task][0])
        pos_max = torch.FloatTensor(loc_bounds[task][1])
        qpos[..., :3] = (qpos[..., :3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
        action[..., :3] = (action[..., :3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

        qpos[..., 3:7] = torch.nn.functional.normalize(qpos[..., 3:7], dim=-1)
        action[..., 3:7] = torch.nn.functional.normalize(action[..., 3:7], dim=-1)

        if self.rot_type == "6d":
            qpos_rot = matrix_to_rotation_6d(quaternion_to_matrix(qpos[..., 3:7]))
            action_rot = matrix_to_rotation_6d(quaternion_to_matrix(action[..., 3:7]))
        else:
            raise NotImplementedError

        qpos = torch.cat([qpos[..., :3], qpos_rot, qpos[..., 7:]], dim=-1)
        obs_dict["qpos"] = qpos.float()
        action = torch.cat([action[..., :3], action_rot, action[..., 7:]], dim=-1)
        return_dict["obs"] = obs_dict
        return_dict["action"] = action.float()
        return_dict["goal"] = dict(task_emb=goal_cond.reshape(-1))

        return return_dict
