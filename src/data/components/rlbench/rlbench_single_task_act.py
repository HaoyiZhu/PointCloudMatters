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
from tqdm import tqdm

import src.utils as U
from src.data.components.rlbench.constants import SCENE_BOUNDS, loc_bounds
from src.data.components.transformpcd import ComposePCD
from src.utils.rotation_conversions import matrix_to_rotation_6d, quaternion_to_matrix

log = U.RankedLogger(__name__, rank_zero_only=True)


class RLBenchSingleTaskACTRGBDDataset(Dataset):
    def __init__(
        self,
        root: str = "data/rlbench/processed/train/",
        task_names: List[str] = [],
        chunk_size: int = 16,
        max_episodes_per_task: int = 100,
        cameras: Tuple[str] = ("front",),
        action_dim: int = 11,
        include_depth: bool = False,
        rot_type: str = "6d",
        collision: bool = True,
        use_mask=False,
        invalid_mask_values=[201, 204, 208, 246],
        loop: int = 1,
        cache_episode: bool = True,
        use_pcd=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cameras = cameras
        self.chunk_size = chunk_size
        self.task_names = task_names
        self.action_dim = action_dim
        self.root = root
        self.include_depth = include_depth
        assert rot_type in [
            "6d",
        ], f"rot_type {rot_type} not supported. Choose from ['6d']"
        self.rot_type = rot_type
        self.collision = collision
        self.use_mask = use_mask
        self.invalid_mask_values = invalid_mask_values
        self.loop = loop
        self.cache_episode = cache_episode
        self.use_pcd = use_pcd

        # File-names of episodes per-task and variation
        self.episodes = []
        for task_n in task_names:
            count = 0
            for filename in tqdm(list(U.io_utils.listdir(os.path.join(root, task_n)))):
                if filename.endswith("npy") and "old" not in filename:
                    if self.cache_episode:
                        data = U.io_utils.load_numpy_pickle(
                            os.path.join(root, task_n, filename),
                        )
                        for demo in data["demo"]:
                            if not self.include_depth:
                                for camera_name in self.cameras:
                                    demo.pop(f"{camera_name}_depth")
                            if not self.use_pcd:
                                for camera_name in self.cameras:
                                    demo.pop(f"{camera_name}_point_cloud")

                        self.episodes.append(
                            (
                                task_n,
                                data,
                            )
                        )
                    else:
                        self.episodes.append(
                            (task_n, os.path.join(root, task_n, filename))
                        )
                    count += 1
                    if count >= max_episodes_per_task:
                        break

        log.info(f"Created dataset from {root} with {len(self.episodes)}.")

    def __len__(self):
        return len(self.episodes) * self.loop

    def __getitem__(self, idx):
        idx %= len(self.episodes)
        if not self.cache_episode:
            task, episode = self.episodes[idx]
            episode = U.io_utils.load_numpy_pickle(episode)
        else:
            task, episode = self.episodes[idx]

        demo, goal_cond = episode["demo"], episode["task_goal"]
        episode_len = len(demo)
        start_ts = np.random.choice(episode_len - 1)
        # get observation at start_ts only
        obs = demo[start_ts]
        if not self.collision:
            qpos = np.concatenate([obs["gripper_pose"], [obs["gripper_open"]]])
        else:
            qpos = np.concatenate(
                [obs["gripper_pose"], [obs["gripper_open"]], [obs["ignore_collisions"]]]
            )
        image_dict = dict()
        for camera_name in self.cameras:
            image = obs[f"{camera_name}_rgb"].astype(float)
            if self.include_depth:
                depth = obs[f"{camera_name}_depth"].astype(float)[:, :, None]
                image = np.concatenate([image, depth], axis=-1)
            image_dict[camera_name] = image

        if not self.collision:
            action = np.stack(
                [
                    np.concatenate([d["gripper_pose"], [d["gripper_open"]]])
                    for d in demo[start_ts + 1 : start_ts + 1 + self.chunk_size]
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
                    for d in demo[start_ts + 1 : start_ts + 1 + self.chunk_size]
                ]
            )
        action_len = action.shape[0]
        padded_action = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.cameras:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data[:, :3] = image_data[:, :3] / 255.0

        pos_min = torch.FloatTensor(loc_bounds[task][0])
        pos_max = torch.FloatTensor(loc_bounds[task][1])
        qpos_data[:3] = (qpos_data[:3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
        action_data[:action_len, :3] = (action_data[:action_len, :3] - pos_min) / (
            pos_max - pos_min
        ) * 2.0 - 1.0

        qpos_data[3:7] = torch.nn.functional.normalize(qpos_data[3:7], dim=-1)
        action_data[:action_len, 3:7] = torch.nn.functional.normalize(
            action_data[:action_len, 3:7], dim=-1
        )

        if self.rot_type == "6d":
            qpos_rot = matrix_to_rotation_6d(quaternion_to_matrix(qpos_data[3:7]))
            action_rot = matrix_to_rotation_6d(
                quaternion_to_matrix(action_data[:action_len, 3:7])
            )
            action_rot = torch.cat(
                [action_rot, torch.zeros((action_data.shape[0] - action_len, 6))]
            )
        else:
            raise NotImplementedError

        qpos_data = torch.cat([qpos_data[:3], qpos_rot, qpos_data[7:]], dim=-1)
        action_data = torch.cat(
            [action_data[..., :3], action_rot, action_data[..., 7:]], dim=-1
        )

        data_dict = dict(
            image=image_data,
            qpos=qpos_data,
            actions=action_data,
            is_pad=is_pad,
            goal_cond=torch.from_numpy(goal_cond).float(),
        )

        return data_dict


class RLBenchSingleTaskACTPCDDataset(RLBenchSingleTaskACTRGBDDataset):
    def __init__(
        self,
        root: str = "data/rlbench/processed/train/",
        task_names: List[str] = [],
        chunk_size: int = 16,
        transform_pcd: List[Dict[str, Any]] = None,
        max_episodes_per_task: int = 100,
        cameras: Tuple[str] = ("front",),
        action_dim: int = 11,
        rot_type: str = "6d",
        collision: bool = True,
        use_mask=False,
        invalid_mask_values=[201, 204, 208, 246],
        loop: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            task_names=task_names,
            chunk_size=chunk_size,
            max_episodes_per_task=max_episodes_per_task,
            cameras=cameras,
            action_dim=action_dim,
            rot_type=rot_type,
            collision=collision,
            use_mask=use_mask,
            invalid_mask_values=invalid_mask_values,
            loop=loop,
            use_pcd=True,
            **kwargs,
        )

        self.transform_pcd = ComposePCD(transform_pcd)

    def __getitem__(self, idx):
        idx %= len(self.episodes)
        try:
            if not self.cache_episode:
                task, episode = self.episodes[idx]
                episode = U.io_utils.load_numpy_pickle(episode)
            else:
                task, episode = self.episodes[idx]
        except:
            print(os.path.join(self.root, self.episodes[idx]))
            raise

        demo, goal_cond = episode["demo"], episode["task_goal"]
        episode_len = len(demo)
        start_ts = np.random.choice(episode_len - 1)
        # get observation at start_ts only
        obs = demo[start_ts]
        if not self.collision:
            qpos = np.concatenate([obs["gripper_pose"], [obs["gripper_open"]]])
        else:
            qpos = np.concatenate(
                [obs["gripper_pose"], [obs["gripper_open"]], [obs["ignore_collisions"]]]
            )
        colors, coords = [], []
        if self.use_mask:
            masks = []
        for camera_name in self.cameras:
            colors.append(obs[f"{camera_name}_rgb"].astype(float))
            coords.append(obs[f"{camera_name}_point_cloud"].astype(float))
            if self.use_mask:
                masks.append(obs[f"{camera_name}_mask"].astype(float))

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
            pcds_ = [
                self.transform_pcd(
                    dict(
                        coord=coords,
                        color=np.concatenate([colors, masks[:, None]], axis=-1),
                    )
                )
            ]
            pcds = []
            for pcd in pcds_:
                pcd["mask"] = pcd["feat"][:, -1].bool()
                pcd["feat"] = pcd["feat"][:, :-1]
                pcds.append(pcd)
        else:
            pcds = [self.transform_pcd(dict(coord=coords, color=colors))]

        if not self.collision:
            action = np.stack(
                [
                    np.concatenate([d["gripper_pose"], [d["gripper_open"]]])
                    for d in demo[start_ts + 1 : start_ts + 1 + self.chunk_size]
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
                    for d in demo[start_ts + 1 : start_ts + 1 + self.chunk_size]
                ]
            )
        action_len = action.shape[0]
        padded_action = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        pos_min = torch.FloatTensor(loc_bounds[task][0])
        pos_max = torch.FloatTensor(loc_bounds[task][1])
        qpos_data[:3] = (qpos_data[:3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
        action_data[:action_len, :3] = (action_data[:action_len, :3] - pos_min) / (
            pos_max - pos_min
        ) * 2.0 - 1.0

        qpos_data[3:7] = torch.nn.functional.normalize(qpos_data[3:7], dim=-1)
        action_data[:action_len, 3:7] = torch.nn.functional.normalize(
            action_data[:action_len, 3:7], dim=-1
        )

        if self.rot_type == "6d":
            qpos_rot = matrix_to_rotation_6d(quaternion_to_matrix(qpos_data[3:7]))
            action_rot = matrix_to_rotation_6d(
                quaternion_to_matrix(action_data[:action_len, 3:7])
            )
            action_rot = torch.cat(
                [action_rot, torch.zeros((action_data.shape[0] - action_len, 6))]
            )
        else:
            raise NotImplementedError

        qpos_data = torch.cat([qpos_data[:3], qpos_rot, qpos_data[7:]], dim=-1)
        action_data = torch.cat(
            [action_data[..., :3], action_rot, action_data[..., 7:]], dim=-1
        )

        data_dict = dict(
            pcds=pcds,
            qpos=qpos_data,
            actions=action_data,
            is_pad=is_pad,
            goal_cond=torch.from_numpy(goal_cond).float(),
        )

        return data_dict
