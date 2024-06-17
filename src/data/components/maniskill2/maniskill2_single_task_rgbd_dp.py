import os
from os.path import expanduser

import h5py
import numpy as np
import torch
from tqdm import tqdm

import src.utils as U
from src.utils.diffusion_policy import LinearNormalizer, SingleFieldLinearNormalizer
from src.utils.normalize_utils import get_range_normalizer_from_stat

from .maniskill2_single_task_rgbd_act import ManiSkill2GoalPosSingleTaskACTRGBDDataset

log = U.RankedLogger(__name__, rank_zero_only=True)


class ManiSkill2GoalPosSingleTaskDiffusionPolicyRGBDDataset(
    ManiSkill2GoalPosSingleTaskACTRGBDDataset
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_keys = ["qpos"]
        for cam_name in self.camera_names:
            self.obs_keys.append(f"{cam_name}_rgb")
            if self.include_depth:
                self.obs_keys.append(f"{cam_name}_depth")

    def get_norm_stats(
        self,
    ):
        if os.path.exists(
            os.path.join(self.cache_dir, f"{self.env_id}_norm_stats_dp.pt")
        ):
            log.info("Loading normalization stats from cache...")
            return torch.load(
                os.path.join(self.cache_dir, f"{self.env_id}_norm_stats_dp.pt")
            )

        log.info("Calculating normalization stats...")

        all_qpos_data = []
        all_action_data = []
        for episode_idx in tqdm(range(self.load_count)):
            eps = (
                self.episodes[episode_idx]
                if self.load_count == len(self.episodes)
                else self.episodes[
                    :: int(np.floor(len(self.episodes) / self.load_count))
                ][episode_idx]
            )
            if self.cache_traj:
                trajectory = self.trajectories[episode_idx]
            else:
                trajectory = self.data[f"traj_{eps['episode_id']}"]
                trajectory = U.io_utils.load_h5_data(trajectory)
            all_qpos_data.append(torch.from_numpy(trajectory["obs"]["agent"]["qpos"]))
            all_action_data.append(torch.from_numpy(trajectory["actions"]))

        all_qpos_data = torch.cat(all_qpos_data, dim=0)
        all_action_data = torch.cat(all_action_data, dim=0)

        # normalize action data
        action_min = all_action_data.min(dim=0, keepdim=True).values
        action_max = all_action_data.max(dim=0, keepdim=True).values
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        # action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
        action_std[action_std < 1e-2] = 1e-2  # clipping

        # normalize qpos data
        qpos_min = all_qpos_data.min(dim=0, keepdim=True).values
        qpos_max = all_qpos_data.max(dim=0, keepdim=True).values
        qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
        qpos_std = all_qpos_data.std(dim=0, keepdim=True)
        # qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
        qpos_std[qpos_std < 1e-2] = 1e-2  # clipping

        stats = dict(
            action=dict(
                min=action_min.numpy().squeeze(),
                max=action_max.numpy().squeeze(),
                mean=action_mean.numpy().squeeze(),
                std=action_std.numpy().squeeze(),
            ),
            qpos=dict(
                min=qpos_min.numpy().squeeze(),
                max=qpos_max.numpy().squeeze(),
                mean=qpos_mean.numpy().squeeze(),
                std=qpos_std.numpy().squeeze(),
            ),
        )

        torch.save(
            stats, os.path.join(self.cache_dir, f"{self.env_id}_norm_stats_dp.pt")
        )

        return stats

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        stats = self.get_norm_stats()
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(stats["action"], **kwargs)
        for k in self.obs_keys:
            if "rgb" in k or "depth" in k:
                normalizer[k] = SingleFieldLinearNormalizer.create_identity()
            elif "qpos" in k:
                normalizer[k] = get_range_normalizer_from_stat(stats["qpos"], **kwargs)
            else:
                raise ValueError(f"Unknown key {k}")
        return normalizer

    def __getitem__(self, idx):
        idx = idx % self.load_count
        if hasattr(self, "subset_indices"):
            idx = self.subset_indices[idx]

        if self.cache_traj:
            trajectory = self.trajectories[idx]
        else:
            eps = (
                self.episodes[idx]
                if self.load_count == len(self.episodes)
                else self.episodes[
                    :: int(np.floor(len(self.episodes) / self.load_count))
                ][idx]
            )
            if self.data is None:
                self.data = h5py.File(self.dataset_file, "r")
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = U.io_utils.load_h5_data(trajectory)

        original_action_shape = trajectory["actions"].shape
        episode_len = original_action_shape[0]
        start_ts = np.random.choice(episode_len)

        return_dict = dict()
        obs_dict = dict()

        qpos = trajectory["obs"]["agent"]["qpos"][start_ts : start_ts + self.chunk_size]
        if len(qpos) < self.chunk_size:
            qpos = np.pad(
                qpos,
                [[0, self.chunk_size - len(qpos)]]
                + [[0, 0] for _ in range(qpos.ndim - 1)],
                mode="edge",
            )
        obs_dict["qpos"] = torch.from_numpy(qpos).float()
        for camera_name in self.camera_names:
            data_cam = camera_name
            if data_cam not in trajectory["obs"]["image"].keys():
                data_cam = data_cam.replace("base", "front")
            assert (
                data_cam in trajectory["obs"]["image"].keys()
            ), f"Camera {camera_name} not found in trajectory, available cameras: {trajectory['obs']['image'].keys()}"
            if self.only_depth or self.include_depth:
                depth = trajectory["obs"]["image"][data_cam]["depth"].astype(float)[
                    start_ts : start_ts + self.chunk_size
                ] / (2**10)
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
            if not self.only_depth:
                rgb = trajectory["obs"]["image"][data_cam]["rgb"].astype(float)[
                    start_ts : start_ts + self.chunk_size
                ]
                rgb = rgb / 255.0
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

        return_dict["obs"] = obs_dict
        action = trajectory["actions"][start_ts : start_ts + self.chunk_size]
        if len(action) < self.chunk_size:
            action = np.pad(
                action,
                [[0, self.chunk_size - len(action)]]
                + [[0, 0] for _ in range(action.ndim - 1)],
                mode="edge",
            )
        return_dict["action"] = torch.from_numpy(action).float()

        goal_cond = self.get_goal(trajectory["obs"])
        if goal_cond is not None:
            return_dict["goal"] = dict(task_emb=goal_cond[start_ts])

        return return_dict


class ManiSkill2NullGoalSingleTaskDiffusionPolicyRGBDDataset(
    ManiSkill2GoalPosSingleTaskDiffusionPolicyRGBDDataset
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_goal(self, obs):
        return None
