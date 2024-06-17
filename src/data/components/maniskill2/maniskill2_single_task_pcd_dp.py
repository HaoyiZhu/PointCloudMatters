import os
from os.path import expanduser

import h5py
import numpy as np
import torch
from tqdm import tqdm

import src.utils as U
from src.utils.diffusion_policy import LinearNormalizer, SingleFieldLinearNormalizer
from src.utils.normalize_utils import get_range_normalizer_from_stat

from .maniskill2_single_task_pcd_act import ManiSkill2GoalPosSingleTaskACTPCDDataset

log = U.RankedLogger(__name__, rank_zero_only=True)


class ManiSkill2GoalPosSingleTaskDiffusionPolicyPCDDataset(
    ManiSkill2GoalPosSingleTaskACTPCDDataset
):
    def __init__(self, n_obs_steps=2, **kwargs):
        super().__init__(**kwargs)
        self.n_obs_steps = n_obs_steps
        self.obs_keys = ["qpos", "pcds"]

    def get_norm_stats(self, suffix=""):
        if os.path.exists(
            os.path.join(self.cache_dir, f"{self.env_id}_norm_stats_dp{suffix}.pt")
        ):
            log.info("Loading normalization stats from cache...")
            return torch.load(
                os.path.join(self.cache_dir, f"{self.env_id}_norm_stats_dp{suffix}.pt")
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
            stats,
            os.path.join(self.cache_dir, f"{self.env_id}_norm_stats_dp{suffix}.pt"),
        )

        return stats

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        stats = self.get_norm_stats()
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(stats["action"], **kwargs)
        for k in self.obs_keys:
            if "pcd" in k:
                if self.pointmap:
                    normalizer["base_camera_rgb"] = (
                        SingleFieldLinearNormalizer.create_identity()
                    )
                continue
                # normalizer[k] = SingleFieldLinearNormalizer.create_identity()
            elif "qpos" in k:
                normalizer[k] = get_range_normalizer_from_stat(stats["qpos"], **kwargs)
            else:
                raise ValueError(f"Unknown key {k}")
        return normalizer

    def __getitem__(self, idx):
        idx = idx % self.load_count
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

        obs_pcds = []
        for obs_step_idx in range(self.n_obs_steps):
            if start_ts + obs_step_idx >= len(trajectory["obs"]["pointcloud"]["xyzw"]):
                assert len(obs_pcds), (
                    obs_step_idx,
                    len(trajectory["obs"]["pointcloud"]["xyzw"]),
                )
                obs_pcds.append(obs_pcds[-1])
            else:
                coords = trajectory["obs"]["pointcloud"]["xyzw"][start_ts].reshape(
                    -1, 128, 128, 4
                )[self.camera_ids]
                if not self.pointmap:
                    if self.rand_crop:
                        crop_size = 112
                        crop_start_x = np.random.randint(0, 128 - crop_size)
                        crop_start_y = np.random.randint(0, 128 - crop_size)
                        coords[:, :crop_start_x] = 0
                        coords[:, crop_start_x + crop_size :] = 0
                        coords[:, :, :crop_start_y] = 0
                        coords[:, :, crop_start_y + crop_size :] = 0
                    coords = coords.reshape(-1, 4)
                    colors = (
                        trajectory["obs"]["pointcloud"]["rgb"][start_ts + obs_step_idx]
                        .reshape(-1, self.point_num_per_cam, 3)[self.camera_ids]
                        .reshape(-1, 3)
                    )
                    colors = colors[coords[..., -1] > 0]  # / 255.0
                    coords = coords[coords[..., -1] > 0][:, :3]
                    if not self.include_ground:
                        colors = colors[coords[..., -1] > 0.005]
                        coords = coords[coords[..., -1] > 0.005]
                    else:
                        colors = colors[coords[..., 0] > -0.8]
                        coords = coords[coords[..., 0] > -0.8]

                    pcd = self.transform_pcd(dict(coord=coords, color=colors))
                    if self.include_ground:
                        pcd["mask"] = pcd["coord"][:, -1] > 0.005
                    obs_pcds.append(pcd)
                else:
                    colors = (
                        trajectory["obs"]["pointcloud"]["rgb"][
                            start_ts + obs_step_idx
                        ].reshape(-1, 128, 128, 3)[self.camera_ids]
                    ).astype(float) / 255.0
                    colors[coords[..., -1] == 0] = 0
                    coords[coords[..., -1] == 0] = 0
                    coords = coords[..., :3]
                    image = np.concatenate([colors, coords], axis=-1).reshape(
                        -1, 128, 128, 6
                    )
                    obs_pcds.append(image)

        if not self.pointmap:
            obs_dict["pcds"] = obs_pcds
        else:
            obs_pcds = np.concatenate(obs_pcds, axis=0)
            obs_dict["base_camera_rgb"] = torch.einsum(
                "k h w c -> k c h w", torch.from_numpy(obs_pcds).float()
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


class ManiSkill2NullGoalSingleTaskDiffusionPolicyPCDDataset(
    ManiSkill2GoalPosSingleTaskDiffusionPolicyPCDDataset
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_goal(self, obs):
        return None
