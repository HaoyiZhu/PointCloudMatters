import os
from os.path import expanduser

import h5py
import numpy as np
import torch

# from mani_skill2.utils.io_utils import load_json
from torch.utils.data import Dataset
from tqdm import tqdm

import src.utils as U
from src.data.components.transformpcd import ComposePCD

log = U.RankedLogger(__name__, rank_zero_only=True)


class ManiSkill2GoalPosSingleTaskACTPCDDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        load_count=-1,
        goal_cond_keys=None,
        chunk_size=100,
        transform_pcd=None,
        cache_dir=os.path.join(expanduser("~"), ".cache", "pcm"),
        camera_ids=(0,),  # (0, 2, 3, 4) for 4-camera
        point_num_per_cam=16384,  # 128 * 128
        include_ground=False,
        cache_traj=True,
        rand_crop=False,
        pointmap=False,
        loop=1,
    ) -> None:
        super().__init__()
        self.dataset_file = dataset_file

        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = U.io_utils.load_json(json_path)

        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.loop = loop

        self.goal_cond_keys = goal_cond_keys
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_traj = cache_traj

        self.camera_ids = list(camera_ids)
        self.point_num_per_cam = 16384
        self.include_ground = include_ground
        self.rand_crop = rand_crop
        self.pointmap = pointmap

        self.obs_state = []
        self.obs_rgbd = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        elif isinstance(load_count, float):
            load_count = int(load_count * len(self.episodes))
        self.load_count = load_count

        if self.cache_traj:
            self.trajectories = []
            for eps in tqdm(self.episodes[:load_count]):
                traj = self.data[f"traj_{eps['episode_id']}"]
                traj = U.io_utils.load_h5_data(traj)
                try:
                    traj["obs"]["agent"].pop("qvel")
                    traj["obs"]["agent"].pop("base_pose")
                    traj["obs"].pop("camera_param")
                except:
                    pass
                self.trajectories.append(traj)

        self.norm_stats = self.get_norm_stats()

        self.transform_pcd = ComposePCD(transform_pcd)

        self.data.close()
        self.data = None

    def __len__(self):
        return self.load_count * self.loop

    def get_norm_stats(
        self,
    ):
        suffix = (
            ""
            if self.load_count == len(self.episodes)
            else f"_sample_{self.load_count}.pt"
        )
        if os.path.exists(
            os.path.join(self.cache_dir, f"{self.env_id}_norm_stats{suffix}.pt")
        ):
            log.info("Loading normalization stats from cache...")
            return torch.load(
                os.path.join(self.cache_dir, f"{self.env_id}_norm_stats{suffix}.pt")
            )

        log.info(
            f"Calculating normalization stats and saving to {os.path.join(self.cache_dir, f'{self.env_id}_norm_stats.pt')}..."
        )

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

            # trajectory = self.trajectorys[episode_idx]
            all_qpos_data.append(torch.from_numpy(trajectory["obs"]["agent"]["qpos"]))
            all_action_data.append(torch.from_numpy(trajectory["actions"]))

        all_qpos_data = torch.cat(all_qpos_data, dim=0)
        all_action_data = torch.cat(all_action_data, dim=0)

        # normalize action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

        # normalize qpos data
        qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
        qpos_std = all_qpos_data.std(dim=0, keepdim=True)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

        stats = {
            "action_mean": action_mean.numpy().squeeze(),
            "action_std": action_std.numpy().squeeze(),
            "qpos_mean": qpos_mean.numpy().squeeze(),
            "qpos_std": qpos_std.numpy().squeeze(),
        }

        torch.save(
            stats, os.path.join(self.cache_dir, f"{self.env_id}_norm_stats{suffix}.pt")
        )

        return stats

    def get_goal(self, obs):
        goal_conds = []
        for goal_cond_key in self.goal_cond_keys:
            goal_cond = torch.from_numpy(obs["extra"][goal_cond_key]).float()
            if goal_cond_key == "target_angle_diff":
                goal_cond = goal_cond.unsqueeze(dim=-1)
            if "target_angle_diff" in self.goal_cond_keys:
                if goal_cond.dim() == 1:
                    goal_cond = goal_cond.unsqueeze(dim=0)
            goal_conds.append(goal_cond)
        goal_conds = torch.cat(goal_conds, dim=-1)
        return goal_conds

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
        # get observation at start_ts only
        qpos = trajectory["obs"]["agent"]["qpos"][start_ts]
        # qvel = trajectory["obs"]["agent"]["qvel"][start_ts]

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
                trajectory["obs"]["pointcloud"]["rgb"][start_ts]
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

            pcds = [self.transform_pcd(dict(coord=coords, color=colors))]
            if self.include_ground:
                pcds_ = []
                for pcd in pcds:
                    pcd["mask"] = pcd["coord"][:, -1] > 0.005
                    pcds_.append(pcd)
                pcds = pcds_
        else:
            colors = (
                trajectory["obs"]["pointcloud"]["rgb"][start_ts].reshape(
                    -1, 128, 128, 3
                )[self.camera_ids]
            ).astype(float) / 255.0
            colors[coords[..., -1] == 0] = 0
            coords[coords[..., -1] == 0] = 0
            coords = coords[..., :3]
            image = np.concatenate([colors, coords], axis=-1).reshape(-1, 128, 128, 6)
            image_data = torch.from_numpy(image).float()
            # channel last
            image_data = torch.einsum("k h w c -> k c h w", image_data)

        action = trajectory["actions"][start_ts : start_ts + self.chunk_size]
        action_len = action.shape[0]
        padded_action = np.zeros(
            (self.chunk_size, original_action_shape[1]), dtype=np.float32
        )
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        goal_cond = self.get_goal(trajectory["obs"])[start_ts]
        if not self.pointmap:
            data_dict = dict(
                pcds=pcds,
                qpos=qpos_data,
                actions=action_data,
                is_pad=is_pad,
                goal_cond=goal_cond,
            )
        else:
            data_dict = dict(
                image=image_data,
                qpos=qpos_data,
                actions=action_data,
                is_pad=is_pad,
                goal_cond=goal_cond,
            )

        return data_dict


class ManiSkill2NullGoalSingleTaskACTPCDDataset(
    ManiSkill2GoalPosSingleTaskACTPCDDataset
):
    def __init__(
        self,
        dataset_file: str,
        load_count=-1,
        chunk_size=20,
        transform_pcd=None,
        cache_dir="~/.cache/pcm",
        camera_ids=(0,),
        point_num_per_cam=16384,
        include_ground=False,
        loop=1,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_file=dataset_file,
            load_count=load_count,
            chunk_size=chunk_size,
            transform_pcd=transform_pcd,
            include_ground=include_ground,
            loop=loop,
            **kwargs,
        )

    def get_goal(self, obs):
        return torch.zeros(
            1000,
        ).float()
