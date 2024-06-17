import os
from os.path import expanduser

import h5py
import numpy as np
import torch

# from mani_skill2.utils.io_utils import load_json
from torch.utils.data import Dataset
from tqdm import tqdm

import src.utils as U

log = U.RankedLogger(__name__, rank_zero_only=True)


class ManiSkill2GoalPosSingleTaskACTRGBDDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        load_count=-1,
        camera_names=("base_camera",),
        include_depth=False,  # `True` for RGB-D
        scale_rgb_only=False,
        goal_cond_keys=(
            "goal_pos",
            "obj_start_pos",
        ),
        chunk_size=100,
        cache_dir=os.path.join(expanduser("~"), ".cache", "pcm"),
        only_depth=False,
        cache_traj=True,
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

        self.camera_names = camera_names
        self.include_depth = include_depth
        self.scale_rgb_only = scale_rgb_only
        self.goal_cond_keys = goal_cond_keys
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.only_depth = only_depth
        self.cache_traj = cache_traj

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
            else f"sample_{self.load_count}.pt"
        )
        if os.path.exists(
            os.path.join(self.cache_dir, f"{self.env_id}_norm_stats{suffix}.pt")
        ):
            log.info("Loading normalization stats from cache...")
            return torch.load(
                os.path.join(self.cache_dir, f"{self.env_id}_norm_stats{suffix}.pt")
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
                if self.data is None:
                    self.data = h5py.File(self.dataset_file, "r")
                trajectory = self.data[f"traj_{eps['episode_id']}"]
                trajectory = U.io_utils.load_h5_data(trajectory)
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
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = U.io_utils.load_h5_data(trajectory)

        original_action_shape = trajectory["actions"].shape
        episode_len = original_action_shape[0]
        start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        qpos = trajectory["obs"]["agent"]["qpos"][start_ts]
        # qvel = trajectory["obs"]["agent"]["qvel"][start_ts]
        image_dict = dict()
        for camera_name in self.camera_names:
            data_cam = camera_name
            if data_cam not in trajectory["obs"]["image"].keys():
                data_cam = data_cam.replace("base", "front")
            assert (
                data_cam in trajectory["obs"]["image"].keys()
            ), f"Camera {camera_name} not found in trajectory, available cameras: {trajectory['obs']['image'].keys()}"

            if not self.only_depth:
                image = trajectory["obs"]["image"][data_cam]["rgb"].astype(float)
                if self.include_depth:
                    depth = trajectory["obs"]["image"][data_cam]["depth"].astype(float)
                    image = np.concatenate([image, depth], axis=-1)
            else:
                depth = trajectory["obs"]["image"][data_cam]["depth"].astype(float)
                image = depth
            image_dict[camera_name] = image[start_ts]

        action = trajectory["actions"][start_ts : start_ts + self.chunk_size]
        action_len = action.shape[0]
        padded_action = np.zeros(
            (self.chunk_size, original_action_shape[1]), dtype=np.float32
        )
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])

        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        if not self.only_depth:
            # normalize image and change dtype to float
            image_data[:, :3] = image_data[:, :3] / 255.0
            if self.include_depth and not self.scale_rgb_only:
                image_data[:, 3:] = image_data[:, 3:] / (2**10)
        else:
            image_data[:, :1] = image_data[:, :1] / (2**10)
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        goal_cond = self.get_goal(trajectory["obs"])[start_ts]

        data_dict = dict(
            image=image_data,
            qpos=qpos_data,
            actions=action_data,
            is_pad=is_pad,
            goal_cond=goal_cond,
        )

        return data_dict


class ManiSkill2NullGoalSingleTaskACTRGBDDataset(
    ManiSkill2GoalPosSingleTaskACTRGBDDataset
):
    def __init__(
        self,
        dataset_file: str,
        load_count=-1,
        camera_names=("base_camera",),
        include_depth=False,  # `True` for RGB-D
        scale_rgb_only=False,
        goal_cond_keys=None,
        only_depth=False,
        chunk_size=20,
        loop=1,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_file=dataset_file,
            load_count=load_count,
            camera_names=camera_names,
            include_depth=include_depth,
            scale_rgb_only=scale_rgb_only,
            goal_cond_keys=goal_cond_keys,
            chunk_size=chunk_size,
            only_depth=only_depth,
            loop=loop,
            **kwargs,
        )

    def get_goal(self, obs):
        return torch.zeros(
            1000,
        ).float()
