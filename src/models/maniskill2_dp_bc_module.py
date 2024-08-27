from __future__ import annotations

from collections import defaultdict
from typing import Any

import gymnasium as gym
import mani_skill2.envs
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from mani_skill2.utils.wrappers import RecordEpisode

from src import utils as U
from src.envs import custom_maniskill2
from src.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ManiSkill2DiffusionPolicyBCModule(LightningModule):
    def __init__(
        self,
        policy,
        env_id,
        optimizer,
        lr_scheduler,
        train_metrics,
        val_metrics,
        best_val_metrics,
        compile: bool = False,
        obs_mode: str = "rgbd",
        control_mode: str = "pd_ee_delta_pose",
        render_mode: str = "cameras",
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["policy", "train_metrics", "val_metrics", "best_val_metrics"],
        )
        self.policy = policy

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        # for tracking best so far validation metrics
        self.best_val_metrics = best_val_metrics

    def setup(self, stage: str) -> None:
        self.policy.set_normalizer(self.trainer.datamodule.data_train.get_normalizer())
        if stage == "fit" and self.hparams.compile:
            self.policy = torch.compile(self.policy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.policy.compute_loss(x)
        else:
            return self.policy.predict_action(x)

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            return self.policy.compute_loss(batch)
        else:
            return self.policy.predict_action(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.best_val_metrics.reset()
        super().on_train_start()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss_dict = self.model_step(batch)

        # update and log metrics
        self.train_metrics(loss_dict)
        self.log_dict(
            self.train_metrics.metrics_dict(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["action"].shape[0],
        )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    def on_validation_epoch_start(self) -> None:
        env = gym.make(
            (
                self.hparams.env_id
                if self.hparams.env_id != "PegInsertionSide-v0"
                else "PegInsertionSide-3steps-v0"
            ),
            obs_mode=self.hparams.obs_mode,
            control_mode=self.hparams.control_mode,
            render_mode=self.hparams.render_mode,
            shader_dir=self.hparams.shader_dir,
            render_config={
                "rt_samples_per_pixel": self.hparams.rt_samples_per_pixel,
                "rt_use_denoiser": self.hparams.rt_use_denoiser,
            },
            camera_cfgs={"use_stereo_depth": self.hparams.use_stereo_depth},
        )
        self.env = RecordEpisode(
            env,
            f"{self.trainer.log_dir.split('tensorboard')[0]}/video/epoch_{self.current_epoch}",
            save_trajectory=False,
        )
        self.env.reset()

    def raw_obs_to_tensor_obs(self, obs, task_emb):
        """
        Prepare the tensor observations as input for the algorithm.
        """

        data = {
            "obs": defaultdict(list),
            "task_emb": (
                task_emb.reshape(1, -1).to(self.device)
                if task_emb is not None
                else None
            ),
        }
        use_pcd = False
        all_pcds = []

        for obs_ in obs:
            qpos = torch.from_numpy(obs_["agent"]["qpos"]).float()
            data["obs"]["qpos"].append(qpos)
            if "image" in obs_.keys():
                for camera_name in self.trainer.datamodule.data_train.camera_names:
                    rgb = obs_["image"][camera_name]["rgb"].astype(float)
                    rgb = rgb / 255.0
                    rgb = rgb.transpose(2, 0, 1)
                    data["obs"][f"{camera_name}_rgb"].append(torch.from_numpy(rgb))
                    if self.trainer.datamodule.data_train.include_depth:
                        depth = obs_["image"][camera_name]["depth"].astype(float)
                        depth = depth.transpose(2, 0, 1)
                        data["obs"][f"{camera_name}_depth"].append(
                            torch.from_numpy(depth)
                        )
            elif "pointcloud" in obs_.keys():
                use_pcd = True
                coords = obs_["pointcloud"]["xyzw"].reshape(-1, 128, 128, 4)[
                    self.trainer.datamodule.data_train.camera_ids
                ]
                if not self.trainer.datamodule.data_train.pointmap:
                    if self.trainer.datamodule.data_train.rand_crop:
                        # center crop
                        crop_size = 112
                        crop_start_x = (128 - crop_size) // 2
                        crop_start_y = (128 - crop_size) // 2
                        coords[:, :crop_start_x] = 0
                        coords[:, crop_start_x + crop_size :] = 0
                        coords[:, :, :crop_start_y] = 0
                        coords[:, :, crop_start_y + crop_size :] = 0
                    coords = coords.reshape(-1, 4)
                    colors = (
                        obs_["pointcloud"]["rgb"]
                        .reshape(
                            -1, self.trainer.datamodule.data_train.point_num_per_cam, 3
                        )[self.trainer.datamodule.data_train.camera_ids]
                        .reshape(-1, 3)
                    )
                    colors = colors[coords[..., -1] > 0]  # / 255.0
                    coords = coords[coords[..., -1] > 0][:, :3]
                    colors = colors[coords[..., -1] > 0.005]
                    coords = coords[coords[..., -1] > 0.005]
                    pcds = self.trainer.datamodule.data_train.transform_pcd(
                        dict(coord=coords, color=colors), mode="test"
                    )
                    all_pcds.append(pcds)
                else:
                    colors = (
                        obs_["pointcloud"]["rgb"].reshape(-1, 128, 128, 3)[
                            self.trainer.datamodule.data_train.camera_ids
                        ]
                    ).astype(float) / 255.0
                    colors[coords[..., -1] == 0] = 0
                    coords[coords[..., -1] == 0] = 0
                    coords = coords[..., :3]
                    image = np.concatenate([colors, coords], axis=-1).reshape(
                        -1, 128, 128, 6
                    )
                    all_pcds.append(image)
            else:
                raise NotImplementedError

        for key in data["obs"]:
            data["obs"][key] = (
                torch.stack(data["obs"][key]).unsqueeze(0).to(self.device)
            )
        if use_pcd:
            if not self.trainer.datamodule.data_train.pointmap:
                pcds = U.point_collate_fn(all_pcds)
                for k in pcds.keys():
                    pcds[k] = pcds[k].to(self.device)
                data["obs"]["pcds"] = pcds
            else:
                all_pcds = np.stack(all_pcds, axis=0)
                data["obs"]["base_camera_rgb"] = torch.einsum(
                    "b k h w c -> b k c h w",
                    torch.from_numpy(all_pcds).float().to(self.device),
                )

        return data

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        if self.hparams.env_id == "TurnFaucet-v0":
            model_ids = [
                "5002",
                "5021",
                "5023",
                "5028",
                "5029",
                "5045",
                "5047",
                "5051",
                "5056",
                "5063",
            ]
            options = {"model_id": model_ids[batch_idx // 40]}
            obs, _ = self.env.reset(seed=10240 + batch_idx, options=options)
        else:
            obs, _ = self.env.reset(seed=10240 + batch_idx)

        rewards = []
        success = None
        self.policy.reset()
        if self.hparams.env_id == "PegInsertionSide-v0":
            grasp, align = False, False

        n_obs_steps = self.policy.n_obs_steps
        hist_obs = [obs for _ in range(n_obs_steps - 1)]

        while True:
            hist_obs.append(obs)
            assert len(hist_obs) == n_obs_steps
            goal_cond = self.trainer.datamodule.data_train.get_goal(obs)

            data = self.raw_obs_to_tensor_obs(hist_obs, goal_cond)
            hist_obs = hist_obs[1:]
            if "task_emb" in data and data["task_emb"] is not None:
                data["goal"] = dict(task_emb=data.pop("task_emb"))
            actions = self.policy.predict_action(data)["action"]

            if actions.dim() == 2:
                actions = actions[:, None, :]

            for action_idx in range(actions.shape[1]):
                obs, reward, terminated, truncated, info = self.env.step(
                    actions[:, action_idx, :].cpu().numpy().squeeze(0)
                )
                hist_obs.append(obs)
                hist_obs = hist_obs[1:]
                if self.hparams.env_id == "PegInsertionSide-v0":
                    grasp = info["is_grasped"] | grasp
                    align = info["pre_inserted"] | align
                rewards.append(reward)
                if terminated or truncated:
                    success = info["success"]
                    break
            if terminated or truncated:
                break
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rewards)
        self.logger.experiment.add_figure(
            f"val/rewards_{batch_idx}", fig, global_step=self.global_step
        )
        plt.close()

        mean_success = torch.tensor(success).float().mean()
        print(batch_idx, "mean_success:", mean_success)
        val_metrics_dict = dict(mean_success=mean_success)
        if self.hparams.env_id == "PegInsertionSide-v0":
            mean_grasp = 1 if grasp else 0
            mean_align = 1 if align else 0
            val_metrics_dict.update(dict(mean_grasp=mean_grasp, mean_align=mean_align))

        # update and log metrics
        self.val_metrics(val_metrics_dict)
        self.log_dict(
            self.val_metrics.metrics_dict(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metrics = self.val_metrics.compute()  # get current val metrics
        self.best_val_metrics(metrics)  # update best so far val metrics
        # log `best_val_metrics` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log_dict(self.best_val_metrics.compute(), sync_dist=True, prog_bar=True)
        self.env.close()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = U.build_optimizer_v2(self.hparams.optimizer, self.policy)
        if self.hparams.lr_scheduler is not None:
            self.hparams.lr_scheduler.scheduler.total_steps = (
                self.trainer.estimated_stepping_batches
            )
            scheduler = U.build_scheduler(
                self.hparams.lr_scheduler.scheduler, optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.lr_scheduler.get("monitor", "val/loss"),
                    "interval": self.hparams.lr_scheduler.get("interval", "step"),
                    "frequency": self.hparams.lr_scheduler.get("frequency", 1),
                },
            }
        return optimizer
