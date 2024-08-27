from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import mani_skill2.envs
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from mani_skill2.utils.wrappers import RecordEpisode

from src import utils as U
from src.envs import custom_maniskill2


class ManiSkill2ACTBCModule(LightningModule):
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
        temporal_agg: bool = False,
        param_dicts: list = None,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.best_val_metrics.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy(batch)

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
            batch_size=batch["actions"].shape[0],
        )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        if self.hparams.get("num_envs", 1) == 1:
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
        else:
            raise NotImplementedError

        self.env.reset()

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
            assert self.hparams.get("num_envs", 1) == 1
            options = {"model_id": model_ids[batch_idx // 40]}
            obs, _ = self.env.reset(seed=10240 + batch_idx, options=options)
        else:
            if self.hparams.get("num_envs", 1) == 1:
                obs, _ = self.env.reset(seed=10240 + batch_idx)
            else:
                seeds = [
                    10240 + self.hparams.get("num_envs", 1) * batch_idx + i
                    for i in range(self.hparams.get("num_envs", 1))
                ]
                obs, _ = self.env.reset(seed=seeds)

        rewards = []
        success = None
        if self.hparams.env_id == "PegInsertionSide-v0":
            grasp, align = False, False

        temp_agg = [
            U.TemporalAgg(
                apply=self.hparams.temporal_agg,
                action_dim=self.hparams.policy.action_dim,
                chunk_size=self.hparams.policy.num_queries,
                k=0.01,
            )
            for _ in range(self.hparams.get("num_envs", 1))
        ]

        while True:
            qpos = obs["agent"]["qpos"]

            if "image" in obs.keys():
                all_cam_images = []
                if self.hparams.get("num_envs", 1) == 1:
                    for camera_name in self.trainer.datamodule.data_train.camera_names:
                        image = obs["image"][camera_name]["rgb"].astype(float)
                        if self.trainer.datamodule.data_train.include_depth:
                            depth = obs["image"][camera_name]["depth"].astype(float)
                            image = np.concatenate([image, depth], axis=-1)
                        all_cam_images.append(image)

                    all_cam_images = np.stack(all_cam_images)
                    # construct observations
                    image_data = torch.from_numpy(all_cam_images).float()
                else:
                    for camera_name in self.trainer.datamodule.data_train.camera_names:
                        image = obs["image"][camera_name]["rgb"].float()
                        if self.trainer.datamodule.data_train.include_depth:
                            depth = obs["image"][camera_name]["depth"].float()
                            image = torch.cat([image, depth], axis=-1)
                        all_cam_images.append(image)
                    image_data = torch.stack(
                        all_cam_images, dim=1
                    )  # num_env, num_cam, h, w, c

                # channel last -> channel first
                image_data = torch.einsum("... h w c -> ... c h w", image_data)
                # normalize image and change dtype to float
                image_data[..., :3, :, :] = image_data[..., :3, :, :] / 255.0
            elif "pointcloud" in obs.keys():
                coords = obs["pointcloud"]["xyzw"].reshape(-1, 128, 128, 4)[
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
                        obs["pointcloud"]["rgb"]
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
                    for k in pcds.keys():
                        pcds[k] = pcds[k].to(self.device)
                else:
                    colors = (
                        obs["pointcloud"]["rgb"].reshape(-1, 128, 128, 3)[
                            self.trainer.datamodule.data_train.camera_ids
                        ]
                    ).astype(float) / 255.0
                    colors[coords[..., -1] == 0] = 0
                    coords[coords[..., -1] == 0] = 0
                    coords = coords[..., :3]
                    image = np.concatenate([colors, coords], axis=-1).reshape(
                        -1, 128, 128, 6
                    )
                    image_data = torch.from_numpy(image).float()
                    image_data = torch.einsum("... h w c -> ... c h w", image_data)

            qpos_data = torch.from_numpy(qpos).float()
            qpos_data = (
                qpos_data - self.trainer.datamodule.data_train.norm_stats["qpos_mean"]
            ) / self.trainer.datamodule.data_train.norm_stats["qpos_std"]
            goal_cond = self.trainer.datamodule.data_train.get_goal(obs)

            if self.hparams.get("num_envs", 1) == 1:
                qpos_data = qpos_data.unsqueeze(0)
                if "image" in obs.keys() or self.trainer.datamodule.data_train.pointmap:
                    image_data = image_data.unsqueeze(0)
                if goal_cond is not None:
                    goal_cond = goal_cond.unsqueeze(0)

            input_dict = dict(
                qpos=qpos_data.to(self.device),
                actions=None,
                is_pad=None,
                goal_cond=goal_cond.to(self.device) if goal_cond is not None else None,
            )
            if "image" in obs.keys() or self.trainer.datamodule.data_train.pointmap:
                input_dict.update(
                    dict(
                        image=image_data.to(self.device),
                    )
                )
            elif "pointcloud" in obs.keys():
                input_dict.update(
                    dict(
                        pcds=pcds,
                    )
                )

            pred_action = self.policy(input_dict)["a_hat"].cpu().numpy()
            pred_action = np.stack(
                [
                    temp_agg[i](pred_action[i])
                    for i in range(self.hparams.get("num_envs", 1))
                ]
            )

            pred_action = (
                pred_action
                * self.trainer.datamodule.data_train.norm_stats["action_std"]
                + self.trainer.datamodule.data_train.norm_stats["action_mean"]
            )

            obs, reward, terminated, truncated, info = self.env.step(
                pred_action.squeeze(0)
            )
            if self.hparams.env_id == "PegInsertionSide-v0":
                grasp = info["is_grasped"] | grasp
                align = info["pre_inserted"] | align
            rewards.append(reward)
            if terminated or truncated:
                success = info["success"]
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
        if "3steps" in self.hparams.env_id:
            mean_grasp = 1 if grasp else 0
            mean_align = 1 if align else 0
            val_metrics_dict.update(dict(mean_grasp=mean_grasp, mean_align=mean_align))

        # update and log metrics
        self.val_metrics(val_metrics_dict)
        self.log_dict(
            self.val_metrics.metrics_dict(),
            on_step=False,
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

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.policy = torch.compile(self.policy)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = U.build_optimizer(
            self.hparams.optimizer, self.policy, self.hparams.param_dicts
        )
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
        return {"optimizer": optimizer}
