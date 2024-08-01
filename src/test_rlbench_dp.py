from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import clip
import einops
import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError

OmegaConf.register_new_resolver("eval", eval)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

import src.utils.rlbench_utils as RU
from src.data.components.rlbench.constants import loc_bounds
from src.models.rlbench_dp_bc_module import RLBenchDiffusionPolicyBCModule
from src.utils import RankedLogger, extras
from src.utils.rotation_conversions import matrix_to_quaternion, rotation_6d_to_matrix

log = RankedLogger(__name__, rank_zero_only=True)


def get_image_data(
    cfg,
    obs,
    device,
):
    obs_dict = dict()
    for camera_name in cfg.data.train.cameras:
        rgb = np.stack(
            [getattr(obs_, f"{camera_name}_rgb").astype(float) / 255.0 for obs_ in obs]
        )
        obs_dict[f"{camera_name}_rgb"] = einops.rearrange(
            torch.from_numpy(rgb).float().to(device),
            "k h w c -> 1 k c h w",
        )
        if cfg.data.train.include_depth:
            depth = np.stack(
                [
                    getattr(obs_, f"{camera_name}_depth").astype(float)[:, :, None]
                    for obs_ in obs
                ]
            )
            obs_dict[f"{camera_name}_depth"] = einops.rearrange(
                torch.from_numpy(depth).float().to(device),
                "k h w c -> 1 k c h w",
            )

    return obs_dict


def convert_obs(cfg, obs, n_obs_steps, device):
    qpos_data = RU.get_qpos_data(cfg, obs, cfg.rlbench_task, cfg.data.train.collision)
    data_dict = dict(
        obs=dict(
            qpos=qpos_data.unsqueeze(0).to(device),
        ),
    )

    if "pcd" in cfg.data.train._target_.lower():
        pcds = RU.get_pcd(
            obs,
            cfg.data.train.cameras,
            cfg.data.train.transform_pcd,
            cfg.data.train.get("use_mask", False),
            n_obs_steps=n_obs_steps,
        )
        for k in pcds:
            pcds[k] = pcds[k].to(device)
        data_dict["obs"]["pcds"] = pcds
    else:
        obs_dict = get_image_data(cfg, obs, device=device)
        data_dict["obs"].update(**obs_dict)

    return data_dict


@torch.no_grad()
def eval(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    env, task = RU.build_env_and_task(cfg)
    log.info(f"Testing RLBench {cfg.rlbench_task} task...")

    clip_model, device = RU.build_clip_model()
    model = RLBenchDiffusionPolicyBCModule.load_from_checkpoint(
        cfg.ckpt_path, map_location=lambda storage, loc: storage.cuda(0)
    )
    model.eval()
    model = model.to(device)
    n_obs_steps = model.policy.n_obs_steps

    pos_min = np.array(loc_bounds[cfg.rlbench_task][0])
    pos_max = np.array(loc_bounds[cfg.rlbench_task][1])

    success_rate = 0
    for from_episode_number in range(cfg.episodes_num):
        task, demos, description, obs = RU.reset_task(task, cfg, from_episode_number)

        hist_obs = [obs for _ in range(n_obs_steps)]
        reward = None
        max_reward = 0.0
        total_reward = 0

        log.info(
            f"Episode {from_episode_number + 1} / {cfg.episodes_num}: {description[0]}"
        )

        if cfg.offline:
            gt_qpos_data = RU.get_gt_action(cfg, demos[0], cfg.data.train.collision)
            max_steps = len(gt_qpos_data)
            for step_id in range(max_steps):
                action = gt_qpos_data[step_id]

                action[..., -1] = torch.round(action[..., -1])
                action = action.squeeze(0)
                obs, reward, terminate = task.step(action)
                try:
                    obs, reward, terminate = task.step(action)
                    if reward == 1:
                        success_rate += 1
                        break
                    if terminate:
                        print("The episode has terminated!")
                        break
                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(e)
                    reward = 0
        else:
            step_id = 0
            while step_id < cfg.max_steps:
                log.info(f"Step {step_id + 1} / {cfg.max_steps}")

                assert len(hist_obs) == n_obs_steps

                data_dict = convert_obs(cfg, hist_obs, n_obs_steps, device)

                description_token = clip.tokenize(description[0]).to(device)
                task_goal = (
                    clip_model.encode_text(description_token).reshape(1, -1).float()
                )
                data_dict["goal"] = dict(task_emb=task_goal)

                pred_action = model(data_dict)["action"]

                if pred_action.dim() == 2:
                    pred_action = pred_action[:, None, :]

                rot_6d = pred_action[..., 3:9]
                quat = matrix_to_quaternion(rotation_6d_to_matrix(rot_6d))
                pred_action = torch.cat(
                    [pred_action[..., :3], quat, pred_action[..., 9:]], dim=-1
                )

                pred_action = pred_action.cpu().numpy()
                pred_action[..., :3] = (pred_action[..., :3] + 1) / 2 * (
                    pos_max - pos_min
                ) + pos_min

                if cfg.data.train.collision:
                    pred_action[..., -1] = pred_action[..., -1] = (
                        pred_action[..., -1] > 0.5
                    ).astype(bool)
                    pred_action[..., -2] = (pred_action[..., -2] > 0.5).astype(float)
                else:
                    pred_action[..., -1] = (pred_action[..., -1] > 0.5).astype(float)

                pred_action = pred_action.squeeze(0)

                for action_idx in range(pred_action.shape[0]):
                    try:
                        obs, reward, terminate = task.step(pred_action[action_idx])

                        max_reward = max(max_reward, reward)

                        step_id += 1

                        if reward == 1:
                            success_rate += 1
                            break

                        if terminate:
                            log.info("The episode has terminated!")

                    except (IKError, ConfigurationPathError, InvalidActionError) as e:
                        print(e)
                        reward = 0
                        step_id += 1
                        continue

                    hist_obs = hist_obs[1:]
                    hist_obs.append(obs)

                if reward > 0:
                    break

            log.info(f"success num: {success_rate}")
            total_reward += max_reward

    success_rate = success_rate / 25.0
    log.info(f"total_reward: {total_reward}, success_rate: {success_rate}")

    full_path = os.path.join(cfg.result_path, cfg.result_file)
    os.makedirs(cfg.result_path, exist_ok=True)

    with open(full_path, "a") as file:
        file.write(f"{cfg.result_name}: {success_rate}\n")

    log.info(f"Results saved to {full_path}")

    env.shutdown()


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="test_rlbench_dp.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    OmegaConf.set_struct(cfg, False)
    extras(cfg)

    eval(cfg)


if __name__ == "__main__":
    main()
