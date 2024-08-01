import os
import pickle

import clip
import einops
import hydra
import numpy as np
import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

import src.utils as U
from src.data.components.rlbench.constants import SCENE_BOUNDS, loc_bounds
from src.data.components.transformpcd import ComposePCD

from .rotation_conversions import matrix_to_rotation_6d, quaternion_to_matrix

ALL_TASKS = [
    "basketball_in_hoop",
    "beat_the_buzz",
    "change_channel",
    "change_clock",
    "close_box",
    "close_door",
    "close_drawer",
    "close_fridge",
    "close_grill",
    "close_jar",
    "close_laptop_lid",
    "close_microwave",
    "hang_frame_on_hanger",
    "insert_onto_square_peg",
    "insert_usb_in_computer",
    "lamp_off",
    "lamp_on",
    "lift_numbered_block",
    "light_bulb_in",
    "meat_off_grill",
    "meat_on_grill",
    "move_hanger",
    "open_box",
    "open_door",
    "open_drawer",
    "open_fridge",
    "open_grill",
    "open_microwave",
    "open_oven",
    "open_window",
    "open_wine_bottle",
    "phone_on_base",
    "pick_and_lift",
    "pick_and_lift_small",
    "pick_up_cup",
    "place_cups",
    "place_hanger_on_rack",
    "place_shape_in_shape_sorter",
    "place_wine_at_rack_location",
    "play_jenga",
    "plug_charger_in_power_supply",
    "press_switch",
    "push_button",
    "push_buttons",
    "put_books_on_bookshelf",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_knife_on_chopping_board",
    "put_money_in_safe",
    "put_rubbish_in_bin",
    "put_umbrella_in_umbrella_stand",
    "reach_and_drag",
    "reach_target",
    "scoop_with_spatula",
    "screw_nail",
    "setup_checkers",
    "slide_block_to_color_target",
    "slide_block_to_target",
    "slide_cabinet_open_and_place_cups",
    "stack_blocks",
    "stack_cups",
    "stack_wine",
    "straighten_rope",
    "sweep_to_dustpan",
    "sweep_to_dustpan_of_size",
    "take_frame_off_hanger",
    "take_lid_off_saucepan",
    "take_money_out_safe",
    "take_plate_off_colored_dish_rack",
    "take_shoes_out_of_box",
    "take_toilet_roll_off_stand",
    "take_umbrella_out_of_umbrella_stand",
    "take_usb_out_of_computer",
    "toilet_seat_down",
    "toilet_seat_up",
    "tower3",
    "turn_oven_on",
    "turn_tap",
    "tv_on",
    "unplug_charger",
    "water_plants",
    "wipe_desk",
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_TASKS)}


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def build_clip_model(clip_model: str = "ViT-B/16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(clip_model, device=device, download_root="./.cache/clip")
    clip_model.requires_grad_(False)
    clip_model.eval()
    return clip_model, device


def get_qpos_data(cfg, obs, task_name, collision=False):
    if not collision:
        qpos = np.stack(
            [np.concatenate([obs_.gripper_pose, [obs_.gripper_open]]) for obs_ in obs]
        )
    else:
        qpos = np.stack(
            [
                np.concatenate(
                    [obs_.gripper_pose, [obs_.gripper_open], [obs_.ignore_collisions]]
                )
                for obs_ in obs
            ]
        )
    qpos = torch.from_numpy(qpos).float()

    pos_min = torch.FloatTensor(loc_bounds[task_name][0])
    pos_max = torch.FloatTensor(loc_bounds[task_name][1])
    qpos[:, :3] = (qpos[:, :3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
    qpos[:, 3:7] = torch.nn.functional.normalize(qpos[:, 3:7], dim=-1)
    if cfg.data.train.rot_type == "6d":
        qpos_rot = matrix_to_rotation_6d(quaternion_to_matrix(qpos[:, 3:7]))
    else:
        raise NotImplementedError
    qpos = torch.cat([qpos[:, :3], qpos_rot, qpos[:, 7:]], dim=-1)
    return qpos.float()


def get_pcd(
    obs,
    cameras,
    transform_pcd,
    use_mask=False,
    n_obs_steps=1,
    invalid_mask_values=[
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        246,
    ],
):
    # transform_pcd = ComposePCD(transform_pcd)
    transform_pcd = ComposePCD(hydra.utils.instantiate(transform_pcd))
    obs_pcds = []
    for obs_step_idx in range(n_obs_steps):
        obs_ = obs[obs_step_idx]

        colors, coords = [], []
        if use_mask:
            masks = []
        for camera_name in cameras:
            colors.append(getattr(obs_, f"{camera_name}_rgb").astype(float))
            coords.append(getattr(obs_, f"{camera_name}_point_cloud").astype(float))
            if use_mask:
                masks.append(getattr(obs_, f"{camera_name}_mask").astype(float))

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
        if use_mask:
            masks = np.stack(masks)
            masks = einops.rearrange(masks, "n h w -> (n h w)")
            masks = masks[scene_mask]
            for v in invalid_mask_values:
                masks[masks == v] = 0
            masks[masks > 0] = 1
            masks = masks.astype(float)
            pcds = transform_pcd(
                dict(
                    coord=coords,
                    color=np.concatenate([colors, masks[:, None]], axis=-1),
                )
            )
            pcds["mask"] = pcds["feat"][:, -1].bool()
            pcds["feat"] = pcds["feat"][:, :-1]
        else:
            pcds = transform_pcd(dict(coord=coords, color=colors))

        obs_pcds.append(pcds)

    pcds = U.point_collate_fn(obs_pcds)

    return pcds


def get_gt_action(cfg, demo, collision=False):
    action_ls = []
    for i, obs in enumerate(demo):
        if not collision:
            qpos_data = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        else:
            qpos_data = np.concatenate(
                [obs.gripper_pose, [obs.gripper_open], [obs.ignore_collisions]]
            )
        qpos_data = torch.from_numpy(qpos_data).float()
        action_ls.append(qpos_data.unsqueeze(0))
    return action_ls


def rotation_matrix_x(theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def rotation_matrix_y(theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rotation_matrix_z(theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def apply_rotation(matrix, theta, axis="x"):
    if axis == "x":
        rot_matrix = rotation_matrix_x(theta)
    elif axis == "y":
        rot_matrix = rotation_matrix_y(theta)
    elif axis == "z":
        rot_matrix = rotation_matrix_z(theta)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    return np.dot(rot_matrix, matrix)


def translation_matrix(dx, dy, dz):
    return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])


# Function to apply translation to a 4x4 matrix
def apply_translation(matrix, dx, dy, dz):
    trans_matrix = translation_matrix(dx, dy, dz)
    return np.dot(trans_matrix, matrix)


def build_env_and_task(cfg):
    assert not cfg.live_demos, "Live demos are not supported in this script."

    obs_config = ObservationConfig()
    obs_config.set_all(False)

    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.front_camera.set_all(True)
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        ),
        dataset_root="" if cfg.live_demos else cfg.data_root,
        obs_config=ObservationConfig(),
        headless=cfg.headless,
    )
    env.launch()

    if cfg.camera_view_test.apply:
        fron_cam_matrix = env._scene._cam_front.get_matrix()
        env._scene._cam_front.set_matrix(
            apply_translation(
                apply_rotation(
                    fron_cam_matrix,
                    cfg.camera_view_test.rot_angle,
                    axis=cfg.camera_view_test.rot_axis,
                ),
                *cfg.camera_view_test.transl,
            )
        )

    task_type = task_file_to_task_class(cfg.rlbench_task)
    task = env.get_task(task_type)

    return env, task


def reset_task(task, cfg, from_episode_number):
    var_num = pickle.load(
        open(
            os.path.join(
                cfg.data_root,
                cfg.rlbench_task,
                "all_variations",
                "episodes",
                f"episode{from_episode_number}",
                "variation_number.pkl",
            ),
            "rb",
        )
    )
    task.set_variation(-1)
    demos = task.get_demos(
        1,
        random_selection=False,
        live_demos=cfg.live_demos,
        from_episode_number=from_episode_number,
    )
    task.set_variation(var_num)
    description, obs = task.reset_to_demo(demos[0])

    return task, demos, description, obs
