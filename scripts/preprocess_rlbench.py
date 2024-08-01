import argparse
import os
import pickle

import clip
import numpy as np
import torch
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_names",
    type=str,
    default="turn_tap",
    help="comma separated list of task names",
)
parser.add_argument(
    "--camera_views",
    type=str,
    default="front",
    help="comma separated list of camera views",
)
parser.add_argument(
    "--modalities",
    type=str,
    default="rgb,depth,mask,point_cloud",
    help="comma separated list of modalities",
)
parser.add_argument(
    "--low_dim_states",
    type=str,
    default="joint_velocities,joint_positions,joint_forces,task_low_dim_state",
    help="comma separated list of low dim states",
)
parser.add_argument(
    "--gripper_states",
    type=str,
    default="gripper_open,gripper_pose,gripper_matrix,gripper_joint_positions,gripper_touch_forces",
    help="comma separated list of low dim states",
)
parser.add_argument(
    "--headless",
    action="store_true",
    default=True,
    help="run in headless mode",
)

args = parser.parse_args()


root = "data/rlbench/raw"
save_root = "data/rlbench/processed"
device = "cuda" if torch.cuda.is_available() else "cpu"


clip_model = "ViT-B/16"
clip_model, _ = clip.load(
    clip_model, device=device, download_root=os.path.expanduser("~/.cache/clip")
)
clip_model.requires_grad_(False)
clip_model.eval()


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


task_names = args.task_names.split(",")
for stage in ["train", "val"]:
    DATASET = os.path.join(root, stage)
    episodes_num = 100 if stage == "train" else 25

    for task_name in task_names:
        print(f"Processing {stage} data of task {task_name}...")
        obs_config = ObservationConfig()
        obs_config.set_all(True)

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
                gripper_action_mode=Discrete(),
            ),
            dataset_root=os.path.join(root, stage),
            obs_config=ObservationConfig(),
            headless=args.headless,
        )
        env.launch()

        task_type = task_file_to_task_class(task_name)
        task = env.get_task(task_type)

        for from_episode_number in tqdm(range(episodes_num)):
            var_num = pickle.load(
                open(
                    os.path.join(
                        DATASET,
                        task_name,
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
                live_demos=False,
                from_episode_number=from_episode_number,
            )

            task.set_variation(var_num)
            description, obs = task.reset_to_demo(demos[0])

            gt_decription = pickle.load(
                open(
                    os.path.join(
                        DATASET,
                        task_name,
                        "all_variations",
                        "episodes",
                        f"episode{from_episode_number}",
                        "variation_descriptions.pkl",
                    ),
                    "rb",
                )
            )

            assert (
                gt_decription[0] == description[0]
            ), f"gt: {gt_decription[0]}, task desc: {description[0]}"
            description_token = clip.tokenize(description[0]).to(device)
            task_goal = clip_model.encode_text(description_token).cpu().numpy()

            demo = np.array(demos[0]).flatten()
            demo_array = []
            for frame in demo:
                frame_dict = {
                    "ignore_collisions": frame.ignore_collisions,
                }
                for view in args.camera_views.split(","):
                    for modality in args.modalities.split(","):
                        frame_dict[f"{view}_{modality}"] = getattr(
                            frame, f"{view}_{modality}"
                        )
                for state in args.low_dim_states.split(","):
                    frame_dict[state] = getattr(frame, state)
                for state in args.gripper_states.split(","):
                    frame_dict[state] = getattr(frame, state)

                demo_array.append(frame_dict)

            os.makedirs(os.path.join(save_root, stage, task_name), exist_ok=True)
            np.save(
                os.path.join(
                    save_root, stage, task_name, f"ep{from_episode_number}.npy"
                ),
                dict(demo=demo_array, task_goal=task_goal),
                allow_pickle=True,
            )
        env.shutdown()
