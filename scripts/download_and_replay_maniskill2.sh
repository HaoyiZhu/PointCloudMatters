# !/bin/bash


# 1) download demonstrations
# reference: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
mkdir data/maniskill2
cd data/maniskill2
mkdir demos
echo "Downloading demonstrations to $(pwd)/demos"
python -m mani_skill2.utils.download_demo -o ./demos/ all


# 2) download assets
echo "Downloading assets to $(pwd)/data"
export MS2_ASSET_DIR=$(pwd)/data
echo "
export MS2_ASSET_DIR=$MS2_ASSET_DIR
" >> ~/.bashrc
if [ -f ~/.zshrc ]; then
    echo "
export MS2_ASSET_DIR=$MS2_ASSET_DIR
" >> ~/.zshrc
fi
python -m mani_skill2.utils.download_asset all -y


# 3) replay demonstrations
# reference: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html
# we use `pd_ee_delta_pose` as the target control mode by default
echo "Replaying demonstrations..."
for rigid_task in {"PegInsertionSide-v0","StackCube-v0","PickCube-v0"}
do
for obs_mode in {"rgbd","pointcloud"}
do
python -m mani_skill2.trajectory.replay_trajectory \
  --traj-path demos/v0/rigid_body/$rigid_task/trajectory.h5 \
  --save-traj --target-control-mode pd_ee_delta_pose --obs-mode $obs_mode --num-procs 10
done
done

export CUDA_VISIBLE_DEVICES=0
python -m mani_skill2.utils.precompile_mpm
for soft_task in {"Hang-v0","Pour-v0","Excavate-v0","Fill-v0"}
do
for obs_mode in {"rgbd","pointcloud"}
do
python -m mani_skill2.trajectory.replay_trajectory \
  --traj-path demos/v0/soft_body/$soft_task/trajectory.h5 \
  --save-traj --target-control-mode pd_ee_delta_pose --obs-mode $obs_mode --num-procs 10
done
done

# For `TurnFaucet-v0`, we only use the demonstrations from the following models:
rigid_task=TurnFaucet-v0
for model_id in {"5002","5021","5023","5028","5029","5045","5047","5051","5056","5063"}
do
for obs_mode in {"rgbd","pointcloud"}
do
python -m mani_skill2.trajectory.replay_trajectory \
  --traj-path demos/v0/rigid_body/${rigid_task}/${model_id}.h5 \
  --save-traj --target-control-mode pd_ee_delta_pose --obs-mode $obs_mode --num-procs 10
done
done
# then we need to merge the demonstrations
mkdir demos/v0/rigid_body/TurnFaucet-v0/merged_pointcloud/
mv demos/v0/rigid_body/TurnFaucet-v0/50*.pointcloud.* demos/v0/rigid_body/TurnFaucet-v0/merged_pointcloud/
python -m mani_skill2.trajectory.merge_trajectory \
    -i demos/v0/rigid_body/TurnFaucet-v0/merged_pointcloud -p "*.pointcloud.pd_ee_delta_pose.h5" \
    -o demos/v0/rigid_body/TurnFaucet-v0/trajectory.pointcloud.pd_ee_delta_pose.h5
mkdir demos/v0/rigid_body/TurnFaucet-v0/merged_rgbd/
mv demos/v0/rigid_body/TurnFaucet-v0/50*.rgbd.* demos/v0/rigid_body/TurnFaucet-v0/merged_rgbd/
python -m mani_skill2.trajectory.merge_trajectory \
    -i demos/v0/rigid_body/TurnFaucet-v0/merged_rgbd -p "*.rgbd.pd_ee_delta_pose.h5" \
    -o demos/v0/rigid_body/TurnFaucet-v0/trajectory.rgbd.pd_ee_delta_pose.h5
