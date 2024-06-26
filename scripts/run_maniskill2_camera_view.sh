ckpt_path=$1
task=$2 
model=$3
seed=$4

task_name=${task%-v0}
if [ "$task_name" = "PegInsertionSide" ];then
    model_env_id=${task_name}-3steps-MultiView
  else
    model_env_id=${task_name}-MultiView
fi

echo "Starting validation for task: ${task} with model: ${model} and checkpoint path: ${ckpt_path}"
echo "Model environment ID: ${model_env_id}"
echo "Seed: ${seed}"

if [[ "$model" == *"pcd"* ]]; then
    echo "Running validation for PCD camera views..."
    # Change the camera view for PCD by specifying the `camera_ids` parameter
    # Mapping of camera names to IDs:
    # 4:left_camera_5
    # 5:down_camera_5
    # 6:left_camera_10
    # 7:down_camera_10
    for camera in 4 5 6 7
    do
    echo "Validating with camera ID: ${camera}"
    python src/validate.py exp_maniskill2_act_policy=base \
        exp_maniskill2_act_policy/maniskill2_model@maniskill2_task=${task} \
        exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} \
        ckpt_path=${ckpt_path} \
        model.env_id=${model_env_id} \
        data.train.camera_ids=${camera} seed=${seed}
    done
else
    echo "Running validation for RGB or RGBD camera views..."
    # Change the camera view for RGB and RGBD by specifying the `camera_names` parameter
    for camera in left_camera_5 left_camera_10 down_camera_5 down_camera_10
    do 
    echo "Validating with camera view: ${camera}"
    python src/validate.py exp_maniskill2_act_policy=base \
        exp_maniskill2_act_policy/maniskill2_model@maniskill2_task=${task} \
        exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} \
        ckpt_path=${ckpt_path} \
        model.env_id=${model_env_id} \
        data.train.camera_names=${camera} seed=${seed}
    done
fi

echo "Validation completed."











