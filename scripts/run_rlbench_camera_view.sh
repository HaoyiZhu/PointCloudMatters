#!/bin/bash

policy=$1   # policy can be either "diffusion" or act
ckpt_path=$2
model=$3
task=$4
seed=$5

if [ "$policy" != "diffusion" ] && [ "$policy" != "act" ]; then
    echo "Policy should be either diffusion or act, but got $policy"
    exit 1
fi

echo "Starting testing ${policy} policy for task: ${task} with model: ${model} and checkpoint path: ${ckpt_path}"
echo "Seed: ${seed}"


rot_axes=("y" "z")
rot_angles=(-5 -10)

for rot_axis in "${rot_axes[@]}"; do
    for rot_angle in "${rot_angles[@]}"; do
        transl=(0 0 0)

        if [ "$rot_axis" == "y" ]; then
            transl[0]=$(echo "$rot_angle * -0.02" | bc)
            transl[2]=$(echo "$rot_angle * -0.02" | bc)
        fi

        echo "Rotation Axis: $rot_axis, Rotation Angle: $rot_angle, Translation: ${transl[@]}"

        if [ "$policy" == "diffusion" ]; then
            test_script=test_rlbench_dp.py
        elif [ "$policy" == "act" ]; then
            test_script=test_rlbench_act.py
        fi

        python src/${test_script} exp_rlbench_${policy}_policy=base \
            rlbench_task=$task exp_rlbench_${policy}_policy/rlbench_model@rlbench_model=$model \
            ckpt_path=$ckpt_path seed=$seed \
            camera_view_test.apply=true camera_view_test.rot_axis=${rot_axis} camera_view_test.rot_angle=${rot_angle} \
            camera_view_test.transl="[${transl[0]},${transl[1]},${transl[2]}]" \
            result_file=test_camera_view_${rot_axis}_${rot_angle}.txt

    done
done

echo "Testing completed."











