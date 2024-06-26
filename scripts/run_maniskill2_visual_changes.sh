ckpt_path=$1
model=$2
seed=$3

echo "Starting validation with model: ${model} and checkpoint path: ${ckpt_path}"
echo "Seed: ${seed}"
# 1) lighting intensity evaluation
light_intensities=("0.03" "0.05" "0.15" "0.6" "1.8" "3")
for light in "${light_intensities[@]}"
do
echo "Running validation for light intensity: ${light}"
python src/validate.py exp_maniskill2_act_policy=base \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_task=StackCube-v0 \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} \
    ckpt_path=${ckpt_path} \
    model.env_id=StackCube-light-${light} \
    task_name=${model}-light-${light} \
    seed=${seed}
done

# 2) noise level evaluation
noise_levels=("2" "16" "32" "64")
for noise in "${noise_levels[@]}"
do
echo "Running validation for noise level: ${noise}"
python src/validate.py exp_maniskill2_act_policy=base \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_task=StackCube-v0 \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} \
    ckpt_path=${ckpt_path} \
    model.env_id=StackCube-v0 \
    model.shader_dir="rt" \
    model.rt_samples_per_pixel=${noise}  model.rt_use_denoiser=false \
    task_name=${model}-noise-${noise} \
    seed=${seed}
done

# background color evaluation
colors=("0.2" "0.6" "1.0")

for red_color in "${colors[@]}"
do
echo "Running validation for red background with R value: ${red_color}"
python src/validate.py exp_maniskill2_act_policy=base \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_task=StackCube-v0 \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} \
    ckpt_path=${ckpt_path} \
    model.env_id=StackCube-background-red-${red_color} \
    task_name=${model}-red_color-${red_color} \
    seed=${seed}
done

for green_color in "${colors[@]}"
do
echo "Running validation for green background with G value: ${green_color}"
python src/validate.py exp_maniskill2_act_policy=base \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_task=StackCube-v0 \
    exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} \
    ckpt_path=${ckpt_path} \
    model.env_id=StackCube-background-green-${green_color} \
    task_name=${model}-green_color-${green_color} \
    seed=${seed}
done













