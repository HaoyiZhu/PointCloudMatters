<div align="center">

# Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[**Project Page**](https://haoyizhu.github.io/pcm/) | [**Arxiv**](https://arxiv.org/abs/2402.02500)

[Haoyi Zhu](https://www.haoyizhu.site/), [Yating Wang](https://scholar.google.com/citations?hl=zh-CN&user=5SuBWh0AAAAJ), [Di Huang](https://dihuang.me/), [Weicai Ye](https://ywcmaike.github.io/), [Wanli Ouyang](https://wlouyang.github.io/), [Tong He](http://tonghe90.github.io/)
</div>

<p align="center">
    <img src="assets/overview.png" alt="overview" width="90%" />
</p>

This is the official implementation of **NeurIPS 2024 D&B track** paper "Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning". Real-world codes can be found in [RealRobot](https://github.com/HaoyiZhu/RealRobot).

In robot learning, the observation space is crucial due to the distinct characteristics of different modalities, which can potentially become a bottleneck alongside policy design. In this study, we explore the influence of various observation spaces on robot learning, focusing on three predominant modalities: RGB, RGB-D, and point cloud. We introduce OBSBench, a benchmark comprising two simulators and 125 tasks, along with standardized pipelines for various encoders and policy baselines. Extensive experiments on diverse contact-rich manipulation tasks reveal a notable trend: point cloud-based methods, even those with the simplest designs, frequently outperform their RGB and RGB-D counterparts. This trend persists in both scenarios: training from scratch and utilizing pre-training. Furthermore, our findings demonstrate that point cloud observations often yield better policy performance and significantly stronger generalization capabilities across various geometric and visual conditions. These outcomes suggest that the 3D point cloud is a valuable observation modality for intricate robotic tasks. We also suggest that incorporating both appearance and coordinate information can enhance the performance of point cloud methods. We hope our work provides valuable insights and guidance for designing more generalizable and robust robotic models.


## :clipboard: Contents

- [Project Structure](#telescope-project-structure)
- [Installation](#installation)
- [Data Preparation](#mag-data-preparation)
- [Training and Evaluation](#rocket-training-and-evaluation)
- [Gotchas](#tada-gotchas)
- [Trouble Shooting](#bulb-trouble-shooting)
- [License](#books-license)
- [Acknowledgement](#sparkles-acknowledgement)
- [Citation](#pencil-citation)


## :telescope: Project Structure

Our codebase draws significant inspiration from the excellent [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template). The directory structure of this project is organized as follows:

<details>
<summary><b>Show directory structure</b></summary>

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                         <- Callbacks configs
│   ├── data                              <- Data configs
│   ├── debug                             <- Debugging configs
│   ├── exp_maniskill2_act_policy         <- ManiSkill2 w. ACT policy experiment configs
|   ├── exp_maniskill2_diffusion_policy   <- ManiSkill2 w. diffusion policy experiment configs
│   ├── extras                            <- Extra utilities configs
│   ├── hydra                             <- Hydra configs
│   ├── local                             <- Local configs
│   ├── logger                            <- Logger configs
│   ├── model                             <- Model configs
│   ├── paths                             <- Project paths configs
│   ├── trainer                           <- Trainer configs
|   |
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data, e.g. ManiSkill2 replayed trajectories
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── scripts                <- Shell scripts
|
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── validate.py              <- Run evaluation
│   └── train.py                 <- Run training
│
├── .gitignore                <- List of files ignored by git
├── .project-root             <- File for inferring the position of project root directory
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

</details>

## :hammer: Installation

<details>
<summary><b>Basics</b></summary>

```bash
# clone project
git clone https://github.com/HaoyiZhu/PointCloudMatters.git
cd PointCloudMatters

# crerate conda environment
conda create -n pcm python=3.11 -y
conda activate pcm

# install PyTorch, please refer to https://pytorch.org/ for other CUDA versions
# e.g. cuda 11.8:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install basic packages
pip3 install -r requirements.txt
```
</details>

<details>
<summary><b> Point cloud related</b></summary>

```bash
# please install with your PyTorch and CUDA version
# e.g. torch 2.3.0 + cuda 118:
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu118
```
> **Note**: `spconv` must matches your CUDA version, see [official Github](https://github.com/traveller59/spconv) for more information.
```bash
# e.g. for CUDA 11.8:
pip3 install spconv-cu118
```
```bash
# build FPS sampling operations (CUDA required)
cd libs/pointops
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python setup.py install
cd ../..
```

</details>

<details>
<summary><b> ManiSkill2 </b></summary>

```bash
pip install mani-skill2==0.5.3 && pip cache purge
```

You can test whether your `ManiSkill2` is installed successfully by running:
```bash
python -m mani_skill2.examples.demo_random_action
```

</details>

<details>
<summary><b> RLBench </b></summary>

> **Note**: Installing RLbench can be challenging. We recommend referring to [PerAct's installation guides](https://github.com/peract/peract?tab=readme-ov-file#installation) for more assistance.

#### 1. PyRep and Coppelia Simulator

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

#### 2. RLBench

We use [PerAct's RLBench fork](https://github.com/MohitShridhar/RLBench/tree/peract). 

```bash
cd <install_dir>
git clone -b peract https://github.com/MohitShridhar/RLBench.git # note: 'peract' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).


</details>


## :mag: Data Preparation

<details>
<summary><b> ManiSkill2 </b></summary>

You can simply run the following to download and replay demonstrations:
```bash
bash scripts/download_and_replay_maniskill2.sh
```

</details>

<details>
<summary><b> RLBench </b></summary>

#### 1. Quick Start with PerAct's Pre-generated Datasets

[PerAct](https://github.com/peract/peract?tab=readme-ov-file#pre-generated-datasets) has provided [pre-generated RLBench demonstrations](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ&usp=share_link) for the 18 tasks it used. Each task contains 100 episodes for training, and 25 for testing and validation. Please download and extract them into `./data/rlbench/raw`. Your data directory structure may look like the following:

```
├── data
│   ├── ...
│   ├── rlbench
│   │   ├── raw
|   |   |   ├── train
|   |   |   |   ├── close_jar
|   |   |   |   |   ├── all_variations
|   |   |   |   |   |   ├── episodes
|   |   |   |   |   |   |   ├── episode0
|   |   |   |   |   |   |   ├── episode1
|   |   |   |   |   |   |   ├── ...
|   |   |   |   ├── open_drawer
|   |   |   |   ├── ...
|   |   |   ├── val
|   |   |   |   ├── ...
|   |   |   ├── test
|   |   |   |   ├── ...
│   └── ...
```

To facilite the data loading speed during training, we provide a script to pre-process the raw data. You can run the following example command and it will generate processed data under `./data/rlbench/processed`.

```bash
# e.g. to pre-process task turn_tap with front camera:
python scripts/preprocess_rlbench.py --task_names turn_tap --camera_views front
```

#### 2. Data Generation by Your Own

You can also generate your own data on [all tasks RLBench supported](https://github.com/stepjam/RLBench/tree/master/rlbench/tasks).

Coming soon.

</details>


## :rocket: Training and Evaluation

<details>
<summary><b> ManiSkill2 </b></summary>

- Train with RGB(-D) image observation:
  ```bash
  # ACT policy example:
  python src/train.py exp_maniskill2_act_policy=base exp_maniskill2_act_policy/maniskill2_task@maniskill2_task=${task} exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} seed=${seed}
  # Diffusion policy example:
  python src/train.py exp_maniskill2_diffusion_policy=base exp_maniskill2_diffusion_policy/maniskill2_task@maniskill2_task=${task} exp_maniskill2_diffusion_policy/maniskill2_model@maniskill2_model=${model} seed=${seed}
  ```

- Train with point cloud observation:
  ```bash
  # ACT policy example:
  python src/train.py exp_maniskill2_act_policy=base exp_maniskill2_act_policy/maniskill2_pcd_task@maniskill2_pcd_task=${task} exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} seed=${seed}
  # Diffusion policy example:
  python src/train.py exp_maniskill2_diffusion_policy=base exp_maniskill2_diffusion_policy/maniskill2_pcd_task@maniskill2_pcd_task=${task} exp_maniskill2_diffusion_policy/maniskill2_model@maniskill2_model=${model} seed=${seed}
  ```

- Evaluate a checkpoint:
  ```bash
  python src/validate.py exp_maniskill2_act_policy=base exp_maniskill2_act_policy/maniskill2_pcd_task@maniskill2_pcd_task=${task} exp_maniskill2_act_policy/maniskill2_model@maniskill2_model=${model} ckpt_path=${path/to/checkpoint} seed=${seed}
  ```

- Zero-shot generalization evaluation:
  - To evaluate camera view generalization experiments, run [scripts/run_maniskill2_camera_view.sh](scripts/run_maniskill2_camera_view.sh). The script evaluates the given `checkpoint` of the given `model` on the given `task` with four different camera views, using the specified `seed`. See the script for more details. For example:
  ```bash
  bash scripts/run_maniskill2_camera_view.sh ${path/to/checkpoint} ${task} ${model} ${seed}
  ```
  - To evaluate visual changes generalization experiments, run [scripts/run_maniskill2_visual_changes.sh](scripts/run_maniskill2_visual_changes.sh). The script evaluates the given `checkpoint` of the given `model` with different lighting conditions, noise levels and background colors, using the specified `seed`. See the script for more details. Note that currently only `StackCube` task is supported. For example:
  ```bash
  bash scripts/run_maniskill2_visual_changes.sh ${path/to/checkpoint} ${model} ${seed}
  ```
  
Detailed configurations can be found in [configs/exp_maniskill2_act_policy](configs/exp_maniskill2_act_policy) and [configs/exp_maniskill2_diffusion_policy](configs/exp_maniskill2_diffusion_policy).

Currently supported tasks can be found in [configs/exp_maniskill2_act_policy/maniskill2_task](configs/exp_maniskill2_act_policy/maniskill2_task), [configs/exp_maniskill2_act_policy/maniskill2_pcd_task](configs/exp_maniskill2_act_policy/maniskill2_pcd_task), [configs/exp_maniskill2_diffusion_policy/maniskill2_task](configs/exp_maniskill2_diffusion_policy/maniskill2_task) and [configs/exp_maniskill2_diffusion_policy/maniskill2_pcd_task](configs/exp_maniskill2_diffusion_policy/maniskill2_pcd_task).

Currently supported models can be found in [configs/exp_maniskill2_act_policy/maniskill2_model](configs/exp_maniskill2_act_policy/maniskill2_model) and [configs/exp_maniskill2_diffusion_policy/maniskill2_model](configs/exp_maniskill2_diffusion_policy/maniskill2_model).


</details>

<details>
<summary><b> RLBench </b></summary>

- Train with RGB(-D) image observation:
  ```bash
  # ACT policy example:
  python src/train.py exp_rlbench_act_policy=base rlbench_task=${task} exp_rlbench_act_policy/rlbench_model@rlbench_model=${model} seed=${seed}
  # Diffusion policy example:
  python src/train.py exp_rlbench_diffusion_policy=base rlbench_task=${task} exp_rlbench_diffusion_policy/rlbench_model@rlbench_model=${model} seed=${seed}
  ```

- Train with point cloud observation:
  ```bash
  # ACT policy example:
  python src/train.py exp_rlbench_act_policy=base rlbench_task=${task} exp_rlbench_act_policy/rlbench_model@rlbench_model=${model} seed=${seed}
  # Diffusion policy example:
  python src/train.py exp_rlbench_diffusion_policy=base rlbench_task=${task} exp_rlbench_diffusion_policy/rlbench_model@rlbench_model=${model} seed=${seed}
  ```

- Evaluate a checkpoint:
  ```bash
  # ACT policy example:
  python src/test_rlbench_act.py exp_rlbench_act_policy=base rlbench_task=${task} exp_rlbench_act_policy/rlbench_model@rlbench_model=${model} seed=${seed} ckpt_path=${path/to/checkpoint}
  ```

- Zero-shot camera-view generalization evaluation:
  To evaluate camera view generalization experiments, run [scripts/run_rlbench_camera_view.sh](scripts/run_rlbench_camera_view.sh). The script evaluates the given `checkpoint` of the given `policy` and `model` on the given `task` with four different camera views, using the specified `seed`. See the script for more details. For example:
  ```bash
  # policy: either diffusion or act
  bash scripts/run_rlbench_camera_view.sh ${policy} ${path/to/checkpoint} ${task} ${model} ${seed}
  ```

Detailed configurations can be found in [configs/exp_rlbench_act_policy](configs/exp_rlbench_act_policy) and [configs/exp_rlbench_diffusion_policy](configs/exp_rlbench_diffusion_policy).

Currently supported models can be found in [configs/exp_rlbench_act_policy/rlbench_model](configs/exp_rlbench_act_policy/rlbench_model) and [configs/exp_rlbench_diffusion_policy/rlbench_model](configs/exp_rlbench_diffusion_policy/rlbench_model).

</details>

## :tada: Gotchas

<details>
<summary><b> Override any config parameter from command line </b></summary>

This codebase is based on [Hydra](https://github.com/facebookresearch/hydra), which allows for convenient configuration overriding:
```bash
python src/train.py trainer.max_epochs=20 seed=300
```
> **Note**: You can also add new parameters with `+` sign.
```bash
python src/train.py +some_new_param=some_new_value
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

```bash
# train on CPU
python src/train.py trainer=cpu

# train on 1 GPU
python src/train.py trainer=gpu

# train on TPU
python src/train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python src/train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python src/train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes
python src/train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac
python src/train.py trainer=mps
```

</details>

<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python src/train.py trainer=gpu +trainer.precision=16
```

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python src/train.py trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python src/train.py +trainer.val_check_interval=0.25

# accumulate gradients
python src/train.py trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python src/train.py +trainer.max_time="00:12:00:00"
```

> **Note**: PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Easily debug</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python src/train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python src/train.py debug=fdr

# print execution time profiling
python src/train.py debug=profiler

# try overfitting to 1 batch
python src/train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python src/train.py +trainer.detect_anomaly=true

# use only 20% of the data
python src/train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: Visit [configs/debug/](configs/debug/) for different debugging configs.

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python src/train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 9 experiments one after the other,
# each with different combination of seed and learning rate
python src/train.py -m seed=100,200,300 model.optimizer.lr=0.0001,0.00005,0.00001
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python src/train.py -m 'exp_maniskill2_act_policy/maniskill2_task@maniskill2_task=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all task experiments from [configs/exp_maniskill2_act_policy/maniskill2_task](configs/experiment/).

</details>

<details>
<summary><b>Execute run for multiple different seeds</b></summary>

```bash
python src/train.py -m seed=100,200,300 trainer.deterministic=True
```

> **Note**: `trainer.deterministic=True` makes pytorch more deterministic but impacts the performance.

</details>

For more instructions, refer to the official documentation for [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [Hydra](https://github.com/facebookresearch/hydra), and [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template).

## :bulb: Trouble Shooting

See [TroubleShooting.md](TroubleShooting.md).

## :books: License

This repository is released under the [MIT license](LICENSE).

## :sparkles: Acknowledgement

Our code is primarily built upon [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [Hydra](https://github.com/facebookresearch/hydra), [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template), [ManiSkill2](https://github.com/haosulab/ManiSkill), [RLBench](https://github.com/stepjam/RLBench), [PerAct](https://github.com/peract/peract), [ACT](https://github.com/tonyzhaozh/act), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [TIMM](https://github.com/huggingface/pytorch-image-models), [PonderV2](https://github.com/OpenGVLab/PonderV2), [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE), [Pointcept](https://github.com/Pointcept/Pointcept), [VC1](https://github.com/facebookresearch/eai-vc), [R3M](https://github.com/facebookresearch/r3m). We extend our gratitude to all these authors for their generously open-sourced code and their significant contributions to the community.

## :pencil: Citation

```bib
@article{zhu2024point,
  title={Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning},
  author={Zhu, Haoyi and Wang, Yating and Huang, Di and Ye, Weicai and Ouyang, Wanli and He, Tong},
  journal={arXiv preprint arXiv:2402.02500},
  year={2024}
}
```
