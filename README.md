<div align="center">

# Point Cloud matters: Rethinking the Impact of Different Observation Spaces on Robot Learning

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[**Project Page**](https://haoyizhu.github.io/PointCloudMatters.github.io/) | [**Arxiv**](https://arxiv.org/abs/2402.02500)

[Haoyi Zhu](https://www.haoyizhu.site/), [Yating Wang](https://scholar.google.com/citations?hl=zh-CN&user=5SuBWh0AAAAJ), [Di Huang](https://dihuang.me/), [Weicai Ye](https://ywcmaike.github.io/), [Wanli Ouyang](https://wlouyang.github.io/), [Tong He](http://tonghe90.github.io/)
</div>

<p align="center">
    <img src="assets/overview.png" alt="overview" width="90%" />
</p>

This is the official implementation of paper "Point Cloud matters: Rethinking the Impact of Different Observation Spaces on Robot Learning".

In robot learning, the observation space is crucial due to the distinct characteristics of different modalities, which can potentially become a bottleneck alongside policy design. In this study, we explore the influence of various observation spaces on robot learning, focusing on three predominant modalities: RGB, RGB-D, and point cloud. We introduce OBSBench, a benchmark comprising two simulators and 125 tasks, along with standardized pipelines for various encoders and policy baselines. Extensive experiments on diverse contact-rich manipulation tasks reveal a notable trend: point cloud-based methods, even those with the simplest designs, frequently outperform their RGB and RGB-D counterparts. This trend persists in both scenarios: training from scratch and utilizing pre-training. Furthermore, our findings demonstrate that point cloud observations often yield better policy performance and significantly stronger generalization capabilities across various geometric and visual conditions. These outcomes suggest that the 3D point cloud is a valuable observation modality for intricate robotic tasks. We also suggest that incorporating both appearance and coordinate information can enhance the performance of point cloud methods. We hope our work provides valuable insights and guidance for designing more generalizable and robust robotic models.


## :clipboard: Contents

- [Installation](#installation)
- [Data Preparation](#mag-data-preparation)
- [Training and Evaluation](#rocket-training-and-evaluation)
- [License](#books-license)
- [Acknowledgement](#sparkles-acknowledgement)
- [Citation](#pencil-citation)

## :hammer: Installation

Coming soon.

## :mag: Data Preparation

Coming soon.

## :rocket: Training and Evaluation

Coming soon.

## :books: License

This repository is released under the [MIT license](LICENSE).

## :sparkles: Acknowledgement

Our code is primarily built upon [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [Hydra](https://github.com/facebookresearch/hydra), [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template), [ManiSkill2](https://github.com/haosulab/ManiSkill), [RLBench](https://github.com/stepjam/RLBench), [PerAct](https://github.com/peract/peract), [ACT](https://github.com/tonyzhaozh/act), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [TIMM](https://github.com/huggingface/pytorch-image-models), [PonderV2](https://github.com/OpenGVLab/PonderV2), [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE), [VC1](https://github.com/facebookresearch/eai-vc), [R3M](https://github.com/facebookresearch/r3m). We extend our gratitude to all these authors for their generously open-sourced code and their significant contributions to the community.

## :pencil: Citation

```bib
@article{zhu2024point,
  title={Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning},
  author={Zhu, Haoyi and Wang, Yating and Huang, Di and Ye, Weicai and Ouyang, Wanli and He, Tong},
  journal={arXiv preprint arXiv:2402.02500},
  year={2024}
}
```