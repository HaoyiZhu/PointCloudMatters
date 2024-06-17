#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning",
    author="Haoyi Zhu",
    author_email="hyizhu1108@gmail.com",
    url="https://github.com/HaoyiZhu/PointCloudMatters",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
