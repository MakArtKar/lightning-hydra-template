#!/usr/bin/env python

from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup


def read_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    with requirements_file.open() as f:
        requirements = []
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            requirement = line.split("#", 1)[0].strip()
            if requirement:
                requirements.append(requirement)
        return requirements


setup(
    name="ml_core",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=read_requirements(),
    packages=find_packages() + find_namespace_packages(include=["hydra_plugins*"]),
    include_package_data=True,
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = ml_core.train:main",
            "eval_command = ml_core.eval:main",
        ],
        "hydra_plugins": [
            "custom_searchpath_plugin = hydra_plugins.hydra_custom_searchpath_plugin.custom_searchpath_plugin:CustomSearchPathPlugin",
            "hydra_colorlog_plugin = hydra_plugins.hydra_colorlog.colorlog:HydraColorlogSearchPathPlugin",
        ],
    },
)
