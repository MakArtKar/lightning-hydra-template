"""Core ML package providing data modules, models, training and utilities."""

from omegaconf import OmegaConf

# Register eval resolver for conditional expressions in configs
OmegaConf.register_new_resolver("eval", eval, replace=True)
