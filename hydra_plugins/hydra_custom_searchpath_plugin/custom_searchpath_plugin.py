# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class CustomSearchPathPlugin(SearchPathPlugin):
    """Plugin to add custom config search paths."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Adds the default configs directory to the search path.

        This allows the package to work with both:
        1. Default configs from PROJECT_ROOT/configs (when installed)
        2. Custom configs via --config-path (which takes priority)
        """
        # Try to find configs directory
        # First try PROJECT_ROOT env var (set by rootutils)
        project_root = os.environ.get("PROJECT_ROOT")
        configs_dir = None

        if project_root:
            configs_dir = Path(project_root) / "configs"

        # Only use fallback if PROJECT_ROOT didn't work
        if not configs_dir or not configs_dir.exists():
            # Fallback: find configs relative to this plugin file
            # hydra_plugins/hydra_custom_searchpath_plugin/... -> go up to project root
            plugin_dir = Path(__file__).parent.parent.parent
            configs_dir = plugin_dir / "configs"

        if configs_dir and configs_dir.exists():
            # Append to search path (so --config-path takes priority)
            search_path.append(provider="custom-searchpath-plugin", path=str(configs_dir))
