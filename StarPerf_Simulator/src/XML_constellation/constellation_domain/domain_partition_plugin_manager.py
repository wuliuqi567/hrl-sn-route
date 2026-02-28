"""
Domain partition plugin manager.

Follows the same plugin architecture as StarPerf's connectivity_mode_plugin_manager:
  - Scans domain_partition_plugin/ folder for .py files
  - Each .py file exports a function with the same name as the file
  - The function signature: f(shell, n_domains) -> DomainInfo
"""

import importlib
import os


class domain_partition_plugin_manager:
    def __init__(self):
        self.plugins = {}
        package_name = "src.XML_constellation.constellation_domain.domain_partition_plugin"
        # Use __file__-relative path so it works regardless of cwd
        plugins_path = os.path.join(os.path.dirname(__file__), "domain_partition_plugin")
        for plugin_name in os.listdir(plugins_path):
            if plugin_name.endswith(".py") and not plugin_name.startswith("_"):
                plugin_name = plugin_name[:-3]
                plugin = importlib.import_module(package_name + "." + plugin_name)
                if hasattr(plugin, plugin_name) and callable(getattr(plugin, plugin_name)):
                    function = getattr(plugin, plugin_name)
                    self.plugins[plugin_name] = function
        self.current_partition_mode = "by_orbit_group"

    def set_partition_mode(self, plugin_name):
        """Switch the domain partition mode."""
        if plugin_name not in self.plugins:
            raise ValueError(
                f"Unknown partition mode '{plugin_name}'. "
                f"Available: {list(self.plugins.keys())}"
            )
        self.current_partition_mode = plugin_name

    def execute_partition(self, shell, n_domains):
        """Execute the current partition mode.

        Parameters
        ----------
        shell : shell object
            A StarPerf shell with orbits and satellites.
        n_domains : int
            Number of domains to partition into.

        Returns
        -------
        DomainInfo
            Complete domain partition result.
        """
        function = self.plugins[self.current_partition_mode]
        return function(shell, n_domains)
