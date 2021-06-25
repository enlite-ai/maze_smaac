"""Hydra plugin to register additional config packages in the search path."""
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class SMAACL2RPNSearchPathPlugin(SearchPathPlugin):
    """Hydra plugin to register additional config packages in the search path.
    Be aware that hydra uses an unconventional way to import this object: ``imp.load_module``, which forces a reload
    of this Python module. This breaks the singleton semantics of Python modules and makes it impossible to mock
    this class during testing. Therefore the actual paths are provided by ``loop_envs.__init__``.
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """implement the SearchPathPlugin interface"""

        search_path.append("project", "pkg://maze.conf")
        search_path.append("project", "pkg://maze_smaac.conf")
