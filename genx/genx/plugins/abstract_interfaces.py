"""
Collection of abstract interfaces used for GUI calls from plugins.
"""

from typing import Protocol, Tuple

from genx.plugins.data_loader_framework import Template

class AbstractSettingsDialog(Protocol):

    def __init__(self, plugin: Template, col_values: dict, misc_values: dict):
        ...

    def get_results(self) -> Tuple[bool, dict, dict]:
        """
        Shows the dialog and returns success and list of results.
        """
        ...