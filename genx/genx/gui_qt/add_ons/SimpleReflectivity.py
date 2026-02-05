""" <h1>Simple Reflectivity plugin </h1>
Qt port uses the standard Reflectivity UI as a fallback.
"""

from .Reflectivity import Plugin as ReflectivityPlugin


class Plugin(ReflectivityPlugin):
    def __init__(self, parent):
        super().__init__(parent)
        self.StatusMessage("Simple Reflectivity plugin loaded (Reflectivity UI in Qt).")
