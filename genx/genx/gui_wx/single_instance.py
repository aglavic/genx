"""wxPython single-instance helpers."""

from __future__ import annotations

import wx

from genx.gui_generic.single_instance import SingleInstanceClient
from genx.gui_generic.single_instance import SingleInstanceServer as _SingleInstanceServer


class SingleInstanceServer(_SingleInstanceServer):
    """Dispatch received messages on wxPython's main thread."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("run_on_main_thread", wx.CallAfter)
        super().__init__(*args, **kwargs)


__all__ = ["SingleInstanceClient", "SingleInstanceServer"]
