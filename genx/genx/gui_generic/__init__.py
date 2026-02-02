"""
Generic package that acts as placeholder to import directly from the selected GUI toolkit.

After calling set_gui_toolkit("TOOLKIT"), you can import genx.gui_generic.xyz to load genx.gui_TOOLKIT.xyz.
"""

import importlib
import importlib.abc
import importlib.util
import sys
from typing import Optional


class _GuiGenericRedirectLoader(importlib.abc.Loader):
    def __init__(self, public_name: str, target_name: str):
        self.public_name = public_name
        self.target_name = target_name

    def create_module(self, spec):
        return None  # use default module creation semantics

    def exec_module(self, module):
        target = importlib.import_module(self.target_name)
        module.__dict__.clear()
        module.__dict__.update(target.__dict__)
        module.__dict__["__name__"] = self.public_name
        module.__dict__["__package__"] = self.public_name.rpartition(".")[0]


class _GuiGenericRedirectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, toolkit: str):
        self._toolkit = toolkit

    PREFIX = __name__ + "."

    def find_spec(self, fullname: str, path, target=None) -> Optional[importlib.machinery.ModuleSpec]:
        if not fullname.startswith(self.PREFIX):
            return None

        # Only redirect submodules (not the package itself)
        subname = fullname[len(self.PREFIX) :]
        if not subname:
            return None

        target_pkg = f"genx.gui_{self._toolkit}"
        target_name = f"{target_pkg}.{subname}"

        # Let import fail naturally if target doesn't exist; this keeps error messages meaningful.
        spec = importlib.util.find_spec(target_name)
        if spec is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _GuiGenericRedirectLoader(public_name=fullname, target_name=target_name),
            is_package=spec.submodule_search_locations is not None,
        )


def _get_existing_redirector() -> Optional[_GuiGenericRedirectFinder]:
    for finder in sys.meta_path:
        if isinstance(finder, _GuiGenericRedirectFinder):
            return finder
    return None


def set_gui_toolkit(toolkit: str) -> None:
    """
    Select GUI toolkit implementation for imports under `genx.gui_generic.*`.

    Must be called before importing any `genx.gui_generic.<submodule>`.
    """
    toolkit = str(toolkit).strip().lower()
    if toolkit not in {"wx", "qt"}:
        raise ValueError(f"Unknown GUI toolkit: {toolkit!r}. Expected 'wx' or 'qt'.")

    # If any gui_generic submodules are already imported, switching toolkits would
    # produce a mixed sys.modules cache (part wx, part qt). Fail fast.
    prefix = __name__ + "."
    already_loaded = [name for name in sys.modules.keys() if name.startswith(prefix)]
    if already_loaded:
        raise RuntimeError(
            "set_gui_toolkit() must be called before importing any genx.gui_generic submodules. "
            f"Already imported: {already_loaded[:5]}{'...' if len(already_loaded) > 5 else ''}"
        )

    existing = _get_existing_redirector()
    if existing is not None:
        # Replace existing redirector (if any) to ensure the toolkit is updated deterministically.
        sys.meta_path = [f for f in sys.meta_path if not isinstance(f, _GuiGenericRedirectFinder)]

    sys.meta_path.insert(0, _GuiGenericRedirectFinder(toolkit=toolkit))