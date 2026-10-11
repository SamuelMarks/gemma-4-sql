"""Provide module docstring."""

from __future__ import annotations

import sys
from importlib import import_module
from unittest import mock


def exec_import(mod_name: str, mock_dict: dict[str, object], func_name: str | None = None, *args: object, **kwargs: object) -> None:
    """Initialize function exec_import.

    Args:
    ----
    mod_name: Description of mod_name.
    mock_dict: Description of mock_dict.
    func_name: Description of func_name.
    args: Description of args.
    kwargs: Description of kwargs.

    """
    orig_mod = sys.modules.get(mod_name)
    parent_name, _, child_name = mod_name.rpartition(".")
    parent_mod = sys.modules.get(parent_name) if parent_name else None
    orig_child = getattr(parent_mod, child_name, None) if parent_mod is not None else None

    with mock.patch.dict(sys.modules, mock_dict):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        try:
            mod = import_module(mod_name)
            if func_name:
                getattr(mod, func_name)(*args, **kwargs)
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass
        finally:
            if orig_mod is not None:
                sys.modules[mod_name] = orig_mod
            elif mod_name in sys.modules:
                del sys.modules[mod_name]
            if parent_mod is not None:
                if orig_child is not None:
                    setattr(parent_mod, child_name, orig_child)
                elif hasattr(parent_mod, child_name):
                    delattr(parent_mod, child_name)


def test_exec_import():
    """Test exec_import edge cases."""
    import sys
    from unittest import mock

    # False branch for `if mod_name in sys.modules:`
    exec_import("does_not_exist_xyz", {})

    # func call exception (ValueError from math.sqrt(-1))
    exec_import("math", {}, "sqrt", -1)

    # orig_mod is None, mod_name in sys.modules
    # Let's clear something obscure like 'csv'
    sys.modules.pop("csv", None)
    exec_import("csv", {})

    # parent_mod exists, orig_child is None, but import adds it
    sys.modules["my_parent"] = type("MyParent", (), {})()

    def fake_import(name):
        sys.modules["my_parent"].child = "mocked_child"
        sys.modules["my_parent.child"] = "mocked_child"
        return "mocked_child"

    with mock.patch("tests.backends.test_missing_backends_edge_cases.import_module", fake_import):
        exec_import("my_parent.child", {})

    # parent_mod exists, orig_child exists
    sys.modules["my_parent2"] = type("MyParent", (), {})()
    sys.modules["my_parent2.child2"] = type("MyChild", (), {})()
    sys.modules["my_parent2"].child2 = "orig_child"
    exec_import("my_parent2.child2", {})

    # parent_mod exists, orig_child is None, import does NOT add it
    sys.modules["my_parent3"] = type("MyParent", (), {})()

    def fake_import_3(name):
        return "mocked_child3"

    with mock.patch("tests.backends.test_missing_backends_edge_cases.import_module", fake_import_3):
        exec_import("my_parent3.child3", {})
