"""HBTXR deployment package — the host side of SPEC §8.

    payload   the M-AXI weight blob and the AXIS streams. Stdlib only, no board needed
    overlay   the PYNQ runtime. Needs pynq and a bitstream

`payload` imports on any Python; `overlay` guards its `pynq` and `numpy` imports so this
package can be inspected and its blob built on a workstation.
"""
from . import payload  # noqa: F401

__all__ = ["payload", "overlay"]
