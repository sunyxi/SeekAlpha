"""Optional Qlib research boundary.

Importing this package does not import ``qlib``. Call :func:`require_qlib`
only for operations that need the external runtime.
"""

from .dataset import to_qlib_frame
from .runtime import QlibUnavailableError, require_qlib

__all__ = ["QlibUnavailableError", "require_qlib", "to_qlib_frame"]
