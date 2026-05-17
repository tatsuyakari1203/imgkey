"""Optional AI backend adapters for ImgKey.

Importing this package must stay lightweight: no torch, transformers, timm,
or model-runtime imports are allowed at package import time.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
