"""Higher-order connectome control analysis package."""

from .threads import pin_threads

pin_threads()

from .config import load_config  # noqa: E402

__all__ = ["load_config"]
