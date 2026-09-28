"""IF Processes aligned with the original QCFS conversion implementation."""
from .process import QCFSIF, QCFSIFFixed
from . import models  # Register implementations when importing this package.

__all__ = ["QCFSIF", "QCFSIFFixed"]
