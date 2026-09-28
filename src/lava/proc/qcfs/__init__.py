"""IF Processes aligned with the original QCFS conversion implementation."""
from .process import QCFSIF, QCFSIFFixed, QCFSSpikeDelay
from . import models  # Register implementations when importing this package.

__all__ = ["QCFSIF", "QCFSIFFixed", "QCFSSpikeDelay"]
