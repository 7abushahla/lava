"""IF Processes matching the SNN branch of the original QCFS implementation."""
import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var

V_MIN = -(1 << 23)
V_MAX = (1 << 23) - 1
I_MIN = -(1 << 15)
I_MAX = (1 << 15) - 1


class QCFSIF(AbstractProcess):
    """Float32 IF matching the original QCFS SNN branch.

    a_in supplies numeric current once per update. Optional bias is added
    once per update, corresponding to an upstream affine-layer contribution.
    State persists across RunSteps. No refractory period, saturation,
    persistent synaptic current, or lower voltage floor is introduced.
    """
    def __init__(self, *, shape, threshold=1., bias=0.):
        super().__init__(shape=shape)
        if (not isinstance(shape, tuple) or not shape or
                any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                    or n <= 0 for n in shape)):
            raise ValueError('shape must be a nonempty tuple of positive integers')
        threshold = np.broadcast_to(np.asarray(threshold, dtype=np.float32), shape).copy()
        bias = np.broadcast_to(np.asarray(bias, dtype=np.float32), shape).copy()
        if not np.all(np.isfinite(threshold)) or np.any(threshold <= 0):
            raise ValueError('threshold must be finite and positive')
        if not np.all(np.isfinite(bias)):
            raise ValueError('bias must be finite')
        initial = threshold * np.float32(.5)
        if np.any(initial == 0):
            raise ValueError('half-threshold must be representable in float32')
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=initial)
        self.threshold = Var(shape=shape, init=threshold)
        self.bias = Var(shape=shape, init=bias)

    def reset_state(self):
        """Restore half-threshold voltage while the created runtime is paused.

        Input/readout buffers and sample timing remain the caller's
        responsibility. Call this after runtime creation, between runs.
        """
        self.v.set(self.threshold.get() * np.float32(.5))



class QCFSIFFixed(AbstractProcess):
    """Integer IF candidate matching the QCFS SNN branch on a chosen grid.

    Threshold must be even, so theta/2 is exactly representable. Current and
    bias are integers. Integration clips to signed 24-bit before comparison.
    This is a declared CPU mapping candidate, not a verified Loihi 2 model.
    ``valid_start`` and ``valid_stop`` form a zero-based half-open window of
    numbered updates. Outside it, the neuron consumes input but neither
    integrates bias nor changes membrane state nor spikes. The default window
    includes every practical update. This scheduling rule is a candidate for
    retiming buffered feedforward layers, not a verified Loihi 2 mechanism.
    """
    def __init__(self, *, shape, threshold, bias=0,
                 valid_start=0, valid_stop=(1 << 31)-1):
        super().__init__(shape=shape)
        if (not isinstance(shape, tuple) or not shape or
                any(not isinstance(n, (int, np.integer)) or isinstance(n, bool)
                    or n <= 0 for n in shape)):
            raise ValueError('shape must be a nonempty tuple of positive integers')
        theta = np.broadcast_to(np.asarray(threshold), shape)
        bias_arr = np.broadcast_to(np.asarray(bias), shape)
        if (not np.issubdtype(theta.dtype, np.integer) or
                not np.issubdtype(bias_arr.dtype, np.integer)):
            raise ValueError('threshold and bias must be integers')
        if (np.any(theta <= 0) or np.any(theta > V_MAX) or
                np.any(theta % 2 != 0)):
            raise ValueError('threshold must be positive, even, and fit signed 24-bit')
        if np.any(bias_arr < I_MIN) or np.any(bias_arr > I_MAX):
            raise ValueError('bias must fit signed 16-bit')
        if (isinstance(valid_start, (bool, np.bool_)) or
                isinstance(valid_stop, (bool, np.bool_)) or
                not isinstance(valid_start, (int, np.integer)) or
                not isinstance(valid_stop, (int, np.integer)) or
                not 0 <= valid_start < valid_stop <= (1 << 31)-1):
            raise ValueError('valid update window must be zero-based [start, stop)')
        theta = theta.astype(np.int32)
        bias_arr = bias_arr.astype(np.int32)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=theta // 2)
        self.threshold = Var(shape=shape, init=theta)
        self.bias = Var(shape=shape, init=bias_arr)
        # Candidate retiming schedule. This is not a verified Loihi 2 feature.
        self.valid_start = Var(shape=(1,), init=int(valid_start))
        self.valid_stop = Var(shape=(1,), init=int(valid_stop))

    def reset_state(self):
        self.v.set(self.threshold.get() // 2)
