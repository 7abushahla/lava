"""CPU ProcessModels for IF dynamics used in QCFS conversion."""
import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.qcfs.process import (QCFSIF, QCFSIFFixed, QCFSSpikeDelay,
                                    I_MIN, I_MAX, V_MIN, V_MAX)


@implements(proc=QCFSIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyQCFSIFModelFloat(PyLoihiProcessModel):
    """CPU float32 specification, not a hardware-bit-accurate model."""
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, np.float32)
    threshold: np.ndarray = LavaPyType(np.ndarray, np.float32)
    bias: np.ndarray = LavaPyType(np.ndarray, np.float32)

    def run_spk(self):
        current = np.asarray(self.a_in.recv(), dtype=np.float32)
        self.v[:] = self.v + (current + self.bias)
        spike = self.v >= self.threshold
        self.v[:] = self.v - self.threshold * spike
        self.s_out.send(spike)


@implements(proc=QCFSIFFixed, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyQCFSIFFixed(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=16)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    threshold: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    valid_start: np.ndarray = LavaPyType(np.ndarray, np.int32)
    valid_stop: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._trace_prefix = proc_params.get('trace_prefix')
        self._trace_spikes = None
        self._trace_voltage = None

    def run_spk(self):
        current = np.asarray(self.a_in.recv(), dtype=np.int64)
        if np.any(current < I_MIN) or np.any(current > I_MAX):
            raise ValueError('input current exceeds declared signed 16-bit range')
        step = self.time_step - 1
        if not self.valid_start.item() <= step < self.valid_stop.item():
            self.s_out.send(np.zeros(self.v.shape, dtype=bool))
            return
        integrated = self.v.astype(np.int64) + current + self.bias
        integrated = np.clip(integrated, V_MIN, V_MAX).astype(np.int32)
        spike = integrated >= self.threshold
        self.v[:] = integrated - self.threshold * spike
        self.s_out.send(spike)
        if self._trace_prefix is not None:
            if self._trace_spikes is None:
                shape = (int(self.valid_stop.item() - self.valid_start.item()),
                         *self.v.shape)
                self._trace_spikes = np.memmap(self._trace_prefix + '.spikes',
                                               mode='w+', dtype=np.int8,
                                               shape=shape)
                self._trace_voltage = np.memmap(self._trace_prefix + '.voltage',
                                                mode='w+', dtype=np.int32,
                                                shape=shape)
            local = step - int(self.valid_start.item())
            self._trace_spikes[local] = spike
            self._trace_voltage[local] = self.v
            self._trace_spikes.flush()
            self._trace_voltage.flush()


@implements(proc=QCFSSpikeDelay, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyQCFSSpikeDelay(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    buffer: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=1)

    def run_spk(self):
        self.s_out.send(self.buffer.astype(bool))
        self.buffer[:] = self.s_in.recv().astype(np.int32)
