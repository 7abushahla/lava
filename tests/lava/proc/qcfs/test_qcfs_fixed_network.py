"""Integer QCFS graph, weight scaling, and step-delay checks."""
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.dense.process import Dense
from lava.proc.io import source, sink
from lava.proc.qcfs import QCFSIFFixed
from lava.proc.qcfs.process import V_MIN, V_MAX


def scalar_trace(current, theta):
    v = int(theta) // 2
    spikes = []
    states = []
    for x in current:
        v = max(V_MIN, min(V_MAX, v + int(x)))
        spike = int(v >= theta)
        v -= theta * spike
        spikes.append(spike)
        states.append(v)
    return np.array(spikes), np.array(states)


class TestFixedNetwork(unittest.TestCase):
    def test_connected_integer_graph(self):
        currents = np.array([[15, 15, 15, 15, 0], [-8, -8, -8, -8, 0]], dtype=np.int32)
        theta1 = np.array([8, 16], dtype=np.int32)
        weights = np.array([[8, -16]], dtype=np.int32)
        src = source.RingBuffer(data=currents)
        first = QCFSIFFixed(shape=(2,), threshold=theta1)
        dense = Dense(weights=weights)
        second = QCFSIFFixed(shape=(1,), threshold=np.array([16], dtype=np.int32))
        out1 = sink.RingBuffer(shape=(2,), buffer=5)
        out2 = sink.RingBuffer(shape=(1,), buffer=5)
        src.s_out.connect(first.a_in)
        first.s_out.connect(dense.s_in)
        first.s_out.connect(out1.a_in)
        dense.a_out.connect(second.a_in)
        second.s_out.connect(out2.a_in)
        try:
            first.run(RunSteps(num_steps=5), Loihi2SimCfg(select_tag='fixed_pt'))
            actual1 = out1.data.get().copy()
            actual2 = out2.data.get().copy()
            effective = dense.weights.get().copy()
            expected1 = np.stack([scalar_trace(currents[i], theta1[i])[0] for i in range(2)])
            expected_current2 = np.zeros(5, dtype=np.int32)
            expected_current2[1:] = effective @ expected1[:, :-1]
            expected2, expected_v2 = scalar_trace(expected_current2, 16)
            assert_array_equal(actual1, expected1)
            assert_array_equal(actual2[0], expected2)
            assert_array_equal(second.v.get(), [expected_v2[-1]])
            assert_array_equal(effective, weights)
        finally:
            first.stop()

    def test_limits_and_half_threshold(self):
        neuron = QCFSIFFixed(shape=(2,), threshold=np.array([8, 16]))
        assert_array_equal(neuron.v.init, [4, 8])
        for bad in (0, 1, -2, V_MAX + 1):
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                QCFSIFFixed(shape=(1,), threshold=bad)
        with self.assertRaises(ValueError):
            QCFSIFFixed(shape=(1,), threshold=8., bias=0)


if __name__ == '__main__':
    unittest.main()
