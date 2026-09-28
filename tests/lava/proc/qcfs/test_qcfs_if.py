"""Run with python -m unittest -v tests.lava.proc.qcfs.test_qcfs_if from the Lava root."""
from fractions import Fraction
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io import source, sink
from lava.proc.qcfs import QCFSIF


def exact_trace(currents, threshold=1., bias=0.):
    """Independent scalar, rational oracle; binary64 inputs are exact fractions."""
    currents = np.asarray(currents, dtype=float)
    neurons, steps = currents.shape
    thresholds = np.broadcast_to(threshold, (neurons,))
    biases = np.broadcast_to(bias, (neurons,))
    spikes = np.zeros_like(currents)
    voltage = np.zeros_like(currents)
    for n in range(neurons):
        theta = Fraction(float(thresholds[n]))
        v = theta / 2
        for t in range(steps):
            v += Fraction(float(currents[n, t])) + Fraction(float(biases[n]))
            fired = v >= theta
            if fired:
                v -= theta
            spikes[n, t] = int(fired)
            voltage[n, t] = float(v)
    return spikes, voltage


def execute(currents, threshold=1., bias=0., chunks=None, reset_between=False):
    currents = np.asarray(currents, dtype=float)
    shape = currents.shape[:-1]
    steps = currents.shape[-1]
    src = source.RingBuffer(data=currents)
    neuron = QCFSIF(shape=shape, threshold=threshold, bias=bias)
    dst = sink.RingBuffer(shape=shape, buffer=steps)
    src.s_out.connect(neuron.a_in)
    neuron.s_out.connect(dst.a_in)
    cfg = Loihi2SimCfg(select_tag='floating_pt')
    snapshots = []
    chunks = chunks or [1] * steps
    try:
        for length in chunks:
            neuron.run(condition=RunSteps(num_steps=length), run_cfg=cfg)
            snapshots.append(neuron.v.get().copy())
        first = dst.data.get().copy()
        if reset_between:
            neuron.reset_state()
            assert_array_equal(neuron.v.get(), np.broadcast_to(threshold, shape) / 2)
            neuron.run(condition=RunSteps(num_steps=steps), run_cfg=cfg)
            return first, dst.data.get().copy()
        return first, np.stack(snapshots, axis=-1)
    finally:
        neuron.stop()


class TestQCFSIF(unittest.TestCase):
    def test_half_threshold_and_equality(self):
        spikes, v = execute([[.5, 0., .5, .5]])
        assert_array_equal(spikes, [[1, 0, 0, 1]])
        assert_array_equal(v, [[0., 0., .5, 0.]])

    def test_soft_reset_keeps_residual(self):
        spikes, v = execute([[.75, .75, .75, .75]])
        assert_array_equal(spikes, [[1, 1, 0, 1]])
        assert_array_equal(v, [[.25, 0., .75, .5]])

    def test_one_spike_per_step_and_backlog(self):
        spikes, v = execute([[2.5, 0., 0., 0.]])
        assert_array_equal(spikes, [[1, 1, 1, 0]])
        assert_array_equal(v, [[2., 1., 0., 0.]])

    def test_negative_state_is_not_clipped(self):
        spikes, v = execute([[-1., .5, .5, .5]])
        assert_array_equal(spikes, [[0, 0, 0, 1]])
        assert_array_equal(v, [[-.5, 0., .5, 0.]])

    def test_no_leak_or_persistent_synaptic_current(self):
        spikes, v = execute([[.25, 0., 0., 0.]])
        assert_array_equal(spikes, [[0, 0, 0, 0]])
        assert_array_equal(v, [[.75, .75, .75, .75]])

    def test_constant_drive_rate_and_qcfs_rounding(self):
        levels = np.arange(-2, 19) / 16
        for steps in (1, 2, 4, 8, 16):
            with self.subTest(T=steps):
                current = np.repeat(levels[:, None], steps, axis=1)
                spikes, voltage = execute(current)
                expected_s, expected_v = exact_trace(current)
                assert_array_equal(spikes, expected_s)
                assert_array_equal(voltage, expected_v)
                expected_counts = np.clip(np.floor(steps * levels + .5), 0, steps)
                assert_array_equal(spikes.sum(axis=-1), expected_counts)
                # For held drive and T=L this is the scalar QCFS ANN quantizer.
                quantized = np.floor(np.clip(levels, 0, 1) * steps + .5) / steps
                assert_array_equal(spikes.mean(axis=-1), quantized)

    def test_signed_inputs_scaled_thresholds_and_bias(self):
        rng = np.random.default_rng(42)
        currents = rng.integers(-8, 25, size=(6, 16)) / 16
        thresholds = np.array([.5, 1., 2., .5, 1., 2.])
        bias = np.array([0., .125, -.125, .25, -.25, 0.])
        spikes, v = execute(currents, thresholds, bias)
        expected_s, expected_v = exact_trace(currents, thresholds, bias)
        assert_array_equal(spikes, expected_s)
        assert_array_equal(v, expected_v)
        assert_array_equal(thresholds[:, None] * spikes,
                           thresholds[:, None] * expected_s)

    def test_runsteps_partition_preserves_state(self):
        currents = [[.75] * 8]
        single, states = execute(currents)
        chunked, last = execute(currents, chunks=[3, 1, 4])
        assert_array_equal(single, chunked)
        assert_array_equal(last[:, -1], states[:, -1])

    def test_reset_and_repeat_sample(self):
        first, second = execute([[.125, .5, 0., .125]], reset_between=True)
        assert_array_equal(first, second)
        assert_array_equal(first, [[0, 1, 0, 0]])

    def test_reject_invalid_parameters(self):
        for theta in (0., -1., np.nan, np.inf):
            with self.subTest(threshold=theta), self.assertRaises(ValueError):
                QCFSIF(shape=(1,), threshold=theta)
        for shape in ((), (0,), (-1,), (1.5,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                QCFSIF(shape=shape)
        with self.assertRaises(ValueError):
            QCFSIF(shape=(2,), threshold=np.ones(3))
        with self.assertRaises(ValueError):
            QCFSIF(shape=(1,), bias=np.nan)


if __name__ == '__main__':
    unittest.main()
