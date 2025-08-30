from __future__ import annotations

import pytest


def build_clifford_meas():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def test_stim_vs_aer_counts_close() -> None:
    pytest.importorskip("qiskit")
    try:
        import stim  # noqa: F401
    except Exception:
        pytest.skip("Stim not installed")

    from ariadne_mac.route.execute import _to_stim
    from qiskit_aer import AerSimulator

    circ = build_clifford_meas()

    # Stim sampling
    stim_circ = _to_stim(circ)
    sampler = stim_circ.compile_sampler()
    s_arr = sampler.sample(2000, rand_seed=1234)
    st_counts = {}
    for row in s_arr:
        key = f"{int(row[1])}{int(row[0])}"  # reverse order to match qiskit bitstring
        st_counts[key] = st_counts.get(key, 0) + 1

    # Aer sampling
    sim = AerSimulator(method="qasm")
    job = sim.run(circ, shots=2000, seed_simulator=1234)
    ae_counts = job.result().get_counts()

    # Distributions should be close for a deterministic Clifford circuit
    # Expect roughly half 00 and half 11
    p00_st = st_counts.get("00", 0) / 2000
    p11_st = st_counts.get("11", 0) / 2000
    p00_ae = ae_counts.get("00", 0) / 2000
    p11_ae = ae_counts.get("11", 0) / 2000

    assert abs((p00_st + p11_st) - 1.0) < 0.05
    assert abs(p00_st - p00_ae) < 0.1
    assert abs(p11_st - p11_ae) < 0.1

