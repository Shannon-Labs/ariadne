"""
Quantum Network Coordination Platform - Shannon Labs Quantum Bell Labs Initiative

This module provides the foundational infrastructure layer for quantum networks,
integrating Driftlock's 22ps synchronization, Entruptor's anomaly detection,
and Ariadne's quantum program analysis to create the quantum Bell Labs.

Core capabilities:
- 22ps precision quantum timing coordination
- Quantum entanglement distribution with integrity monitoring
- Real-time quantum state anomaly detection
- Distributed quantum computing coordination
- Quantum key distribution network synchronization
- Quantum sensor network coordination for GW detection

This positions Shannon Labs as the Bell Labs of the Quantum Era by solving
the fundamental quantum coordination problems that enable quantum internet
and distributed quantum computing.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Shannon Labs Quantum Team"

from .coordinator import QuantumNetworkCoordinator
from .timing import QuantumTimingCoordinator
from .entanglement import QuantumEntanglementCoordinator
from .anomaly import QuantumStateAnomalyDetector
from .distributed import DistributedQuantumCoordinator
from .qkd import QKDSynchronizationCoordinator
from .sensors import QuantumSensorNetworkCoordinator

__all__ = [
    "QuantumNetworkCoordinator",
    "QuantumTimingCoordinator",
    "QuantumEntanglementCoordinator",
    "QuantumStateAnomalyDetector",
    "DistributedQuantumCoordinator",
    "QKDSynchronizationCoordinator",
    "QuantumSensorNetworkCoordinator",
]