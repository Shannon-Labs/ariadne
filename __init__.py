"""
Ariadne: Quantum Security for the Classical World
=================================================

Your Classical Computer Can Detect Quantum Attacks.

This package provides real-time quantum threat detection using
compression-based anomaly detection (CbAD) running on your laptop.
"""

from .quantum_detector import QuantumThreatDetector
from .cbad_integration import CompressionAnomalyDetector
from .driftlock import DriftlockSynchronizer
from .api import QuantumDetectionAPI

__version__ = "1.0.0"
__author__ = "Shannon Labs"
__description__ = "Your Classical Computer Can Detect Quantum Attacks"

__all__ = [
    "QuantumThreatDetector",
    "CompressionAnomalyDetector",
    "DriftlockSynchronizer",
    "QuantumDetectionAPI",
]
