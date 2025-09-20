"""
Quantum Network Coordinator - Main coordination engine for quantum networks

This module provides the central coordination engine that integrates all quantum
network components including timing, entanglement, anomaly detection, and
distributed computing coordination.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table

from .timing import QuantumTimingCoordinator
from .entanglement import QuantumEntanglementCoordinator
from .anomaly import QuantumStateAnomalyDetector
from .distributed import DistributedQuantumCoordinator
from .qkd import QKDSynchronizationCoordinator
from .sensors import QuantumSensorNetworkCoordinator


class QuantumNetworkStatus(Enum):
    """Status of the quantum network coordination system."""
    INITIALIZING = "initializing"
    SYNCHRONIZING = "synchronizing"
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class QuantumNetworkConfig:
    """Configuration for quantum network coordination."""
    timing_precision_ps: int = 22  # 22ps precision from Driftlock
    max_nodes: int = 64
    entanglement_fidelity_threshold: float = 0.95
    anomaly_detection_sensitivity: float = 0.85
    qkd_key_rate_min: float = 1e3  # bits per second
    sensor_sync_tolerance_ps: int = 50
    enable_adaptive_thresholds: bool = True
    enable_fusion_analysis: bool = True
    log_level: str = "INFO"


@dataclass
class NetworkNode:
    """Represents a node in the quantum network."""
    node_id: str
    node_type: str  # "quantum_processor", "sensor", "repeater", "qkd_node"
    location: tuple[float, float, float]  # lat, lon, elevation
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    last_sync_time: float = 0.0
    timing_offset_ps: int = 0


@dataclass
class CoordinationMetrics:
    """Metrics for quantum network coordination performance."""
    timing_accuracy_ps: float = 0.0
    entanglement_fidelity: float = 0.0
    anomaly_detection_rate: float = 0.0
    network_efficiency: float = 0.0
    qkd_key_rate: float = 0.0
    sensor_coherence: float = 0.0
    total_nodes: int = 0
    active_connections: int = 0


class QuantumNetworkCoordinator:
    """
    Main coordinator for quantum network operations.

    Integrates all quantum network components to provide unified coordination
    across timing, entanglement, anomaly detection, and distributed computing.
    """

    def __init__(self, config: Optional[QuantumNetworkConfig] = None):
        """Initialize the quantum network coordinator."""
        self.config = config or QuantumNetworkConfig()
        self.console = Console()

        # Initialize component coordinators
        self.timing_coordinator = QuantumTimingCoordinator(self.config.timing_precision_ps)
        self.entanglement_coordinator = QuantumEntanglementCoordinator(
            self.config.entanglement_fidelity_threshold
        )
        self.anomaly_detector = QuantumStateAnomalyDetector(
            self.config.anomaly_detection_sensitivity
        )
        self.distributed_coordinator = DistributedQuantumCoordinator(self.config.max_nodes)
        self.qkd_coordinator = QKDSynchronizationCoordinator(self.config.qkd_key_rate_min)
        self.sensor_coordinator = QuantumSensorNetworkCoordinator(
            self.config.sensor_sync_tolerance_ps
        )

        # Network state
        self.status = QuantumNetworkStatus.INITIALIZING
        self.nodes: Dict[str, NetworkNode] = {}
        self.metrics = CoordinationMetrics()
        self.start_time = time.time()

        # Performance tracking
        self.performance_log: List[Dict[str, Any]] = []

    async def initialize_network(self) -> bool:
        """Initialize the quantum network coordination system."""
        try:
            self.console.print("[bold blue]Initializing Quantum Network Coordination Platform...[/bold blue]")

            # Initialize all components
            await self.timing_coordinator.initialize()
            await self.entanglement_coordinator.initialize()
            await self.anomaly_detector.initialize()
            await self.distributed_coordinator.initialize()
            await self.qkd_coordinator.initialize()
            await self.sensor_coordinator.initialize()

            self.status = QuantumNetworkStatus.SYNCHRONIZING
            self.console.print("[green]✓ Quantum network initialization complete[/green]")

            return True

        except Exception as e:
            self.console.print(f"[red]✗ Quantum network initialization failed: {e}[/red]")
            self.status = QuantumNetworkStatus.OFFLINE
            return False

    async def synchronize_network(self) -> bool:
        """Synchronize all nodes in the quantum network."""
        try:
            self.console.print("[bold blue]Synchronizing quantum network nodes...[/bold blue]")

            # Perform timing synchronization
            timing_results = await self.timing_coordinator.synchronize_nodes(list(self.nodes.keys()))

            # Update node timing offsets
            for node_id, offset in timing_results.items():
                if node_id in self.nodes:
                    self.nodes[node_id].timing_offset_ps = offset
                    self.nodes[node_id].last_sync_time = time.time()

            # Coordinate entanglement distribution
            await self.entanglement_coordinator.coordinate_entanglement(self.nodes)

            # Initialize anomaly detection
            await self.anomaly_detector.start_monitoring(self.nodes)

            # Start distributed coordination
            await self.distributed_coordinator.coordinate_computation(self.nodes)

            # Initialize QKD synchronization
            await self.qkd_coordinator.synchronize_qkd_network(self.nodes)

            # Coordinate sensor networks
            await self.sensor_coordinator.coordinate_sensors(self.nodes)

            self.status = QuantumNetworkStatus.OPERATIONAL
            self.console.print("[green]✓ Quantum network synchronization complete[/green]")

            return True

        except Exception as e:
            self.console.print(f"[red]✗ Quantum network synchronization failed: {e}[/red]")
            self.status = QuantumNetworkStatus.DEGRADED
            return False

    def add_node(self, node: NetworkNode) -> bool:
        """Add a node to the quantum network."""
        try:
            self.nodes[node.node_id] = node
            self.console.print(f"[green]✓ Added node {node.node_id} ({node.node_type})[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]✗ Failed to add node: {e}[/red]")
            return False

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the quantum network."""
        try:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.console.print(f"[yellow]✓ Removed node {node_id}[/yellow]")
                return True
            return False
        except Exception as e:
            self.console.print(f"[red]✗ Failed to remove node: {e}[/red]")
            return False

    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        return {
            "status": self.status.value,
            "uptime_seconds": time.time() - self.start_time,
            "total_nodes": len(self.nodes),
            "active_nodes": len([n for n in self.nodes.values() if n.status == "active"]),
            "node_types": self._count_node_types(),
            "coordination_metrics": self._get_metrics_dict(),
            "component_status": {
                "timing": self.timing_coordinator.status,
                "entanglement": self.entanglement_coordinator.status,
                "anomaly_detection": self.anomaly_detector.status,
                "distributed_computing": self.distributed_coordinator.status,
                "qkd": self.qkd_coordinator.status,
                "sensors": self.sensor_coordinator.status,
            }
        }

    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node in self.nodes.values():
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts

    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        return {
            "timing_accuracy_ps": self.metrics.timing_accuracy_ps,
            "entanglement_fidelity": self.metrics.entanglement_fidelity,
            "anomaly_detection_rate": self.metrics.anomaly_detection_rate,
            "network_efficiency": self.metrics.network_efficiency,
            "qkd_key_rate": self.metrics.qkd_key_rate,
            "sensor_coherence": self.metrics.sensor_coherence,
            "total_nodes": self.metrics.total_nodes,
            "active_connections": self.metrics.active_connections,
        }

    def display_status_table(self) -> None:
        """Display network status in a formatted table."""
        table = Table(title="Quantum Network Coordination Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        status = self.get_network_status()

        table.add_row("Network", status["status"], f"{status['total_nodes']} nodes")
        table.add_row("Timing", self.timing_coordinator.status, f"{self.metrics.timing_accuracy_ps".1f"}ps accuracy")
        table.add_row("Entanglement", self.entanglement_coordinator.status, f"{self.metrics.entanglement_fidelity".3f"} fidelity")
        table.add_row("Anomaly Detection", self.anomaly_detector.status, f"{self.metrics.anomaly_detection_rate".2f"}% rate")
        table.add_row("Distributed Computing", self.distributed_coordinator.status, f"{status['total_nodes']} nodes")
        table.add_row("QKD", self.qkd_coordinator.status, f"{self.metrics.qkd_key_rate".0f"} bps")
        table.add_row("Sensors", self.sensor_coordinator.status, f"{self.metrics.sensor_coherence".3f"} coherence")

        self.console.print(table)

    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive network diagnostics."""
        self.console.print("[bold blue]Running quantum network diagnostics...[/bold blue]")

        diagnostics = {
            "timestamp": time.time(),
            "network_status": self.get_network_status(),
            "timing_diagnostics": await self.timing_coordinator.run_diagnostics(),
            "entanglement_diagnostics": await self.entanglement_coordinator.run_diagnostics(),
            "anomaly_diagnostics": await self.anomaly_detector.run_diagnostics(),
            "distributed_diagnostics": await self.distributed_coordinator.run_diagnostics(),
            "qkd_diagnostics": await self.qkd_coordinator.run_diagnostics(),
            "sensor_diagnostics": await self.sensor_coordinator.run_diagnostics(),
        }

        self.console.print("[green]✓ Diagnostics complete[/green]")
        return diagnostics

    def save_configuration(self, path: Union[str, Path]) -> bool:
        """Save current configuration to file."""
        try:
            config_path = Path(path)
            config_data = {
                "quantum_network_config": {
                    "timing_precision_ps": self.config.timing_precision_ps,
                    "max_nodes": self.config.max_nodes,
                    "entanglement_fidelity_threshold": self.config.entanglement_fidelity_threshold,
                    "anomaly_detection_sensitivity": self.config.anomaly_detection_sensitivity,
                    "qkd_key_rate_min": self.config.qkd_key_rate_min,
                    "sensor_sync_tolerance_ps": self.config.sensor_sync_tolerance_ps,
                    "enable_adaptive_thresholds": self.config.enable_adaptive_thresholds,
                    "enable_fusion_analysis": self.config.enable_fusion_analysis,
                    "log_level": self.config.log_level,
                },
                "network_state": {
                    "status": self.status.value,
                    "total_nodes": len(self.nodes),
                    "uptime_seconds": time.time() - self.start_time,
                }
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.console.print(f"[green]✓ Configuration saved to {config_path}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to save configuration: {e}[/red]")
            return False

    def load_configuration(self, path: Union[str, Path]) -> bool:
        """Load configuration from file."""
        try:
            config_path = Path(path)
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Update configuration
            qn_config = config_data.get("quantum_network_config", {})
            self.config = QuantumNetworkConfig(**qn_config)

            self.console.print(f"[green]✓ Configuration loaded from {config_path}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to load configuration: {e}[/red]")
            return False