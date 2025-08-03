"""
Distributed computing module for MarkSix AI Phase 3 implementation.

Provides multi-node distributed training, microservices architecture,
and production-ready deployment capabilities.

Expert Panel Approved Architecture:
- Kubernetes + Ray technology stack
- NCCL backend for multi-GPU coordination  
- NUMA-aware memory architecture
- Microservices production deployment
- 6-node balanced cluster setup
"""

from .training_coordinator import DistributedTrainingCoordinator
from .ray_cluster import RayClusterManager
from .multi_gpu_backend import MultiGPUBackend
from .numa_memory_manager import NUMAMemoryManager

__all__ = [
    'DistributedTrainingCoordinator',
    'RayClusterManager', 
    'MultiGPUBackend',
    'NUMAMemoryManager'
]

__version__ = "3.0.0"
__author__ = "MarkSix AI Expert Panel"