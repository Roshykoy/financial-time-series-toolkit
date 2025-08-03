"""
Ray Cluster Manager for MarkSix AI Phase 3.

Provides Ray-based distributed computing integration with Kubernetes,
enabling scalable training and optimization across multiple nodes.

Expert Panel Approved: Ray technology stack with 6-node balanced setup.
"""

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Mock ray for testing
    class ray:
        @staticmethod
        def init(**kwargs):
            pass
        @staticmethod
        def is_initialized():
            return False
        @staticmethod
        def shutdown():
            pass
        @staticmethod
        def get(future):
            return None
        @staticmethod
        def cluster_resources():
            return {}
        @staticmethod
        def nodes():
            return []
        @staticmethod
        def remote(cls):
            return cls
import os
import time
import logging
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import json
import yaml
from pathlib import Path

try:
    from optimization.hardware_manager import HardwareResourceManager
except ImportError:
    class HardwareResourceManager:
        def __init__(self):
            pass
        def get_hardware_profile(self):
            import psutil
            return type('obj', (object,), {
                'cpu_cores': psutil.cpu_count(),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'gpu_count': 1 if torch.cuda.is_available() else 0
            })


@dataclass
class RayClusterConfig:
    """Configuration for Ray cluster."""
    head_node: bool = False
    redis_address: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    memory: Optional[int] = None
    object_store_memory: Optional[int] = None
    log_to_driver: bool = True
    namespace: str = "marksix_ai"
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8265


class RayClusterManager:
    """
    Manages Ray cluster for distributed MarkSix AI training and optimization.
    
    Features:
    - Automatic cluster discovery and connection
    - Resource-aware task scheduling
    - Integration with Kubernetes service discovery
    - Fault tolerance and node management
    """
    
    def __init__(self, config: Dict[str, Any], ray_config: Optional[RayClusterConfig] = None):
        """
        Initialize Ray cluster manager.
        
        Args:
            config: Main MarkSix configuration
            ray_config: Ray-specific configuration
        """
        self.config = config
        self.ray_config = ray_config or RayClusterConfig()
        self.hardware_manager = HardwareResourceManager()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Cluster state
        self.cluster_initialized = False
        self.available_resources = {}
        self.worker_nodes = []
        
    def setup_logging(self):
        """Setup Ray-aware logging."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[Ray] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def initialize_cluster(self) -> bool:
        """
        Initialize Ray cluster with auto-detection.
        
        Returns:
            bool: True if cluster initialized successfully
        """
        try:
            # Auto-configure based on environment
            self._auto_configure()
            
            # Check if Ray is already initialized
            if ray.is_initialized():
                self.logger.info("Ray already initialized, connecting to existing cluster")
                self.cluster_initialized = True
                return True
                
            # Initialize Ray based on configuration
            ray_init_kwargs = self._build_ray_init_config()
            
            self.logger.info(f"Initializing Ray cluster with config: {ray_init_kwargs}")
            ray.init(**ray_init_kwargs)
            
            # Verify cluster connectivity
            self._verify_cluster_health()
            
            self.cluster_initialized = True
            self.logger.info("Ray cluster initialized successfully")
            
            # Log cluster information
            self._log_cluster_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ray cluster: {e}")
            return False
            
    def _auto_configure(self):
        """Auto-configure Ray settings based on environment."""
        try:
            # Detect if running in Kubernetes
            if os.path.exists('/var/run/secrets/kubernetes.io'):
                self.logger.info("Kubernetes environment detected")
                self._configure_for_kubernetes()
            else:
                self.logger.info("Standalone environment detected")
                self._configure_for_standalone()
                
            # Auto-detect hardware resources
            hardware_profile = self.hardware_manager.get_hardware_profile()
            
            # Set resource limits if not specified
            if self.ray_config.num_cpus is None:
                self.ray_config.num_cpus = max(1, hardware_profile.cpu_cores - 1)
                
            if self.ray_config.num_gpus is None:
                self.ray_config.num_gpus = hardware_profile.gpu_count
                
            if self.ray_config.memory is None:
                # Use 80% of available memory
                available_memory_bytes = int(hardware_profile.available_memory_gb * 0.8 * 1024**3)
                self.ray_config.memory = available_memory_bytes
                
            if self.ray_config.object_store_memory is None:
                # Use 20% of available memory for object store
                object_store_bytes = int(hardware_profile.available_memory_gb * 0.2 * 1024**3)
                self.ray_config.object_store_memory = object_store_bytes
                
            self.logger.info(f"Auto-configured Ray resources: CPUs={self.ray_config.num_cpus}, "
                           f"GPUs={self.ray_config.num_gpus}, Memory={self.ray_config.memory//1024**3}GB")
            
        except Exception as e:
            self.logger.error(f"Auto-configuration failed: {e}")
            
    def _configure_for_kubernetes(self):
        """Configure Ray for Kubernetes environment."""
        # Check for Ray head service
        ray_head_service = os.environ.get('RAY_HEAD_SERVICE', 'ray-head-svc')
        ray_head_port = os.environ.get('RAY_HEAD_PORT', '10001')
        
        # Determine if this is the head node
        pod_name = os.environ.get('HOSTNAME', '')
        if 'head' in pod_name.lower():
            self.ray_config.head_node = True
            self.logger.info("Detected as Ray head node")
        else:
            self.ray_config.head_node = False
            self.ray_config.redis_address = f"ray://{ray_head_service}:{ray_head_port}"
            self.logger.info(f"Detected as Ray worker, connecting to {self.ray_config.redis_address}")
            
    def _configure_for_standalone(self):
        """Configure Ray for standalone environment."""
        # Check for manual Ray address override
        ray_address = os.environ.get('RAY_ADDRESS')
        if ray_address:
            self.ray_config.redis_address = ray_address
            self.ray_config.head_node = False
        else:
            self.ray_config.head_node = True
            
    def _build_ray_init_config(self) -> Dict[str, Any]:
        """Build Ray initialization configuration."""
        config = {
            'log_to_driver': self.ray_config.log_to_driver,
            'namespace': self.ray_config.namespace,
            'ignore_reinit_error': True
        }
        
        # Add address if connecting to existing cluster
        if self.ray_config.redis_address:
            config['address'] = self.ray_config.redis_address
        else:
            # Head node configuration
            config.update({
                'num_cpus': self.ray_config.num_cpus,
                'num_gpus': self.ray_config.num_gpus,
                'memory': self.ray_config.memory,
                'object_store_memory': self.ray_config.object_store_memory,
                'dashboard_host': self.ray_config.dashboard_host,
                'dashboard_port': self.ray_config.dashboard_port
            })
            
        return config
        
    def _verify_cluster_health(self):
        """Verify Ray cluster health and connectivity."""
        try:
            # Test basic Ray functionality
            @ray.remote
            def health_check():
                return "healthy"
                
            result = ray.get(health_check.remote())
            if result != "healthy":
                raise Exception("Health check failed")
                
            # Get cluster resources
            resources = ray.cluster_resources()
            self.available_resources = resources
            
            self.logger.info(f"Cluster health check passed, resources: {resources}")
            
        except Exception as e:
            self.logger.error(f"Cluster health check failed: {e}")
            raise
            
    def _log_cluster_info(self):
        """Log detailed cluster information."""
        try:
            # Get cluster information
            nodes = ray.nodes()
            self.worker_nodes = [node for node in nodes if node['Alive']]
            
            self.logger.info(f"Ray cluster info:")
            self.logger.info(f"  Total nodes: {len(self.worker_nodes)}")
            self.logger.info(f"  Available resources: {self.available_resources}")
            
            for i, node in enumerate(self.worker_nodes):
                self.logger.info(f"  Node {i}: {node.get('NodeManagerAddress', 'unknown')} "
                                f"(CPUs: {node.get('Resources', {}).get('CPU', 0)}, "
                                f"GPUs: {node.get('Resources', {}).get('GPU', 0)})")
                
        except Exception as e:
            self.logger.error(f"Failed to log cluster info: {e}")
            
    @ray.remote
    class DistributedParetoOptimizer:
        """Ray actor for distributed Pareto Front optimization."""
        
        def __init__(self, config: Dict[str, Any], worker_id: int):
            self.config = config
            self.worker_id = worker_id
            
        def run_optimization_trials(self, 
                                   trials: int, 
                                   algorithm: str,
                                   parameter_space: Dict[str, Any]) -> Dict[str, Any]:
            """Run optimization trials on this worker."""
            # Import here to avoid circular imports
            from ..optimization.pareto_integration import ParetoIntegration
            
            pareto_integration = ParetoIntegration()
            results = pareto_integration.run_optimization(
                trials, 
                algorithm,
                node_id=self.worker_id
            )
            
            return {
                'worker_id': self.worker_id,
                'trials_completed': trials,
                'results': results
            }
            
    def distribute_pareto_optimization(self, 
                                     total_trials: int,
                                     algorithm: str = "nsga2",
                                     parameter_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Distribute Pareto Front optimization across Ray cluster.
        
        Args:
            total_trials: Total optimization trials to run
            algorithm: Optimization algorithm ("nsga2" or "tpe")
            parameter_space: Parameter space definition
            
        Returns:
            Merged optimization results from all workers
        """
        if not self.cluster_initialized:
            raise RuntimeError("Ray cluster not initialized")
            
        try:
            # Determine number of workers
            num_workers = len(self.worker_nodes)
            if num_workers == 0:
                num_workers = 1
                
            # Distribute trials across workers
            trials_per_worker = total_trials // num_workers
            remaining_trials = total_trials % num_workers
            
            self.logger.info(f"Distributing {total_trials} trials across {num_workers} workers")
            
            # Create worker actors
            workers = []
            for i in range(num_workers):
                worker = self.DistributedParetoOptimizer.remote(self.config, i)
                workers.append(worker)
                
            # Submit optimization tasks
            futures = []
            for i, worker in enumerate(workers):
                worker_trials = trials_per_worker
                if i < remaining_trials:
                    worker_trials += 1
                    
                if worker_trials > 0:
                    future = worker.run_optimization_trials.remote(
                        worker_trials,
                        algorithm,
                        parameter_space or {}
                    )
                    futures.append(future)
                    
            # Wait for all workers to complete
            self.logger.info("Waiting for distributed optimization to complete...")
            results = ray.get(futures)
            
            # Merge results from all workers
            merged_results = self._merge_worker_results(results, algorithm)
            
            self.logger.info(f"Distributed optimization completed with {len(results)} workers")
            return merged_results
            
        except Exception as e:
            self.logger.error(f"Distributed optimization failed: {e}")
            raise
            
    def _merge_worker_results(self, worker_results: List[Dict[str, Any]], algorithm: str) -> Dict[str, Any]:
        """
        Merge optimization results from multiple workers.
        
        Args:
            worker_results: List of results from each worker
            algorithm: Algorithm used for optimization
            
        Returns:
            Merged Pareto Front results
        """
        all_solutions = []
        all_objectives = []
        total_trials = 0
        
        for result in worker_results:
            worker_data = result['results']
            total_trials += result['trials_completed']
            
            if 'pareto_front' in worker_data:
                all_solutions.extend(worker_data['pareto_front'])
            if 'objectives' in worker_data:
                all_objectives.extend(worker_data['objectives'])
                
        # Re-compute global Pareto Front
        # This is a simplified merge - in real implementation,
        # would apply proper Pareto dominance sorting
        
        # Sort by first objective and take top solutions
        if all_objectives:
            combined = list(zip(all_solutions, all_objectives))
            # Sort by first objective (assuming minimization)
            combined.sort(key=lambda x: x[1][0] if x[1] else float('inf'))
            
            # Take top 50 solutions for final Pareto Front
            final_size = min(50, len(combined))
            final_solutions = [sol for sol, _ in combined[:final_size]]
            final_objectives = [obj for _, obj in combined[:final_size]]
        else:
            final_solutions = all_solutions[:50]
            final_objectives = all_objectives[:50]
            
        return {
            'pareto_front': final_solutions,
            'objectives': final_objectives,
            'algorithm': algorithm,
            'total_trials': total_trials,
            'distributed': True,
            'num_workers': len(worker_results),
            'worker_results': worker_results
        }
        
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get current cluster status and resource information.
        
        Returns:
            Dict containing cluster status
        """
        if not self.cluster_initialized:
            return {'status': 'not_initialized'}
            
        try:
            return {
                'status': 'active',
                'nodes': len(self.worker_nodes),
                'resources': self.available_resources,
                'head_node': self.ray_config.head_node,
                'namespace': self.ray_config.namespace,
                'dashboard_url': f"http://{self.ray_config.dashboard_host}:{self.ray_config.dashboard_port}"
            }
        except Exception as e:
            self.logger.error(f"Failed to get cluster status: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def shutdown_cluster(self):
        """Shutdown Ray cluster gracefully."""
        try:
            if self.cluster_initialized:
                self.logger.info("Shutting down Ray cluster")
                ray.shutdown()
                self.cluster_initialized = False
                
        except Exception as e:
            self.logger.error(f"Error during cluster shutdown: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        self.initialize_cluster()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown_cluster()


# Helper function for easy Ray cluster setup
def create_ray_cluster_manager(config: Dict[str, Any]) -> RayClusterManager:
    """
    Factory function to create Ray cluster manager.
    
    Args:
        config: MarkSix configuration dictionary
        
    Returns:
        Configured RayClusterManager
    """
    ray_config = RayClusterConfig()
    
    # Apply any Ray-specific settings from main config
    if 'ray_cluster' in config:
        ray_settings = config['ray_cluster']
        for key, value in ray_settings.items():
            if hasattr(ray_config, key):
                setattr(ray_config, key, value)
                
    return RayClusterManager(config, ray_config)