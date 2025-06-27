"""
Tests for hardware resource management functionality.
"""

import pytest
import torch
import psutil
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.optimization.hardware_manager import (
    HardwareResourceManager,
    HardwareProfile,
    ResourceConstraints,
    create_hardware_manager
)


class TestHardwareProfile:
    """Test HardwareProfile dataclass."""
    
    def test_hardware_profile_creation(self):
        """Test hardware profile creation."""
        profile = HardwareProfile(
            cpu_cores=8,
            cpu_frequency=3000.0,
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            gpu_available=True,
            gpu_count=1,
            gpu_memory_gb=[8.0],
            gpu_names=["RTX 3080"],
            recommended_batch_size=32,
            recommended_workers=4,
            parallel_jobs=2
        )
        
        assert profile.cpu_cores == 8
        assert profile.gpu_available is True
        assert profile.gpu_count == 1
        assert profile.recommended_batch_size == 32


class TestResourceConstraints:
    """Test ResourceConstraints dataclass."""
    
    def test_resource_constraints_defaults(self):
        """Test default resource constraints."""
        constraints = ResourceConstraints()
        
        assert constraints.max_memory_fraction == 0.8
        assert constraints.max_gpu_memory_fraction == 0.8
        assert constraints.max_cpu_usage == 0.9
        assert constraints.min_free_memory_gb == 2.0
        assert constraints.enable_memory_monitoring is True
        assert constraints.cleanup_frequency == 10


class TestHardwareResourceManager:
    """Test HardwareResourceManager functionality."""
    
    @pytest.fixture
    def mock_hardware_manager(self):
        """Create a hardware manager with mocked hardware detection."""
        with patch.object(HardwareResourceManager, '_detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareProfile(
                cpu_cores=4,
                cpu_frequency=2400.0,
                total_memory_gb=8.0,
                available_memory_gb=4.0,
                gpu_available=False,
                gpu_count=0,
                gpu_memory_gb=[],
                gpu_names=[],
                recommended_batch_size=16,
                recommended_workers=2,
                parallel_jobs=1
            )
            return HardwareResourceManager()
    
    def test_hardware_manager_initialization(self, mock_hardware_manager):
        """Test hardware manager initialization."""
        manager = mock_hardware_manager
        
        assert manager.profile.cpu_cores == 4
        assert manager.profile.total_memory_gb == 8.0
        assert manager.profile.gpu_available is False
        assert manager.constraints.max_memory_fraction == 0.8
    
    def test_hardware_detection_fallback(self):
        """Test hardware detection with fallback."""
        with patch('psutil.cpu_count', side_effect=Exception("Mock error")):
            manager = HardwareResourceManager()
            
            # Should create fallback profile
            assert manager.profile.cpu_cores == 2
            assert manager.profile.total_memory_gb == 8.0
            assert manager.profile.gpu_available is False
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_get_resource_status(self, mock_cpu_percent, mock_virtual_memory, mock_hardware_manager):
        """Test resource status retrieval."""
        # Mock system metrics
        mock_memory = Mock()
        mock_memory.percent = 75.0
        mock_memory.available = 2 * 1024**3  # 2GB
        mock_memory.used = 6 * 1024**3  # 6GB
        mock_virtual_memory.return_value = mock_memory
        mock_cpu_percent.return_value = 60.0
        
        manager = mock_hardware_manager
        status = manager.get_resource_status()
        
        assert 'cpu_usage_percent' in status
        assert 'memory_usage_percent' in status
        assert 'memory_available_gb' in status
        assert 'timestamp' in status
        assert status['cpu_usage_percent'] == 60.0
        assert status['memory_usage_percent'] == 75.0
    
    def test_check_resource_constraints(self, mock_hardware_manager):
        """Test resource constraint checking."""
        manager = mock_hardware_manager
        
        # Mock get_resource_status to return high usage
        with patch.object(manager, 'get_resource_status') as mock_status:
            mock_status.return_value = {
                'memory_usage_percent': 95.0,  # High memory usage
                'memory_available_gb': 0.5,    # Low free memory
                'cpu_usage_percent': 95.0       # High CPU usage
            }
            
            is_ok, issues = manager.check_resource_constraints()
            
            assert is_ok is False
            assert len(issues) > 0
            assert any("Memory usage too high" in issue for issue in issues)
    
    def test_optimize_for_hardware(self, mock_hardware_manager):
        """Test hardware optimization of configuration."""
        manager = mock_hardware_manager
        
        config = {
            'batch_size': 64,
            'num_workers': 8,
            'device': 'auto'
        }
        
        optimized = manager.optimize_for_hardware(config)
        
        # Should be adjusted for hardware limitations
        assert optimized['batch_size'] <= manager.profile.recommended_batch_size
        assert optimized['num_workers'] == manager.profile.recommended_workers
        assert optimized['device'] == 'cpu'  # No GPU available in mock
    
    def test_cleanup_resources(self, mock_hardware_manager):
        """Test resource cleanup."""
        manager = mock_hardware_manager
        
        # Should not raise exception
        manager.cleanup_resources()
    
    def test_monitoring_start_stop(self, mock_hardware_manager):
        """Test resource monitoring start/stop."""
        manager = mock_hardware_manager
        
        # Start monitoring
        manager.start_monitoring()
        assert manager._monitoring_active is True
        assert manager._monitor_thread is not None
        
        # Stop monitoring
        manager.stop_monitoring()
        assert manager._monitoring_active is False
    
    def test_estimate_trial_time(self, mock_hardware_manager):
        """Test trial time estimation."""
        manager = mock_hardware_manager
        
        config = {
            'batch_size': 16,
            'epochs': 5,
            'device': 'cpu',
            'hidden_size': 256
        }
        
        estimated_time = manager.estimate_trial_time(config)
        
        assert isinstance(estimated_time, float)
        assert estimated_time > 0
    
    def test_suggest_optimization_strategy(self, mock_hardware_manager):
        """Test optimization strategy suggestion."""
        manager = mock_hardware_manager
        
        strategy = manager.suggest_optimization_strategy()
        
        assert 'recommended_algorithm' in strategy
        assert 'max_trials' in strategy
        assert 'parallel_jobs' in strategy
        assert strategy['parallel_jobs'] <= manager.profile.parallel_jobs
    
    def test_save_hardware_profile(self, mock_hardware_manager):
        """Test hardware profile saving."""
        manager = mock_hardware_manager
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            manager.save_hardware_profile(temp_path)
            
            # Verify file was created and contains expected data
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'profile' in data
            assert 'constraints' in data
            assert data['profile']['cpu_cores'] == 4
            
        finally:
            temp_path.unlink()


class TestCreateHardwareManager:
    """Test hardware manager factory function."""
    
    def test_create_hardware_manager_default(self):
        """Test creating hardware manager with defaults."""
        with patch.object(HardwareResourceManager, '_detect_hardware'):
            manager = create_hardware_manager()
            assert isinstance(manager, HardwareResourceManager)
    
    def test_create_hardware_manager_with_config(self):
        """Test creating hardware manager with custom config."""
        config = {
            'max_memory_fraction': 0.7,
            'max_gpu_memory_fraction': 0.9,
            'max_cpu_usage': 0.8
        }
        
        with patch.object(HardwareResourceManager, '_detect_hardware'):
            manager = create_hardware_manager(config)
            
            assert manager.constraints.max_memory_fraction == 0.7
            assert manager.constraints.max_gpu_memory_fraction == 0.9
            assert manager.constraints.max_cpu_usage == 0.8


class TestHardwareManagerIntegration:
    """Integration tests for hardware manager."""
    
    def test_real_hardware_detection(self):
        """Test with real hardware detection (may vary by system)."""
        manager = HardwareResourceManager()
        
        # Basic sanity checks
        assert manager.profile.cpu_cores > 0
        assert manager.profile.total_memory_gb > 0
        assert manager.profile.recommended_batch_size > 0
        assert manager.profile.recommended_workers > 0
    
    def test_gpu_detection_when_available(self):
        """Test GPU detection when CUDA is available."""
        manager = HardwareResourceManager()
        
        if torch.cuda.is_available():
            assert manager.profile.gpu_available is True
            assert manager.profile.gpu_count > 0
            assert len(manager.profile.gpu_memory_gb) == manager.profile.gpu_count
            assert len(manager.profile.gpu_names) == manager.profile.gpu_count
        else:
            assert manager.profile.gpu_available is False
            assert manager.profile.gpu_count == 0
    
    def test_resource_status_real(self):
        """Test getting real resource status."""
        manager = HardwareResourceManager()
        status = manager.get_resource_status()
        
        # Should contain expected keys
        required_keys = [
            'cpu_usage_percent', 'memory_usage_percent', 
            'memory_available_gb', 'memory_used_gb', 'timestamp'
        ]
        
        for key in required_keys:
            assert key in status
        
        # Values should be reasonable
        assert 0 <= status['cpu_usage_percent'] <= 100
        assert 0 <= status['memory_usage_percent'] <= 100
        assert status['memory_available_gb'] >= 0
    
    @pytest.mark.slow
    def test_monitoring_integration(self):
        """Test monitoring over short period."""
        manager = HardwareResourceManager()
        
        # Start monitoring
        manager.start_monitoring()
        
        # Wait briefly for some data collection
        time.sleep(2)
        
        # Stop monitoring
        manager.stop_monitoring()
        
        # Should have collected some data
        monitoring_data = manager.get_monitoring_data()
        assert len(monitoring_data) > 0
        
        # Check data structure
        for data_point in monitoring_data:
            assert 'timestamp' in data_point
            assert 'cpu_usage_percent' in data_point


if __name__ == '__main__':
    pytest.main([__file__])