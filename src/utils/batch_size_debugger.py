"""
Batch Size Debugging and Monitoring Infrastructure.

This module provides real-time monitoring and debugging tools for the batch size 
constraint system throughout the optimization pipeline.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import psutil
import torch
from pathlib import Path
import json


@dataclass
class BatchSizeDecision:
    """Record of a batch size decision point in the pipeline."""
    timestamp: float
    location: str  # Function/module where decision was made
    original_batch_size: int
    final_batch_size: int
    reason: str  # Why this batch size was chosen
    memory_info: Optional[Dict[str, Any]] = None
    constraint_applied: bool = False
    source_config: Optional[str] = None


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    timestamp: float
    location: str
    batch_size: int
    expected_minimum: int
    violation_type: str  # 'below_minimum', 'above_ceiling', 'invalid_value'
    stack_trace: Optional[str] = None


class BatchSizeMonitor:
    """Real-time monitoring system for batch size decisions and constraints."""
    
    def __init__(self, enable_memory_tracking: bool = True, log_level: str = "INFO"):
        self.decisions: List[BatchSizeDecision] = []
        self.violations: List[ConstraintViolation] = []
        self.enable_memory_tracking = enable_memory_tracking
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger('batch_size_monitor')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [BatchMonitor] %(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def record_decision(self, 
                       location: str,
                       original_batch_size: int,
                       final_batch_size: int,
                       reason: str,
                       source_config: Optional[str] = None) -> None:
        """Record a batch size decision."""
        with self._lock:
            memory_info = self._get_memory_info() if self.enable_memory_tracking else None
            constraint_applied = (original_batch_size != final_batch_size)
            
            decision = BatchSizeDecision(
                timestamp=time.time(),
                location=location,
                original_batch_size=original_batch_size,
                final_batch_size=final_batch_size,
                reason=reason,
                memory_info=memory_info,
                constraint_applied=constraint_applied,
                source_config=source_config
            )
            
            self.decisions.append(decision)
            
            # Log the decision
            if constraint_applied:
                self.logger.info(f"{location}: {original_batch_size} → {final_batch_size} ({reason})")
            else:
                self.logger.debug(f"{location}: {final_batch_size} (no change: {reason})")
    
    def record_violation(self,
                        location: str,
                        batch_size: int,
                        expected_minimum: int,
                        violation_type: str) -> None:
        """Record a constraint violation."""
        with self._lock:
            # Get stack trace for debugging
            import traceback
            stack_trace = ''.join(traceback.format_stack())
            
            violation = ConstraintViolation(
                timestamp=time.time(),
                location=location,
                batch_size=batch_size,
                expected_minimum=expected_minimum,
                violation_type=violation_type,
                stack_trace=stack_trace
            )
            
            self.violations.append(violation)
            
            # Log violation as warning
            self.logger.warning(f"CONSTRAINT VIOLATION in {location}: "
                              f"batch_size={batch_size}, expected_min={expected_minimum}, "
                              f"type={violation_type}")
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        memory_info = {}
        
        try:
            # System memory
            ram = psutil.virtual_memory()
            memory_info.update({
                'system_memory_total_gb': ram.total / (1024**3),
                'system_memory_available_gb': ram.available / (1024**3),
                'system_memory_percent': ram.percent
            })
            
            # GPU memory if available
            if torch.cuda.is_available():
                total_mem, free_mem = torch.cuda.mem_get_info()
                memory_info.update({
                    'gpu_memory_total_gb': total_mem / (1024**3),
                    'gpu_memory_free_gb': free_mem / (1024**3),
                    'gpu_memory_used_gb': (total_mem - free_mem) / (1024**3),
                    'gpu_memory_percent': ((total_mem - free_mem) / total_mem) * 100
                })
        except Exception as e:
            memory_info['error'] = str(e)
        
        return memory_info
    
    def get_decision_chain(self, include_memory: bool = False) -> List[Dict[str, Any]]:
        """Get the chain of batch size decisions."""
        with self._lock:
            chain = []
            for decision in self.decisions:
                entry = {
                    'timestamp': decision.timestamp,
                    'relative_time': decision.timestamp - self.start_time,
                    'location': decision.location,
                    'original_batch_size': decision.original_batch_size,
                    'final_batch_size': decision.final_batch_size,
                    'reason': decision.reason,
                    'constraint_applied': decision.constraint_applied,
                    'source_config': decision.source_config
                }
                
                if include_memory and decision.memory_info:
                    entry['memory_info'] = decision.memory_info
                
                chain.append(entry)
            
            return chain
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get all recorded constraint violations."""
        with self._lock:
            return [{
                'timestamp': v.timestamp,
                'relative_time': v.timestamp - self.start_time,
                'location': v.location,
                'batch_size': v.batch_size,
                'expected_minimum': v.expected_minimum,
                'violation_type': v.violation_type,
                'stack_trace': v.stack_trace
            } for v in self.violations]
    
    def print_summary(self) -> None:
        """Print a summary of all decisions and violations."""
        print("\n🔍 Batch Size Decision Chain Summary")
        print("=" * 50)
        
        decisions = self.get_decision_chain()
        
        if not decisions:
            print("   No batch size decisions recorded")
            return
        
        print(f"   Total decisions: {len(decisions)}")
        print(f"   Constraints applied: {sum(1 for d in decisions if d['constraint_applied'])}")
        print(f"   Violations detected: {len(self.violations)}")
        print()
        
        # Show decision chain
        for i, decision in enumerate(decisions, 1):
            timestamp = decision['relative_time']
            location = decision['location']
            original = decision['original_batch_size']
            final = decision['final_batch_size']
            reason = decision['reason']
            
            status = "🔧" if decision['constraint_applied'] else "✅"
            print(f"{status} {i:2d}. [{timestamp:6.2f}s] {location}")
            print(f"      {original} → {final} ({reason})")
        
        # Show violations
        violations = self.get_violations()
        if violations:
            print(f"\n❌ Constraint Violations ({len(violations)}):")
            for i, violation in enumerate(violations, 1):
                print(f"   {i}. {violation['location']}: batch_size={violation['batch_size']} "
                      f"(expected min={violation['expected_minimum']})")
    
    def save_debug_report(self, filename: Optional[str] = None) -> str:
        """Save detailed debug report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_size_debug_report_{timestamp}.json"
        
        report = {
            'metadata': {
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration_seconds': time.time() - self.start_time,
                'total_decisions': len(self.decisions),
                'total_violations': len(self.violations),
                'constraints_applied': sum(1 for d in self.decisions if d.constraint_applied),
                'memory_tracking_enabled': self.enable_memory_tracking
            },
            'decisions': self.get_decision_chain(include_memory=True),
            'violations': self.get_violations(),
            'system_info': self._get_memory_info()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Debug report saved to: {filename}")
        return filename


# Global monitor instance
_global_monitor: Optional[BatchSizeMonitor] = None


def get_batch_size_monitor() -> BatchSizeMonitor:
    """Get or create the global batch size monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = BatchSizeMonitor()
    return _global_monitor


def monitor_batch_decision(location: str, 
                          original_batch_size: int,
                          final_batch_size: int,
                          reason: str,
                          source_config: Optional[str] = None) -> None:
    """Convenience function to record a batch size decision."""
    monitor = get_batch_size_monitor()
    monitor.record_decision(location, original_batch_size, final_batch_size, reason, source_config)


def monitor_constraint_violation(location: str,
                                batch_size: int, 
                                expected_minimum: int,
                                violation_type: str = "below_minimum") -> None:
    """Convenience function to record a constraint violation."""
    monitor = get_batch_size_monitor()
    monitor.record_violation(location, batch_size, expected_minimum, violation_type)


def validate_batch_size_constraint(batch_size: int, 
                                  minimum: int = 8,
                                  location: str = "unknown") -> int:
    """Validate and enforce batch size constraint with monitoring."""
    if not isinstance(batch_size, (int, float)) or batch_size < minimum:
        # Record violation before correction
        monitor_constraint_violation(location, batch_size, minimum)
        
        # Apply constraint
        corrected_batch_size = max(minimum, int(batch_size)) if isinstance(batch_size, (int, float)) else minimum
        monitor_batch_decision(location, batch_size, corrected_batch_size, 
                             f"minimum constraint enforcement (min={minimum})")
        return corrected_batch_size
    
    # No constraint needed
    monitor_batch_decision(location, batch_size, batch_size, "no constraint needed")
    return int(batch_size)


class BatchSizeConstraintDecorator:
    """Decorator to automatically monitor batch size constraint enforcement."""
    
    def __init__(self, minimum_batch_size: int = 8, location_override: Optional[str] = None):
        self.minimum_batch_size = minimum_batch_size
        self.location_override = location_override
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            location = self.location_override or f"{func.__module__}.{func.__name__}"
            
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Check if result contains batch size information
            if isinstance(result, dict) and 'batch_size' in result:
                batch_size = result['batch_size']
                validated_batch_size = validate_batch_size_constraint(
                    batch_size, self.minimum_batch_size, location
                )
                result['batch_size'] = validated_batch_size
                
            elif isinstance(result, (int, float)):
                # Assume the result is a batch size
                result = validate_batch_size_constraint(
                    result, self.minimum_batch_size, location
                )
            
            return result
        
        return wrapper


def batch_constraint_monitor(minimum: int = 8, location: Optional[str] = None):
    """Decorator for automatic batch size constraint monitoring."""
    return BatchSizeConstraintDecorator(minimum, location)


# Context manager for batch size monitoring sessions
class BatchSizeMonitoringSession:
    """Context manager for batch size monitoring sessions."""
    
    def __init__(self, session_name: str, save_report: bool = True):
        self.session_name = session_name
        self.save_report = save_report
        self.monitor = get_batch_size_monitor()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.monitor.logger.info(f"Starting batch size monitoring session: {self.session_name}")
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.logger.info(f"Ending batch size monitoring session: {self.session_name} "
                               f"(duration: {duration:.2f}s)")
        
        if self.save_report:
            filename = f"batch_debug_{self.session_name.lower().replace(' ', '_')}.json"
            self.monitor.save_debug_report(filename)
        
        # Print summary for immediate feedback
        print(f"\n🔍 Session '{self.session_name}' Summary:")
        decisions = len(self.monitor.decisions)
        violations = len(self.monitor.violations) 
        constraints = sum(1 for d in self.monitor.decisions if d.constraint_applied)
        print(f"   • Duration: {duration:.2f}s")
        print(f"   • Decisions: {decisions}, Constraints: {constraints}, Violations: {violations}")


# Performance impact measurement
class BatchSizePerformanceMeasurer:
    """Measure performance impact of different batch sizes."""
    
    def __init__(self):
        self.measurements: Dict[int, List[float]] = {}
        self.memory_usage: Dict[int, List[float]] = {}
    
    def measure_batch_performance(self, 
                                 batch_size: int,
                                 operation_func: Callable,
                                 trials: int = 3) -> Dict[str, float]:
        """Measure performance of an operation with given batch size."""
        if batch_size not in self.measurements:
            self.measurements[batch_size] = []
            self.memory_usage[batch_size] = []
        
        trial_times = []
        trial_memory = []
        
        for trial in range(trials):
            # Clear cache if CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Measure memory before
            memory_before = self._get_memory_usage()
            
            # Time the operation
            start_time = time.time()
            operation_func(batch_size)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Measure memory after
            memory_after = self._get_memory_usage()
            
            trial_time = end_time - start_time
            memory_used = memory_after - memory_before
            
            trial_times.append(trial_time)
            trial_memory.append(memory_used)
        
        # Store results
        self.measurements[batch_size].extend(trial_times)
        self.memory_usage[batch_size].extend(trial_memory)
        
        # Return summary
        avg_time = sum(trial_times) / len(trial_times)
        avg_memory = sum(trial_memory) / len(trial_memory)
        throughput = batch_size / avg_time if avg_time > 0 else 0
        
        return {
            'batch_size': batch_size,
            'avg_time_seconds': avg_time,
            'avg_memory_gb': avg_memory,
            'throughput_samples_per_sec': throughput,
            'trials': trials
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            return psutil.Process().memory_info().rss / (1024**3)
    
    def compare_batch_sizes(self, batch_sizes: List[int]) -> Dict[str, Any]:
        """Compare performance across different batch sizes."""
        comparison = {}
        
        for batch_size in batch_sizes:
            if batch_size in self.measurements and self.measurements[batch_size]:
                times = self.measurements[batch_size]
                memories = self.memory_usage[batch_size]
                
                comparison[batch_size] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'avg_memory': sum(memories) / len(memories),
                    'throughput': batch_size / (sum(times) / len(times)),
                    'measurements_count': len(times)
                }
        
        return comparison
    
    def find_optimal_batch_size(self, 
                               max_memory_gb: Optional[float] = None,
                               min_throughput: Optional[float] = None) -> Optional[int]:
        """Find optimal batch size based on constraints."""
        viable_sizes = []
        
        for batch_size, times in self.measurements.items():
            if not times:
                continue
                
            avg_time = sum(times) / len(times)
            avg_memory = sum(self.memory_usage[batch_size]) / len(self.memory_usage[batch_size])
            throughput = batch_size / avg_time
            
            # Check constraints
            memory_ok = max_memory_gb is None or avg_memory <= max_memory_gb
            throughput_ok = min_throughput is None or throughput >= min_throughput
            
            if memory_ok and throughput_ok:
                viable_sizes.append((batch_size, throughput, avg_memory))
        
        if not viable_sizes:
            return None
        
        # Return batch size with highest throughput
        return max(viable_sizes, key=lambda x: x[1])[0]