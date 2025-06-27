"""
Monitoring and visualization components for hyperparameter optimization.
Provides real-time monitoring, progress tracking, and result visualization.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd

from .base_optimizer import OptimizationTrial
from ..infrastructure.logging.logger import get_logger

logger = get_logger(__name__)

# Optional imports for advanced plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("Plotly not available for interactive plots. Install with: pip install plotly")


class OptimizationMonitor:
    """Monitors optimization progress in real-time."""
    
    def __init__(self, update_interval: float = 5.0, max_history: int = 1000):
        self.update_interval = update_interval
        self.max_history = max_history
        
        # Monitoring data
        self.trial_history: List[OptimizationTrial] = []
        self.score_history: List[Tuple[datetime, float]] = []
        self.best_score_history: List[Tuple[datetime, float]] = []
        self.resource_usage_history: List[Dict[str, Any]] = []
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Statistics
        self.start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        self.current_best_score: Optional[float] = None
        
        logger.info("Optimization monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start monitoring in background thread."""
        if self._monitoring_active:
            return
        
        self.start_time = datetime.now()
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Optimization monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Optimization monitoring stopped")
    
    def add_trial(self, trial: OptimizationTrial) -> None:
        """Add a completed trial to monitoring."""
        with self._lock:
            self.trial_history.append(trial)
            
            # Keep only recent trials
            if len(self.trial_history) > self.max_history:
                self.trial_history = self.trial_history[-self.max_history:]
            
            # Update score history
            if trial.status == "completed" and trial.score is not None:
                timestamp = trial.end_time or datetime.now()
                self.score_history.append((timestamp, trial.score))
                
                # Update best score
                if self.current_best_score is None or trial.score > self.current_best_score:
                    self.current_best_score = trial.score
                    self.best_score_history.append((timestamp, trial.score))
            
            self.last_update_time = datetime.now()
    
    def add_resource_usage(self, usage_data: Dict[str, Any]) -> None:
        """Add resource usage data."""
        with self._lock:
            usage_data['timestamp'] = datetime.now()
            self.resource_usage_history.append(usage_data)
            
            # Keep only recent data
            if len(self.resource_usage_history) > self.max_history:
                self.resource_usage_history = self.resource_usage_history[-self.max_history:]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            completed_trials = [t for t in self.trial_history if t.status == "completed" and t.score is not None]
            failed_trials = [t for t in self.trial_history if t.status == "failed"]
            
            stats = {
                'total_trials': len(self.trial_history),
                'completed_trials': len(completed_trials),
                'failed_trials': len(failed_trials),
                'success_rate': len(completed_trials) / len(self.trial_history) if self.trial_history else 0,
                'current_best_score': self.current_best_score,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'elapsed_time_minutes': (datetime.now() - self.start_time).total_seconds() / 60 
                                       if self.start_time else 0
            }
            
            if completed_trials:
                scores = [t.score for t in completed_trials]
                durations = [t.duration_seconds for t in completed_trials if t.duration_seconds]
                
                stats.update({
                    'score_mean': np.mean(scores),
                    'score_std': np.std(scores),
                    'score_min': np.min(scores),
                    'score_max': np.max(scores),
                    'avg_trial_duration_seconds': np.mean(durations) if durations else 0,
                    'total_compute_hours': sum(durations) / 3600 if durations else 0
                })
            
            return stats
    
    def get_progress_report(self) -> str:
        """Generate a human-readable progress report."""
        stats = self.get_summary_stats()
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("OPTIMIZATION PROGRESS REPORT")
        report_lines.append("=" * 60)
        
        # Basic stats
        report_lines.append(f"Total Trials: {stats['total_trials']}")
        report_lines.append(f"Completed: {stats['completed_trials']}")
        report_lines.append(f"Failed: {stats['failed_trials']}")
        report_lines.append(f"Success Rate: {stats['success_rate']:.1%}")
        
        if stats['current_best_score'] is not None:
            report_lines.append(f"Best Score: {stats['current_best_score']:.6f}")
        
        report_lines.append(f"Elapsed Time: {stats['elapsed_time_minutes']:.1f} minutes")
        
        # Performance stats
        if 'score_mean' in stats:
            report_lines.append("")
            report_lines.append("PERFORMANCE STATISTICS")
            report_lines.append("-" * 30)
            report_lines.append(f"Mean Score: {stats['score_mean']:.6f}")
            report_lines.append(f"Score Std: {stats['score_std']:.6f}")
            report_lines.append(f"Score Range: {stats['score_min']:.6f} - {stats['score_max']:.6f}")
            report_lines.append(f"Avg Trial Duration: {stats['avg_trial_duration_seconds']:.1f}s")
            report_lines.append(f"Total Compute Time: {stats['total_compute_hours']:.2f} hours")
        
        return '\n'.join(report_lines)
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Log progress periodically
                if len(self.trial_history) > 0:
                    stats = self.get_summary_stats()
                    logger.info(f"Optimization progress: {stats['completed_trials']} trials, "
                              f"best score: {stats.get('current_best_score', 'N/A')}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def export_data(self, output_dir: Path) -> None:
        """Export monitoring data to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export trial data
        trial_data = []
        for trial in self.trial_history:
            trial_dict = {
                'trial_id': trial.trial_id,
                'parameters': trial.parameters,
                'score': trial.score,
                'status': trial.status,
                'start_time': trial.start_time.isoformat() if trial.start_time else None,
                'end_time': trial.end_time.isoformat() if trial.end_time else None,
                'duration_seconds': trial.duration_seconds,
                'error_message': trial.error_message
            }
            trial_data.append(trial_dict)
        
        with open(output_dir / "trial_data.json", 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        # Export summary statistics
        stats = self.get_summary_stats()
        with open(output_dir / "summary_stats.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Export progress report
        report = self.get_progress_report()
        with open(output_dir / "progress_report.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"Monitoring data exported to {output_dir}")


class OptimizationVisualizer:
    """Creates visualizations for optimization results."""
    
    def __init__(self, output_dir: Path, use_plotly: bool = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_plotly = use_plotly if use_plotly is not None else PLOTLY_AVAILABLE
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Visualization initialized, plotly={'enabled' if self.use_plotly else 'disabled'}")
    
    def plot_optimization_progress(self, trials: List[OptimizationTrial], save_path: Optional[Path] = None) -> None:
        """Plot optimization progress over time."""
        completed_trials = [t for t in trials if t.status == "completed" and t.score is not None]
        
        if len(completed_trials) < 2:
            logger.warning("Not enough completed trials for progress plot")
            return
        
        # Prepare data
        trial_numbers = list(range(1, len(completed_trials) + 1))
        scores = [t.score for t in completed_trials]
        best_scores = np.maximum.accumulate(scores)
        
        if self.use_plotly:
            self._plot_progress_plotly(trial_numbers, scores, best_scores, save_path)
        else:
            self._plot_progress_matplotlib(trial_numbers, scores, best_scores, save_path)
    
    def _plot_progress_matplotlib(self, trial_numbers: List[int], scores: List[float], 
                                best_scores: List[float], save_path: Optional[Path]) -> None:
        """Plot progress using matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Score progression
        ax1.scatter(trial_numbers, scores, alpha=0.6, s=30, label='Trial Scores')
        ax1.plot(trial_numbers, best_scores, 'r-', linewidth=2, label='Best Score')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Score')
        ax1.set_title('Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Score distribution
        ax2.hist(scores, bins=min(20, len(scores)//3), alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
        ax2.axvline(max(scores), color='green', linestyle='--', label=f'Best: {max(scores):.4f}')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Progress plot saved to {save_path}")
        else:
            plt.savefig(self.output_dir / "optimization_progress.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_progress_plotly(self, trial_numbers: List[int], scores: List[float], 
                            best_scores: List[float], save_path: Optional[Path]) -> None:
        """Plot progress using plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Optimization Progress', 'Score Distribution'),
            vertical_spacing=0.1
        )
        
        # Score progression
        fig.add_trace(
            go.Scatter(x=trial_numbers, y=scores, mode='markers', name='Trial Scores',
                      marker=dict(size=6, opacity=0.6)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=trial_numbers, y=best_scores, mode='lines', name='Best Score',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Score distribution
        fig.add_trace(
            go.Histogram(x=scores, nbinsx=min(20, len(scores)//3), name='Score Distribution',
                        showlegend=False),
            row=2, col=1
        )
        
        # Add mean and best lines
        fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {np.mean(scores):.4f}", row=2, col=1)
        fig.add_vline(x=max(scores), line_dash="dash", line_color="green", 
                     annotation_text=f"Best: {max(scores):.4f}", row=2, col=1)
        
        fig.update_xaxes(title_text="Trial Number", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive progress plot saved to {save_path}")
        else:
            fig.write_html(self.output_dir / "optimization_progress.html")
    
    def plot_parameter_importance(self, trials: List[OptimizationTrial], save_path: Optional[Path] = None) -> None:
        """Plot parameter importance analysis."""
        completed_trials = [t for t in trials if t.status == "completed" and t.score is not None]
        
        if len(completed_trials) < 10:
            logger.warning("Not enough trials for parameter importance analysis")
            return
        
        # Extract parameter data
        param_data = {}
        scores = []
        
        for trial in completed_trials:
            scores.append(trial.score)
            for param_name, param_value in trial.parameters.items():
                if param_name not in param_data:
                    param_data[param_name] = []
                param_data[param_name].append(param_value)
        
        # Calculate correlations for numeric parameters
        correlations = {}
        for param_name, param_values in param_data.items():
            try:
                # Convert to numeric if possible
                numeric_values = []
                for val in param_values:
                    if isinstance(val, (int, float)):
                        numeric_values.append(float(val))
                    elif isinstance(val, bool):
                        numeric_values.append(float(val))
                    else:
                        # Skip non-numeric parameters for correlation
                        break
                else:
                    if len(set(numeric_values)) > 1:  # Avoid zero variance
                        correlation = np.corrcoef(numeric_values, scores)[0, 1]
                        correlations[param_name] = abs(correlation)
            except:
                continue
        
        if not correlations:
            logger.warning("No numeric parameters found for importance analysis")
            return
        
        # Sort by importance
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params]
        importance_values = [p[1] for p in sorted_params]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(param_names) * 0.4)))
        
        bars = ax.barh(param_names, importance_values)
        ax.set_xlabel('Parameter Importance (|correlation|)')
        ax.set_title('Hyperparameter Importance Analysis')
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter importance plot saved to {save_path}")
        else:
            plt.savefig(self.output_dir / "parameter_importance.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_parameter_correlations(self, trials: List[OptimizationTrial], save_path: Optional[Path] = None) -> None:
        """Plot correlation matrix of parameters."""
        completed_trials = [t for t in trials if t.status == "completed" and t.score is not None]
        
        if len(completed_trials) < 10:
            logger.warning("Not enough trials for correlation analysis")
            return
        
        # Create DataFrame
        data_rows = []
        for trial in completed_trials:
            row = trial.parameters.copy()
            row['score'] = trial.score
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric parameters for correlation analysis")
            return
        
        correlation_matrix = df[numeric_cols].corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=ax)
        
        ax.set_title('Parameter Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        else:
            plt.savefig(self.output_dir / "parameter_correlations.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_optimization_dashboard(self, trials: List[OptimizationTrial], monitor: OptimizationMonitor) -> None:
        """Create comprehensive optimization dashboard."""
        logger.info("Creating optimization dashboard...")
        
        # Generate all plots
        self.plot_optimization_progress(trials)
        self.plot_parameter_importance(trials)
        self.plot_parameter_correlations(trials)
        
        # Create summary dashboard
        stats = monitor.get_summary_stats()
        
        if self.use_plotly:
            self._create_plotly_dashboard(trials, stats)
        
        # Generate HTML report
        self._create_html_report(trials, stats)
        
        logger.info(f"Dashboard created in {self.output_dir}")
    
    def _create_plotly_dashboard(self, trials: List[OptimizationTrial], stats: Dict[str, Any]) -> None:
        """Create interactive Plotly dashboard."""
        if not self.use_plotly:
            return
        
        completed_trials = [t for t in trials if t.status == "completed" and t.score is not None]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score Progress', 'Parameter vs Score', 'Trial Duration', 'Score Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add plots...
        # (Implementation details for comprehensive dashboard)
        
        fig.update_layout(height=800, showlegend=True, title_text="Optimization Dashboard")
        fig.write_html(self.output_dir / "dashboard.html")
    
    def _create_html_report(self, trials: List[OptimizationTrial], stats: Dict[str, Any]) -> None:
        """Create HTML report with embedded plots."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperparameter Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .stats {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .plot {{ text-align: center; margin: 30px 0; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hyperparameter Optimization Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <h2>Summary Statistics</h2>
                <p><strong>Total Trials:</strong> {stats.get('total_trials', 0)}</p>
                <p><strong>Completed Trials:</strong> {stats.get('completed_trials', 0)}</p>
                <p><strong>Success Rate:</strong> {stats.get('success_rate', 0):.1%}</p>
                <p><strong>Best Score:</strong> {stats.get('current_best_score', 'N/A')}</p>
                <p><strong>Elapsed Time:</strong> {stats.get('elapsed_time_minutes', 0):.1f} minutes</p>
            </div>
            
            <div class="plot">
                <h2>Optimization Progress</h2>
                <img src="optimization_progress.png" alt="Optimization Progress">
            </div>
            
            <div class="plot">
                <h2>Parameter Importance</h2>
                <img src="parameter_importance.png" alt="Parameter Importance">
            </div>
            
            <div class="plot">
                <h2>Parameter Correlations</h2>
                <img src="parameter_correlations.png" alt="Parameter Correlations">
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "report.html", 'w') as f:
            f.write(html_content)


def create_monitor() -> OptimizationMonitor:
    """Factory function to create optimization monitor."""
    return OptimizationMonitor()


def create_visualizer(output_dir: Path) -> OptimizationVisualizer:
    """Factory function to create optimization visualizer."""
    return OptimizationVisualizer(output_dir)