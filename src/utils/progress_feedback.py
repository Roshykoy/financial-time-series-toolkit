"""
User experience improvements for MarkSix forecasting system.
Provides progress indicators, loading states, and helpful feedback.
"""
import time
import sys
import threading
from typing import Optional, Callable, Any, Iterator
from contextlib import contextmanager
import warnings


class ProgressIndicator:
    """Enhanced progress indicator with multiple display modes."""
    
    def __init__(self, total: Optional[int] = None, description: str = "Processing", 
                 show_percentage: bool = True, show_time: bool = True):
        self.total = total
        self.description = description
        self.show_percentage = show_percentage
        self.show_time = show_time
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms
        
    def update(self, increment: int = 1, description: Optional[str] = None):
        """Update progress with optional description change."""
        self.current += increment
        current_time = time.time()
        
        # Throttle updates to avoid overwhelming the terminal
        if current_time - self.last_update < self.update_interval:
            return
        
        if description:
            self.description = description
        
        self._render()
        self.last_update = current_time
    
    def set_progress(self, current: int, description: Optional[str] = None):
        """Set absolute progress value."""
        self.current = current
        if description:
            self.description = description
        self._render()
    
    def _render(self):
        """Render the progress bar."""
        # Build progress string
        parts = [f"\r🔄 {self.description}"]
        
        if self.total and self.show_percentage:
            percentage = min(100, (self.current / self.total) * 100)
            parts.append(f" {percentage:.1f}%")
            
            # Visual progress bar
            bar_length = 20
            filled_length = int(bar_length * self.current / self.total)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            parts.append(f" [{bar}]")
        
        if self.show_time:
            elapsed = time.time() - self.start_time
            parts.append(f" ({elapsed:.1f}s)")
            
            if self.total and self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                if eta > 0:
                    parts.append(f" ETA: {eta:.1f}s")
        
        parts.append(f" ({self.current}")
        if self.total:
            parts.append(f"/{self.total}")
        parts.append(")")
        
        # Print and flush
        progress_str = "".join(parts)
        print(progress_str[:80], end="", flush=True)  # Limit width
    
    def finish(self, message: Optional[str] = None):
        """Finish progress with optional completion message."""
        if message:
            print(f"\r✅ {message}")
        else:
            print(f"\r✅ {self.description} completed!")
        print()  # New line


class SpinnerIndicator:
    """Spinner for indeterminate progress."""
    
    def __init__(self, message: str = "Processing..."):
        self.message = message
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.is_spinning = False
        self.thread = None
        self.current_char = 0
    
    def start(self):
        """Start the spinner."""
        self.is_spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self, success_message: Optional[str] = None):
        """Stop the spinner."""
        self.is_spinning = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if success_message:
            print(f"\r✅ {success_message}")
        else:
            print(f"\r✅ {self.message} completed!")
        print()
    
    def _spin(self):
        """Internal spinning method."""
        while self.is_spinning:
            char = self.spinner_chars[self.current_char % len(self.spinner_chars)]
            print(f"\r{char} {self.message}", end="", flush=True)
            self.current_char += 1
            time.sleep(0.1)


@contextmanager
def progress_context(total: Optional[int] = None, description: str = "Processing"):
    """Context manager for progress indication."""
    if total:
        progress = ProgressIndicator(total, description)
        try:
            yield progress
        finally:
            progress.finish()
    else:
        spinner = SpinnerIndicator(description)
        spinner.start()
        try:
            yield spinner
        finally:
            spinner.stop()


@contextmanager
def loading_state(message: str = "Loading..."):
    """Simple loading state context manager."""
    spinner = SpinnerIndicator(message)
    spinner.start()
    try:
        yield
    finally:
        spinner.stop()


class UserFeedback:
    """Comprehensive user feedback system."""
    
    @staticmethod
    def info(message: str, emoji: str = "ℹ️"):
        """Display informational message."""
        print(f"{emoji} {message}")
    
    @staticmethod
    def success(message: str, emoji: str = "✅"):
        """Display success message."""
        print(f"{emoji} {message}")
    
    @staticmethod
    def warning(message: str, emoji: str = "⚠️"):
        """Display warning message."""
        print(f"{emoji} {message}")
    
    @staticmethod
    def error(message: str, emoji: str = "❌"):
        """Display error message."""
        print(f"{emoji} {message}")
    
    @staticmethod
    def step(step_num: int, total_steps: int, description: str):
        """Display step progress."""
        print(f"🔹 Step {step_num}/{total_steps}: {description}")
    
    @staticmethod
    def section_header(title: str, width: int = 50):
        """Display section header."""
        print(f"\n{'=' * width}")
        print(f"🎯 {title.upper()}")
        print(f"{'=' * width}")
    
    @staticmethod
    def subsection(title: str, width: int = 35):
        """Display subsection header."""
        print(f"\n{'-' * width}")
        print(f"📋 {title}")
        print(f"{'-' * width}")
    
    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """Get user confirmation for an action."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"❓ {message} ({default_str}): ").strip().lower()
        
        if not response:
            return default
        
        return response in ['y', 'yes', 'true', '1']
    
    @staticmethod
    def show_summary(title: str, items: dict, width: int = 40):
        """Display a formatted summary."""
        print(f"\n📊 {title}")
        print("─" * width)
        for key, value in items.items():
            key_str = f"{key}:"
            print(f"{key_str:<20} {value}")
        print("─" * width)
    
    @staticmethod
    def show_tips(tips: list):
        """Display helpful tips."""
        print("\n💡 Helpful Tips:")
        for i, tip in enumerate(tips, 1):
            print(f"   {i}. {tip}")
        print()
    
    @staticmethod
    def show_warning_box(message: str, width: int = 60):
        """Display a warning in a box."""
        lines = message.split('\n')
        print(f"\n┌{'─' * (width - 2)}┐")
        print(f"│{'⚠️  WARNING':^{width - 2}}│")
        print(f"├{'─' * (width - 2)}┤")
        for line in lines:
            # Word wrap if needed
            while len(line) > width - 4:
                split_point = line.rfind(' ', 0, width - 4)
                if split_point == -1:
                    split_point = width - 4
                print(f"│ {line[:split_point]:<{width - 3}}│")
                line = line[split_point:].lstrip()
            print(f"│ {line:<{width - 3}}│")
        print(f"└{'─' * (width - 2)}┘\n")


class ValidationFeedback:
    """Specialized feedback for validation errors."""
    
    @staticmethod
    def validation_error(field_name: str, error_message: str, suggestion: Optional[str] = None):
        """Display validation error with helpful suggestions."""
        UserFeedback.error(f"Invalid {field_name}: {error_message}")
        if suggestion:
            UserFeedback.info(f"Suggestion: {suggestion}")
    
    @staticmethod
    def range_error(field_name: str, value: Any, min_val: Any, max_val: Any):
        """Display range validation error."""
        UserFeedback.error(f"{field_name} value '{value}' is out of range [{min_val}, {max_val}]")
        UserFeedback.info(f"Please enter a value between {min_val} and {max_val}")
    
    @staticmethod
    def format_error(field_name: str, expected_format: str, example: str):
        """Display format validation error."""
        UserFeedback.error(f"{field_name} format is invalid")
        UserFeedback.info(f"Expected format: {expected_format}")
        UserFeedback.info(f"Example: {example}")


class PerformanceFeedback:
    """Feedback for performance-related information."""
    
    @staticmethod
    def time_estimate(operation: str, estimated_seconds: float):
        """Display time estimate for an operation."""
        if estimated_seconds < 60:
            time_str = f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            time_str = f"{estimated_seconds / 60:.1f} minutes"
        else:
            time_str = f"{estimated_seconds / 3600:.1f} hours"
        
        UserFeedback.info(f"Estimated time for {operation}: {time_str}")
    
    @staticmethod
    def memory_warning(operation: str, estimated_mb: float, available_mb: float):
        """Display memory usage warning."""
        if estimated_mb > available_mb * 0.8:
            UserFeedback.warning(
                f"{operation} may use {estimated_mb:.0f} MB memory "
                f"(Available: {available_mb:.0f} MB)"
            )
            UserFeedback.info("Consider reducing batch size or using CPU mode")
    
    @staticmethod
    def optimization_suggestion(current_config: dict, suggested_changes: dict):
        """Display optimization suggestions."""
        UserFeedback.info("Performance optimization suggestions:")
        for key, (current, suggested, reason) in suggested_changes.items():
            print(f"  • {key}: {current} → {suggested} ({reason})")


def display_startup_banner():
    """Display enhanced startup banner with system info."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                    MARKSIX PREDICTION SYSTEM v3.0                   ║
║                Enhanced with Comprehensive Error Handling            ║
╠══════════════════════════════════════════════════════════════════════╣
║  🧠 CVAE + Graph Neural Networks + Meta-Learning                    ║
║  🛡️  Enhanced Error Handling & Input Validation                    ║
║  📊 Advanced Progress Monitoring & User Feedback                    ║
║  🔧 Comprehensive Testing & Debugging Tools                         ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def display_feature_highlights():
    """Display key features and improvements."""
    features = [
        "🔒 Robust input validation with helpful error messages",
        "🛠️  Automatic error recovery and fallback mechanisms", 
        "📈 Real-time progress indicators for long operations",
        "💡 Intelligent suggestions and performance tips",
        "🔍 Comprehensive system diagnostics and health checks",
        "⚡ Optimized performance with memory management"
    ]
    
    UserFeedback.info("Key Features & Improvements:")
    for feature in features:
        print(f"  {feature}")
    print()


class OperationTimer:
    """Timer for measuring and reporting operation duration."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            UserFeedback.success(f"{self.operation_name} completed in {duration:.2f} seconds")
        else:
            UserFeedback.error(f"{self.operation_name} failed after {duration:.2f} seconds")
    
    def get_duration(self) -> Optional[float]:
        """Get operation duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def create_progress_callback(total_steps: int, description: str = "Processing"):
    """Create a progress callback function for use in other modules."""
    progress = ProgressIndicator(total_steps, description)
    
    def callback(step: int = 1, new_description: Optional[str] = None):
        progress.update(step, new_description)
    
    callback.finish = progress.finish
    return callback


def enhanced_input_prompt(
    prompt: str, 
    input_type: type = str,
    validator: Optional[Callable] = None,
    default: Any = None,
    help_text: Optional[str] = None
) -> Any:
    """Enhanced input prompt with validation and help."""
    
    if help_text:
        UserFeedback.info(help_text)
    
    if default is not None:
        prompt += f" (default: {default})"
    
    prompt += ": "
    
    while True:
        try:
            user_input = input(prompt).strip()
            
            # Handle default
            if not user_input and default is not None:
                return default
            
            # Type conversion
            if input_type != str:
                user_input = input_type(user_input)
            
            # Validation
            if validator:
                user_input = validator(user_input)
            
            return user_input
            
        except KeyboardInterrupt:
            UserFeedback.info("Operation cancelled by user")
            raise
        except Exception as e:
            ValidationFeedback.validation_error("input", str(e))
            if help_text:
                UserFeedback.info(f"Help: {help_text}")


# Convenience functions for common patterns
def show_loading(message: str = "Loading..."):
    """Decorator to show loading spinner for functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with loading_state(message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def timed_operation(operation_name: str):
    """Decorator to time and report operation duration."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with OperationTimer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator