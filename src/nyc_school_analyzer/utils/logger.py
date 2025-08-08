"""
Logging configuration for NYC School Analyzer.

Provides centralized logging setup with consistent formatting,
multiple output destinations, and production-ready configuration.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        if record.levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
            record.levelname = colored_levelname
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_string: Optional[str] = None,
    enable_colors: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> Dict[str, Any]:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file name
        log_dir: Directory for log files
        format_string: Custom format string
        enable_colors: Whether to enable colored output for console
        max_file_size_mb: Maximum size for log files before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Dictionary with logging configuration details
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format string
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatters
    file_formatter = logging.Formatter(format_string)
    
    if enable_colors:
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    config_info = {
        'level': level,
        'handlers': ['console'],
        'log_dir': None,
        'log_file': None,
    }
    
    # File handler (if requested)
    if log_file or log_dir:
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            if log_file:
                log_path = log_dir / log_file
            else:
                timestamp = datetime.now().strftime("%Y%m%d")
                log_path = log_dir / f"nyc_school_analyzer_{timestamp}.log"
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        config_info.update({
            'handlers': ['console', 'file'],
            'log_dir': str(log_dir) if log_dir else str(log_path.parent),
            'log_file': str(log_path),
        })
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, Handlers: {config_info['handlers']}")
    
    return config_info


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def configure_library_loggers(level: str = "WARNING"):
    """
    Configure third-party library loggers to reduce noise.
    
    Args:
        level: Log level for library loggers
    """
    library_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'pandas',
        'numpy',
    ]
    
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    
    for lib_name in library_loggers:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(numeric_level)


def create_analysis_logger(
    analysis_name: str,
    output_dir: Optional[Path] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Create a dedicated logger for a specific analysis.
    
    Args:
        analysis_name: Name of the analysis
        output_dir: Directory to save analysis logs
        level: Logging level
        
    Returns:
        Configured logger for the analysis
    """
    logger_name = f"nyc_school_analyzer.analysis.{analysis_name}"
    logger = logging.getLogger(logger_name)
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create file handler if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"{analysis_name}_analysis_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Analysis logger created: {log_file}")
    
    return logger


class LoggerContext:
    """Context manager for temporary logger configuration."""
    
    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize logger context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), logging.INFO)
        self.original_level = None
    
    def __enter__(self):
        """Enter context - set new log level."""
        self.original_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original log level."""
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)


def silence_warnings():
    """Temporarily silence warning-level messages."""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Also configure pandas to reduce warnings
    try:
        import pandas as pd
        pd.options.mode.chained_assignment = None
    except ImportError:
        pass


def log_performance(func):
    """Decorator to log function performance."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        logger.debug(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed_time:.2f}s")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed_time:.2f}s: {e}")
            raise
    
    return wrapper


def log_memory_usage():
    """Log current memory usage if psutil is available."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        logger = get_logger(__name__)
        logger.debug(f"Memory usage: {memory_mb:.1f} MB")
        
        return memory_mb
    except ImportError:
        return None