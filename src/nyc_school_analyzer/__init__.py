"""
NYC School Analyzer - Production-ready school data analysis tool.

This package provides comprehensive analysis capabilities for NYC high school data,
including data processing, statistical analysis, and visualization components.
"""

__version__ = "1.0.0"
__author__ = "NYC School Analytics Team"
__email__ = "analytics@nycschools.edu"

from .data.processor import DataProcessor
from .analysis.analyzer import SchoolAnalyzer
from .visualization.visualizer import SchoolVisualizer
from .utils.config import Config
from .utils.logger import get_logger

__all__ = [
    "DataProcessor",
    "SchoolAnalyzer", 
    "SchoolVisualizer",
    "Config",
    "get_logger",
]