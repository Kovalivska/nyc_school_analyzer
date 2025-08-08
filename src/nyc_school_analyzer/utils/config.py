"""
Configuration management for NYC School Analyzer.

Provides centralized configuration loading and validation
with support for YAML files, environment variables, and defaults.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

from .exceptions import ConfigurationError
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """Data processing configuration."""
    input_file: str = "high-school-directory.csv"
    input_path: str = "data/"
    output_path: str = "outputs/"
    backup_path: str = "data/backup/"


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    target_borough: str = "BROOKLYN"
    grade_of_interest: int = 9
    include_expanded_grades: bool = True
    validation_enabled: bool = True


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    style: str = "seaborn"
    figure_size: tuple = (12, 8)
    dpi: int = 300
    color_palette: str = "husl"
    font_size: int = 11
    export_formats: list = field(default_factory=lambda: ["png", "pdf"])


@dataclass
class OutputConfig:
    """Output configuration."""
    export_csv: bool = True
    export_json: bool = True
    export_excel: bool = True
    include_timestamps: bool = True
    create_summary_report: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    chunk_size: int = 10000
    memory_limit_mb: int = 1024
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class QualityConfig:
    """Data quality configuration."""
    min_data_completeness: float = 0.7
    max_missing_percentage: float = 30.0
    validate_data_types: bool = True
    check_outliers: bool = True
    outlier_threshold: float = 3.0


class Config:
    """
    Main configuration class for NYC School Analyzer.
    
    Loads configuration from YAML files, environment variables,
    and provides validated access to all settings.
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        self.raw_config = {}
        
        # Initialize configuration sections
        self.data = DataConfig()
        self.analysis = AnalysisConfig()
        self.visualization = VisualizationConfig()
        self.output = OutputConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        self.quality = QualityConfig()
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        try:
            # Load from file if specified
            if self.config_file:
                self._load_from_file(self.config_file)
            else:
                # Try to find default config files
                default_locations = [
                    Path("config/config.yaml"),
                    Path("config.yaml"),
                    Path.home() / ".nyc_school_analyzer" / "config.yaml"
                ]
                
                for config_path in default_locations:
                    if config_path.exists():
                        self._load_from_file(config_path)
                        break
            
            # Override with environment variables
            self._load_from_environment()
            
            # Update dataclass instances
            self._update_config_objects()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _load_from_file(self, config_file: Union[str, Path]):
        """Load configuration from YAML file."""
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self.raw_config = file_config
                logger.debug(f"Loaded configuration from {config_file}")
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")
    
    def _load_from_environment(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'NYC_SCHOOLS_DATA_PATH': ('data', 'input_path'),
            'NYC_SCHOOLS_OUTPUT_PATH': ('output', 'output_path'),
            'NYC_SCHOOLS_TARGET_BOROUGH': ('analysis', 'target_borough'),
            'NYC_SCHOOLS_GRADE': ('analysis', 'grade_of_interest'),
            'NYC_SCHOOLS_LOG_LEVEL': ('logging', 'level'),
            'NYC_SCHOOLS_DPI': ('visualization', 'dpi'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self.raw_config:
                    self.raw_config[section] = {}
                
                # Type conversion
                if key in ['grade_of_interest', 'dpi', 'max_workers']:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue
                elif key in ['min_data_completeness', 'max_missing_percentage', 'outlier_threshold']:
                    try:
                        value = float(value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_var}: {value}")
                        continue
                elif key in ['validation_enabled', 'export_csv', 'parallel_processing']:
                    value = str(value).lower() in ['true', '1', 'yes', 'on']
                
                self.raw_config[section][key] = value
                logger.debug(f"Environment override: {env_var} = {value}")
    
    def _update_config_objects(self):
        """Update dataclass instances with loaded configuration."""
        config_sections = {
            'data': self.data,
            'analysis': self.analysis,
            'visualization': self.visualization,
            'output': self.output,
            'logging': self.logging,
            'performance': self.performance,
            'quality': self.quality,
        }
        
        for section_name, config_obj in config_sections.items():
            if section_name in self.raw_config:
                section_config = self.raw_config[section_name]
                
                for key, value in section_config.items():
                    if hasattr(config_obj, key):
                        # Special handling for tuples (like figure_size)
                        if key == 'figure_size' and isinstance(value, list):
                            value = tuple(value)
                        
                        setattr(config_obj, key, value)
                        logger.debug(f"Set {section_name}.{key} = {value}")
                    else:
                        logger.warning(f"Unknown configuration key: {section_name}.{key}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        try:
            section_obj = getattr(self, section, None)
            if section_obj is not None:
                return getattr(section_obj, key, default)
            
            # Fallback to raw config
            return self.raw_config.get(section, {}).get(key, default)
            
        except Exception:
            return default
    
    def set(self, section: str, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        try:
            section_obj = getattr(self, section, None)
            if section_obj is not None and hasattr(section_obj, key):
                setattr(section_obj, key, value)
                logger.debug(f"Updated configuration: {section}.{key} = {value}")
            else:
                logger.warning(f"Cannot set unknown configuration: {section}.{key}")
                
        except Exception as e:
            logger.error(f"Error setting configuration {section}.{key}: {e}")
    
    def validate(self) -> Dict[str, list]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary with validation errors by section
        """
        errors = {}
        
        # Validate data configuration
        data_errors = []
        if not self.data.input_file:
            data_errors.append("input_file cannot be empty")
        
        if data_errors:
            errors['data'] = data_errors
        
        # Validate analysis configuration
        analysis_errors = []
        if not 0 <= self.analysis.grade_of_interest <= 12:
            analysis_errors.append("grade_of_interest must be between 0 and 12")
        
        if not self.analysis.target_borough:
            analysis_errors.append("target_borough cannot be empty")
        
        if analysis_errors:
            errors['analysis'] = analysis_errors
        
        # Validate visualization configuration
        viz_errors = []
        if self.visualization.dpi < 72 or self.visualization.dpi > 600:
            viz_errors.append("dpi must be between 72 and 600")
        
        if self.visualization.font_size < 6 or self.visualization.font_size > 24:
            viz_errors.append("font_size must be between 6 and 24")
        
        if viz_errors:
            errors['visualization'] = viz_errors
        
        # Validate quality configuration
        quality_errors = []
        if not 0 <= self.quality.min_data_completeness <= 1:
            quality_errors.append("min_data_completeness must be between 0 and 1")
        
        if not 0 <= self.quality.max_missing_percentage <= 100:
            quality_errors.append("max_missing_percentage must be between 0 and 100")
        
        if quality_errors:
            errors['quality'] = quality_errors
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'data': self.data.__dict__,
            'analysis': self.analysis.__dict__,
            'visualization': self.visualization.__dict__,
            'output': self.output.__dict__,
            'logging': self.logging.__dict__,
            'performance': self.performance.__dict__,
            'quality': self.quality.__dict__,
        }
    
    def save_to_file(self, output_file: Union[str, Path]):
        """
        Save current configuration to YAML file.
        
        Args:
            output_file: Path to save configuration
        """
        try:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = self.to_dict()
            
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def get_output_paths(self) -> Dict[str, Path]:
        """
        Get all configured output paths as Path objects.
        
        Returns:
            Dictionary of output paths
        """
        base_output = Path(self.data.output_path)
        
        return {
            'base': base_output,
            'data': base_output / 'data',
            'charts': base_output / 'charts',
            'reports': base_output / 'reports',
            'logs': Path(self.logging.file_path) if self.logging.file_enabled else None,
            'backup': Path(self.data.backup_path),
        }
    
    def create_output_directories(self):
        """Create all configured output directories."""
        try:
            paths = self.get_output_paths()
            
            for path_name, path in paths.items():
                if path is not None:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created output directory: {path}")
            
            logger.info("Output directories created successfully")
            
        except Exception as e:
            error_msg = f"Failed to create output directories: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e