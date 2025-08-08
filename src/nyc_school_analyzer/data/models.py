"""
Data models and schemas for NYC School Analyzer.

This module defines the data structures used throughout the application
for type safety and data validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd


@dataclass
class SchoolRecord:
    """Individual school record with standardized fields."""
    school_name: str
    borough: str
    city: str
    address: str
    phone: Optional[str] = None
    website: Optional[str] = None
    grade_span_min: Optional[int] = None
    grade_span_max: Optional[int] = None
    total_students: Optional[int] = None
    school_type: Optional[str] = None
    overview_paragraph: Optional[str] = None
    
    def __post_init__(self):
        """Validate and clean data after initialization."""
        if self.borough:
            self.borough = self.borough.upper().strip()
        if self.city:
            self.city = self.city.upper().strip()


@dataclass
class SchoolData:
    """Container for processed school dataset."""
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata."""
        self.metadata.update({
            'total_records': len(self.processed_data),
            'columns': list(self.processed_data.columns),
            'memory_usage_mb': self.processed_data.memory_usage(deep=True).sum() / 1024**2,
        })
    
    @property
    def shape(self) -> tuple:
        """Get dataset shape."""
        return self.processed_data.shape
    
    @property
    def boroughs(self) -> List[str]:
        """Get unique boroughs in dataset."""
        return sorted(self.processed_data['city'].unique().tolist())


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    target_borough: str = "BROOKLYN"
    grade_of_interest: int = 9
    include_expanded_grades: bool = True
    validation_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self.target_borough = self.target_borough.upper().strip()
        if not 0 <= self.grade_of_interest <= 12:
            raise ValueError("Grade of interest must be between 0 and 12")


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    analysis_type: str
    target_borough: str
    grade_analyzed: int
    total_schools: int
    schools_meeting_criteria: int
    percentage: float
    statistics: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    
    @property
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"{self.analysis_type} Analysis: {self.schools_meeting_criteria}/{self.total_schools} "
            f"schools ({self.percentage:.1f}%) in {self.target_borough}"
        )


@dataclass
class ValidationResult:
    """Results from data validation checks."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    @property
    def has_issues(self) -> bool:
        """Check if there are any validation issues."""
        return bool(self.errors or self.warnings)


@dataclass
class ExportConfig:
    """Configuration for data export operations."""
    output_dir: Path
    formats: List[str] = field(default_factory=lambda: ['csv', 'json'])
    include_timestamp: bool = True
    create_summary: bool = True
    compress: bool = False
    
    def __post_init__(self):
        """Validate export configuration."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        valid_formats = {'csv', 'json', 'excel', 'parquet', 'html'}
        invalid_formats = set(self.formats) - valid_formats
        if invalid_formats:
            raise ValueError(f"Invalid export formats: {invalid_formats}")


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    style: str = "seaborn"
    figure_size: tuple = (12, 8)
    dpi: int = 300
    color_palette: str = "husl"
    font_size: int = 11
    export_formats: List[str] = field(default_factory=lambda: ['png', 'pdf'])
    
    def __post_init__(self):
        """Validate visualization configuration."""
        valid_styles = {'seaborn', 'ggplot', 'bmh', 'classic', 'dark_background'}
        if self.style not in valid_styles:
            raise ValueError(f"Invalid style: {self.style}. Must be one of {valid_styles}")
        
        if self.dpi < 72 or self.dpi > 600:
            raise ValueError("DPI must be between 72 and 600")


# Type aliases for better readability
DataFrameType = pd.DataFrame
SeriesType = pd.Series
NumericType = Union[int, float]
PathType = Union[str, Path]