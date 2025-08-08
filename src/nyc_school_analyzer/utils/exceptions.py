"""
Custom exceptions for NYC School Analyzer.

Provides specific exception types for different error categories
to enable proper error handling and user feedback.
"""


class NYCSchoolAnalyzerError(Exception):
    """Base exception for all NYC School Analyzer errors."""
    
    def __init__(self, message: str, error_code: str = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class DataProcessingError(NYCSchoolAnalyzerError):
    """Exception raised during data processing operations."""
    
    def __init__(self, message: str, source_error: Exception = None):
        """
        Initialize data processing exception.
        
        Args:
            message: Error message
            source_error: Original exception that caused this error
        """
        super().__init__(message, "DATA_PROCESSING")
        self.source_error = source_error


class AnalysisError(NYCSchoolAnalyzerError):
    """Exception raised during analysis operations."""
    
    def __init__(self, message: str, analysis_type: str = None):
        """
        Initialize analysis exception.
        
        Args:
            message: Error message
            analysis_type: Type of analysis that failed
        """
        super().__init__(message, "ANALYSIS")
        self.analysis_type = analysis_type


class VisualizationError(NYCSchoolAnalyzerError):
    """Exception raised during visualization operations."""
    
    def __init__(self, message: str, chart_type: str = None):
        """
        Initialize visualization exception.
        
        Args:
            message: Error message
            chart_type: Type of chart that failed to generate
        """
        super().__init__(message, "VISUALIZATION")
        self.chart_type = chart_type


class ValidationError(NYCSchoolAnalyzerError):
    """Exception raised during data validation."""
    
    def __init__(self, message: str, validation_rule: str = None):
        """
        Initialize validation exception.
        
        Args:
            message: Error message
            validation_rule: Validation rule that failed
        """
        super().__init__(message, "VALIDATION")
        self.validation_rule = validation_rule


class ConfigurationError(NYCSchoolAnalyzerError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = None):
        """
        Initialize configuration exception.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
        """
        super().__init__(message, "CONFIGURATION")
        self.config_key = config_key


class FileNotFoundError(DataProcessingError):
    """Exception raised when required files are not found."""
    
    def __init__(self, filepath: str):
        """
        Initialize file not found exception.
        
        Args:
            filepath: Path to file that was not found
        """
        message = f"Required file not found: {filepath}"
        super().__init__(message)
        self.filepath = filepath


class InvalidDataFormatError(DataProcessingError):
    """Exception raised when data format is invalid or unexpected."""
    
    def __init__(self, message: str, expected_format: str = None, actual_format: str = None):
        """
        Initialize invalid data format exception.
        
        Args:
            message: Error message
            expected_format: Expected data format
            actual_format: Actual data format encountered
        """
        super().__init__(message)
        self.expected_format = expected_format
        self.actual_format = actual_format


class InsufficientDataError(AnalysisError):
    """Exception raised when insufficient data is available for analysis."""
    
    def __init__(self, message: str, required_rows: int = None, available_rows: int = None):
        """
        Initialize insufficient data exception.
        
        Args:
            message: Error message
            required_rows: Minimum number of rows required
            available_rows: Number of rows available
        """
        super().__init__(message)
        self.required_rows = required_rows
        self.available_rows = available_rows


class ExportError(NYCSchoolAnalyzerError):
    """Exception raised during export operations."""
    
    def __init__(self, message: str, export_format: str = None, filepath: str = None):
        """
        Initialize export exception.
        
        Args:
            message: Error message
            export_format: Format that failed to export
            filepath: Path where export was attempted
        """
        super().__init__(message, "EXPORT")
        self.export_format = export_format
        self.filepath = filepath