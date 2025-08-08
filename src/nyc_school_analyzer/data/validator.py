"""
Data validation module for NYC School Analyzer.

Provides comprehensive validation capabilities for school datasets
including data quality checks, schema validation, and business rule validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from .models import ValidationResult
from ..utils.exceptions import ValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Comprehensive data validator for school datasets.
    
    Performs multiple levels of validation including:
    - Schema validation
    - Data quality checks
    - Business rule validation
    - Statistical outlier detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with configuration.
        
        Args:
            config: Validation configuration parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'min_data_completeness': 0.7,
            'max_missing_percentage': 30.0,
            'outlier_threshold': 3.0,
            'min_schools_per_borough': 1,
            'valid_grade_range': (0, 12),
            'max_students_per_school': 10000,
            'required_columns': ['school_name', 'city'],
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform comprehensive dataset validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with validation outcomes
        """
        logger.info("Starting comprehensive dataset validation...")
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic structure validation
            self._validate_structure(df, result)
            
            # Data quality validation
            self._validate_data_quality(df, result)
            
            # Business rule validation
            self._validate_business_rules(df, result)
            
            # Statistical validation
            self._validate_statistics(df, result)
            
            # Summary metrics
            result.metrics.update(self._calculate_metrics(df))
            
            logger.info(
                f"Validation completed. Valid: {result.is_valid}, "
                f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Validation process failed: {str(e)}"
            logger.error(error_msg)
            result.add_error(error_msg)
            return result
    
    def _validate_structure(self, df: pd.DataFrame, result: ValidationResult):
        """Validate basic dataset structure."""
        # Check if dataset is empty
        if df.empty:
            result.add_error("Dataset is empty")
            return
        
        # Check for required columns
        missing_required = set(self.config['required_columns']) - set(df.columns)
        if missing_required:
            result.add_error(f"Missing required columns: {missing_required}")
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = df.columns[df.columns.duplicated()].tolist()
            result.add_error(f"Duplicate column names found: {duplicates}")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            result.add_warning(f"Completely empty columns: {empty_cols}")
        
        logger.debug("Structure validation completed")
    
    def _validate_data_quality(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data quality metrics."""
        # Calculate missing data percentage
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        
        if missing_percentage > self.config['max_missing_percentage']:
            result.add_error(
                f"Missing data percentage ({missing_percentage:.1f}%) exceeds "
                f"threshold ({self.config['max_missing_percentage']}%)"
            )
        elif missing_percentage > 15.0:
            result.add_warning(
                f"High missing data percentage: {missing_percentage:.1f}%"
            )
        
        # Check data completeness by column
        high_missing_cols = []
        for col in df.columns:
            col_missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if col_missing_pct > 50:
                high_missing_cols.append(f"{col} ({col_missing_pct:.1f}%)")
        
        if high_missing_cols:
            result.add_warning(f"Columns with >50% missing data: {high_missing_cols}")
        
        # Check for duplicate records
        if 'school_name' in df.columns:
            duplicates = df['school_name'].duplicated().sum()
            if duplicates > 0:
                result.add_warning(f"Found {duplicates} duplicate school names")
        
        logger.debug("Data quality validation completed")
    
    def _validate_business_rules(self, df: pd.DataFrame, result: ValidationResult):
        """Validate business-specific rules."""
        # Validate borough/city data
        if 'city' in df.columns:
            # Check for reasonable number of boroughs
            unique_boroughs = df['city'].nunique()
            if unique_boroughs > 50:
                result.add_warning(
                    f"Unusually high number of unique locations: {unique_boroughs}"
                )
            
            # Check borough distribution
            borough_counts = df['city'].value_counts()
            small_boroughs = borough_counts[
                borough_counts < self.config['min_schools_per_borough']
            ]
            if len(small_boroughs) > 10:
                result.add_warning(
                    f"Many locations with very few schools: {len(small_boroughs)} locations"
                )
        
        # Validate grade data
        grade_columns = [col for col in df.columns if 'grade' in col.lower()]
        for col in grade_columns:
            if col in df.columns:
                self._validate_grade_column(df, col, result)
        
        # Validate student counts
        if 'total_students' in df.columns:
            self._validate_student_counts(df, result)
        
        logger.debug("Business rules validation completed")
    
    def _validate_grade_column(self, df: pd.DataFrame, col: str, result: ValidationResult):
        """Validate grade column data."""
        # Convert to numeric if possible
        numeric_grades = pd.to_numeric(df[col], errors='coerce')
        
        # Check grade range
        valid_range = self.config['valid_grade_range']
        invalid_grades = numeric_grades[
            (numeric_grades < valid_range[0]) | (numeric_grades > valid_range[1])
        ].dropna()
        
        if len(invalid_grades) > 0:
            result.add_warning(
                f"Column '{col}' has {len(invalid_grades)} values outside "
                f"valid grade range {valid_range}"
            )
        
        # Check for logical consistency (min <= max)
        if 'min' in col.lower() and col.replace('min', 'max') in df.columns:
            max_col = col.replace('min', 'max')
            min_vals = pd.to_numeric(df[col], errors='coerce')
            max_vals = pd.to_numeric(df[max_col], errors='coerce')
            
            inconsistent = (min_vals > max_vals).sum()
            if inconsistent > 0:
                result.add_error(
                    f"Found {inconsistent} records where {col} > {max_col}"
                )
    
    def _validate_student_counts(self, df: pd.DataFrame, result: ValidationResult):
        """Validate student count data."""
        students = pd.to_numeric(df['total_students'], errors='coerce')
        
        # Check for negative values
        negative_count = (students < 0).sum()
        if negative_count > 0:
            result.add_error(f"Found {negative_count} schools with negative student counts")
        
        # Check for extremely high values
        max_threshold = self.config['max_students_per_school']
        high_count = (students > max_threshold).sum()
        if high_count > 0:
            result.add_warning(
                f"Found {high_count} schools with >{'max students_per_school'} students"
            )
        
        # Check for zeros (might be valid but worth noting)
        zero_count = (students == 0).sum()
        if zero_count > 0:
            result.add_warning(f"Found {zero_count} schools with zero students")
    
    def _validate_statistics(self, df: pd.DataFrame, result: ValidationResult):
        """Validate statistical properties of the data."""
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['total_students', 'graduation_rate', 'attendance_rate']:
                outliers = self._detect_outliers(df[col], self.config['outlier_threshold'])
                if len(outliers) > 0:
                    outlier_pct = (len(outliers) / len(df)) * 100
                    if outlier_pct > 5:
                        result.add_warning(
                            f"Column '{col}' has {len(outliers)} outliers "
                            f"({outlier_pct:.1f}% of data)"
                        )
        
        logger.debug("Statistical validation completed")
    
    def _detect_outliers(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using z-score method.
        
        Args:
            series: Data series to check
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Series of outlier values
        """
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return pd.Series(dtype=series.dtype)
        
        z_scores = np.abs((clean_series - clean_series.mean()) / clean_series.std())
        return clean_series[z_scores > threshold]
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate validation metrics."""
        return {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'data_completeness': ((df.size - df.isnull().sum().sum()) / df.size),
        }
    
    def validate_column_consistency(
        self, 
        df: pd.DataFrame, 
        column_groups: Dict[str, List[str]]
    ) -> ValidationResult:
        """
        Validate consistency within column groups.
        
        Args:
            df: DataFrame to validate
            column_groups: Dictionary mapping group names to column lists
            
        Returns:
            ValidationResult for consistency checks
        """
        result = ValidationResult(is_valid=True)
        
        for group_name, columns in column_groups.items():
            available_cols = [col for col in columns if col in df.columns]
            
            if len(available_cols) < 2:
                continue
            
            # Check for logical consistency within group
            if group_name == 'grade_span':
                self._validate_grade_span_consistency(df, available_cols, result)
            elif group_name == 'contact_info':
                self._validate_contact_consistency(df, available_cols, result)
        
        return result
    
    def _validate_grade_span_consistency(
        self, 
        df: pd.DataFrame, 
        grade_cols: List[str], 
        result: ValidationResult
    ):
        """Validate grade span logical consistency."""
        min_cols = [col for col in grade_cols if 'min' in col.lower()]
        max_cols = [col for col in grade_cols if 'max' in col.lower()]
        
        for min_col in min_cols:
            for max_col in max_cols:
                if min_col.replace('min', 'max') == max_col:
                    # Check if min <= max
                    min_vals = pd.to_numeric(df[min_col], errors='coerce')
                    max_vals = pd.to_numeric(df[max_col], errors='coerce')
                    
                    violations = ((min_vals > max_vals) & 
                                min_vals.notna() & 
                                max_vals.notna()).sum()
                    
                    if violations > 0:
                        result.add_error(
                            f"Grade span inconsistency: {violations} records "
                            f"where {min_col} > {max_col}"
                        )
    
    def _validate_contact_consistency(
        self, 
        df: pd.DataFrame, 
        contact_cols: List[str], 
        result: ValidationResult
    ):
        """Validate contact information consistency."""
        # Check for schools with no contact information
        contact_data = df[contact_cols]
        no_contact = contact_data.isnull().all(axis=1).sum()
        
        if no_contact > 0:
            result.add_warning(
                f"Found {no_contact} schools with no contact information"
            )
        
        # Validate phone number formats if present
        if 'phone' in contact_cols:
            self._validate_phone_formats(df['phone'], result)
    
    def _validate_phone_formats(self, phone_series: pd.Series, result: ValidationResult):
        """Validate phone number formats."""
        import re
        
        phone_pattern = r'^\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        valid_phones = phone_series.dropna().str.match(phone_pattern, na=False)
        
        invalid_count = (~valid_phones).sum()
        if invalid_count > 0:
            result.add_warning(
                f"Found {invalid_count} phone numbers with non-standard format"
            )