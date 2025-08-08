"""
Data processing and cleaning module for NYC School Analyzer.

This module provides comprehensive data processing capabilities including
loading, cleaning, validation, and transformation of school data.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from .models import SchoolData, ValidationResult
from .validator import DataValidator
from ..utils.exceptions import DataProcessingError, ValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Production-ready data processor for NYC school data.
    
    Handles loading, cleaning, validation, and transformation of school datasets
    with comprehensive error handling and logging.
    """
    
    def __init__(self, validator: Optional[DataValidator] = None):
        """
        Initialize data processor.
        
        Args:
            validator: Optional data validator instance
        """
        self.validator = validator or DataValidator()
        self._grade_patterns = {
            'kindergarten': ['K', 'KINDERGARTEN', 'PRE-K', 'PK'],
            'numeric': r'(\d+)',
        }
    
    def load_dataset(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load dataset with comprehensive error handling and validation.
        
        Args:
            file_path: Path to the dataset file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataProcessingError: If loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            if not file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loading dataset from: {file_path}")
            
            # Load based on file type
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            else:
                df = pd.read_excel(file_path, **kwargs)
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Basic validation
            if len(df.columns) == 0:
                raise ValueError("Dataset has no columns")
            
            logger.info(f"Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to load dataset from {file_path}: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        try:
            logger.info("Cleaning column names...")
            original_columns = df.columns.tolist()
            
            cleaned_columns = []
            for col in df.columns:
                # Convert to lowercase and handle special cases
                clean_col = str(col).lower().strip()
                
                # Replace spaces and hyphens with underscores
                clean_col = re.sub(r'[\s\-]+', '_', clean_col)
                
                # Remove special characters except underscores and numbers
                clean_col = re.sub(r'[^a-z0-9_]', '', clean_col)
                
                # Remove multiple consecutive underscores
                clean_col = re.sub(r'_+', '_', clean_col)
                
                # Remove leading/trailing underscores
                clean_col = clean_col.strip('_')
                
                # Ensure column name is not empty
                if not clean_col:
                    clean_col = f'unnamed_{len(cleaned_columns)}'
                
                # Handle duplicates
                original_clean_col = clean_col
                counter = 1
                while clean_col in cleaned_columns:
                    clean_col = f"{original_clean_col}_{counter}"
                    counter += 1
                
                cleaned_columns.append(clean_col)
            
            df.columns = cleaned_columns
            logger.info(f"Successfully cleaned {len(cleaned_columns)} column names")
            
            # Log significant changes
            changed_columns = [
                (orig, new) for orig, new in zip(original_columns, cleaned_columns)
                if orig != new
            ]
            
            if changed_columns:
                logger.info(f"Changed {len(changed_columns)} column names")
                for orig, new in changed_columns[:5]:  # Log first 5 changes
                    logger.debug(f"  '{orig}' -> '{new}'")
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to clean column names: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e
    
    def parse_grade_value(self, grade_value: Union[str, int, float]) -> Optional[int]:
        """
        Parse grade values with comprehensive handling of different formats.
        
        Args:
            grade_value: Grade value to parse
            
        Returns:
            Parsed grade as integer or None if unparseable
        """
        if pd.isna(grade_value):
            return None
        
        try:
            # Handle direct numeric values
            if isinstance(grade_value, (int, float)):
                if np.isnan(grade_value):
                    return None
                return int(grade_value)
            
            # Handle string values
            grade_str = str(grade_value).strip().upper()
            
            # Handle kindergarten cases
            if any(k in grade_str for k in self._grade_patterns['kindergarten']):
                return 0
            
            # Extract numeric grade
            match = re.search(self._grade_patterns['numeric'], grade_str)
            if match:
                grade = int(match.group(1))
                # Validate reasonable grade range
                if 0 <= grade <= 12:
                    return grade
            
            return None
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse grade value '{grade_value}': {e}")
            return None
    
    def filter_by_borough(
        self, 
        df: pd.DataFrame, 
        borough: str, 
        city_column: str = 'city'
    ) -> pd.DataFrame:
        """
        Filter DataFrame by borough with validation and error handling.
        
        Args:
            df: Input DataFrame
            borough: Target borough name
            city_column: Column name containing city/borough information
            
        Returns:
            Filtered DataFrame
            
        Raises:
            DataProcessingError: If filtering fails
        """
        try:
            if city_column not in df.columns:
                available_columns = [col for col in df.columns if 'city' in col.lower()]
                raise KeyError(
                    f"Column '{city_column}' not found. Available city columns: {available_columns}"
                )
            
            # Clean and standardize borough name
            borough = borough.upper().strip()
            
            # Filter with case-insensitive comparison
            mask = df[city_column].str.upper().str.strip() == borough
            filtered_df = df[mask].copy()
            
            if filtered_df.empty:
                available_boroughs = sorted(df[city_column].dropna().unique())
                raise ValueError(
                    f"No schools found for borough '{borough}'. "
                    f"Available locations: {available_boroughs}"
                )
            
            logger.info(f"Filtered dataset for {borough}: {len(filtered_df)} schools found")
            return filtered_df
            
        except Exception as e:
            error_msg = f"Failed to filter by borough '{borough}': {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e
    
    def process_grade_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and standardize grade-related columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed grade columns
        """
        try:
            logger.info("Processing grade columns...")
            df_processed = df.copy()
            
            # Standard grade column mappings
            grade_mappings = {
                'grade_span_min': ['grade_span_min', 'min_grade', 'start_grade'],
                'grade_span_max': ['grade_span_max', 'max_grade', 'end_grade'],
                'exp_grade_span_min': ['exp_grade_span_min', 'expanded_min_grade'],
                'exp_grade_span_max': ['exp_grade_span_max', 'expanded_max_grade'],
            }
            
            processed_count = 0
            for target_col, possible_cols in grade_mappings.items():
                for col in possible_cols:
                    if col in df_processed.columns:
                        # Create numeric version
                        numeric_col = f"{target_col}_numeric"
                        df_processed[numeric_col] = df_processed[col].apply(
                            self.parse_grade_value
                        )
                        processed_count += 1
                        logger.debug(f"Processed grade column: {col} -> {numeric_col}")
                        break
            
            logger.info(f"Successfully processed {processed_count} grade columns")
            return df_processed
            
        except Exception as e:
            error_msg = f"Failed to process grade columns: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned numeric columns
        """
        try:
            logger.info("Cleaning numeric columns...")
            df_cleaned = df.copy()
            
            # Identify numeric columns
            numeric_columns = [
                'total_students', 'ell_programs', 'school_sports',
                'graduation_rate', 'attendance_rate', 'college_ready_rate'
            ]
            
            cleaned_count = 0
            for col in numeric_columns:
                if col in df_cleaned.columns:
                    # Convert to numeric, handling errors
                    original_dtype = df_cleaned[col].dtype
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    
                    # Log conversion results
                    na_count = df_cleaned[col].isna().sum()
                    if na_count > 0:
                        logger.debug(
                            f"Column '{col}': converted {original_dtype} to numeric, "
                            f"{na_count} values became NaN"
                        )
                    
                    cleaned_count += 1
            
            logger.info(f"Successfully cleaned {cleaned_count} numeric columns")
            return df_cleaned
            
        except Exception as e:
            error_msg = f"Failed to clean numeric columns: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e
    
    def process_dataset(
        self, 
        file_path: Union[str, Path],
        validate: bool = True
    ) -> SchoolData:
        """
        Complete data processing pipeline.
        
        Args:
            file_path: Path to input file
            validate: Whether to perform validation
            
        Returns:
            SchoolData object with processed data
        """
        try:
            logger.info("Starting complete data processing pipeline...")
            
            # Load raw data
            raw_df = self.load_dataset(file_path)
            
            # Processing pipeline
            processed_df = raw_df.copy()
            processed_df = self.clean_column_names(processed_df)
            processed_df = self.process_grade_columns(processed_df)
            processed_df = self.clean_numeric_columns(processed_df)
            
            # Validation
            if validate:
                validation_result = self.validator.validate_dataset(processed_df)
                if not validation_result.is_valid:
                    logger.warning(f"Validation issues found: {len(validation_result.errors)} errors")
                    for error in validation_result.errors[:3]:  # Log first 3 errors
                        logger.warning(f"Validation error: {error}")
            
            # Create SchoolData object
            school_data = SchoolData(
                raw_data=raw_df,
                processed_data=processed_df,
                metadata={
                    'source_file': str(file_path),
                    'processing_timestamp': pd.Timestamp.now(),
                    'validation_performed': validate,
                }
            )
            
            logger.info(
                f"Data processing completed successfully. "
                f"Final dataset: {school_data.shape[0]} rows × {school_data.shape[1]} columns"
            )
            
            return school_data
            
        except Exception as e:
            error_msg = f"Data processing pipeline failed: {str(e)}"
            logger.error(error_msg)
            raise DataProcessingError(error_msg) from e
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing summary statistics
        """
        try:
            summary = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'missing_data': {
                    'total_missing': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
                    'columns_with_missing': df.isnull().any().sum(),
                    'completely_empty_columns': (df.isnull().all()).sum(),
                },
                'data_types': df.dtypes.value_counts().to_dict(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
            }
            
            # Add column-specific statistics for key columns
            if 'city' in df.columns:
                summary['boroughs'] = {
                    'unique_count': df['city'].nunique(),
                    'most_common': df['city'].value_counts().head().to_dict(),
                }
            
            if 'total_students' in df.columns:
                student_stats = df['total_students'].describe()
                summary['student_statistics'] = {
                    'mean': student_stats['mean'],
                    'median': student_stats['50%'],
                    'std': student_stats['std'],
                    'min': student_stats['min'],
                    'max': student_stats['max'],
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate data summary: {str(e)}")
            raise DataProcessingError(f"Failed to generate data summary: {str(e)}") from e