"""
Metrics calculation module for NYC School Analyzer.

Provides specialized calculations for educational metrics and statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EducationalMetrics:
    """Container for educational performance metrics."""
    enrollment_metrics: Dict[str, float]
    accessibility_metrics: Dict[str, float]
    distribution_metrics: Dict[str, float]
    quality_indicators: Dict[str, float]


class MetricsCalculator:
    """
    Specialized calculator for educational metrics and statistics.
    
    Provides methods for calculating various educational performance
    indicators and accessibility metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_enrollment_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate enrollment-related metrics.
        
        Args:
            df: DataFrame with school data
            
        Returns:
            Dictionary of enrollment metrics
        """
        try:
            metrics = {}
            
            if 'total_students' in df.columns:
                students = pd.to_numeric(df['total_students'], errors='coerce')
                valid_students = students.dropna()
                
                if len(valid_students) > 0:
                    with np.errstate(invalid='ignore'):
                        metrics.update({
                            'total_enrollment': valid_students.sum(),
                            'average_school_size': valid_students.mean(),
                            'median_school_size': valid_students.median(),
                            'enrollment_std': valid_students.std(),
                            'min_school_size': valid_students.min(),
                            'max_school_size': valid_students.max(),
                            'enrollment_coefficient_variation': (
                                valid_students.std() / valid_students.mean() 
                                if valid_students.mean() > 0 else 0
                            ),
                        })
                        
                        # Calculate percentile-based metrics
                        percentiles = [10, 25, 75, 90]
                        for p in percentiles:
                            metrics[f'enrollment_p{p}'] = np.percentile(valid_students, p)
                        
                        # Size categories
                        small_schools = (valid_students < 300).sum()
                        medium_schools = ((valid_students >= 300) & (valid_students < 1000)).sum()
                        large_schools = (valid_students >= 1000).sum()
                        
                        total_schools = len(valid_students)
                        metrics.update({
                            'small_schools_count': small_schools,
                            'medium_schools_count': medium_schools,
                            'large_schools_count': large_schools,
                            'small_schools_percentage': (small_schools / total_schools) * 100,
                            'medium_schools_percentage': (medium_schools / total_schools) * 100,
                            'large_schools_percentage': (large_schools / total_schools) * 100,
                        })
            
            self.logger.debug(f"Calculated {len(metrics)} enrollment metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating enrollment metrics: {e}")
            return {}
    
    def calculate_accessibility_metrics(
        self, 
        df: pd.DataFrame, 
        target_grade: int = 9
    ) -> Dict[str, float]:
        """
        Calculate accessibility and availability metrics.
        
        Args:
            df: DataFrame with school data
            target_grade: Grade level to analyze
            
        Returns:
            Dictionary of accessibility metrics
        """
        try:
            metrics = {}
            
            # Grade availability metrics
            grade_columns = {
                'min': 'grade_span_min_numeric',
                'max': 'grade_span_max_numeric',
            }
            
            if all(col in df.columns for col in grade_columns.values()):
                min_grades = df[grade_columns['min']].dropna()
                max_grades = df[grade_columns['max']].dropna()
                
                # Schools offering target grade
                offering_condition = (
                    (df[grade_columns['min']] <= target_grade) & 
                    (df[grade_columns['max']] >= target_grade)
                )
                schools_offering = offering_condition.sum()
                total_schools = len(df)
                
                metrics.update({
                    f'grade_{target_grade}_availability_count': schools_offering,
                    f'grade_{target_grade}_availability_percentage': (
                        schools_offering / total_schools * 100 if total_schools > 0 else 0
                    ),
                    'average_grade_span': (max_grades - min_grades).mean(),
                    'median_grade_span': (max_grades - min_grades).median(),
                })
                
                # Grade range analysis
                grade_ranges = max_grades - min_grades
                narrow_range = (grade_ranges <= 3).sum()
                wide_range = (grade_ranges >= 8).sum()
                
                metrics.update({
                    'narrow_range_schools': narrow_range,
                    'wide_range_schools': wide_range,
                    'narrow_range_percentage': (narrow_range / len(grade_ranges)) * 100,
                    'wide_range_percentage': (wide_range / len(grade_ranges)) * 100,
                })
            
            # Geographic accessibility (if location data available)
            if 'city' in df.columns:
                borough_counts = df['city'].value_counts()
                metrics.update({
                    'geographic_diversity': len(borough_counts),
                    'largest_borough_concentration': (
                        borough_counts.max() / len(df) * 100 if len(df) > 0 else 0
                    ),
                    'smallest_borough_representation': (
                        borough_counts.min() / len(df) * 100 if len(df) > 0 else 0
                    ),
                })
                
                # Calculate geographic equity (Gini coefficient for borough distribution)
                metrics['geographic_equity_index'] = self._calculate_gini_coefficient(
                    borough_counts.values
                )
            
            self.logger.debug(f"Calculated {len(metrics)} accessibility metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating accessibility metrics: {e}")
            return {}
    
    def calculate_distribution_metrics(
        self, 
        df: pd.DataFrame, 
        group_by: str = 'city'
    ) -> Dict[str, float]:
        """
        Calculate distribution and equity metrics across geographic areas.
        
        Args:
            df: DataFrame with school data
            group_by: Column to group by for distribution analysis
            
        Returns:
            Dictionary of distribution metrics
        """
        try:
            metrics = {}
            
            if group_by not in df.columns:
                self.logger.warning(f"Column '{group_by}' not found for distribution analysis")
                return metrics
            
            # School distribution metrics
            distribution = df[group_by].value_counts()
            
            metrics.update({
                'total_areas': len(distribution),
                'max_schools_per_area': distribution.max(),
                'min_schools_per_area': distribution.min(),
                'avg_schools_per_area': distribution.mean(),
                'distribution_std': distribution.std(),
                'distribution_coefficient_variation': (
                    distribution.std() / distribution.mean() 
                    if distribution.mean() > 0 else 0
                ),
            })
            
            # Equity metrics
            metrics['herfindahl_hirschman_index'] = (
                (distribution / distribution.sum()) ** 2
            ).sum()
            
            metrics['distribution_gini'] = self._calculate_gini_coefficient(distribution.values)
            
            # Student distribution (if available)
            if 'total_students' in df.columns:
                student_dist = df.groupby(group_by)['total_students'].sum()
                
                metrics.update({
                    'student_distribution_gini': self._calculate_gini_coefficient(
                        student_dist.values
                    ),
                    'max_students_per_area': student_dist.max(),
                    'min_students_per_area': student_dist.min(),
                    'avg_students_per_area': student_dist.mean(),
                })
            
            self.logger.debug(f"Calculated {len(metrics)} distribution metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating distribution metrics: {e}")
            return {}
    
    def calculate_quality_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate data quality and completeness indicators.
        
        Args:
            df: DataFrame with school data
            
        Returns:
            Dictionary of quality indicators
        """
        try:
            metrics = {}
            
            # Basic completeness metrics
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            
            metrics.update({
                'data_completeness_percentage': (1 - missing_cells / total_cells) * 100,
                'total_missing_values': missing_cells,
                'columns_with_missing_data': df.isnull().any().sum(),
                'rows_with_missing_data': df.isnull().any(axis=1).sum(),
                'completely_empty_rows': df.isnull().all(axis=1).sum(),
                'completely_empty_columns': df.isnull().all().sum(),
            })
            
            # Column-specific quality metrics
            critical_columns = ['school_name', 'city', 'total_students']
            available_critical = [col for col in critical_columns if col in df.columns]
            
            if available_critical:
                critical_completeness = []
                for col in available_critical:
                    completeness = (1 - df[col].isnull().sum() / len(df)) * 100
                    critical_completeness.append(completeness)
                    metrics[f'{col}_completeness'] = completeness
                
                metrics['critical_data_completeness'] = np.mean(critical_completeness)
            
            # Data consistency indicators
            if 'school_name' in df.columns:
                unique_names = df['school_name'].nunique()
                total_records = len(df)
                metrics['name_uniqueness_ratio'] = unique_names / total_records if total_records > 0 else 0
            
            # Numeric data quality
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                negative_values = (df[numeric_cols] < 0).sum().sum()
                zero_values = (df[numeric_cols] == 0).sum().sum()
                
                metrics.update({
                    'negative_values_count': negative_values,
                    'zero_values_count': zero_values,
                    'numeric_data_quality_score': max(0, 100 - (negative_values + zero_values) * 2),
                })
            
            self.logger.debug(f"Calculated {len(metrics)} quality indicators")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality indicators: {e}")
            return {}
    
    def calculate_comprehensive_metrics(
        self, 
        df: pd.DataFrame, 
        target_grade: int = 9,
        group_by: str = 'city'
    ) -> EducationalMetrics:
        """
        Calculate all educational metrics in one comprehensive analysis.
        
        Args:
            df: DataFrame with school data
            target_grade: Grade level to analyze
            group_by: Column for distribution analysis
            
        Returns:
            EducationalMetrics object with all calculated metrics
        """
        try:
            self.logger.info("Calculating comprehensive educational metrics...")
            
            enrollment_metrics = self.calculate_enrollment_metrics(df)
            accessibility_metrics = self.calculate_accessibility_metrics(df, target_grade)
            distribution_metrics = self.calculate_distribution_metrics(df, group_by)
            quality_indicators = self.calculate_quality_indicators(df)
            
            metrics = EducationalMetrics(
                enrollment_metrics=enrollment_metrics,
                accessibility_metrics=accessibility_metrics,
                distribution_metrics=distribution_metrics,
                quality_indicators=quality_indicators
            )
            
            total_metrics = (
                len(enrollment_metrics) + len(accessibility_metrics) + 
                len(distribution_metrics) + len(quality_indicators)
            )
            
            self.logger.info(f"Calculated {total_metrics} comprehensive metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            # Return empty metrics object
            return EducationalMetrics(
                enrollment_metrics={},
                accessibility_metrics={},
                distribution_metrics={},
                quality_indicators={}
            )
    
    def calculate_system_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate system-wide educational metrics.
        
        Args:
            df: DataFrame with school data
            
        Returns:
            Dictionary of system-wide metrics
        """
        try:
            metrics = {}
            
            # System capacity metrics
            if 'total_students' in df.columns:
                total_students = pd.to_numeric(df['total_students'], errors='coerce').sum()
                total_schools = len(df)
                
                metrics.update({
                    'system_total_students': total_students,
                    'system_total_schools': total_schools,
                    'system_average_school_size': total_students / total_schools if total_schools > 0 else 0,
                })
            
            # System diversity metrics
            if 'city' in df.columns:
                borough_diversity = df['city'].nunique()
                metrics['system_geographic_diversity'] = borough_diversity
            
            # Grade coverage analysis
            grade_columns = ['grade_span_min_numeric', 'grade_span_max_numeric']
            if all(col in df.columns for col in grade_columns):
                min_grade_system = df['grade_span_min_numeric'].min()
                max_grade_system = df['grade_span_max_numeric'].max()
                
                metrics.update({
                    'system_grade_range_min': min_grade_system,
                    'system_grade_range_max': max_grade_system,
                    'system_total_grade_span': max_grade_system - min_grade_system,
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating system metrics: {e}")
            return {}
    
    # Private helper methods
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for measuring inequality.
        
        Args:
            values: Array of values to calculate Gini coefficient for
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        try:
            values = np.array(values, dtype=float)
            values = values[~np.isnan(values)]  # Remove NaN values
            
            if len(values) == 0:
                return 0.0
            
            # Sort values
            sorted_values = np.sort(values)
            n = len(sorted_values)
            
            if n == 1:
                return 0.0
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(sorted_values)
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_values) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
            
            return max(0.0, min(1.0, gini))  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.warning(f"Error calculating Gini coefficient: {e}")
            return 0.0
    
    def _calculate_statistical_summary(self, series: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive statistical summary for a series."""
        try:
            clean_series = series.dropna()
            if len(clean_series) == 0:
                return {}
            
            return {
                'count': len(clean_series),
                'mean': clean_series.mean(),
                'median': clean_series.median(),
                'std': clean_series.std(),
                'min': clean_series.min(),
                'max': clean_series.max(),
                'q25': clean_series.quantile(0.25),
                'q75': clean_series.quantile(0.75),
                'iqr': clean_series.quantile(0.75) - clean_series.quantile(0.25),
                'skewness': clean_series.skew(),
                'kurtosis': clean_series.kurtosis(),
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating statistical summary: {e}")
            return {}