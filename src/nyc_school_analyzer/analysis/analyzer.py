"""
School analysis module for NYC School Analyzer.

Provides comprehensive analysis capabilities for school datasets including
statistical analysis, trend identification, and comparative studies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from ..data.models import SchoolData, AnalysisResult, AnalysisConfig
from ..data.processor import DataProcessor
from .metrics import MetricsCalculator
from .insights import InsightGenerator
from ..utils.exceptions import AnalysisError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SchoolAnalyzer:
    """
    Comprehensive school data analyzer.
    
    Performs statistical analysis, trend identification, and generates
    actionable insights from NYC school data.
    """
    
    def __init__(
        self, 
        data_processor: Optional[DataProcessor] = None,
        config: Optional[AnalysisConfig] = None
    ):
        """
        Initialize school analyzer.
        
        Args:
            data_processor: Data processor instance
            config: Analysis configuration
        """
        self.processor = data_processor or DataProcessor()
        self.config = config or AnalysisConfig()
        self.metrics_calculator = MetricsCalculator()
        self.insight_generator = InsightGenerator()
    
    def analyze_borough_distribution(
        self, 
        df: pd.DataFrame, 
        city_column: str = 'city'
    ) -> Dict[str, Any]:
        """
        Analyze school distribution across boroughs with comprehensive metrics.
        
        Args:
            df: Input DataFrame
            city_column: Column containing borough information
            
        Returns:
            Dictionary containing borough analysis results
        """
        try:
            logger.info("Analyzing borough distribution...")
            
            if city_column not in df.columns:
                raise AnalysisError(f"Column '{city_column}' not found in DataFrame")
            
            # Basic school counts
            school_counts = df[city_column].value_counts().sort_values(ascending=False)
            
            # Student statistics by borough (if available)
            student_stats = None
            if 'total_students' in df.columns:
                # Convert to numeric
                df_copy = df.copy()
                df_copy['total_students'] = pd.to_numeric(
                    df_copy['total_students'], errors='coerce'
                )
                
                student_stats = df_copy.groupby(city_column)['total_students'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max', 'sum'
                ]).round(1)
                
                student_stats.columns = [
                    'school_count', 'avg_students', 'median_students',
                    'std_students', 'min_students', 'max_students', 'total_students'
                ]
            
            # Calculate market concentration
            total_schools = len(df)
            concentration_metrics = self._calculate_market_concentration(school_counts, total_schools)
            
            # Generate analysis results
            results = {
                'school_counts': school_counts.to_dict(),
                'student_statistics': student_stats.to_dict() if student_stats is not None else None,
                'concentration_metrics': concentration_metrics,
                'total_boroughs': len(school_counts),
                'largest_borough': school_counts.index[0],
                'smallest_borough': school_counts.index[-1],
                'analysis_metadata': {
                    'total_schools_analyzed': total_schools,
                    'analysis_timestamp': pd.Timestamp.now(),
                    'city_column_used': city_column,
                }
            }
            
            logger.info(
                f"Borough analysis completed: {results['total_boroughs']} boroughs, "
                f"largest: {results['largest_borough']} "
                f"({school_counts[results['largest_borough']]} schools)"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Borough distribution analysis failed: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def analyze_grade_availability(
        self, 
        df: pd.DataFrame, 
        target_grade: int,
        include_expanded: bool = True
    ) -> AnalysisResult:
        """
        Analyze grade availability with comprehensive statistical analysis.
        
        Args:
            df: Input DataFrame
            target_grade: Grade to analyze availability for
            include_expanded: Whether to include expanded grade ranges
            
        Returns:
            AnalysisResult containing grade analysis results
        """
        try:
            logger.info(f"Analyzing Grade {target_grade} availability...")
            
            # Process grade columns
            df_processed = self.processor.process_grade_columns(df)
            
            # Determine schools offering target grade
            grade_condition = self._build_grade_condition(
                df_processed, target_grade, include_expanded
            )
            
            schools_offering_grade = df_processed[grade_condition]
            
            # Calculate statistics
            total_schools = len(df_processed)
            schools_count = len(schools_offering_grade)
            percentage = (schools_count / total_schools) * 100 if total_schools > 0 else 0
            
            # Generate detailed statistics
            detailed_stats = self._calculate_grade_statistics(
                df_processed, target_grade, include_expanded
            )
            
            # Generate insights
            insights = self.insight_generator.generate_grade_insights(
                target_grade, percentage, detailed_stats
            )
            
            # Create analysis result
            result = AnalysisResult(
                analysis_type="Grade Availability",
                target_borough=getattr(self.config, 'target_borough', 'All'),
                grade_analyzed=target_grade,
                total_schools=total_schools,
                schools_meeting_criteria=schools_count,
                percentage=percentage,
                statistics=detailed_stats,
                insights=insights
            )
            
            logger.info(
                f"Grade {target_grade} analysis: {schools_count}/{total_schools} schools "
                f"({percentage:.1f}%)"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Grade availability analysis failed: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def analyze_student_populations(
        self, 
        df: pd.DataFrame,
        group_by: str = 'city'
    ) -> Dict[str, Any]:
        """
        Analyze student population distributions and patterns.
        
        Args:
            df: Input DataFrame
            group_by: Column to group analysis by
            
        Returns:
            Dictionary containing student population analysis
        """
        try:
            logger.info("Analyzing student populations...")
            
            if 'total_students' not in df.columns:
                raise AnalysisError("'total_students' column required for population analysis")
            
            # Clean and prepare data
            df_clean = df.copy()
            df_clean['total_students'] = pd.to_numeric(
                df_clean['total_students'], errors='coerce'
            )
            
            # Remove rows with missing or invalid student counts
            df_clean = df_clean.dropna(subset=['total_students'])
            df_clean = df_clean[df_clean['total_students'] >= 0]
            
            # Calculate population statistics
            population_stats = self._calculate_population_statistics(df_clean, group_by)
            
            # Identify patterns and outliers
            patterns = self._identify_population_patterns(df_clean, group_by)
            
            # Calculate system-wide metrics
            system_metrics = self.metrics_calculator.calculate_system_metrics(df_clean)
            
            results = {
                'population_statistics': population_stats,
                'patterns_and_outliers': patterns,
                'system_metrics': system_metrics,
                'analysis_metadata': {
                    'schools_analyzed': len(df_clean),
                    'schools_excluded': len(df) - len(df_clean),
                    'group_by_column': group_by,
                    'analysis_timestamp': pd.Timestamp.now(),
                }
            }
            
            logger.info(
                f"Student population analysis completed: {len(df_clean)} schools analyzed"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Student population analysis failed: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data quality metrics and insights
        """
        try:
            logger.info("Performing data quality analysis...")
            
            # Basic quality metrics
            basic_metrics = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            }
            
            # Missing data analysis
            missing_analysis = self._analyze_missing_data(df)
            
            # Data type analysis
            dtype_analysis = self._analyze_data_types(df)
            
            # Completeness analysis
            completeness_analysis = self._analyze_data_completeness(df)
            
            # Generate quality score
            quality_score = self._calculate_quality_score(
                missing_analysis, dtype_analysis, completeness_analysis
            )
            
            results = {
                'basic_metrics': basic_metrics,
                'missing_data_analysis': missing_analysis,
                'data_type_analysis': dtype_analysis,
                'completeness_analysis': completeness_analysis,
                'quality_score': quality_score,
                'recommendations': self._generate_quality_recommendations(quality_score),
            }
            
            logger.info(
                f"Data quality analysis completed. Quality score: {quality_score:.1f}/100"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Data quality analysis failed: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def generate_comprehensive_report(
        self, 
        school_data: SchoolData,
        target_borough: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for the dataset.
        
        Args:
            school_data: SchoolData object to analyze
            target_borough: Optional specific borough to focus on
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            logger.info("Generating comprehensive analysis report...")
            
            df = school_data.processed_data
            
            # Filter by borough if specified
            if target_borough:
                df = self.processor.filter_by_borough(df, target_borough)
            
            # Perform all analyses
            analyses = {}
            
            # Borough distribution
            analyses['borough_distribution'] = self.analyze_borough_distribution(df)
            
            # Grade availability
            analyses['grade_availability'] = self.analyze_grade_availability(
                df, self.config.grade_of_interest
            )
            
            # Student populations
            analyses['student_populations'] = self.analyze_student_populations(df)
            
            # Data quality
            analyses['data_quality'] = self.analyze_data_quality(df)
            
            # Generate executive summary
            executive_summary = self.insight_generator.generate_executive_summary(analyses)
            
            # Generate recommendations
            recommendations = self.insight_generator.generate_recommendations(analyses)
            
            report = {
                'executive_summary': executive_summary,
                'detailed_analyses': analyses,
                'recommendations': recommendations,
                'metadata': {
                    'target_borough': target_borough,
                    'grade_of_interest': self.config.grade_of_interest,
                    'report_generated_at': pd.Timestamp.now(),
                    'data_source': school_data.metadata.get('source_file', 'Unknown'),
                    'total_schools_analyzed': len(df),
                }
            }
            
            logger.info("Comprehensive analysis report generated successfully")
            return report
            
        except Exception as e:
            error_msg = f"Comprehensive report generation failed: {str(e)}"
            logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    # Private helper methods
    
    def _calculate_market_concentration(
        self, 
        school_counts: pd.Series, 
        total_schools: int
    ) -> Dict[str, float]:
        """Calculate market concentration metrics."""
        top_3_schools = school_counts.head(3).sum()
        top_5_schools = school_counts.head(5).sum()
        
        return {
            'top_3_concentration': (top_3_schools / total_schools) * 100,
            'top_5_concentration': (top_5_schools / total_schools) * 100,
            'herfindahl_index': ((school_counts / total_schools) ** 2).sum() * 10000,
        }
    
    def _build_grade_condition(
        self, 
        df: pd.DataFrame, 
        target_grade: int, 
        include_expanded: bool
    ) -> pd.Series:
        """Build boolean condition for grade availability."""
        # Primary grade condition
        condition = (
            (df.get('grade_span_min_numeric', pd.Series(dtype=float)) <= target_grade) & 
            (df.get('grade_span_max_numeric', pd.Series(dtype=float)) >= target_grade)
        )
        
        # Include expanded grades if requested and available
        if include_expanded:
            exp_condition = (
                (df.get('exp_grade_span_min_numeric', pd.Series(dtype=float)) <= target_grade) & 
                (df.get('exp_grade_span_max_numeric', pd.Series(dtype=float)) >= target_grade)
            )
            condition = condition | exp_condition
        
        return condition.fillna(False)
    
    def _calculate_grade_statistics(
        self, 
        df: pd.DataFrame, 
        target_grade: int, 
        include_expanded: bool
    ) -> Dict[str, Any]:
        """Calculate detailed grade statistics."""
        stats = {}
        
        # Grade distribution
        if 'grade_span_min_numeric' in df.columns:
            stats['min_grade_distribution'] = df['grade_span_min_numeric'].value_counts().to_dict()
        
        if 'grade_span_max_numeric' in df.columns:
            stats['max_grade_distribution'] = df['grade_span_max_numeric'].value_counts().to_dict()
        
        # Grade span analysis
        if all(col in df.columns for col in ['grade_span_min_numeric', 'grade_span_max_numeric']):
            df_grades = df[['grade_span_min_numeric', 'grade_span_max_numeric']].dropna()
            stats['average_grade_span'] = (
                df_grades['grade_span_max_numeric'] - df_grades['grade_span_min_numeric']
            ).mean()
        
        return stats
    
    def _calculate_population_statistics(
        self, 
        df: pd.DataFrame, 
        group_by: str
    ) -> Dict[str, Any]:
        """Calculate student population statistics."""
        if group_by not in df.columns:
            return {}
        
        stats = df.groupby(group_by)['total_students'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max', 'sum'
        ]).round(1)
        
        return stats.to_dict()
    
    def _identify_population_patterns(
        self, 
        df: pd.DataFrame, 
        group_by: str
    ) -> Dict[str, Any]:
        """Identify patterns in student populations."""
        patterns = {}
        
        # Calculate system average
        system_avg = df['total_students'].mean()
        
        # Identify high and low capacity areas
        if group_by in df.columns:
            borough_avgs = df.groupby(group_by)['total_students'].mean()
            
            patterns['high_capacity_areas'] = borough_avgs[
                borough_avgs > system_avg * 1.5
            ].to_dict()
            
            patterns['low_capacity_areas'] = borough_avgs[
                borough_avgs < system_avg * 0.5
            ].to_dict()
        
        # Identify outlier schools
        q1 = df['total_students'].quantile(0.25)
        q3 = df['total_students'].quantile(0.75)
        iqr = q3 - q1
        
        outliers = df[
            (df['total_students'] < q1 - 1.5 * iqr) |
            (df['total_students'] > q3 + 1.5 * iqr)
        ]
        
        patterns['outlier_schools'] = len(outliers)
        patterns['outlier_percentage'] = (len(outliers) / len(df)) * 100
        
        return patterns
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_stats = df.isnull().sum()
        total_cells = df.size
        
        return {
            'total_missing_values': missing_stats.sum(),
            'missing_percentage': (missing_stats.sum() / total_cells) * 100,
            'columns_with_missing': (missing_stats > 0).sum(),
            'completely_empty_columns': (missing_stats == len(df)).sum(),
            'worst_columns': missing_stats.nlargest(5).to_dict(),
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data type distribution."""
        return {
            'data_type_counts': df.dtypes.value_counts().to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
        }
    
    def _analyze_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data completeness."""
        completeness = 1 - (df.isnull().sum() / len(df))
        
        return {
            'average_completeness': completeness.mean(),
            'median_completeness': completeness.median(),
            'best_columns': completeness.nlargest(5).to_dict(),
            'worst_columns': completeness.nsmallest(5).to_dict(),
        }
    
    def _calculate_quality_score(
        self, 
        missing_analysis: Dict, 
        dtype_analysis: Dict, 
        completeness_analysis: Dict
    ) -> float:
        """Calculate overall data quality score (0-100)."""
        # Components of quality score
        completeness_score = completeness_analysis['average_completeness'] * 40
        missing_penalty = min(missing_analysis['missing_percentage'], 30) * -1
        type_diversity_bonus = min(len(dtype_analysis['data_type_counts']), 5) * 5
        
        score = max(0, min(100, 50 + completeness_score + missing_penalty + type_diversity_bonus))
        return score
    
    def _generate_quality_recommendations(self, quality_score: float) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        if quality_score < 50:
            recommendations.append("Critical data quality issues require immediate attention")
            recommendations.append("Consider data cleaning and validation procedures")
        elif quality_score < 70:
            recommendations.append("Data quality improvements recommended")
            recommendations.append("Focus on reducing missing data and improving consistency")
        elif quality_score < 85:
            recommendations.append("Good data quality with room for minor improvements")
        else:
            recommendations.append("Excellent data quality - maintain current standards")
        
        return recommendations