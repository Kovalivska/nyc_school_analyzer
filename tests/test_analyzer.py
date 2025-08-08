"""
Tests for analysis functionality.

Comprehensive tests for SchoolAnalyzer class including statistical analysis,
grade availability analysis, and insight generation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.nyc_school_analyzer.analysis.analyzer import SchoolAnalyzer
from src.nyc_school_analyzer.data.models import AnalysisResult, AnalysisConfig
from src.nyc_school_analyzer.data.processor import DataProcessor
from src.nyc_school_analyzer.utils.exceptions import AnalysisError


class TestSchoolAnalyzer:
    """Test cases for SchoolAnalyzer class."""
    
    def test_init(self):
        """Test SchoolAnalyzer initialization."""
        analyzer = SchoolAnalyzer()
        assert analyzer is not None
        assert analyzer.processor is not None
        assert analyzer.config is not None
        assert analyzer.metrics_calculator is not None
        assert analyzer.insight_generator is not None
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = AnalysisConfig(target_borough="QUEENS", grade_of_interest=10)
        analyzer = SchoolAnalyzer(config=config)
        
        assert analyzer.config.target_borough == "QUEENS"
        assert analyzer.config.grade_of_interest == 10
    
    def test_analyze_borough_distribution(self, sample_school_data):
        """Test borough distribution analysis."""
        analyzer = SchoolAnalyzer()
        
        result = analyzer.analyze_borough_distribution(sample_school_data)
        
        assert 'school_counts' in result
        assert 'total_boroughs' in result
        assert 'largest_borough' in result
        assert 'concentration_metrics' in result
        
        # Check specific values from sample data
        school_counts = result['school_counts']
        assert school_counts['NEW YORK'] == 3  # 3 NYC schools in sample
        assert school_counts['BROOKLYN'] == 2  # 2 Brooklyn schools in sample
        assert result['total_boroughs'] == 5  # 5 unique boroughs
        assert result['largest_borough'] == 'NEW YORK'
    
    def test_analyze_borough_distribution_with_students(self, sample_school_data):
        """Test borough distribution analysis including student statistics."""
        analyzer = SchoolAnalyzer()
        
        result = analyzer.analyze_borough_distribution(sample_school_data)
        
        assert 'student_statistics' in result
        assert result['student_statistics'] is not None
        
        # Check that student statistics are calculated
        student_stats = result['student_statistics']
        assert 'BROOKLYN' in student_stats
        assert 'NEW YORK' in student_stats
        
        # Brooklyn stats should show average of the two Brooklyn schools
        brooklyn_stats = student_stats['BROOKLYN']
        expected_brooklyn_avg = (5858 + 585) / 2  # Average of two Brooklyn schools
        assert abs(brooklyn_stats['avg_students'] - expected_brooklyn_avg) < 1
    
    def test_analyze_borough_distribution_missing_column(self):
        """Test borough analysis with missing city column."""
        analyzer = SchoolAnalyzer()
        df = pd.DataFrame({'name': ['School A'], 'students': [500]})
        
        with pytest.raises(AnalysisError, match="Column 'city' not found"):
            analyzer.analyze_borough_distribution(df)
    
    def test_analyze_grade_availability(self, data_processor, sample_school_data):
        """Test grade availability analysis."""
        analyzer = SchoolAnalyzer(data_processor=data_processor)
        
        # Process grade columns first
        processed_df = data_processor.process_grade_columns(sample_school_data)
        
        result = analyzer.analyze_grade_availability(processed_df, target_grade=9)
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "Grade Availability"
        assert result.grade_analyzed == 9
        assert result.total_schools == 8
        
        # All schools in sample data should offer grade 9
        assert result.schools_meeting_criteria == 7  # 7 schools start at grade 9
        assert result.percentage > 80  # Should be high percentage
    
    def test_analyze_grade_availability_different_grades(self, data_processor, sample_school_data):
        """Test grade availability for different grade levels."""
        analyzer = SchoolAnalyzer(data_processor=data_processor)
        processed_df = data_processor.process_grade_columns(sample_school_data)
        
        # Test grade 6 (should be available in fewer schools)
        result_grade_6 = analyzer.analyze_grade_availability(processed_df, target_grade=6)
        assert result_grade_6.grade_analyzed == 6
        assert result_grade_6.schools_meeting_criteria < result_grade_6.total_schools
        
        # Test grade 12 (should be available in most schools)
        result_grade_12 = analyzer.analyze_grade_availability(processed_df, target_grade=12)
        assert result_grade_12.grade_analyzed == 12
        assert result_grade_12.schools_meeting_criteria == 8  # All schools go to grade 12
    
    def test_analyze_student_populations(self, sample_school_data):
        """Test student population analysis."""
        analyzer = SchoolAnalyzer()
        
        result = analyzer.analyze_student_populations(sample_school_data)
        
        assert 'population_statistics' in result
        assert 'patterns_and_outliers' in result
        assert 'system_metrics' in result
        assert 'analysis_metadata' in result
        
        # Check metadata
        metadata = result['analysis_metadata']
        assert metadata['schools_analyzed'] == 8
        assert metadata['group_by_column'] == 'city'
    
    def test_analyze_student_populations_missing_column(self):
        """Test student population analysis without total_students column."""
        analyzer = SchoolAnalyzer()
        df = pd.DataFrame({'name': ['School A'], 'city': ['BROOKLYN']})
        
        with pytest.raises(AnalysisError, match="'total_students' column required"):
            analyzer.analyze_student_populations(df)
    
    def test_analyze_data_quality(self, sample_school_data):
        """Test data quality analysis."""
        analyzer = SchoolAnalyzer()
        
        result = analyzer.analyze_data_quality(sample_school_data)
        
        assert 'basic_metrics' in result
        assert 'missing_data_analysis' in result
        assert 'data_type_analysis' in result
        assert 'completeness_analysis' in result
        assert 'quality_score' in result
        assert 'recommendations' in result
        
        # Check basic metrics
        basic_metrics = result['basic_metrics']
        assert basic_metrics['total_records'] == 8
        assert basic_metrics['total_columns'] == 9
        
        # Quality score should be high for clean sample data
        assert result['quality_score'] > 80
    
    def test_generate_comprehensive_report(self, processed_school_data):
        """Test comprehensive report generation."""
        analyzer = SchoolAnalyzer()
        
        report = analyzer.generate_comprehensive_report(processed_school_data)
        
        assert 'executive_summary' in report
        assert 'detailed_analyses' in report
        assert 'recommendations' in report
        assert 'metadata' in report
        
        # Check that all expected analyses are present
        analyses = report['detailed_analyses']
        assert 'borough_distribution' in analyses
        assert 'grade_availability' in analyses
        assert 'student_populations' in analyses
        assert 'data_quality' in analyses
        
        # Check metadata
        metadata = report['metadata']
        assert 'report_generated_at' in metadata
        assert 'total_schools_analyzed' in metadata
    
    def test_generate_comprehensive_report_with_borough_filter(self, processed_school_data):
        """Test comprehensive report with borough filtering."""
        analyzer = SchoolAnalyzer()
        
        report = analyzer.generate_comprehensive_report(
            processed_school_data, 
            target_borough="BROOKLYN"
        )
        
        # Metadata should reflect the borough filter
        assert report['metadata']['target_borough'] == "BROOKLYN"
        
        # Schools analyzed should be fewer (only Brooklyn schools)
        brooklyn_count = report['metadata']['total_schools_analyzed']  
        assert brooklyn_count < processed_school_data.shape[0]  # Should be subset
    
    def test_market_concentration_calculation(self):
        """Test market concentration metrics calculation."""
        analyzer = SchoolAnalyzer()
        
        # Create test data
        school_counts = pd.Series({'A': 100, 'B': 50, 'C': 30, 'D': 20})
        total_schools = 200
        
        concentration = analyzer._calculate_market_concentration(school_counts, total_schools)
        
        assert 'top_3_concentration' in concentration
        assert 'top_5_concentration' in concentration
        assert 'herfindahl_index' in concentration
        
        # Top 3 should be (100 + 50 + 30) / 200 = 90%
        assert abs(concentration['top_3_concentration'] - 90.0) < 0.1
    
    def test_grade_condition_building(self, data_processor, sample_school_data):
        """Test grade condition building logic."""
        analyzer = SchoolAnalyzer(data_processor=data_processor)
        processed_df = data_processor.process_grade_columns(sample_school_data)
        
        # Test grade 9 condition
        condition = analyzer._build_grade_condition(processed_df, 9, include_expanded=False)
        
        assert isinstance(condition, pd.Series)
        assert condition.dtype == bool
        assert len(condition) == len(processed_df)
        
        # Should be True for schools that include grade 9
        # Most schools in sample start at grade 9, so most should be True
        assert condition.sum() >= 6  # At least 6 schools should offer grade 9
    
    def test_error_handling_empty_dataset(self):
        """Test error handling with empty dataset."""
        analyzer = SchoolAnalyzer()
        empty_df = pd.DataFrame()
        
        with pytest.raises(AnalysisError):
            analyzer.analyze_borough_distribution(empty_df)
    
    def test_error_handling_invalid_grade(self, sample_school_data):
        """Test error handling with invalid grade values."""
        analyzer = SchoolAnalyzer()
        
        # Test with out-of-range grade
        result = analyzer.analyze_grade_availability(sample_school_data, target_grade=15)
        
        # Should complete without error but with 0 schools meeting criteria
        assert isinstance(result, AnalysisResult)
        assert result.schools_meeting_criteria == 0
    
    @pytest.mark.slow
    def test_performance_large_dataset(self, test_data_helper):
        """Test performance with large dataset."""
        analyzer = SchoolAnalyzer()
        
        # Create large dataset
        large_df = test_data_helper.create_large_dataset(2000)
        
        # Analysis should complete in reasonable time
        result = analyzer.analyze_borough_distribution(large_df)
        
        assert 'school_counts' in result
        assert result['total_boroughs'] == 5  # Expected number of boroughs
        assert sum(result['school_counts'].values()) == 2000


class TestAnalysisResult:
    """Test cases for AnalysisResult model."""
    
    def test_analysis_result_creation(self):
        """Test AnalysisResult creation and properties."""
        result = AnalysisResult(
            analysis_type="Test Analysis",
            target_borough="BROOKLYN",
            grade_analyzed=9,
            total_schools=100,
            schools_meeting_criteria=85,
            percentage=85.0,
            statistics={'mean': 500, 'std': 100},
            insights=['Good coverage', 'Room for improvement']
        )
        
        assert result.analysis_type == "Test Analysis"
        assert result.target_borough == "BROOKLYN"
        assert result.grade_analyzed == 9
        assert result.total_schools == 100
        assert result.schools_meeting_criteria == 85
        assert result.percentage == 85.0
        assert len(result.statistics) == 2
        assert len(result.insights) == 2
    
    def test_analysis_result_summary(self):
        """Test AnalysisResult summary property."""
        result = AnalysisResult(
            analysis_type="Grade Availability",
            target_borough="QUEENS",
            grade_analyzed=10,
            total_schools=50,
            schools_meeting_criteria=45,
            percentage=90.0
        )
        
        summary = result.summary
        assert "Grade Availability Analysis" in summary
        assert "45/50 schools" in summary
        assert "90.0%" in summary
        assert "QUEENS" in summary


@pytest.mark.integration
class TestSchoolAnalyzerIntegration:
    """Integration tests for SchoolAnalyzer with real workflows."""
    
    def test_complete_analysis_workflow(self, processed_school_data):
        """Test complete analysis workflow from data to insights."""
        analyzer = SchoolAnalyzer()
        
        # Run borough analysis
        borough_result = analyzer.analyze_borough_distribution(
            processed_school_data.processed_data
        )
        assert 'school_counts' in borough_result
        
        # Run grade analysis
        grade_result = analyzer.analyze_grade_availability(
            processed_school_data.processed_data, 
            target_grade=9
        )
        assert isinstance(grade_result, AnalysisResult)
        
        # Run student population analysis
        student_result = analyzer.analyze_student_populations(
            processed_school_data.processed_data
        )
        assert 'population_statistics' in student_result
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(processed_school_data)
        assert 'detailed_analyses' in report
        assert 'recommendations' in report
    
    def test_analysis_with_missing_data(self, test_data_helper, data_processor):
        """Test analysis robustness with missing data."""
        # Create dataset with missing values
        missing_df = test_data_helper.create_missing_data_dataset()
        
        analyzer = SchoolAnalyzer(data_processor=data_processor)
        
        # Should handle missing data gracefully
        result = analyzer.analyze_data_quality(missing_df)
        
        assert 'missing_data_analysis' in result
        assert result['missing_data_analysis']['total_missing_values'] > 0
        assert result['quality_score'] < 100  # Should reflect missing data in score
    
    def test_cross_analysis_consistency(self, processed_school_data):
        """Test consistency across different analysis methods."""
        analyzer = SchoolAnalyzer()
        
        # Get total schools from different analyses
        borough_result = analyzer.analyze_borough_distribution(
            processed_school_data.processed_data
        )
        
        grade_result = analyzer.analyze_grade_availability(
            processed_school_data.processed_data, 
            target_grade=9
        )
        
        quality_result = analyzer.analyze_data_quality(
            processed_school_data.processed_data
        )
        
        # Total schools should be consistent across analyses
        total_from_borough = sum(borough_result['school_counts'].values())
        total_from_grade = grade_result.total_schools
        total_from_quality = quality_result['basic_metrics']['total_records']
        
        assert total_from_borough == total_from_grade == total_from_quality