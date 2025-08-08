"""
Tests for data processing functionality.

Comprehensive tests for DataProcessor class including data loading,
cleaning, validation, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.nyc_school_analyzer.data.processor import DataProcessor
from src.nyc_school_analyzer.data.models import SchoolData
from src.nyc_school_analyzer.utils.exceptions import DataProcessingError


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor is not None
        assert processor.validator is not None
    
    def test_load_dataset_success(self, sample_csv_file, data_processor):
        """Test successful dataset loading."""
        df = data_processor.load_dataset(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 8  # Sample data has 8 schools
        assert 'school_name' in df.columns
        assert 'city' in df.columns
    
    def test_load_dataset_file_not_found(self, data_processor):
        """Test loading non-existent file."""
        with pytest.raises(DataProcessingError, match="Dataset file not found"):
            data_processor.load_dataset("nonexistent_file.csv")
    
    def test_load_dataset_empty_file(self, data_processor, tmp_path):
        """Test loading empty CSV file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        
        with pytest.raises(DataProcessingError, match="Dataset is empty"):
            data_processor.load_dataset(empty_file)
    
    def test_clean_column_names(self, data_processor):
        """Test column name cleaning."""
        # Create DataFrame with problematic column names
        df = pd.DataFrame({
            'School Name': [1, 2, 3],
            'Grade Span-Min': [1, 2, 3],
            'Total Students!!!': [1, 2, 3],
            '  Extra Spaces  ': [1, 2, 3],
            '': [1, 2, 3],  # Empty column name
            'School Name': [4, 5, 6]  # Duplicate name
        })
        
        cleaned_df = data_processor.clean_column_names(df)
        
        expected_columns = [
            'school_name', 'grade_span_min', 'total_students', 
            'extra_spaces', 'unnamed_4', 'school_name_1'
        ]
        
        assert list(cleaned_df.columns) == expected_columns
        assert len(set(cleaned_df.columns)) == len(cleaned_df.columns)  # No duplicates
    
    @pytest.mark.parametrize("input_value,expected", [
        ('9', 9),
        ('12', 12),
        ('K', 0),
        ('KINDERGARTEN', 0),
        ('PRE-K', 0),
        (9, 9),
        (9.0, 9),
        ('Grade 10', 10),
        ('invalid', None),
        (None, None),
        (np.nan, None),
        ('15', None),  # Out of range
        (-1, None)     # Out of range
    ])
    def test_parse_grade_value(self, data_processor, input_value, expected):
        """Test grade value parsing with various inputs."""
        result = data_processor.parse_grade_value(input_value)
        assert result == expected
    
    def test_filter_by_borough_success(self, data_processor, sample_school_data):
        """Test successful borough filtering."""
        filtered_df = data_processor.filter_by_borough(sample_school_data, "BROOKLYN")
        
        assert len(filtered_df) == 2  # Two Brooklyn schools in sample data
        assert all(filtered_df['city'] == 'BROOKLYN')
    
    def test_filter_by_borough_case_insensitive(self, data_processor, sample_school_data):
        """Test borough filtering is case insensitive."""
        filtered_df = data_processor.filter_by_borough(sample_school_data, "brooklyn")
        assert len(filtered_df) == 2
        
        filtered_df = data_processor.filter_by_borough(sample_school_data, "Brooklyn")
        assert len(filtered_df) == 2
    
    def test_filter_by_borough_not_found(self, data_processor, sample_school_data):
        """Test filtering for non-existent borough."""
        with pytest.raises(DataProcessingError, match="No schools found for borough"):
            data_processor.filter_by_borough(sample_school_data, "NONEXISTENT")
    
    def test_filter_by_borough_missing_column(self, data_processor):
        """Test filtering with missing city column."""
        df = pd.DataFrame({'name': ['School A'], 'location': ['Somewhere']})
        
        with pytest.raises(DataProcessingError, match="Column 'city' not found"):
            data_processor.filter_by_borough(df, "BROOKLYN")
    
    def test_process_grade_columns(self, data_processor, sample_school_data):
        """Test grade column processing."""
        processed_df = data_processor.process_grade_columns(sample_school_data)
        
        # Check that numeric columns were created
        expected_numeric_columns = [
            'grade_span_min_numeric', 'grade_span_max_numeric'
        ]
        
        for col in expected_numeric_columns:
            assert col in processed_df.columns
            # Check that values are numeric where original was valid
            assert processed_df[col].dtype in [np.int64, np.float64]
    
    def test_clean_numeric_columns(self, data_processor):
        """Test numeric column cleaning."""
        df = pd.DataFrame({
            'total_students': ['500', '1000', 'invalid', '750'],
            'graduation_rate': ['95.5', '88.2', '92.0', 'N/A'],
            'text_column': ['A', 'B', 'C', 'D']  # Should not be affected
        })
        
        cleaned_df = data_processor.clean_numeric_columns(df)
        
        # Check total_students column
        assert pd.api.types.is_numeric_dtype(cleaned_df['total_students'])
        assert cleaned_df['total_students'].iloc[0] == 500
        assert pd.isna(cleaned_df['total_students'].iloc[2])  # 'invalid' became NaN
        
        # Check graduation_rate column
        assert pd.api.types.is_numeric_dtype(cleaned_df['graduation_rate'])
        assert cleaned_df['graduation_rate'].iloc[0] == 95.5
        assert pd.isna(cleaned_df['graduation_rate'].iloc[3])  # 'N/A' became NaN
        
        # Check text column unchanged
        assert cleaned_df['text_column'].iloc[0] == 'A'
    
    def test_process_dataset_full_pipeline(self, data_processor, sample_csv_file):
        """Test complete data processing pipeline."""
        school_data = data_processor.process_dataset(sample_csv_file, validate=True)
        
        assert isinstance(school_data, SchoolData)
        assert isinstance(school_data.raw_data, pd.DataFrame)
        assert isinstance(school_data.processed_data, pd.DataFrame)
        assert 'source_file' in school_data.metadata
        assert 'processing_timestamp' in school_data.metadata
        
        # Check that processing was applied
        assert len(school_data.processed_data) == len(school_data.raw_data)
        assert school_data.processed_data.columns[0].islower()  # Column names cleaned
    
    def test_get_data_summary(self, data_processor, sample_school_data):
        """Test data summary generation."""
        summary = data_processor.get_data_summary(sample_school_data)
        
        assert 'shape' in summary
        assert 'memory_usage_mb' in summary
        assert 'missing_data' in summary
        assert 'data_types' in summary
        
        # Check specific values
        assert summary['shape'] == (8, 9)  # 8 schools, 9 columns
        assert summary['missing_data']['total_missing'] == 0  # Sample data has no missing values
        assert 'boroughs' in summary  # Should include borough analysis
    
    def test_error_handling_invalid_file_format(self, data_processor, tmp_path):
        """Test error handling for unsupported file formats."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("some text content")
        
        with pytest.raises(DataProcessingError, match="Unsupported file format"):
            data_processor.load_dataset(invalid_file)
    
    @pytest.mark.slow
    def test_performance_large_dataset(self, data_processor, test_data_helper, tmp_path):
        """Test performance with large dataset."""
        # Create large dataset
        large_df = test_data_helper.create_large_dataset(5000)
        large_file = tmp_path / "large_dataset.csv"
        large_df.to_csv(large_file, index=False)
        
        # Process dataset - should complete without errors
        school_data = data_processor.process_dataset(large_file, validate=False)
        
        assert len(school_data.processed_data) == 5000
        assert school_data.metadata['total_records'] == 5000
    
    def test_missing_data_handling(self, data_processor, test_data_helper, tmp_path):
        """Test handling of datasets with missing data."""
        missing_df = test_data_helper.create_missing_data_dataset()
        missing_file = tmp_path / "missing_data.csv"
        missing_df.to_csv(missing_file, index=False)
        
        school_data = data_processor.process_dataset(missing_file, validate=False)
        
        # Should process successfully despite missing data
        assert isinstance(school_data, SchoolData)
        assert len(school_data.processed_data) == 4
        
        # Check summary includes missing data information
        summary = data_processor.get_data_summary(school_data.processed_data)
        assert summary['missing_data']['total_missing'] > 0
    
    def test_data_consistency_validation(self, data_processor, invalid_school_data, tmp_path):
        """Test data consistency validation."""
        invalid_file = tmp_path / "invalid_data.csv" 
        invalid_school_data.to_csv(invalid_file, index=False)
        
        # Should process but may have validation warnings
        school_data = data_processor.process_dataset(invalid_file, validate=True)
        
        # Processing should complete
        assert isinstance(school_data, SchoolData)
        
        # Check that invalid numeric values were handled
        processed_df = school_data.processed_data
        total_students = processed_df['total_students']
        
        # Negative value should become NaN
        assert pd.isna(total_students.iloc[0]) or total_students.iloc[0] >= 0
        # Non-numeric value should become NaN
        assert pd.isna(total_students.iloc[1])


class TestSchoolData:
    """Test cases for SchoolData model."""
    
    def test_school_data_creation(self, sample_school_data):
        """Test SchoolData object creation."""
        raw_data = sample_school_data.copy()
        processed_data = sample_school_data.copy()
        processed_data.columns = [col.lower() for col in processed_data.columns]
        
        school_data = SchoolData(
            raw_data=raw_data,
            processed_data=processed_data
        )
        
        assert school_data.shape == (8, 9)
        assert len(school_data.boroughs) > 0
        assert 'total_records' in school_data.metadata
        assert school_data.metadata['total_records'] == 8
    
    def test_school_data_properties(self, processed_school_data):
        """Test SchoolData properties."""
        assert processed_school_data.shape[0] > 0
        assert processed_school_data.shape[1] > 0
        assert isinstance(processed_school_data.boroughs, list)
        assert len(processed_school_data.boroughs) > 0


@pytest.mark.integration
class TestDataProcessorIntegration:
    """Integration tests for DataProcessor with real-world scenarios."""
    
    def test_end_to_end_processing(self, sample_csv_file, temp_output_dir):
        """Test end-to-end data processing workflow."""
        processor = DataProcessor()
        
        # Process data
        school_data = processor.process_dataset(sample_csv_file)
        
        # Verify processing
        assert isinstance(school_data, SchoolData)
        assert len(school_data.processed_data) > 0
        
        # Test filtering
        brooklyn_schools = processor.filter_by_borough(
            school_data.processed_data, "BROOKLYN"
        )
        assert len(brooklyn_schools) > 0
        
        # Test summary generation
        summary = processor.get_data_summary(school_data.processed_data)
        assert summary['shape'][0] > 0
    
    def test_error_recovery(self, data_processor):
        """Test error recovery and graceful handling."""
        # Test with various error conditions
        with pytest.raises(DataProcessingError):
            data_processor.load_dataset("nonexistent.csv")
        
        # Processor should still work after error
        df = pd.DataFrame({'test': [1, 2, 3]})
        cleaned = data_processor.clean_column_names(df)
        assert 'test' in cleaned.columns