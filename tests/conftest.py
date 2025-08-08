"""
Pytest configuration and shared fixtures for NYC School Analyzer tests.

Provides common test fixtures, sample data, and test utilities
for comprehensive testing of all application components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any
import logging

from src.nyc_school_analyzer.utils.config import Config
from src.nyc_school_analyzer.data.processor import DataProcessor
from src.nyc_school_analyzer.data.validator import DataValidator


# Suppress logging during tests
logging.getLogger().setLevel(logging.CRITICAL)


@pytest.fixture
def sample_school_data():
    """Create sample school data for testing."""
    data = {
        'school_name': [
            'Brooklyn Technical High School',
            'Stuyvesant High School',
            'Bronx Science High School',
            'LaGuardia High School',
            'Brooklyn Latin School',
            'Queens High School for Sciences',
            'Manhattan Center for Science',
            'Staten Island Technical High School'
        ],
        'city': [
            'BROOKLYN', 'NEW YORK', 'BRONX', 'NEW YORK', 
            'BROOKLYN', 'QUEENS', 'NEW YORK', 'STATEN ISLAND'
        ],
        'address': [
            '29 Fort Greene Pl, Brooklyn, NY 11217',
            '345 Chambers St, New York, NY 10282',
            '75 W 205th St, Bronx, NY 10468',
            '100 Amsterdam Ave, New York, NY 10023',
            '223 Graham Ave, Brooklyn, NY 11206',
            '105-07 207th St, Queens, NY 11412',
            '260 Pleasant Ave, New York, NY 10029',
            '485 Clawson St, Staten Island, NY 10306'
        ],
        'phone': [
            '718-804-6400', '212-312-4800', '718-817-7700', '212-496-0700',
            '718-366-0154', '718-658-6500', '212-860-5858', '718-667-3222'
        ],
        'grade_span_min': ['9', '9', '9', '9', '6', '9', '6', '9'],
        'grade_span_max': ['12', '12', '12', '12', '12', '12', '12', '12'],
        'total_students': [5858, 3336, 3010, 2730, 585, 446, 410, 378],
        'overview_paragraph': [
            'Specialized high school focusing on engineering and technology',
            'Specialized high school with rigorous academic program',
            'Specialized high school emphasizing science and mathematics',
            'Specialized high school for music, art and performing arts',
            'College preparatory school with classical curriculum',
            'STEM-focused high school in Queens',
            'Science and mathematics specialized program',
            'Technical high school on Staten Island'
        ]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_school_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_schools.csv"
    sample_school_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = Config()
    
    # Override with test-specific settings
    config.analysis.target_borough = "BROOKLYN"
    config.analysis.grade_of_interest = 9
    config.visualization.dpi = 100  # Lower DPI for faster tests
    config.logging.level = "CRITICAL"
    
    return config


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def data_processor():
    """Create DataProcessor instance for testing."""
    return DataProcessor()


@pytest.fixture
def data_validator():
    """Create DataValidator instance for testing."""
    return DataValidator()


@pytest.fixture
def processed_school_data(data_processor, sample_csv_file):
    """Create processed school data for testing."""
    return data_processor.process_dataset(sample_csv_file, validate=False)


@pytest.fixture
def sample_analysis_results():
    """Create sample analysis results for testing."""
    return {
        'borough_distribution': {
            'school_counts': {
                'BROOKLYN': 2,
                'NEW YORK': 3,
                'BRONX': 1,
                'QUEENS': 1,
                'STATEN ISLAND': 1
            },
            'total_boroughs': 5,
            'largest_borough': 'NEW YORK',
            'smallest_borough': 'BRONX'
        },
        'grade_availability': {
            'total_schools': 8,
            'schools_offering_grade': 7,
            'percentage_offering': 87.5,
            'target_grade': 9
        },
        'student_populations': {
            'population_statistics': {
                'BROOKLYN': {
                    'avg_students': 3221.5,
                    'total_students': 6443,
                    'school_count': 2
                },
                'NEW YORK': {
                    'avg_students': 2826.0,
                    'total_students': 8478,
                    'school_count': 3
                }
            }
        }
    }


@pytest.fixture
def invalid_school_data():
    """Create invalid school data for testing error handling."""
    data = {
        'school_name': ['Test School', '', 'Another School'],
        'city': ['BROOKLYN', None, 'INVALID_BOROUGH'],
        'grade_span_min': ['invalid', '9', '15'],  # Invalid grade values
        'grade_span_max': ['12', 'also_invalid', '8'],  # Invalid and inconsistent
        'total_students': [-100, 'not_a_number', 50000]  # Invalid student counts
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_visualization_config():
    """Create mock visualization configuration for testing."""
    from src.nyc_school_analyzer.data.models import VisualizationConfig
    
    return VisualizationConfig(
        style="seaborn",
        figure_size=(8, 6),
        dpi=100,
        color_palette="husl",
        font_size=10,
        export_formats=["png"]
    )


class TestDataHelper:
    """Helper class for creating test data."""
    
    @staticmethod
    def create_large_dataset(num_schools: int = 1000) -> pd.DataFrame:
        """Create large dataset for performance testing."""
        np.random.seed(42)  # For reproducible results
        
        boroughs = ['BROOKLYN', 'QUEENS', 'MANHATTAN', 'BRONX', 'STATEN ISLAND']
        
        data = {
            'school_name': [f'Test School {i+1}' for i in range(num_schools)],
            'city': np.random.choice(boroughs, num_schools),
            'grade_span_min': np.random.choice([6, 9, 'K'], num_schools),
            'grade_span_max': np.random.choice([8, 12], num_schools),
            'total_students': np.random.randint(100, 3000, num_schools),
            'address': [f'{i+1} Test Street, NY' for i in range(num_schools)]
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_missing_data_dataset() -> pd.DataFrame:
        """Create dataset with various missing data patterns."""
        data = {
            'school_name': ['School A', 'School B', None, 'School D'],
            'city': ['BROOKLYN', None, 'QUEENS', 'MANHATTAN'],
            'grade_span_min': ['9', '6', '9', None],
            'grade_span_max': ['12', None, '12', '12'],
            'total_students': [500, None, 750, 600],
            'phone': [None, None, '718-123-4567', '212-987-6543']
        }
        
        return pd.DataFrame(data)


@pytest.fixture
def test_data_helper():
    """Provide TestDataHelper instance."""
    return TestDataHelper()


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_dataframe_structure(df: pd.DataFrame, expected_columns: list):
        """Assert that DataFrame has expected structure."""
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in expected_columns)
    
    @staticmethod
    def assert_file_exists_and_not_empty(file_path: Path):
        """Assert that file exists and is not empty."""
        assert file_path.exists()
        assert file_path.stat().st_size > 0
    
    @staticmethod
    def create_temp_config_file(tmp_path: Path, config_data: Dict[str, Any]) -> Path:
        """Create temporary configuration file."""
        import yaml
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return config_file


@pytest.fixture
def test_utils():
    """Provide TestUtils instance."""
    return TestUtils()


# Cleanup helpers
@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Cleanup matplotlib figures after each test."""
    yield
    # Close all matplotlib figures to prevent memory leaks
    import matplotlib.pyplot as plt
    plt.close('all')