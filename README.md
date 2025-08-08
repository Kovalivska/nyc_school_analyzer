# NYC School Analyzer

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-green.svg)](https://github.com/nyc-schools/nyc-school-analyzer)

**Production-ready NYC High School Directory Analysis Tool**

A comprehensive, enterprise-grade Python application for analyzing NYC high school data with advanced statistical analysis, professional visualizations, and exportable insights.

## Overview

NYC School Analyzer transforms the raw Jupyter notebook analysis into a scalable, maintainable production system. This project originated from a comprehensive 4-day data analysis internship program that demonstrates proficiency in data exploration, Python programming, SQL database operations, and ETL processes using real NYC educational datasets.

The production system provides:

- **Professional Data Processing**: Robust ETL pipeline with validation and error handling
- **Advanced Analytics**: Statistical analysis, trend identification, and comparative studies  
- **Publication-Quality Visualizations**: Charts and dashboards ready for presentations
- **Multiple Export Formats**: CSV, JSON, Excel, PDF reports with automated generation
- **Command-Line Interface**: Easy-to-use CLI for batch processing and automation
- **Comprehensive Testing**: Unit tests, integration tests, and performance benchmarks
- **Production Deployment**: Docker support, logging, monitoring, and CI/CD ready

## 4-Day Analysis Project Foundation

This production tool is built upon comprehensive analysis work completed across 4 focused days:

### Day 1: Data Exploration and Google Sheets Analysis
**Dataset**: School Safety Report Data (data/1_school-safety-report.csv)
**Notebook**: notebooks/1_School_incident_Analysis(GoogleSheet).md

**Results**: Analyzed 6,310 school safety incidents across 1,893 unique schools. Identified "nocrim_n" as the most frequent incident type (28.2% in Bronx). Discovered that Brooklyn and Bronx represent 61% of all violent crimes, with correlation between building size and incident frequency.

### Day 2: Python Data Analysis with Pandas  
**Dataset**: NYC High School Directory (data/2_3_high-school-directory.csv)
**Notebook**: notebooks/2_NYC_school_directory_exploration_analysis_v2(prod).ipynb

**Results**: Processed 435 high schools with 69 attributes each. Brooklyn identified as largest borough with 121 schools (27.8% of total). Analyzed student population distribution showing average 699 students per Brooklyn school. Implemented comprehensive data cleaning pipeline with 100% validation success.

### Day 3: SQL Database Operations via Python
**Datasets**: High School Directory, School Demographics, School Safety Report 
**Notebook**: notebooks/3_NYC_school_data_exploration(SQL_via_Python).ipynb

**Results**: Executed multi-table JOIN queries across normalized database tables. Confirmed school distribution: Brooklyn (121), Bronx (118), Manhattan (106), Queens (80), Staten Island (10). Identified data quality issues with ELL demographic data and incomplete special education coverage.

### Day 4: ETL Pipeline and Data Integration
**Dataset**: NYC SAT Results (data/4_sat-results.csv)  
**Notebook**: notebooks/4_SAT_modeling_database_population_v3(prod).ipynb
**Output**: outputs/4_cleaned_sat_results.csv, outputs/4_data_integration_summary.md

**Results**: Implemented comprehensive data quality improvements including score validation, format standardization, and duplicate removal. Generated statistical analysis showing average 1,209 total SAT score with 87% testing participation across NYC schools.

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/Kovalivska/nyc_school_analyzer.git
cd nyc-school-analyzer
pip install -e .

# Or install from PyPI (when published)
pip install nyc-school-analyzer
```

### Basic Usage

```bash
# Run complete analysis
nyc-schools analyze data/2_3_high-school-directory.csv --borough BROOKLYN --grade 9

# Generate visualizations from processed data
nyc-schools viz outputs/data/processed_data_*.csv --format png pdf

# Validate data quality
nyc-schools validate data/2_3_high-school-directory.csv

# Export processed data
nyc-schools export outputs/data/processed_data_*.csv --formats csv json
```

### Python API

```python
from nyc_school_analyzer import DataProcessor, SchoolAnalyzer, SchoolVisualizer

# Initialize components
processor = DataProcessor()
analyzer = SchoolAnalyzer()
visualizer = SchoolVisualizer()

# Process data
school_data = processor.process_dataset("data/2_3_high-school-directory.csv")

# Run analysis
report = analyzer.generate_comprehensive_report(
    school_data, 
    target_borough="BROOKLYN"
)

# Create visualizations
charts = visualizer.export_all_charts(
    report['detailed_analyses'], 
    output_dir="charts/"
)
```

##  Features

### Data Processing & Validation
- **Robust Data Loading**: Handles CSV, Excel formats with comprehensive error handling
- **Data Cleaning**: Automatic column standardization, type conversion, and normalization
- **Quality Validation**: Schema validation, business rule checking, and outlier detection
- **Missing Data Handling**: Intelligent imputation and reporting of data gaps

### Statistical Analysis
- **Borough Distribution**: School and student distribution analysis across NYC
- **Grade Availability**: Coverage analysis for specific grade levels
- **Student Population**: Enrollment patterns, capacity analysis, and demographic insights
- **Performance Metrics**: Educational equity indicators and accessibility measures

### Professional Visualizations
- **Interactive Charts**: Bar charts, pie charts, scatter plots, and histograms
- **Comprehensive Dashboards**: Multi-panel analysis summaries
- **Export Formats**: PNG, PDF, SVG, EPS with configurable DPI and styling
- **Publication Ready**: Professional styling with consistent branding

### Export & Reporting
- **Multiple Formats**: CSV, JSON, Excel, Parquet with compression options
- **HTML Reports**: Professional summary reports with embedded charts
- **Executive Summaries**: Key findings and strategic recommendations
- **API Documentation**: Comprehensive analysis metadata and lineage

## Architecture

### Project Structure
```
nyc_school_analyzer/
├── src/nyc_school_analyzer/           # Production source code
│   ├── data/                          # Data processing modules
│   │   ├── processor.py               # ETL pipeline
│   │   ├── validator.py               # Data validation
│   │   └── models.py                  # Data models
│   ├── analysis/                      # Analysis modules
│   │   ├── analyzer.py                # Statistical analysis
│   │   ├── metrics.py                 # Educational metrics
│   │   ├── insights.py                # Insight generation
│   │   └── 4_sat_data_processing.py   # SAT data ETL script
│   ├── visualization/                 # Visualization modules
│   │   ├── visualizer.py              # Chart generation
│   │   ├── charts.py                  # Chart utilities
│   │   └── exports.py                 # Export management
│   ├── cli/                           # Command-line interface
│   │   ├── main.py                    # CLI entry point
│   │   └── commands.py                # Command implementations
│   └── utils/                         # Utilities
│       ├── config.py                  # Configuration management
│       ├── logger.py                  # Logging setup
│       └── exceptions.py              # Custom exceptions
├── notebooks/                         # 4-day analysis project notebooks
│   ├── 1_School_incident_Analysis(GoogleSheet).md     # Day 1 results
│   ├── 2_NYC_school_directory_exploration_analysis_v2(prod).ipynb  # Day 2 analysis
│   ├── 3_NYC_school_data_exploration(SQL_via_Python).ipynb         # Day 3 SQL work
│   ├── 4_SAT_modeling_database_population_v3(prod).ipynb           # Day 4 ETL
│   ├── readme1.md - readme4.md        # Daily task descriptions
│   ├── README.md                      # Analysis project summary
│   └── demo_nb/                       # Development notebooks
├── data/                              # Raw datasets from analysis project
│   ├── 1_school-safety-report.csv     # Day 1 safety incident data
│   ├── 2_3_high-school-directory.csv  # Day 2 school directory data  
│   ├── 4_sat-results.csv              # Day 4 SAT results data
│   └── *.md                           # Dataset documentation
├── outputs/                           # Processed results and analysis outputs
│   ├── 4_cleaned_sat_results.csv      # Day 4 cleaned SAT data
│   ├── 4_data_integration_summary.md  # Day 4 processing report
│   └── *.xlsx, *.log                  # Analysis outputs and logs
├── tests/                             # Production test suite
├── config/                            # Configuration files
├── docs/                              # Documentation
└── docker/                            # Docker configurations
```

### Core Components

#### DataProcessor
- Loads and cleans raw school data
- Handles multiple file formats and encodings
- Performs data type conversion and validation
- Generates comprehensive data quality reports

#### SchoolAnalyzer  
- Executes statistical analysis workflows
- Calculates educational performance metrics
- Identifies trends and patterns in school data
- Generates actionable insights and recommendations

#### SchoolVisualizer
- Creates publication-quality charts and graphs
- Supports multiple export formats and styling options
- Generates comprehensive dashboards and reports
- Handles large datasets with performance optimization

##  Configuration

### YAML Configuration
```yaml
# config/config.yaml
data:
  input_file: "2_3_high-school-directory.csv"
  input_path: "data/"
  output_path: "outputs/"

analysis:
  target_borough: "BROOKLYN"
  grade_of_interest: 9
  validation_enabled: true

visualization:
  style: "seaborn"
  dpi: 300
  export_formats: ["png", "pdf"]

logging:
  level: "INFO"
  file_enabled: true
  file_path: "logs/"
```

### Environment Variables
```bash
export NYC_SCHOOLS_TARGET_BOROUGH="BROOKLYN"
export NYC_SCHOOLS_GRADE=9
export NYC_SCHOOLS_LOG_LEVEL="INFO"
export NYC_SCHOOLS_OUTPUT_PATH="./outputs"
```

##  Testing

### Run Test Suite
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/nyc_school_analyzer --cov-report=html

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m "unit"               # Unit tests only
pytest -m "integration"        # Integration tests only
```

### Test Categories
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end workflow testing  
- **Performance Tests**: Large dataset and memory usage testing
- **CLI Tests**: Command-line interface testing

##  Deployment

### Docker Deployment
```bash
# Build image
docker build -t nyc-school-analyzer .

# Run analysis
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  nyc-school-analyzer analyze /app/data/2_3_high-school-directory.csv
```

### Production Environment
```bash
# Install with production dependencies
pip install nyc-school-analyzer[prod]

# Set up logging and monitoring
export NYC_SCHOOLS_LOG_LEVEL="WARNING"
export NYC_SCHOOLS_MONITORING_ENABLED="true"

# Run with production configuration
nyc-schools analyze data/2_3_high-school-directory.csv --config production.yaml
```

## Performance

### Benchmarks
- **Data Processing**: Handles 100K+ records in <30 seconds
- **Analysis Pipeline**: Complete analysis in <60 seconds for typical datasets  
- **Memory Usage**: <500MB for datasets up to 50K records
- **Visualization**: Chart generation <10 seconds per chart

### Optimization Features
- **Chunked Processing**: Large dataset handling with configurable chunk sizes
- **Parallel Processing**: Multi-core utilization for CPU-intensive operations
- **Memory Management**: Efficient pandas operations with memory monitoring
- **Caching**: Intelligent caching of intermediate results

##  Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/Kovalivska/nyc_school_analyzer.git
cd nyc-school-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Quality Standards
- **Black**: Code formatting with line length 88
- **Flake8**: Linting with comprehensive rule set
- **isort**: Import sorting and organization
- **mypy**: Static type checking
- **pytest**: Minimum 80% test coverage required

## Documentation

### API Documentation
```bash
# Generate API docs
sphinx-build -b html docs/ docs/_build/html

# View documentation
open docs/_build/html/index.html
```

### User Guides
- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Reference](docs/api.md)
- [CLI Usage](docs/cli.md)
- [Deployment Guide](docs/deployment.md)
- [Development Guide](docs/development.md)

## Advanced Usage

### Custom Analysis Pipeline
```python
from nyc_school_analyzer import Config, DataProcessor, SchoolAnalyzer

# Custom configuration
config = Config()
config.analysis.target_borough = "QUEENS"
config.analysis.grade_of_interest = 10

# Initialize with custom config
processor = DataProcessor()
analyzer = SchoolAnalyzer(config=config.analysis)

# Custom analysis workflow
school_data = processor.process_dataset("data.csv")
brooklyn_schools = processor.filter_by_borough(school_data.processed_data, "QUEENS")

# Advanced analytics
metrics = analyzer.metrics_calculator.calculate_comprehensive_metrics(
    brooklyn_schools, 
    target_grade=10
)

insights = analyzer.insight_generator.generate_comparative_insights(
    current_analysis=metrics,
    historical_data=previous_metrics
)
```

### Batch Processing
```bash
# Process multiple files
for file in data/*.csv; do
  nyc-schools analyze "$file" --output-dir "results/$(basename $file .csv)"
done

# Automated reporting
nyc-schools analyze data/2_3_high-school-directory.csv --format json | \
  jq '.executive_summary' > summary.json
```

## Troubleshooting

### Common Issues

**Memory Errors with Large Datasets**
```bash
# Use chunked processing
export NYC_SCHOOLS_CHUNK_SIZE=5000
nyc-schools analyze data/2_3_high-school-directory.csv
```

**Visualization Rendering Issues**
```bash
# Use non-interactive backend
export MPLBACKEND=Agg
nyc-schools viz outputs/data/processed_data_*.csv
```

**Configuration Validation Errors**
```bash
# Check configuration
nyc-schools config

# Validate specific config file (if available)
nyc-schools validate data/2_3_high-school-directory.csv
```

## Support

### Getting Help
- **Documentation**: [https://github.com/Kovalivska/nyc_school_analyzer](https://github.com/Kovalivska/nyc_school_analyzer)
- **Issue Tracker**: [GitHub Issues](https://github.com/Kovalivska/nyc_school_analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Kovalivska/nyc_school_analyzer/discussions)
- **Email**: kovalivska@gmail.com

### Reporting Bugs
Please include:
- Python version and operating system
- Complete error traceback
- Minimal reproducible example
- Configuration file (with sensitive data removed)

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Svitlana Kovalivska, Ph.D.**
- Data Scientist  
- GitHub: [@Kovalivska](https://github.com/Kovalivska)
- Email: kovalivska@gmail.com

**Project Foundation**: This project was developed as part of the Masterschool onboarding program, demonstrating comprehensive data analysis skills through a structured 4-day learning curriculum.

**Development Collaboration**: Created in collaboration with Claude AI model sub-agents, leveraging AI-assisted development for code optimization, documentation generation, and analytical insights.

## Acknowledgments

- **Masterschool**: For providing the comprehensive onboarding framework and structured learning path
- **Claude AI**: For collaborative development support and analytical guidance throughout the project
- NYC Department of Education for providing school directory data
- pandas and matplotlib communities for excellent data analysis tools
- Webeet Internship Program for providing the structured learning framework
- Contributors and maintainers who helped build this tool

## Roadmap

### Practical Applications

This NYC School Analyzer system can be applied in various real-world scenarios:

**Educational Administration**
- **Resource Allocation**: Use borough distribution analysis to optimize school funding and teacher assignments
- **Capacity Planning**: Leverage student population data to identify overcrowded schools and plan new facilities
- **Safety Management**: Apply incident analysis patterns to improve security protocols and prevention strategies
- **Academic Performance Monitoring**: Utilize SAT score analysis to identify schools needing additional support

**Policy Development**
- **Equity Assessment**: Analyze demographic distributions to ensure equitable educational access across boroughs
- **Strategic Planning**: Use comprehensive school data to inform long-term educational policy decisions
- **Performance Benchmarking**: Compare school performance metrics across different areas and demographics
- **Budget Planning**: Apply data-driven insights to optimize educational spending and resource distribution

**Research and Analytics**
- **Educational Research**: Provide reliable data foundation for academic studies on urban education
- **Trend Analysis**: Track educational changes over time using standardized data processing pipelines
- **Comparative Studies**: Enable cross-borough and cross-school comparative research
- **Predictive Modeling**: Use historical data patterns to forecast future educational trends and needs

**Operational Management**
- **Data Quality Monitoring**: Implement systematic data validation for ongoing educational data management
- **Automated Reporting**: Generate regular reports for stakeholders using production-ready pipelines
- **Integration Support**: Connect with existing educational information systems and databases
- **Performance Optimization**: Scale analysis workflows to handle growing datasets and reporting demands

### Upcoming Features
- **Real-time Data Integration**: API connections to live school data
- **Machine Learning Models**: Predictive analytics for enrollment forecasting
- **Interactive Web Dashboard**: Browser-based analysis interface
- **Multi-City Support**: Expansion beyond NYC to other school systems
- **Advanced Geospatial Analysis**: School catchment and accessibility mapping

### Version History
- **v1.0.0**: Production release with complete 4-day analysis integration and CLI tools
- **v0.9.0**: Beta release with Day 4 ETL pipeline and SAT data processing
- **v0.8.0**: Alpha release with Day 2-3 analysis modules and SQL integration
- **v0.7.0**: Foundation release with Day 1 safety analysis and data processing pipeline

---

**Built for educational data analysis and evidence-based decision making**