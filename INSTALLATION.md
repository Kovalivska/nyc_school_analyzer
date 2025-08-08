# Installation and Quick Start Guide

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB RAM (4GB recommended)
- **Storage**: 1GB free space for installation and outputs

### Required Data
- NYC High School Directory CSV file (`high-school-directory.csv`)
- Place the data file in the `data/` directory or specify custom path

## Installation Options

### Option 1: Standard Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/nyc-schools/nyc-school-analyzer.git
cd nyc-school-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -r requirements-minimal.txt
pip install -e .
```

### Option 2: Development Installation

```bash
# Clone and install with development dependencies
git clone https://github.com/nyc-schools/nyc-school-analyzer.git
cd nyc-school-analyzer

# Use Makefile for complete setup
make setup-dev
```

### Option 3: Docker Installation

```bash
# Clone repository
git clone https://github.com/nyc-schools/nyc-school-analyzer.git
cd nyc-school-analyzer

# Build Docker image
make docker-build

# Or use docker-compose
docker-compose build
```

## Quick Start

### 1. Prepare Your Data

```bash
# Create data directory and copy your CSV file
mkdir -p data
cp /path/to/your/high-school-directory.csv data/
```

### 2. Run Basic Analysis

```bash
# Complete analysis for Brooklyn schools, Grade 9
nyc-schools analyze data/high-school-directory.csv --borough BROOKLYN --grade 9

# Output will be saved to outputs/ directory
```

### 3. View Results

```bash
# Check generated files
ls -la outputs/
ls -la outputs/charts/
ls -la outputs/reports/
```

## Usage Examples

### Command Line Interface

```bash
# Complete analysis with custom output directory
nyc-schools analyze data/schools.csv \
  --output-dir results/brooklyn_analysis \
  --borough BROOKLYN \
  --grade 9 \
  --format csv json excel

# Generate only visualizations
nyc-schools visualize processed_data.csv \
  --output-dir charts/ \
  --format png pdf svg \
  --dpi 300

# Validate data quality
nyc-schools validate data/schools.csv \
  --output-file validation_report.html \
  --strict

# Export processed data
nyc-schools export processed_data.csv \
  --output-dir exports/ \
  --formats csv json parquet \
  --compress
```

### Python API

```python
from nyc_school_analyzer import Config, DataProcessor, SchoolAnalyzer, SchoolVisualizer

# Load configuration
config = Config('config/config.yaml')

# Initialize components
processor = DataProcessor()
analyzer = SchoolAnalyzer(config=config.analysis)
visualizer = SchoolVisualizer(config.visualization)

# Process data
school_data = processor.process_dataset('data/high-school-directory.csv')

# Run analysis
report = analyzer.generate_comprehensive_report(
    school_data, 
    target_borough='BROOKLYN'
)

# Generate visualizations
charts = visualizer.export_all_charts(
    report['detailed_analyses'],
    output_dir='outputs/charts'
)

print(f"Analysis complete: {len(charts)} chart types generated")
```

### Docker Usage

```bash
# Run analysis in Docker container
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
  nyc-school-analyzer analyze /app/data/high-school-directory.csv \
  --borough BROOKLYN --grade 9

# Using docker-compose
docker-compose run nyc-school-analyzer analyze \
  /app/data/high-school-directory.csv --borough QUEENS --grade 10
```

## Configuration

### Environment Variables

```bash
# Set common configuration via environment
export NYC_SCHOOLS_TARGET_BOROUGH="BROOKLYN"
export NYC_SCHOOLS_GRADE=9
export NYC_SCHOOLS_LOG_LEVEL="INFO"
export NYC_SCHOOLS_OUTPUT_PATH="./results"
```

### Configuration File

Create `config/custom_config.yaml`:

```yaml
data:
  input_file: "my-school-data.csv"
  output_path: "custom_outputs/"

analysis:
  target_borough: "QUEENS"
  grade_of_interest: 10
  validation_enabled: true

visualization:
  style: "ggplot"
  dpi: 300
  export_formats: ["png", "pdf", "svg"]

logging:
  level: "DEBUG"
  file_enabled: true
```

Use custom configuration:

```bash
nyc-schools --config config/custom_config.yaml analyze data/schools.csv
```

## Verification

### Test Installation

```bash
# Run basic functionality test
python -c "
import nyc_school_analyzer
print(f'NYC School Analyzer v{nyc_school_analyzer.__version__} installed successfully')
"

# Run CLI help
nyc-schools --help
```

### Run Test Suite

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/nyc_school_analyzer --cov-report=html
```

### Sample Analysis

```bash
# Use Makefile for sample analysis (requires data file)
make sample-analysis
```

## Troubleshooting

### Common Issues

### Common Issues

**1. Import Error: Module not found**
```bash
# Ensure package is installed
pip install -e .

# Check Python path
echo $PYTHONPATH
```

**2. Permission Error on Windows**
```bash
# Run as administrator or use:
pip install --user -e .
```

**3. Python Version Compatibility Issues**
```bash
# For Python 3.12+ use minimal requirements first
pip install -r requirements-minimal.txt
pip install -e .

# Alternative: use conda environment with specific Python version
conda create -n nyc-schools python=3.11
conda activate nyc-schools
pip install -r requirements-minimal.txt
pip install -e .
```

**4. Memory Error with Large Datasets**
```bash
# Use chunked processing
export NYC_SCHOOLS_CHUNK_SIZE=5000
nyc-schools analyze large_dataset.csv
```

**4. Matplotlib Backend Error**
```bash
# Use non-interactive backend
export MPLBACKEND=Agg
nyc-schools visualize data.csv
```

**5. Docker Permission Issues**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER outputs/
```

### Getting Help

- **Documentation**: Check `README.md` and `docs/` directory
- **Configuration**: Run `nyc-schools config` to see current settings
- **Validation**: Run `nyc-schools validate data.csv` to check data quality
- **Verbose Output**: Use `--verbose` flag for detailed logging
- **Debug Mode**: Set `NYC_SCHOOLS_LOG_LEVEL=DEBUG`

### Performance Optimization

```bash
# For large datasets
export NYC_SCHOOLS_CHUNK_SIZE=10000
export NYC_SCHOOLS_MEMORY_LIMIT_MB=2048
export NYC_SCHOOLS_MAX_WORKERS=4

# Disable validation for speed
nyc-schools analyze data.csv --no-validate
```

## Next Steps

1. **Explore Results**: Check the generated outputs in `outputs/` directory
2. **Customize Analysis**: Modify configuration files for your needs
3. **Integrate**: Use the Python API in your existing workflows
4. **Scale**: Use Docker for production deployments
5. **Extend**: Add custom analysis modules as needed

For advanced usage, see the full documentation in the `docs/` directory or visit the project repository.