# NYC Schools Data Analysis Project

## Project Overview

This project represents a comprehensive 4-day data analysis internship program focused on analyzing New York City school data. The project demonstrates proficiency in data exploration, Python programming, SQL database operations, and ETL processes using real NYC educational datasets.

## Timeline and Task Structure

### Day 1: Data Exploration and Google Sheets Analysis
**Focus**: Basic data exploration and spreadsheet analysis
**Dataset**: School Safety Report Data (1_school-safety-report.csv)

**Objectives**:
- Load and preview the school safety incident dataset
- Perform basic data exploration using Google Sheets
- Answer exploratory questions about school safety incidents
- Submit findings in a markdown format with Google Sheets links

**Results Achieved**:
- Analyzed 6,310 school safety incidents across 1,893 unique schools
- Identified "nocrim_n" (non-criminal incidents) as the most frequent incident type
- Determined that Bronx accounts for 28.2% of reported incidents
- Discovered that Brooklyn and Bronx together represent 61% of all violent crimes
- Found correlation between building size (number of schools) and incident frequency

### Day 2: Python Data Analysis with Pandas
**Focus**: Python programming for data analysis and visualization
**Dataset**: NYC High School Directory (2_3_high-school-directory.csv)

**Objectives**:
- Use Jupyter Notebook for comprehensive data analysis
- Clean and preprocess column names
- Perform filtering and group-based summaries
- Create visualizations using matplotlib and seaborn
- Generate insights from high school directory data

**Results Achieved**:
- Processed 435 high schools across NYC with 69 attributes per school
- Implemented comprehensive data cleaning pipeline with column standardization
- Brooklyn identified as largest borough with 121 schools (27.8% of total)
- Analyzed student population distribution: average 699 students per Brooklyn school
- Created professional visualizations for borough distribution and student demographics
- Implemented quality validation checks ensuring 100% data consistency

### Day 3: SQL Database Operations via Python
**Focus**: PostgreSQL database integration and SQL query execution
**Datasets**: High School Directory, School Demographics, School Safety Report

**Objectives**:
- Connect to PostgreSQL database using Python
- Write complex SQL queries with joins across multiple tables
- Analyze school patterns across boroughs and demographics
- Import query results into pandas DataFrames for further analysis

**Results Achieved**:
- Successfully established database connections to NYC schools PostgreSQL instance
- Executed multi-table JOIN queries across 3 normalized database tables
- Analyzed school distribution: Brooklyn (121), Bronx (118), Manhattan (106), Queens (80), Staten Island (10)
- Identified data quality issues with ELL (English Language Learners) demographic data
- Discovered incomplete special education data coverage across boroughs
- Demonstrated proficiency in combining SQL data access with Python analysis workflows

### Day 4: ETL Pipeline and Data Integration
**Focus**: Data integration, schema design, and production ETL processes
**Dataset**: NYC SAT Results (4_sat-results.csv)

**Objectives**:
- Inspect and understand messy real-world dataset structure
- Design data cleaning and preprocessing pipeline
- Integrate new dataset into existing PostgreSQL database
- Create production-ready Python scripts for data loading

**Results Achieved**:
- Processed 493 SAT result records, cleaned to 478 valid entries (15 duplicates removed)
- Implemented comprehensive data quality improvements:
  - Score validation ensuring 200-800 SAT range compliance
  - Standardized percentage formats from strings to decimals
  - Handled suppressed data ('s' values) appropriately
  - Identified and corrected 5 invalid math scores
- Generated statistical analysis of SAT performance:
  - 416 schools (87%) with complete SAT data
  - Average total SAT score: 1,209 points (competitive performance)
  - Score distribution: 887-2,096 point range with 175.3 standard deviation
- Created production ETL script with database integration capabilities
- Established data validation and quality monitoring processes

## Technical Implementation

### Technology Stack
- **Programming Language**: Python 3.11+
- **Data Analysis**: pandas, numpy, matplotlib, seaborn
- **Database**: PostgreSQL with psycopg2 connectivity
- **Development Environment**: Jupyter Notebook
- **Version Control**: Git with structured branching strategy

### Key Components

#### Data Processing Pipeline
- Automated column name standardization
- Comprehensive data type validation
- Missing value detection and handling
- Duplicate record identification and removal
- Statistical outlier detection

#### Database Integration
- PostgreSQL connection management
- Multi-table JOIN operations
- Schema design and validation
- ETL script automation
- Data integrity checks

#### Visualization and Reporting
- Professional matplotlib/seaborn visualizations
- Borough distribution analysis charts
- Student population distribution plots
- Executive summary reporting
- Quality validation dashboards

### Data Quality Metrics
- **Day 1**: 6,310 incident records processed with 100% coverage
- **Day 2**: 435 schools analyzed with 25.8% missing data rate, 100% validation passed
- **Day 3**: Multi-table queries across 3 database tables with JOIN operations
- **Day 4**: 97% data retention rate (478/493) after quality improvements

## Project Structure

```
nyc_school_analyzer/
├── data/                          # Raw datasets
├── notebooks/                     # Jupyter analysis notebooks
│   ├── 1_School_incident_Analysis(GoogleSheet).md
│   ├── 2_NYC_school_directory_exploration_analysis_v2(prod).ipynb
│   ├── 3_NYC_school_data_exploration(SQL_via_Python).ipynb
│   └── 4_SAT_modeling_database_population_v3(prod).ipynb
├── outputs/                       # Processed results and reports
├── src/nyc_school_analyzer/       # Production Python modules
│   ├── analysis/                  # Analysis algorithms
│   ├── data/                     # Data processing utilities
│   └── visualization/            # Charting and reporting
└── config/                       # Configuration management
```

## Key Findings and Insights

### School Distribution Analysis
- Brooklyn leads with 121 schools, followed by Bronx (118) and Manhattan (106)
- Staten Island significantly underrepresented with only 10 schools
- Student population averages 699 per school in Brooklyn with high variance

### Academic Performance Insights
- NYC SAT scores average 1,209 points, indicating competitive academic standards
- 87% of schools provide complete SAT data, suggesting good testing participation
- Performance distribution shows healthy spread across achievement levels

### Data Quality Assessment
- Identified significant gaps in demographic data collection (ELL statistics)
- Safety incident reporting shows comprehensive coverage across all boroughs
- Database integration reveals need for improved data standardization

### Operational Recommendations
1. **Educational Access**: Brooklyn provides comprehensive grade coverage
2. **Resource Planning**: Focus capacity planning on high-enrollment boroughs  
3. **Data Enhancement**: Improve demographic data collection consistency
4. **Quality Monitoring**: Implement systematic data validation processes

## Technical Skills Demonstrated

### Data Analysis Proficiency
- Advanced pandas DataFrame manipulation
- Complex SQL query development
- Statistical analysis and interpretation
- Data visualization best practices

### Engineering Capabilities  
- ETL pipeline design and implementation
- Database schema design and optimization
- Production code development with logging and error handling
- Git version control and collaborative development

### Analytical Thinking
- Root cause analysis for data quality issues
- Strategic recommendations based on quantitative findings
- Cross-dataset integration and validation
- Performance metric design and monitoring

This project successfully demonstrates comprehensive data analysis capabilities from basic exploration through production-ready ETL systems, showcasing readiness for advanced data science and analytics roles.
