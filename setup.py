"""
NYC School Analyzer Package Setup
Production-ready application for analyzing NYC high school data.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="nyc-school-analyzer",
    version="1.0.0",
    description="Production-ready NYC High School Directory Analysis Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NYC School Analytics Team",
    author_email="analytics@nycschools.edu",
    url="https://github.com/nyc-schools/nyc-school-analyzer",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.18.0",
        ],
        "cloud": [
            "boto3>=1.26.0",
            "google-cloud-storage>=2.7.0",
        ],
        "database": [
            "sqlalchemy>=1.4.0",
            "psycopg2-binary>=2.9.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "nyc-schools=nyc_school_analyzer.cli.main:main",
            "school-analyzer=nyc_school_analyzer.cli.main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Government",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Spreadsheet",
    ],
    
    keywords="nyc schools education analysis data visualization borough statistics",
    
    project_urls={
        "Bug Reports": "https://github.com/nyc-schools/nyc-school-analyzer/issues",
        "Source": "https://github.com/nyc-schools/nyc-school-analyzer",
        "Documentation": "https://nyc-school-analyzer.readthedocs.io/",
    },
    
    include_package_data=True,
    package_data={
        "nyc_school_analyzer": [
            "config/*.yaml",
            "templates/*.html",
            "static/*",
        ],
    },
    
    zip_safe=False,
)