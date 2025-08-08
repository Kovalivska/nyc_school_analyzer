"""
CLI command implementations for NYC School Analyzer.

Contains the actual implementation functions for all CLI commands
with comprehensive error handling and professional output.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import traceback

from ..data.processor import DataProcessor
from ..data.validator import DataValidator
from ..analysis.analyzer import SchoolAnalyzer
from ..visualization.visualizer import SchoolVisualizer
from ..utils.config import Config
from ..utils.logger import get_logger, create_analysis_logger
from ..utils.exceptions import (
    NYCSchoolAnalyzerError,
    DataProcessingError,
    AnalysisError,
    VisualizationError,
    ExportError
)

logger = get_logger(__name__)


def analyze(
    input_file: Path,
    config: Config,
    export_formats: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive school data analysis.
    
    Args:
        input_file: Path to input data file
        config: Application configuration
        export_formats: List of export formats
        
    Returns:
        Dictionary with analysis results and metadata
    """
    try:
        logger.info(f"Starting comprehensive analysis of {input_file}")
        
        # Create output directories
        config.create_output_directories()
        output_paths = config.get_output_paths()
        
        # Create analysis-specific logger
        analysis_logger = create_analysis_logger(
            "comprehensive_analysis",
            output_paths['logs']
        )
        
        # Initialize components
        processor = DataProcessor()
        analyzer = SchoolAnalyzer(processor, config.analysis)
        visualizer = SchoolVisualizer(config.visualization)
        
        # Process data
        analysis_logger.info("Loading and processing data...")
        school_data = processor.process_dataset(
            input_file,
            validate=config.analysis.validation_enabled
        )
        
        # Run comprehensive analysis
        analysis_logger.info("Running comprehensive analysis...")
        report = analyzer.generate_comprehensive_report(
            school_data,
            target_borough=config.analysis.target_borough
        )
        
        # Generate visualizations
        analysis_logger.info("Generating visualizations...")
        chart_files = visualizer.export_all_charts(
            report['detailed_analyses'],
            output_paths['charts'],
            config.visualization.export_formats
        )
        
        # Export data
        analysis_logger.info("Exporting analysis results...")
        export_files = _export_analysis_results(
            report,
            school_data,
            output_paths['data'],
            export_formats or ['csv', 'json']
        )
        
        # Generate summary report
        analysis_logger.info("Generating summary report...")
        summary_file = _generate_summary_report(
            report,
            output_paths['reports'] / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        
        # Calculate results
        total_charts = sum(len(files) for files in chart_files.values())
        total_exports = len(export_files)
        
        result = {
            'success': True,
            'output_dir': str(output_paths['base']),
            'charts_count': total_charts,
            'exports_count': total_exports,
            'summary_report': str(summary_file),
            'analysis_metadata': {
                'input_file': str(input_file),
                'target_borough': config.analysis.target_borough,
                'grade_of_interest': config.analysis.grade_of_interest,
                'total_schools_analyzed': report['metadata']['total_schools_analyzed'],
                'analysis_timestamp': report['metadata']['report_generated_at'].isoformat(),
            }
        }
        
        analysis_logger.info("Analysis completed successfully")
        logger.info(f"Analysis completed: {total_charts} charts, {total_exports} exports")
        
        return result
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def visualize(
    data_file: Path,
    output_dir: Path,
    formats: List[str],
    config: Config
) -> Dict[str, Any]:
    """
    Generate visualizations from processed data.
    
    Args:
        data_file: Path to processed data file
        output_dir: Output directory for charts
        formats: List of chart formats
        config: Application configuration
        
    Returns:
        Dictionary with visualization results
    """
    try:
        logger.info(f"Generating visualizations from {data_file}")
        
        # Load data
        if data_file.suffix.lower() == '.json':
            with open(data_file, 'r') as f:
                data = json.load(f)
        elif data_file.suffix.lower() == '.csv':
            data = pd.read_csv(data_file)
        else:
            raise VisualizationError(f"Unsupported data file format: {data_file.suffix}")
        
        # Initialize visualizer
        visualizer = SchoolVisualizer(config.visualization)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate charts based on data type
        if isinstance(data, dict) and 'detailed_analyses' in data:
            # This is analysis results JSON
            chart_files = visualizer.export_all_charts(
                data['detailed_analyses'],
                output_dir,
                formats
            )
        elif isinstance(data, pd.DataFrame):
            # This is raw data CSV
            # Create basic visualizations
            chart_files = _create_basic_visualizations(data, visualizer, output_dir, formats)
        else:
            raise VisualizationError("Unsupported data format for visualization")
        
        total_charts = sum(len(files) for files in chart_files.values())
        
        result = {
            'success': True,
            'output_dir': str(output_dir),
            'charts_count': total_charts,
            'chart_files': {name: [str(f) for f in files] for name, files in chart_files.items()}
        }
        
        logger.info(f"Generated {total_charts} visualizations")
        return result
        
    except Exception as e:
        error_msg = f"Visualization failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def export(
    input_file: Path,
    output_dir: Path,
    formats: List[str],
    compress: bool,
    config: Config
) -> Dict[str, Any]:
    """
    Export processed data in various formats.
    
    Args:
        input_file: Path to input data file
        output_dir: Output directory
        formats: Export formats
        compress: Whether to compress files
        config: Application configuration
        
    Returns:
        Dictionary with export results
    """
    try:
        logger.info(f"Exporting data from {input_file}")
        
        # Load data
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
        else:
            raise ExportError(f"Unsupported input file format: {input_file.suffix}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate base filename
        base_name = input_file.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        export_files = []
        
        # Export in requested formats
        for format_type in formats:
            output_file = output_dir / f"{base_name}_export_{timestamp}.{format_type}"
            
            if format_type == 'csv':
                df.to_csv(output_file, index=False)
            elif format_type == 'json':
                df.to_json(output_file, orient='records', indent=2)
            elif format_type == 'excel':
                df.to_excel(output_file, index=False)
            elif format_type == 'parquet':
                df.to_parquet(output_file, index=False)
            else:
                logger.warning(f"Unsupported export format: {format_type}")
                continue
            
            # Compress if requested
            if compress and format_type != 'parquet':  # Parquet is already compressed
                import gzip
                compressed_file = output_file.with_suffix(f"{output_file.suffix}.gz")
                
                with open(output_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # Remove uncompressed file
                output_file.unlink()
                export_files.append(compressed_file)
            else:
                export_files.append(output_file)
        
        result = {
            'success': True,
            'output_dir': str(output_dir),
            'files_count': len(export_files),
            'export_files': [str(f) for f in export_files]
        }
        
        logger.info(f"Exported {len(export_files)} files")
        return result
        
    except Exception as e:
        error_msg = f"Export failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def validate(
    input_file: Path,
    output_file: Optional[Path],
    strict: bool,
    config: Config
) -> Dict[str, Any]:
    """
    Validate school data file.
    
    Args:
        input_file: Path to input data file
        output_file: Optional output file for report
        strict: Whether to use strict validation
        config: Application configuration
        
    Returns:
        Dictionary with validation results
    """
    try:
        logger.info(f"Validating data file: {input_file}")
        
        # Load data
        processor = DataProcessor()
        df = processor.load_dataset(input_file)
        df = processor.clean_column_names(df)
        
        # Initialize validator
        validator_config = config.quality.__dict__.copy()
        if strict:
            validator_config.update({
                'min_data_completeness': 0.9,
                'max_missing_percentage': 10.0,
                'outlier_threshold': 2.0
            })
        
        validator = DataValidator(validator_config)
        
        # Run validation
        validation_result = validator.validate_dataset(df)
        
        # Generate validation report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'strict_mode': strict,
            'is_valid': validation_result.is_valid,
            'records_count': len(df),
            'errors_count': len(validation_result.errors),
            'warnings_count': len(validation_result.warnings),
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'metrics': validation_result.metrics
        }
        
        # Save report if output file specified
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix.lower() == '.json':
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
            else:
                # Generate HTML report
                _generate_validation_html_report(report, output_file)
        
        result = {
            'success': True,
            'is_valid': validation_result.is_valid,
            'records_count': len(df),
            'errors_count': len(validation_result.errors),
            'warnings_count': len(validation_result.warnings),
            'report_file': str(output_file) if output_file else None
        }
        
        logger.info(
            f"Validation completed: {'PASSED' if validation_result.is_valid else 'FAILED'} "
            f"({len(validation_result.errors)} errors, {len(validation_result.warnings)} warnings)"
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


# Private helper functions

def _export_analysis_results(
    report: Dict[str, Any],
    school_data,
    output_dir: Path,
    formats: List[str]
) -> List[Path]:
    """Export analysis results in various formats."""
    export_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for format_type in formats:
        if format_type == 'json':
            # Export full report as JSON
            json_file = output_dir / f"analysis_report_{timestamp}.json"
            
            # Convert non-serializable objects
            serializable_report = _make_json_serializable(report)
            
            with open(json_file, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            export_files.append(json_file)
        
        elif format_type == 'csv':
            # Export processed data as CSV
            csv_file = output_dir / f"processed_data_{timestamp}.csv"
            school_data.processed_data.to_csv(csv_file, index=False)
            export_files.append(csv_file)
        
        elif format_type == 'excel':
            # Export multiple sheets in Excel
            excel_file = output_dir / f"analysis_results_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_file) as writer:
                # Write processed data
                school_data.processed_data.to_excel(writer, sheet_name='Data', index=False)
                
                # Write summary statistics if available
                if 'detailed_analyses' in report:
                    analyses = report['detailed_analyses']
                    
                    # Borough distribution
                    if 'borough_distribution' in analyses:
                        borough_data = analyses['borough_distribution']
                        if 'school_counts' in borough_data:
                            pd.Series(borough_data['school_counts']).to_excel(
                                writer, sheet_name='Borough_Distribution'
                            )
            
            export_files.append(excel_file)
    
    return export_files


def _generate_summary_report(report: Dict[str, Any], output_file: Path) -> Path:
    """Generate HTML summary report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NYC School Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>NYC School Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            {_format_executive_summary(report.get('executive_summary', {}))}
        </div>
        
        <div class="section">
            <h2>Key Findings</h2>
            {_format_key_findings(report.get('executive_summary', {}).get('key_findings', []))}
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            {_format_recommendations(report.get('recommendations', {}))}
        </div>
        
        <div class="section">
            <h2>Analysis Metadata</h2>
            {_format_metadata(report.get('metadata', {}))}
        </div>
    </body>
    </html>
    """
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file


def _generate_validation_html_report(report: Dict[str, Any], output_file: Path):
    """Generate HTML validation report."""
    status_color = "#d4edda" if report['is_valid'] else "#f8d7da"
    status_text = "PASSED" if report['is_valid'] else "FAILED"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: {status_color}; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .error {{ color: #721c24; background-color: #f8d7da; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .warning {{ color: #856404; background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Validation Report - {status_text}</h1>
            <p>Validation completed on: {report['validation_timestamp']}</p>
            <p>Input file: {report['input_file']}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <div class="metric">Records validated: {report['records_count']}</div>
            <div class="metric">Errors found: {report['errors_count']}</div>
            <div class="metric">Warnings found: {report['warnings_count']}</div>
        </div>
        
        <div class="section">
            <h2>Errors</h2>
            {_format_validation_issues(report['errors'], 'error')}
        </div>
        
        <div class="section">
            <h2>Warnings</h2>
            {_format_validation_issues(report['warnings'], 'warning')}
        </div>
        
        <div class="section">
            <h2>Metrics</h2>
            {_format_validation_metrics(report.get('metrics', {}))}
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)


def _create_basic_visualizations(
    df: pd.DataFrame, 
    visualizer: SchoolVisualizer, 
    output_dir: Path, 
    formats: List[str]
) -> Dict[str, List[Path]]:
    """Create basic visualizations from raw data."""
    chart_files = {}
    
    # Borough distribution if city column exists
    if 'city' in df.columns:
        borough_counts = df['city'].value_counts()
        fig = visualizer.create_borough_distribution_chart(borough_counts)
        
        chart_files['borough_distribution'] = []
        for fmt in formats:
            file_path = output_dir / f"borough_distribution.{fmt}"
            visualizer.export_manager.save_figure(fig, file_path)
            chart_files['borough_distribution'].append(file_path)
    
    return chart_files


def _make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    import numpy as np
    import pandas as pd
    
    if hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (pd.Index, pd.CategoricalIndex)):
        return obj.tolist()
    elif hasattr(obj, 'dtype') and 'object' in str(obj.dtype):
        # Handle pandas object types
        try:
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        except:
            return str(obj)
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'to_dict'):  # custom objects with to_dict method
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # custom objects
        return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()}
    else:
        try:
            # Try to convert to basic types
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            return obj
        except:
            return str(obj)


def _format_executive_summary(summary: Dict[str, Any]) -> str:
    """Format executive summary for HTML."""
    if not summary:
        return "<p>No executive summary available.</p>"
    
    html = ""
    for key, value in summary.items():
        if key == 'key_findings':
            continue  # Handle separately
        html += f"<div class='metric'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
    
    return html


def _format_key_findings(findings: List[Dict[str, Any]]) -> str:
    """Format key findings for HTML."""
    if not findings:
        return "<p>No key findings available.</p>"
    
    html = "<ul>"
    for finding in findings:
        category = finding.get('category', 'General')
        finding_text = finding.get('finding', 'No description')
        detail = finding.get('detail', '')
        
        html += f"<li><strong>{category}:</strong> {finding_text}"
        if detail:
            html += f" ({detail})"
        html += "</li>"
    
    html += "</ul>"
    return html


def _format_recommendations(recommendations: Dict[str, List[str]]) -> str:
    """Format recommendations for HTML."""
    if not recommendations:
        return "<p>No recommendations available.</p>"
    
    html = ""
    for category, items in recommendations.items():
        html += f"<h3>{category.replace('_', ' ').title()}</h3><ul>"
        for item in items:
            html += f"<li>{item}</li>"
        html += "</ul>"
    
    return html


def _format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata for HTML."""
    html = ""
    for key, value in metadata.items():
        html += f"<div class='metric'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
    
    return html


def _format_validation_issues(issues: List[str], issue_type: str) -> str:
    """Format validation issues for HTML."""
    if not issues:
        return f"<p>No {issue_type}s found.</p>"
    
    html = ""
    for issue in issues:
        html += f"<div class='{issue_type}'>{issue}</div>"
    
    return html


def _format_validation_metrics(metrics: Dict[str, Any]) -> str:
    """Format validation metrics for HTML."""
    html = ""
    for key, value in metrics.items():
        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
        html += f"<div class='metric'><strong>{key.replace('_', ' ').title()}:</strong> {formatted_value}</div>"
    
    return html