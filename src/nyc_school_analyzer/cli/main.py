"""
Main CLI interface for NYC School Analyzer.

Provides command-line interface for running school data analysis
with professional output and comprehensive error handling.
"""

import click
import sys
from pathlib import Path
from typing import Optional
import traceback

from ..utils.config import Config
from ..utils.logger import setup_logging, get_logger
from ..utils.exceptions import NYCSchoolAnalyzerError
from .commands import analyze, visualize, export, validate


@click.group(name='nyc-schools')
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    default='INFO',
    help='Set logging level'
)
@click.option(
    '--log-file',
    type=click.Path(path_type=Path),
    help='Path to log file'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress non-error output'
)
@click.pass_context
def cli(ctx, config, log_level, log_file, verbose, quiet):
    """
    NYC School Analyzer - Production-ready school data analysis tool.
    
    Analyze NYC high school data with comprehensive statistics,
    visualizations, and export capabilities.
    """
    # Adjust log level based on flags
    if verbose:
        log_level = 'DEBUG'
    elif quiet:
        log_level = 'ERROR'
    
    # Setup logging
    log_config = setup_logging(
        level=log_level,
        log_file=str(log_file) if log_file else None,
        enable_colors=not quiet
    )
    
    # Initialize configuration
    try:
        app_config = Config(config_file=config)
        validation_errors = app_config.validate()
        
        if validation_errors:
            logger = get_logger(__name__)
            logger.error("Configuration validation errors:")
            for section, errors in validation_errors.items():
                for error in errors:
                    logger.error(f"  {section}: {error}")
            
            if not ctx.obj:
                ctx.obj = {}
            ctx.obj['config_errors'] = validation_errors
        
        # Store config in context
        if not ctx.obj:
            ctx.obj = {}
        ctx.obj['config'] = app_config
        ctx.obj['log_config'] = log_config
        
    except Exception as e:
        click.echo(f"Error: Failed to initialize configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for results'
)
@click.option(
    '--borough', '-b',
    help='Target borough for analysis'
)
@click.option(
    '--grade', '-g',
    type=int,
    help='Grade of interest for analysis'
)
@click.option(
    '--format',
    'export_formats',
    multiple=True,
    type=click.Choice(['csv', 'json', 'excel', 'html']),
    help='Export formats (can be specified multiple times)'
)
@click.option(
    '--no-validate',
    is_flag=True,
    help='Skip data validation'
)
@click.pass_context
def analyze_command(ctx, input_file, output_dir, borough, grade, export_formats, no_validate):
    """
    Run comprehensive school data analysis.
    
    INPUT_FILE: Path to the school data CSV file
    """
    try:
        config = ctx.obj['config']
        
        # Override config with command line options
        if borough:
            config.analysis.target_borough = borough.upper()
        if grade is not None:
            config.analysis.grade_of_interest = grade
        if output_dir:
            config.data.output_path = str(output_dir)
        if export_formats:
            # This would need to be handled in the analyze function
            pass
        if no_validate:
            config.analysis.validation_enabled = False
        
        # Run analysis
        result = analyze(
            input_file=input_file,
            config=config,
            export_formats=list(export_formats) if export_formats else None
        )
        
        if result['success']:
            click.echo(f"✅ Analysis completed successfully!")
            click.echo(f"📊 Results saved to: {result['output_dir']}")
            click.echo(f"📈 Generated {result['charts_count']} visualizations")
            click.echo(f"📄 Exported {result['exports_count']} data files")
        else:
            click.echo(f"❌ Analysis failed: {result['error']}", err=True)
            sys.exit(1)
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Analysis command failed: {e}")
        if ctx.obj.get('log_config', {}).get('level') == 'DEBUG':
            logger.error(traceback.format_exc())
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='outputs/charts',
    help='Output directory for charts'
)
@click.option(
    '--format',
    'chart_formats',
    multiple=True,
    type=click.Choice(['png', 'pdf', 'svg', 'eps']),
    default=['png'],
    help='Chart formats (can be specified multiple times)'
)
@click.option(
    '--dpi',
    type=int,
    help='DPI for raster formats'
)
@click.option(
    '--style',
    type=click.Choice(['seaborn', 'ggplot', 'bmh', 'classic']),
    help='Visualization style'
)
@click.pass_context
def visualize_command(ctx, data_file, output_dir, chart_formats, dpi, style):
    """
    Generate visualizations from processed data.
    
    DATA_FILE: Path to processed data file
    """
    try:
        config = ctx.obj['config']
        
        # Override config with command line options
        if dpi:
            config.visualization.dpi = dpi
        if style:
            config.visualization.style = style
        
        # Run visualization
        result = visualize(
            data_file=data_file,
            output_dir=output_dir,
            formats=list(chart_formats),
            config=config
        )
        
        if result['success']:
            click.echo(f"✅ Visualizations created successfully!")
            click.echo(f"📊 Generated {result['charts_count']} charts")
            click.echo(f"💾 Saved to: {result['output_dir']}")
        else:
            click.echo(f"❌ Visualization failed: {result['error']}", err=True)
            sys.exit(1)
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Visualization command failed: {e}")
        click.echo(f"❌ Visualization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='outputs/exports',
    help='Output directory for exports'
)
@click.option(
    '--format',
    'export_formats',
    multiple=True,
    type=click.Choice(['csv', 'json', 'excel', 'parquet']),
    default=['csv', 'json'],
    help='Export formats (can be specified multiple times)'
)
@click.option(
    '--compress',
    is_flag=True,
    help='Compress exported files'
)
@click.pass_context
def export_command(ctx, input_file, output_dir, export_formats, compress):
    """
    Export processed data in various formats.
    
    INPUT_FILE: Path to processed data file
    """
    try:
        config = ctx.obj['config']
        
        # Run export
        result = export(
            input_file=input_file,
            output_dir=output_dir,
            formats=list(export_formats),
            compress=compress,
            config=config
        )
        
        if result['success']:
            click.echo(f"✅ Export completed successfully!")
            click.echo(f"📄 Exported {result['files_count']} files")
            click.echo(f"💾 Saved to: {result['output_dir']}")
        else:
            click.echo(f"❌ Export failed: {result['error']}", err=True)
            sys.exit(1)
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Export command failed: {e}")
        click.echo(f"❌ Export failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-file', '-o',
    type=click.Path(path_type=Path),
    help='Output file for validation report'
)
@click.option(
    '--strict',
    is_flag=True,
    help='Use strict validation rules'
)
@click.pass_context
def validate_command(ctx, input_file, output_file, strict):
    """
    Validate school data file.
    
    INPUT_FILE: Path to school data file to validate
    """
    try:
        config = ctx.obj['config']
        
        # Run validation
        result = validate(
            input_file=input_file,
            output_file=output_file,
            strict=strict,
            config=config
        )
        
        if result['is_valid']:
            click.echo(f"✅ Data validation passed!")
            click.echo(f"📊 Validated {result['records_count']} records")
            if result['warnings_count'] > 0:
                click.echo(f"⚠️  {result['warnings_count']} warnings found")
        else:
            click.echo(f"❌ Data validation failed!")
            click.echo(f"🚫 {result['errors_count']} errors found")
            if result['warnings_count'] > 0:
                click.echo(f"⚠️  {result['warnings_count']} warnings found")
        
        if output_file:
            click.echo(f"📄 Validation report saved to: {output_file}")
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Validation command failed: {e}")
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def config_command(ctx):
    """Show current configuration."""
    try:
        config = ctx.obj['config']
        config_dict = config.to_dict()
        
        click.echo("📋 Current Configuration:")
        click.echo("=" * 50)
        
        for section, settings in config_dict.items():
            click.echo(f"\n[{section.upper()}]")
            for key, value in settings.items():
                click.echo(f"  {key}: {value}")
        
        # Show validation errors if any
        if 'config_errors' in ctx.obj:
            click.echo("\n⚠️  Configuration Issues:")
            for section, errors in ctx.obj['config_errors'].items():
                for error in errors:
                    click.echo(f"  {section}: {error}")
                    
    except Exception as e:
        click.echo(f"❌ Failed to show configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    try:
        from .. import __version__
        click.echo(f"NYC School Analyzer v{__version__}")
        click.echo("Production-ready school data analysis tool")
    except ImportError:
        click.echo("NYC School Analyzer")
        click.echo("Version information not available")


# Add command aliases
cli.add_command(analyze_command, name='analyze')
cli.add_command(visualize_command, name='viz')
cli.add_command(export_command, name='export')
cli.add_command(validate_command, name='validate')
cli.add_command(config_command, name='config')


def main():
    """Main entry point for CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n❌ Operation cancelled by user", err=True)
        sys.exit(1)
    except NYCSchoolAnalyzerError as e:
        click.echo(f"❌ {e.message}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()