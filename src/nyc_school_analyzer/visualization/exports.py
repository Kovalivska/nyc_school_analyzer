"""
Export management module for NYC School Analyzer.

Handles saving charts and visualizations in multiple formats with
consistent naming and organization.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

from ..data.models import VisualizationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExportManager:
    """
    Manages export of visualizations to various formats.
    
    Provides consistent file naming, organization, and format handling
    for all visualization exports.
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize export manager.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Supported formats and their settings
        self.format_settings = {
            'png': {'dpi': config.dpi, 'bbox_inches': 'tight', 'pad_inches': 0.1},
            'pdf': {'bbox_inches': 'tight', 'pad_inches': 0.1},
            'svg': {'bbox_inches': 'tight', 'pad_inches': 0.1},
            'eps': {'bbox_inches': 'tight', 'pad_inches': 0.1},
            'jpg': {'dpi': config.dpi, 'bbox_inches': 'tight', 'pad_inches': 0.1, 'quality': 95},
            'jpeg': {'dpi': config.dpi, 'bbox_inches': 'tight', 'pad_inches': 0.1, 'quality': 95},
        }
    
    def save_figure(
        self,
        fig: plt.Figure,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        dpi: Optional[int] = None,
        **kwargs
    ) -> Path:
        """
        Save a matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure to save
            filepath: Path to save file (with or without extension)
            format: File format (inferred from filepath if not provided)
            dpi: DPI override for raster formats
            **kwargs: Additional arguments for plt.savefig
            
        Returns:
            Path to saved file
        """
        try:
            filepath = Path(filepath)
            
            # Infer format from extension if not provided
            if format is None:
                format = filepath.suffix.lower().lstrip('.')
                if not format:
                    format = 'png'  # Default format
                    filepath = filepath.with_suffix('.png')
            else:
                # Ensure filepath has correct extension
                if not filepath.suffix:
                    filepath = filepath.with_suffix(f'.{format}')
            
            # Validate format
            if format not in self.format_settings:
                raise ValueError(f"Unsupported format: {format}. Supported: {list(self.format_settings.keys())}")
            
            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Get format-specific settings
            save_kwargs = self.format_settings[format].copy()
            
            # Override DPI if specified
            if dpi is not None and format in ['png', 'jpg', 'jpeg']:
                save_kwargs['dpi'] = dpi
            
            # Update with any additional kwargs
            save_kwargs.update(kwargs)
            
            # Save figure
            fig.savefig(filepath, format=format, **save_kwargs)
            
            self.logger.debug(f"Saved figure to {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Failed to save figure to {filepath}: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def save_figure_multiple_formats(
        self,
        fig: plt.Figure,
        base_filepath: Union[str, Path],
        formats: List[str],
        add_timestamp: bool = False
    ) -> List[Path]:
        """
        Save a figure in multiple formats.
        
        Args:
            fig: Matplotlib figure to save
            base_filepath: Base filepath (without extension)
            formats: List of formats to save
            add_timestamp: Whether to add timestamp to filename
            
        Returns:
            List of paths to saved files
        """
        try:
            base_filepath = Path(base_filepath)
            
            # Add timestamp if requested
            if add_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filepath = base_filepath.with_name(f"{base_filepath.stem}_{timestamp}")
            
            saved_paths = []
            
            for format in formats:
                filepath = base_filepath.with_suffix(f'.{format}')
                saved_path = self.save_figure(fig, filepath, format=format)
                saved_paths.append(saved_path)
            
            self.logger.info(f"Saved figure in {len(formats)} formats: {formats}")
            return saved_paths
            
        except Exception as e:
            error_msg = f"Failed to save figure in multiple formats: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def create_export_directory(
        self,
        base_dir: Union[str, Path],
        analysis_name: str,
        add_timestamp: bool = True
    ) -> Path:
        """
        Create organized directory structure for exports.
        
        Args:
            base_dir: Base directory for exports
            analysis_name: Name of the analysis
            add_timestamp: Whether to add timestamp to directory name
            
        Returns:
            Path to created export directory
        """
        try:
            base_dir = Path(base_dir)
            
            # Create directory name
            if add_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir_name = f"{analysis_name}_{timestamp}"
            else:
                dir_name = analysis_name
            
            export_dir = base_dir / dir_name
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different types
            subdirs = ['charts', 'data', 'reports']
            for subdir in subdirs:
                (export_dir / subdir).mkdir(exist_ok=True)
            
            self.logger.info(f"Created export directory: {export_dir}")
            return export_dir
            
        except Exception as e:
            error_msg = f"Failed to create export directory: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def export_chart_collection(
        self,
        chart_collection: Dict[str, plt.Figure],
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        add_timestamp: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Export a collection of charts with organized naming.
        
        Args:
            chart_collection: Dictionary mapping chart names to figures
            output_dir: Directory to save charts
            formats: List of formats to export (uses config default if None)
            add_timestamp: Whether to add timestamp to filenames
            
        Returns:
            Dictionary mapping chart names to lists of saved file paths
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            formats = formats or self.config.export_formats
            exported_files = {}
            
            for chart_name, figure in chart_collection.items():
                # Create safe filename
                safe_name = self._create_safe_filename(chart_name)
                base_path = output_dir / safe_name
                
                # Save in multiple formats
                saved_paths = self.save_figure_multiple_formats(
                    figure, base_path, formats, add_timestamp=add_timestamp
                )
                
                exported_files[chart_name] = saved_paths
            
            total_files = sum(len(paths) for paths in exported_files.values())
            self.logger.info(
                f"Exported {len(chart_collection)} charts in {len(formats)} formats "
                f"({total_files} total files)"
            )
            
            return exported_files
            
        except Exception as e:
            error_msg = f"Failed to export chart collection: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def generate_export_manifest(
        self,
        exported_files: Dict[str, List[Path]],
        output_dir: Union[str, Path],
        include_metadata: bool = True
    ) -> Path:
        """
        Generate a manifest file listing all exported files.
        
        Args:
            exported_files: Dictionary of exported file paths
            output_dir: Directory to save manifest
            include_metadata: Whether to include file metadata
            
        Returns:
            Path to manifest file
        """
        try:
            import json
            from datetime import datetime
            
            output_dir = Path(output_dir)
            manifest_path = output_dir / "export_manifest.json"
            
            # Create manifest data
            manifest = {
                "export_timestamp": datetime.now().isoformat(),
                "total_charts": len(exported_files),
                "total_files": sum(len(paths) for paths in exported_files.values()),
                "charts": {}
            }
            
            for chart_name, file_paths in exported_files.items():
                chart_info = {
                    "files": [],
                    "formats": []
                }
                
                for file_path in file_paths:
                    file_info = {
                        "path": str(file_path),
                        "format": file_path.suffix.lstrip('.'),
                        "filename": file_path.name
                    }
                    
                    # Add metadata if requested
                    if include_metadata and file_path.exists():
                        stat = file_path.stat()
                        file_info.update({
                            "size_bytes": stat.st_size,
                            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                    
                    chart_info["files"].append(file_info)
                    chart_info["formats"].append(file_path.suffix.lstrip('.'))
                
                # Remove duplicate formats
                chart_info["formats"] = list(set(chart_info["formats"]))
                manifest["charts"][chart_name] = chart_info
            
            # Save manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Generated export manifest: {manifest_path}")
            return manifest_path
            
        except Exception as e:
            error_msg = f"Failed to generate export manifest: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def cleanup_old_exports(
        self,
        export_dir: Union[str, Path],
        keep_days: int = 30,
        dry_run: bool = False
    ) -> List[Path]:
        """
        Clean up old export files.
        
        Args:
            export_dir: Directory containing exports
            keep_days: Number of days to keep files
            dry_run: If True, only list files that would be deleted
            
        Returns:
            List of files that were (or would be) deleted
        """
        try:
            from datetime import datetime, timedelta
            
            export_dir = Path(export_dir)
            if not export_dir.exists():
                return []
            
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            files_to_delete = []
            
            # Find old files
            for file_path in export_dir.rglob('*'):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        files_to_delete.append(file_path)
            
            # Delete files if not dry run
            if not dry_run:
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        self.logger.debug(f"Deleted old export file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_path}: {e}")
            
            action = "Would delete" if dry_run else "Deleted"
            self.logger.info(f"{action} {len(files_to_delete)} old export files")
            
            return files_to_delete
            
        except Exception as e:
            error_msg = f"Failed to cleanup old exports: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    # Private helper methods
    
    def _create_safe_filename(self, name: str) -> str:
        """
        Create a safe filename from a chart name.
        
        Args:
            name: Original name
            
        Returns:
            Safe filename
        """
        import re
        
        # Replace spaces and special characters
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        
        # Remove multiple underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # Remove leading/trailing underscores
        safe_name = safe_name.strip('_')
        
        # Ensure it's not empty
        if not safe_name:
            safe_name = "chart"
        
        return safe_name.lower()
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0