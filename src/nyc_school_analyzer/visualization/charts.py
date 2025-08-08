"""
Chart generation utilities for NYC School Analyzer.

Provides specialized chart creation functions with consistent styling
and professional appearance for educational data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from ..data.models import VisualizationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """
    Specialized chart generator for educational data visualization.
    
    Creates consistent, publication-quality charts with professional styling
    specifically designed for school data analysis.
    """
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize chart generator.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Define color palettes
        self.palettes = {
            'categorical': sns.color_palette(config.color_palette, 12),
            'sequential': sns.color_palette("Blues", 10),
            'diverging': sns.color_palette("RdYlBu", 11),
            'qualitative': sns.color_palette("Set2", 8),
            'education': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }
    
    def create_bar_chart(
        self,
        data: Union[pd.Series, Dict],
        title: str,
        xlabel: str,
        ylabel: str,
        color_palette: str = 'categorical',
        figsize: Optional[Tuple[float, float]] = None,
        rotation: int = 0,
        show_values: bool = True
    ) -> plt.Figure:
        """
        Create a professional bar chart.
        
        Args:
            data: Series or dict with data to plot
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color_palette: Color palette to use
            figsize: Figure size override
            rotation: X-axis label rotation
            show_values: Whether to show values on bars
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Convert to Series if dict
            if isinstance(data, dict):
                data = pd.Series(data)
            
            # Create figure
            figsize = figsize or self.config.figure_size
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create bar chart
            colors = self.palettes.get(color_palette, self.palettes['categorical'])
            bars = ax.bar(
                range(len(data)),
                data.values,
                color=colors[:len(data)],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.8
            )
            
            # Customize chart
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            
            # Set x-axis labels
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=rotation, ha='right' if rotation > 0 else 'center')
            
            # Add value labels on bars if requested
            if show_values:
                self._add_bar_labels(ax, bars, data.values)
            
            # Add grid and styling
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add statistics box
            if len(data) > 1:
                stats_text = f"Total: {data.sum()}\\nMean: {data.mean():.1f}\\nStd: {data.std():.1f}"
                ax.text(
                    0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=self.config.font_size - 2
                )
            
            plt.tight_layout()
            self.logger.debug(f"Created bar chart with {len(data)} bars")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating bar chart: {e}")
            raise
    
    def create_bar_chart_on_axis(
        self,
        ax: plt.Axes,
        data: Union[pd.Series, Dict],
        title: str,
        xlabel: str,
        ylabel: str,
        color_palette: str = 'categorical',
        rotation: int = 0,
        show_values: bool = True
    ):
        """
        Create a bar chart on existing axis.
        
        Args:
            ax: Matplotlib axes to plot on
            data: Series or dict with data to plot
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color_palette: Color palette to use
            rotation: X-axis label rotation
            show_values: Whether to show values on bars
        """
        try:
            # Convert to Series if dict
            if isinstance(data, dict):
                data = pd.Series(data)
            
            # Create bar chart
            colors = self.palettes.get(color_palette, self.palettes['categorical'])
            bars = ax.bar(
                range(len(data)),
                data.values,
                color=colors[:len(data)],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.8
            )
            
            # Customize chart
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold')
            
            # Set x-axis labels
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=rotation, ha='right' if rotation > 0 else 'center')
            
            # Add value labels on bars if requested
            if show_values:
                self._add_bar_labels(ax, bars, data.values)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            self.logger.debug(f"Created bar chart on axis with {len(data)} bars")
            
        except Exception as e:
            self.logger.error(f"Error creating bar chart on axis: {e}")
            raise
    
    def create_pie_chart(
        self,
        data: Union[pd.Series, Dict],
        title: str,
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        explode: Optional[List[float]] = None
    ) -> plt.Figure:
        """
        Create a professional pie chart.
        
        Args:
            data: Series or dict with data to plot
            title: Chart title
            colors: Custom colors for pie slices
            figsize: Figure size override
            explode: Explode values for pie slices
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Convert to Series if dict
            if isinstance(data, dict):
                data = pd.Series(data)
            
            # Create figure
            figsize = figsize or (10, 8)
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use default colors if none provided
            if colors is None:
                colors = self.palettes['categorical'][:len(data)]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                data.values,
                labels=data.index,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                textprops={'fontweight': 'bold', 'fontsize': self.config.font_size - 1}
            )
            
            # Customize chart
            ax.set_title(title, fontweight='bold', fontsize=self.config.font_size + 2, pad=20)
            
            # Ensure equal aspect ratio
            ax.axis('equal')
            
            self.logger.debug(f"Created pie chart with {len(data)} slices")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating pie chart: {e}")
            raise
    
    def create_histogram(
        self,
        data: Union[pd.Series, np.ndarray, List],
        title: str,
        xlabel: str,
        ylabel: str = "Frequency",
        bins: Union[int, str] = 'auto',
        color: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show_stats: bool = True
    ) -> plt.Figure:
        """
        Create a professional histogram.
        
        Args:
            data: Data to plot
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of bins or binning strategy
            color: Color for bars
            figsize: Figure size override
            show_stats: Whether to show statistics
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Convert to numpy array
            if isinstance(data, pd.Series):
                data = data.dropna().values
            elif isinstance(data, list):
                data = np.array(data)
            
            # Remove NaN values
            data = data[~np.isnan(data)]
            
            # Create figure
            figsize = figsize or self.config.figure_size
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use default color if none provided
            if color is None:
                color = self.palettes['education'][0]
            
            # Create histogram
            n, bins_edges, patches = ax.hist(
                data,
                bins=bins,
                alpha=0.7,
                color=color,
                edgecolor='black',
                linewidth=0.8
            )
            
            # Customize chart
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add statistics if requested
            if show_stats and len(data) > 0:
                stats_text = (
                    f"Count: {len(data)}\\n"
                    f"Mean: {np.mean(data):.1f}\\n"
                    f"Median: {np.median(data):.1f}\\n"
                    f"Std: {np.std(data):.1f}"
                )
                ax.text(
                    0.98, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=self.config.font_size - 2
                )
            
            plt.tight_layout()
            self.logger.debug(f"Created histogram with {len(data)} data points")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating histogram: {e}")
            raise
    
    def create_scatter_plot(
        self,
        x_data: Union[pd.Series, np.ndarray, List],
        y_data: Union[pd.Series, np.ndarray, List],
        title: str,
        xlabel: str,
        ylabel: str,
        color: Optional[str] = None,
        size: Optional[Union[int, List, np.ndarray]] = None,
        alpha: float = 0.7,
        figsize: Optional[Tuple[float, float]] = None,
        show_correlation: bool = True
    ) -> plt.Figure:
        """
        Create a professional scatter plot.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Color for points
            size: Size for points
            alpha: Transparency
            figsize: Figure size override
            show_correlation: Whether to show correlation
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Convert to numpy arrays
            if isinstance(x_data, pd.Series):
                x_data = x_data.values
            if isinstance(y_data, pd.Series):
                y_data = y_data.values
            
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            
            # Remove NaN values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            # Create figure
            figsize = figsize or self.config.figure_size
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use default color if none provided
            if color is None:
                color = self.palettes['education'][0]
            
            # Use default size if none provided
            if size is None:
                size = 60
            
            # Create scatter plot
            scatter = ax.scatter(
                x_data,
                y_data,
                c=color,
                s=size,
                alpha=alpha,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Customize chart
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Show correlation if requested
            if show_correlation and len(x_data) > 1:
                correlation = np.corrcoef(x_data, y_data)[0, 1]
                if not np.isnan(correlation):
                    ax.text(
                        0.05, 0.95,
                        f"Correlation: {correlation:.3f}",
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=self.config.font_size - 1
                    )
            
            plt.tight_layout()
            self.logger.debug(f"Created scatter plot with {len(x_data)} points")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")
            raise
    
    def create_box_plot(
        self,
        data: Union[pd.DataFrame, Dict[str, List]],
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Optional[Tuple[float, float]] = None,
        show_outliers: bool = True
    ) -> plt.Figure:
        """
        Create a professional box plot.
        
        Args:
            data: Data to plot (DataFrame or dict of lists)
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size override
            show_outliers: Whether to show outliers
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
            
            # Create figure
            figsize = figsize or self.config.figure_size
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create box plot
            box_plot = ax.boxplot(
                [data[col].dropna().values for col in data.columns],
                labels=data.columns,
                patch_artist=True,
                showfliers=show_outliers
            )
            
            # Color the boxes
            colors = self.palettes['categorical'][:len(data.columns)]
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize chart
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            self.logger.debug(f"Created box plot with {len(data.columns)} boxes")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating box plot: {e}")
            raise
    
    # Private helper methods
    
    def _add_bar_labels(self, ax: plt.Axes, bars, values):
        """Add value labels on top of bars."""
        try:
            max_value = max(values) if values else 0
            for bar, value in zip(bars, values):
                if pd.notna(value):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + max_value * 0.01,
                        f'{value:.0f}' if isinstance(value, (int, float)) else str(value),
                        ha='center',
                        va='bottom',
                        fontweight='bold',
                        fontsize=self.config.font_size - 2
                    )
        except Exception as e:
            self.logger.warning(f"Error adding bar labels: {e}")
    
    def _format_number(self, value: Union[int, float]) -> str:
        """Format number for display."""
        try:
            if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                if abs(value) >= 1000000:
                    return f"{value/1000000:.1f}M"
                elif abs(value) >= 1000:
                    return f"{value/1000:.1f}K"
                else:
                    return f"{int(value)}"
            else:
                return f"{value:.1f}"
        except Exception:
            return str(value)