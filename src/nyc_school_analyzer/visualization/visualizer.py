"""
Main visualization module for NYC School Analyzer.

Provides comprehensive visualization capabilities for school data analysis
including statistical charts, geographic distributions, and trend analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from ..data.models import VisualizationConfig, SchoolData, AnalysisResult
from .charts import ChartGenerator
from .exports import ExportManager
from ..utils.exceptions import VisualizationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SchoolVisualizer:
    """
    Comprehensive visualization system for school data analysis.
    
    Creates publication-quality charts and exports for analysis results
    with customizable styling and multiple output formats.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize school visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.chart_generator = ChartGenerator(self.config)
        self.export_manager = ExportManager(self.config)
        
        # Initialize matplotlib and seaborn styling
        self._setup_styling()
        
        # Color palettes for different chart types
        self.color_palettes = {
            'categorical': sns.color_palette(self.config.color_palette, 12),
            'sequential': sns.color_palette("Blues", 10),
            'diverging': sns.color_palette("RdYlBu", 11),
            'qualitative': sns.color_palette("Set2", 8),
        }
    
    def create_borough_distribution_chart(
        self, 
        school_counts: Union[pd.Series, Dict],
        title: str = "School Distribution by Borough",
        export_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create enhanced borough distribution visualization.
        
        Args:
            school_counts: Series or dict with borough school counts
            title: Chart title
            export_path: Optional path to save chart
            
        Returns:
            matplotlib Figure object
        """
        try:
            logger.info("Creating borough distribution chart...")
            
            # Convert to Series if dict
            if isinstance(school_counts, dict):
                school_counts = pd.Series(school_counts)
            
            # Sort by values descending
            school_counts = school_counts.sort_values(ascending=False)
            
            # Create figure
            fig = self.chart_generator.create_bar_chart(
                data=school_counts,
                title=title,
                xlabel="Borough/Location",
                ylabel="Number of Schools",
                color_palette='categorical'
            )
            
            # Export if path provided
            if export_path:
                self.export_manager.save_figure(fig, export_path)
            
            logger.info(f"Borough distribution chart created with {len(school_counts)} boroughs")
            return fig
            
        except Exception as e:
            error_msg = f"Failed to create borough distribution chart: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e
    
    def create_student_population_charts(
        self,
        student_stats: pd.DataFrame,
        title: str = "Student Population Analysis",
        export_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive student population visualizations.
        
        Args:
            student_stats: DataFrame with student statistics by area
            title: Chart title
            export_path: Optional path to save chart
            
        Returns:
            matplotlib Figure with multiple subplots
        """
        try:
            logger.info("Creating student population charts...")
            
            # Create multi-panel figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
            
            # Chart 1: Average students per school
            if 'avg_students' in student_stats.columns:
                avg_students = student_stats['avg_students'].sort_values(ascending=False)
                self.chart_generator.create_bar_chart_on_axis(
                    ax=axes[0, 0],
                    data=avg_students.head(10),
                    title="Average Students per School (Top 10)",
                    xlabel="Borough/Location",
                    ylabel="Average Students",
                    color_palette='sequential'
                )
            
            # Chart 2: Total students by area
            if 'total_students' in student_stats.columns:
                total_students = student_stats['total_students'].sort_values(ascending=False)
                self.chart_generator.create_bar_chart_on_axis(
                    ax=axes[0, 1],
                    data=total_students.head(10),
                    title="Total Students by Area (Top 10)",
                    xlabel="Borough/Location",
                    ylabel="Total Students",
                    color_palette='categorical'
                )
            
            # Chart 3: School count vs student population scatter
            if all(col in student_stats.columns for col in ['school_count', 'total_students']):
                axes[1, 0].scatter(
                    student_stats['school_count'],
                    student_stats['total_students'],
                    alpha=0.7,
                    s=60,
                    color=self.color_palettes['qualitative'][0]
                )
                axes[1, 0].set_xlabel('Number of Schools')
                axes[1, 0].set_ylabel('Total Students')
                axes[1, 0].set_title('Schools vs Student Population')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Chart 4: Distribution of school sizes
            if 'avg_students' in student_stats.columns:
                axes[1, 1].hist(
                    student_stats['avg_students'].dropna(),
                    bins=15,
                    alpha=0.7,
                    color=self.color_palettes['sequential'][5],
                    edgecolor='black'
                )
                axes[1, 1].set_xlabel('Average School Size')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Distribution of School Sizes')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Export if path provided
            if export_path:
                self.export_manager.save_figure(fig, export_path)
            
            logger.info("Student population charts created successfully")
            return fig
            
        except Exception as e:
            error_msg = f"Failed to create student population charts: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e
    
    def create_grade_availability_chart(
        self,
        grade_analysis: AnalysisResult,
        export_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create grade availability visualization.
        
        Args:
            grade_analysis: AnalysisResult from grade availability analysis
            export_path: Optional path to save chart
            
        Returns:
            matplotlib Figure with grade analysis charts
        """
        try:
            logger.info(f"Creating grade {grade_analysis.grade_analyzed} availability chart...")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Chart 1: Availability pie chart
            offering_count = grade_analysis.schools_meeting_criteria
            not_offering_count = grade_analysis.total_schools - offering_count
            
            sizes = [offering_count, not_offering_count]
            labels = [
                f'Offer Grade {grade_analysis.grade_analyzed}',
                f'Do Not Offer Grade {grade_analysis.grade_analyzed}'
            ]
            colors = ['#2ecc71', '#e74c3c']
            
            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontweight': 'bold', 'fontsize': 10}
            )
            
            ax1.set_title(
                f'Grade {grade_analysis.grade_analyzed} Availability\\n'
                f'({grade_analysis.total_schools} Total Schools)',
                fontweight='bold',
                fontsize=12
            )
            
            # Chart 2: Grade distribution (if available in statistics)
            grade_stats = grade_analysis.statistics
            if 'min_grade_distribution' in grade_stats:
                min_grades = pd.Series(grade_stats['min_grade_distribution'])
                min_grades = min_grades.sort_index()
                
                self.chart_generator.create_bar_chart_on_axis(
                    ax=ax2,
                    data=min_grades,
                    title="Distribution of Minimum Grades",
                    xlabel="Minimum Grade",
                    ylabel="Number of Schools",
                    color_palette='categorical'
                )
            else:
                # Create summary statistics chart
                summary_data = pd.Series({
                    'Schools Offering': offering_count,
                    'Schools Not Offering': not_offering_count,
                    'Availability %': grade_analysis.percentage
                })
                
                bars = ax2.bar(
                    range(2),
                    [offering_count, not_offering_count],
                    color=['#2ecc71', '#e74c3c'],
                    alpha=0.8,
                    edgecolor='black'
                )
                
                ax2.set_xlabel('Category')
                ax2.set_ylabel('Number of Schools')
                ax2.set_title(f'Grade {grade_analysis.grade_analyzed} Summary Statistics')
                ax2.set_xticks(range(2))
                ax2.set_xticklabels(['Offering', 'Not Offering'])
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, [offering_count, not_offering_count]):
                    ax2.text(
                        bar.get_x() + bar.get_width()/2.,
                        bar.get_height() + max([offering_count, not_offering_count]) * 0.01,
                        f'{value}',
                        ha='center',
                        va='bottom',
                        fontweight='bold'
                    )
            
            plt.tight_layout()
            
            # Export if path provided
            if export_path:
                self.export_manager.save_figure(fig, export_path)
            
            logger.info(f"Grade {grade_analysis.grade_analyzed} availability chart created")
            return fig
            
        except Exception as e:
            error_msg = f"Failed to create grade availability chart: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e
    
    def create_comprehensive_dashboard(
        self,
        analysis_results: Dict[str, Any],
        export_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive analysis dashboard.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            export_path: Optional path to save dashboard
            
        Returns:
            matplotlib Figure with comprehensive dashboard
        """
        try:
            logger.info("Creating comprehensive analysis dashboard...")
            
            # Create large figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Dashboard title
            fig.suptitle(
                'NYC School Analysis Dashboard',
                fontsize=20,
                fontweight='bold',
                y=0.95
            )
            
            # Chart 1: Borough distribution (top-left)
            if 'borough_distribution' in analysis_results:
                ax1 = fig.add_subplot(gs[0, 0])
                borough_data = analysis_results['borough_distribution']
                school_counts = pd.Series(borough_data.get('school_counts', {}))
                
                if not school_counts.empty:
                    self.chart_generator.create_bar_chart_on_axis(
                        ax=ax1,
                        data=school_counts.head(8),
                        title="Schools by Borough",
                        xlabel="Borough",
                        ylabel="Count",
                        color_palette='categorical',
                        rotation=45
                    )
            
            # Chart 2: Grade availability (top-middle)
            if 'grade_availability' in analysis_results:
                ax2 = fig.add_subplot(gs[0, 1])
                grade_result = analysis_results['grade_availability']
                
                if hasattr(grade_result, 'percentage'):
                    # Pie chart for availability
                    offering = grade_result.schools_meeting_criteria
                    not_offering = grade_result.total_schools - offering
                    
                    ax2.pie(
                        [offering, not_offering],
                        labels=['Available', 'Not Available'],
                        colors=['#2ecc71', '#e74c3c'],
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax2.set_title(f'Grade {grade_result.grade_analyzed} Availability')
            
            # Chart 3: Data quality metrics (top-right)
            if 'data_quality' in analysis_results:
                ax3 = fig.add_subplot(gs[0, 2])
                quality_data = analysis_results['data_quality']
                
                quality_metrics = {
                    'Completeness': quality_data.get('completeness_analysis', {}).get('average_completeness', 0) * 100,
                    'Quality Score': quality_data.get('quality_score', 0),
                }
                
                bars = ax3.bar(
                    quality_metrics.keys(),
                    quality_metrics.values(),
                    color=self.color_palettes['sequential'][:len(quality_metrics)],
                    alpha=0.8,
                    edgecolor='black'
                )
                ax3.set_title('Data Quality Metrics')
                ax3.set_ylabel('Score/Percentage')
                ax3.set_ylim(0, 100)
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, quality_metrics.values()):
                    ax3.text(
                        bar.get_x() + bar.get_width()/2.,
                        bar.get_height() + 2,
                        f'{value:.1f}',
                        ha='center',
                        va='bottom',
                        fontweight='bold'
                    )
            
            # Chart 4-6: Student population analysis (middle row)
            if 'student_populations' in analysis_results:
                pop_data = analysis_results['student_populations']
                pop_stats = pop_data.get('population_statistics', {})
                
                if pop_stats:
                    # Convert to DataFrame for easier handling
                    stats_df = pd.DataFrame(pop_stats)
                    
                    # Average students chart
                    if 'avg_students' in stats_df.columns:
                        ax4 = fig.add_subplot(gs[1, 0])
                        top_avg = stats_df['avg_students'].sort_values(ascending=False).head(8)
                        self.chart_generator.create_bar_chart_on_axis(
                            ax=ax4,
                            data=top_avg,
                            title="Avg Students per School",
                            xlabel="Area",
                            ylabel="Students",
                            color_palette='sequential',
                            rotation=45
                        )
                    
                    # Total students chart
                    if 'total_students' in stats_df.columns:
                        ax5 = fig.add_subplot(gs[1, 1])
                        top_total = stats_df['total_students'].sort_values(ascending=False).head(8)
                        self.chart_generator.create_bar_chart_on_axis(
                            ax=ax5,
                            data=top_total,
                            title="Total Students by Area",
                            xlabel="Area",
                            ylabel="Students",
                            color_palette='categorical',
                            rotation=45
                        )
                    
                    # School count distribution
                    if 'school_count' in stats_df.columns:
                        ax6 = fig.add_subplot(gs[1, 2])
                        ax6.hist(
                            stats_df['school_count'].dropna(),
                            bins=15,
                            alpha=0.7,
                            color=self.color_palettes['qualitative'][2],
                            edgecolor='black'
                        )
                        ax6.set_title('Distribution of School Counts')
                        ax6.set_xlabel('Schools per Area')
                        ax6.set_ylabel('Frequency')
                        ax6.grid(True, alpha=0.3)
            
            # Chart 7-9: Summary statistics and insights (bottom row)
            # Key metrics summary
            ax7 = fig.add_subplot(gs[2, :])
            ax7.axis('off')
            
            # Create summary text
            summary_text = self._generate_dashboard_summary(analysis_results)
            ax7.text(
                0.05, 0.95, summary_text,
                transform=ax7.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8)
            )
            
            # Export if path provided
            if export_path:
                self.export_manager.save_figure(fig, export_path, dpi=self.config.dpi)
            
            logger.info("Comprehensive dashboard created successfully")
            return fig
            
        except Exception as e:
            error_msg = f"Failed to create comprehensive dashboard: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e
    
    def export_all_charts(
        self,
        analysis_results: Dict[str, Any],
        output_dir: Path,
        format_list: Optional[List[str]] = None
    ) -> Dict[str, List[Path]]:
        """
        Export all charts for analysis results.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            output_dir: Directory to save charts
            format_list: List of formats to export
            
        Returns:
            Dictionary mapping chart names to saved file paths
        """
        try:
            logger.info(f"Exporting all charts to {output_dir}")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exported_files = {}
            formats = format_list or self.config.export_formats
            
            # Borough distribution chart
            if 'borough_distribution' in analysis_results:
                borough_data = analysis_results['borough_distribution']
                school_counts = borough_data.get('school_counts', {})
                
                if school_counts:
                    fig = self.create_borough_distribution_chart(school_counts)
                    paths = self.export_manager.save_figure_multiple_formats(
                        fig, output_dir / "borough_distribution", formats
                    )
                    exported_files['borough_distribution'] = paths
                    plt.close(fig)
            
            # Grade availability chart
            if 'grade_availability' in analysis_results:
                grade_result = analysis_results['grade_availability']
                fig = self.create_grade_availability_chart(grade_result)
                paths = self.export_manager.save_figure_multiple_formats(
                    fig, output_dir / "grade_availability", formats
                )
                exported_files['grade_availability'] = paths
                plt.close(fig)
            
            # Student population charts
            if 'student_populations' in analysis_results:
                pop_data = analysis_results['student_populations']
                pop_stats = pop_data.get('population_statistics', {})
                
                if pop_stats:
                    stats_df = pd.DataFrame(pop_stats)
                    fig = self.create_student_population_charts(stats_df)
                    paths = self.export_manager.save_figure_multiple_formats(
                        fig, output_dir / "student_populations", formats
                    )
                    exported_files['student_populations'] = paths
                    plt.close(fig)
            
            # Comprehensive dashboard
            fig = self.create_comprehensive_dashboard(analysis_results)
            paths = self.export_manager.save_figure_multiple_formats(
                fig, output_dir / "comprehensive_dashboard", formats
            )
            exported_files['comprehensive_dashboard'] = paths
            plt.close(fig)
            
            total_files = sum(len(paths) for paths in exported_files.values())
            logger.info(f"Exported {total_files} chart files across {len(exported_files)} chart types")
            
            return exported_files
            
        except Exception as e:
            error_msg = f"Failed to export all charts: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg) from e
    
    # Private helper methods
    
    def _setup_styling(self):
        """Setup matplotlib and seaborn styling."""
        try:
            # Set style
            if self.config.style == 'seaborn':
                sns.set_style("whitegrid")
            else:
                plt.style.use(self.config.style)
            
            # Configure matplotlib parameters
            plt.rcParams.update({
                'figure.figsize': self.config.figure_size,
                'font.size': self.config.font_size,
                'axes.titlesize': self.config.font_size + 2,
                'axes.labelsize': self.config.font_size,
                'xtick.labelsize': self.config.font_size - 1,
                'ytick.labelsize': self.config.font_size - 1,
                'legend.fontsize': self.config.font_size - 1,
                'figure.dpi': self.config.dpi,
                'savefig.dpi': self.config.dpi,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1,
            })
            
            # Set color palette
            sns.set_palette(self.config.color_palette)
            
            logger.debug("Visualization styling configured successfully")
            
        except Exception as e:
            logger.warning(f"Error setting up visualization styling: {e}")
    
    def _generate_dashboard_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate summary text for dashboard."""
        try:
            summary_parts = ["ANALYSIS SUMMARY\\n" + "="*50]
            
            # Borough summary
            if 'borough_distribution' in analysis_results:
                borough_data = analysis_results['borough_distribution']
                total_boroughs = borough_data.get('total_boroughs', 0)
                largest_borough = borough_data.get('largest_borough', 'Unknown')
                summary_parts.append(f"• Geographic Coverage: {total_boroughs} boroughs/areas")
                summary_parts.append(f"• Largest Concentration: {largest_borough}")
            
            # Grade availability summary
            if 'grade_availability' in analysis_results:
                grade_result = analysis_results['grade_availability']
                if hasattr(grade_result, 'percentage'):
                    summary_parts.append(
                        f"• Grade {grade_result.grade_analyzed} Availability: {grade_result.percentage:.1f}%"
                    )
            
            # Student population summary
            if 'student_populations' in analysis_results:
                pop_data = analysis_results['student_populations']
                system_metrics = pop_data.get('system_metrics', {})
                total_students = system_metrics.get('system_total_students', 0)
                total_schools = system_metrics.get('system_total_schools', 0)
                
                if total_students and total_schools:
                    avg_size = total_students / total_schools
                    summary_parts.append(f"• Total Students: {total_students:,}")
                    summary_parts.append(f"• Average School Size: {avg_size:.0f} students")
            
            # Data quality summary
            if 'data_quality' in analysis_results:
                quality_score = analysis_results['data_quality'].get('quality_score', 0)
                summary_parts.append(f"• Data Quality Score: {quality_score:.1f}/100")
            
            return "\\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating dashboard summary: {e}")
            return "Summary generation failed - see logs for details"