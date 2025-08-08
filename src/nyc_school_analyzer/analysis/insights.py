"""
Insight generation module for NYC School Analyzer.

Generates actionable insights and recommendations from analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Insight:
    """Individual insight with metadata."""
    title: str
    description: str
    impact_level: str  # 'high', 'medium', 'low'
    category: str
    supporting_data: Dict[str, Any]
    recommendations: List[str]


class InsightGenerator:
    """
    Generates actionable insights from analysis results.
    
    Transforms statistical analysis into meaningful business insights
    and strategic recommendations for educational planning.
    """
    
    def __init__(self):
        """Initialize insight generator."""
        self.logger = get_logger(__name__)
        self.impact_thresholds = {
            'high': {'percentage': 20, 'schools': 50, 'students': 5000},
            'medium': {'percentage': 10, 'schools': 20, 'students': 2000},
            'low': {'percentage': 5, 'schools': 10, 'students': 1000},
        }
    
    def generate_grade_insights(
        self, 
        target_grade: int, 
        availability_percentage: float,
        detailed_stats: Dict[str, Any]
    ) -> List[str]:
        """
        Generate insights about grade availability.
        
        Args:
            target_grade: Grade level analyzed
            availability_percentage: Percentage of schools offering the grade
            detailed_stats: Detailed statistics from analysis
            
        Returns:
            List of insight strings
        """
        try:
            insights = []
            
            # Availability assessment
            if availability_percentage == 100.0:
                insights.append(
                    f"Excellent Access: ALL schools accommodate Grade {target_grade} students, "
                    f"indicating comprehensive coverage with no access gaps."
                )
                insights.append(
                    f"Policy Implication: No Grade {target_grade} placement issues expected "
                    f"in this geographic area."
                )
            elif availability_percentage >= 90.0:
                insights.append(
                    f"High Access: {availability_percentage:.1f}% of schools offer Grade {target_grade}, "
                    f"with minimal access gaps."
                )
                insights.append(
                    f"Minor Optimization: Review the {100 - availability_percentage:.1f}% of schools "
                    f"not offering Grade {target_grade} for potential expansion."
                )
            elif availability_percentage >= 70.0:
                insights.append(
                    f"Moderate Access: {availability_percentage:.1f}% availability indicates "
                    f"room for improvement in Grade {target_grade} access."
                )
                insights.append(
                    f"Policy Review: Consider incentives for more schools to accommodate "
                    f"Grade {target_grade} students."
                )
            else:
                insights.append(
                    f"Limited Access: Only {availability_percentage:.1f}% of schools offer "
                    f"Grade {target_grade}, indicating significant access gaps."
                )
                insights.append(
                    f"Urgent Action: Priority should be given to expanding Grade {target_grade} "
                    f"availability across the school system."
                )
            
            # Grade span insights
            if 'average_grade_span' in detailed_stats:
                avg_span = detailed_stats['average_grade_span']
                if avg_span < 4:
                    insights.append(
                        f"Specialized Schools: Average grade span of {avg_span:.1f} suggests "
                        f"many schools focus on specific grade levels."
                    )
                elif avg_span > 8:
                    insights.append(
                        f"Comprehensive Schools: Large average grade span of {avg_span:.1f} "
                        f"indicates many schools serve broad age ranges."
                    )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating grade insights: {e}")
            return ["Unable to generate grade insights due to data processing error."]
    
    def generate_borough_insights(self, borough_analysis: Dict[str, Any]) -> List[Insight]:
        """
        Generate insights about borough distribution.
        
        Args:
            borough_analysis: Results from borough distribution analysis
            
        Returns:
            List of Insight objects
        """
        try:
            insights = []
            
            # Market concentration insight
            concentration = borough_analysis.get('concentration_metrics', {})
            if concentration:
                top_3_conc = concentration.get('top_3_concentration', 0)
                
                if top_3_conc > 75:
                    insights.append(Insight(
                        title="High Market Concentration",
                        description=(
                            f"Top 3 boroughs contain {top_3_conc:.1f}% of all schools, "
                            f"indicating high concentration of educational resources."
                        ),
                        impact_level="high",
                        category="geographic_equity",
                        supporting_data={"concentration": top_3_conc},
                        recommendations=[
                            "Monitor resource distribution for equity concerns",
                            "Consider targeted investments in underserved areas",
                            "Evaluate transportation and access barriers"
                        ]
                    ))
                elif top_3_conc < 50:
                    insights.append(Insight(
                        title="Distributed Educational Access",
                        description=(
                            f"Top 3 boroughs contain only {top_3_conc:.1f}% of schools, "
                            f"indicating well-distributed educational resources."
                        ),
                        impact_level="medium",
                        category="geographic_equity",
                        supporting_data={"concentration": top_3_conc},
                        recommendations=[
                            "Maintain current distribution strategy",
                            "Monitor quality consistency across areas",
                            "Leverage distributed model for resilience planning"
                        ]
                    ))
            
            # Student population insights
            student_stats = borough_analysis.get('student_statistics')
            if student_stats:
                insights.extend(self._generate_student_population_insights(student_stats))
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating borough insights: {e}")
            return []
    
    def generate_executive_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary from all analyses.
        
        Args:
            analyses: Dictionary containing all analysis results
            
        Returns:
            Dictionary containing executive summary
        """
        try:
            summary = {
                'key_findings': [],
                'critical_metrics': {},
                'strategic_priorities': [],
                'data_quality_assessment': '',
            }
            
            # Extract key findings from each analysis
            if 'borough_distribution' in analyses:
                borough_data = analyses['borough_distribution']
                summary['key_findings'].append({
                    'category': 'Geographic Distribution',
                    'finding': f"Schools distributed across {borough_data.get('total_boroughs', 0)} locations",
                    'detail': f"Largest concentration: {borough_data.get('largest_borough', 'Unknown')}"
                })
            
            if 'grade_availability' in analyses:
                grade_data = analyses['grade_availability']
                grade_pct = grade_data.percentage if hasattr(grade_data, 'percentage') else 0
                summary['key_findings'].append({
                    'category': 'Grade Access',
                    'finding': f"{grade_pct:.1f}% grade availability",
                    'detail': f"Target grade served by {grade_data.schools_meeting_criteria if hasattr(grade_data, 'schools_meeting_criteria') else 0} schools"
                })
            
            # Critical metrics
            if 'student_populations' in analyses:
                pop_data = analyses['student_populations']
                system_metrics = pop_data.get('system_metrics', {})
                summary['critical_metrics'].update({
                    'total_students': system_metrics.get('system_total_students', 0),
                    'total_schools': system_metrics.get('system_total_schools', 0),
                    'average_school_size': system_metrics.get('system_average_school_size', 0),
                })
            
            # Strategic priorities
            summary['strategic_priorities'] = self._identify_strategic_priorities(analyses)
            
            # Data quality assessment
            if 'data_quality' in analyses:
                quality_score = analyses['data_quality'].get('quality_score', 0)
                if quality_score >= 85:
                    summary['data_quality_assessment'] = "Excellent data quality supports reliable analysis"
                elif quality_score >= 70:
                    summary['data_quality_assessment'] = "Good data quality with minor improvement opportunities"
                else:
                    summary['data_quality_assessment'] = "Data quality concerns require attention before strategic decisions"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return {
                'key_findings': [],
                'critical_metrics': {},
                'strategic_priorities': ["Unable to generate priorities due to analysis error"],
                'data_quality_assessment': 'Assessment unavailable',
            }
    
    def generate_recommendations(self, analyses: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate strategic recommendations from analysis results.
        
        Args:
            analyses: Dictionary containing all analysis results
            
        Returns:
            Dictionary of categorized recommendations
        """
        try:
            recommendations = {
                'immediate_actions': [],
                'strategic_initiatives': [],
                'data_improvements': [],
                'policy_considerations': [],
            }
            
            # Grade availability recommendations
            if 'grade_availability' in analyses:
                grade_analysis = analyses['grade_availability']
                grade_pct = grade_analysis.percentage if hasattr(grade_analysis, 'percentage') else 0
                
                if grade_pct < 70:
                    recommendations['immediate_actions'].append(
                        "Expand grade availability in underserved schools"
                    )
                    recommendations['policy_considerations'].append(
                        "Review grade span policies for equity and access"
                    )
            
            # Borough distribution recommendations
            if 'borough_distribution' in analyses:
                borough_data = analyses['borough_distribution']
                concentration = borough_data.get('concentration_metrics', {})
                
                if concentration.get('top_3_concentration', 0) > 80:
                    recommendations['strategic_initiatives'].append(
                        "Develop strategies to improve geographic distribution of schools"
                    )
                    recommendations['policy_considerations'].append(
                        "Assess transportation and access equity across boroughs"
                    )
            
            # Student population recommendations
            if 'student_populations' in analyses:
                pop_analysis = analyses['student_populations']
                patterns = pop_analysis.get('patterns_and_outliers', {})
                
                if patterns.get('outlier_percentage', 0) > 10:
                    recommendations['immediate_actions'].append(
                        "Investigate schools with unusual enrollment patterns"
                    )
            
            # Data quality recommendations
            if 'data_quality' in analyses:
                quality_data = analyses['data_quality']
                quality_score = quality_data.get('quality_score', 0)
                
                if quality_score < 70:
                    recommendations['data_improvements'].extend([
                        "Implement data quality monitoring and validation processes",
                        "Establish data collection standards and training programs",
                        "Regular data audits and cleanup procedures"
                    ])
            
            # Add general best practice recommendations
            recommendations['strategic_initiatives'].extend([
                "Regular monitoring of educational equity metrics",
                "Stakeholder engagement for strategic planning",
                "Performance benchmarking against peer systems"
            ])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {
                'immediate_actions': ["Analysis error - unable to generate recommendations"],
                'strategic_initiatives': [],
                'data_improvements': [],
                'policy_considerations': [],
            }
    
    def generate_comparative_insights(
        self, 
        current_analysis: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> List[Insight]:
        """
        Generate comparative insights between current and historical data.
        
        Args:
            current_analysis: Current analysis results
            historical_data: Optional historical analysis for comparison
            
        Returns:
            List of comparative insights
        """
        try:
            insights = []
            
            if not historical_data:
                # Generate baseline insights without comparison
                insights.append(Insight(
                    title="Baseline Analysis Complete",
                    description="This analysis establishes baseline metrics for future comparisons.",
                    impact_level="medium",
                    category="baseline",
                    supporting_data=current_analysis,
                    recommendations=[
                        "Store current results for future trend analysis",
                        "Establish regular monitoring schedule",
                        "Define key performance indicators for tracking"
                    ]
                ))
                return insights
            
            # Compare key metrics if historical data available
            # This would be implemented based on specific comparison needs
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating comparative insights: {e}")
            return []
    
    # Private helper methods
    
    def _generate_student_population_insights(
        self, 
        student_stats: Dict[str, Any]
    ) -> List[Insight]:
        """Generate insights from student population statistics."""
        insights = []
        
        try:
            # Population size insights
            if student_stats:
                # This would analyze student population patterns
                # Implementation depends on the structure of student_stats
                pass
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating student population insights: {e}")
            return []
    
    def _identify_strategic_priorities(self, analyses: Dict[str, Any]) -> List[str]:
        """Identify top strategic priorities from all analyses."""
        priorities = []
        
        try:
            # Priority 1: Data Quality
            if 'data_quality' in analyses:
                quality_score = analyses['data_quality'].get('quality_score', 0)
                if quality_score < 70:
                    priorities.append("Improve data quality and collection processes")
            
            # Priority 2: Access Equity
            if 'grade_availability' in analyses:
                grade_analysis = analyses['grade_availability']
                grade_pct = grade_analysis.percentage if hasattr(grade_analysis, 'percentage') else 100
                if grade_pct < 90:
                    priorities.append("Expand educational access and availability")
            
            # Priority 3: Geographic Equity
            if 'borough_distribution' in analyses:
                borough_data = analyses['borough_distribution']
                concentration = borough_data.get('concentration_metrics', {})
                if concentration.get('top_3_concentration', 0) > 75:
                    priorities.append("Address geographic concentration of resources")
            
            # Priority 4: System Optimization
            if 'student_populations' in analyses:
                pop_data = analyses['student_populations']
                patterns = pop_data.get('patterns_and_outliers', {})
                if patterns.get('outlier_percentage', 0) > 15:
                    priorities.append("Optimize enrollment and capacity planning")
            
            # Default priorities if none identified
            if not priorities:
                priorities = [
                    "Maintain current performance standards",
                    "Continue regular monitoring and assessment",
                    "Plan for future growth and changes"
                ]
            
            return priorities[:5]  # Return top 5 priorities
            
        except Exception as e:
            self.logger.error(f"Error identifying strategic priorities: {e}")
            return ["Strategic priority analysis unavailable due to processing error"]
    
    def _determine_impact_level(
        self, 
        metric_value: float, 
        metric_type: str = 'percentage'
    ) -> str:
        """Determine impact level based on metric value and type."""
        try:
            thresholds = self.impact_thresholds
            
            if metric_type in ['percentage']:
                if metric_value >= thresholds['high']['percentage']:
                    return 'high'
                elif metric_value >= thresholds['medium']['percentage']:
                    return 'medium'
                else:
                    return 'low'
            elif metric_type in ['schools']:
                if metric_value >= thresholds['high']['schools']:
                    return 'high'
                elif metric_value >= thresholds['medium']['schools']:
                    return 'medium'
                else:
                    return 'low'
            elif metric_type in ['students']:
                if metric_value >= thresholds['high']['students']:
                    return 'high'
                elif metric_value >= thresholds['medium']['students']:
                    return 'medium'
                else:
                    return 'low'
            
            return 'medium'  # Default
            
        except Exception:
            return 'medium'