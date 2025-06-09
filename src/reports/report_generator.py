"""
Report generator module for Chicago Housing Pipeline & Population Shift Project.

This module generates reports from model outputs in markdown format.
"""
import os
import logging
import json
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Report generator for Chicago Housing Pipeline & Population Shift Project.
    
    This class generates reports from model outputs in markdown format.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize report generator.
        
        Args:
            output_dir (str, optional): Output directory for reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path('output/reports')
        self.template_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'templates' / 'markdown'
        # Add templates_dir attribute for compatibility with validation tests
        self.templates_dir = self.template_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate_multifamily_growth_report(self, model_results, visualization_paths=None):
        """
        Generate multifamily growth report.
        
        Args:
            model_results (dict): Model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Paths to generated reports
        """
        logger.info("Generating multifamily growth report")
        
        # Validate model results
        if not self._validate_model_results(model_results, 'multifamily_growth'):
            logger.warning("Invalid or empty multifamily growth model results, skipping report generation")
            return {}
        
        # Prepare context for template
        context = self._prepare_multifamily_growth_context(model_results, visualization_paths)
        
        # Generate reports
        report_paths = {}
        
        # Generate markdown report
        try:
            markdown_path = self._generate_markdown_report('multifamily_growth_template.md', context, 'multifamily_growth')
            if markdown_path:
                report_paths['markdown'] = markdown_path
        except Exception as e:
            logger.error(f"Error generating markdown report: multifamily_growth_template.md")
            logger.error(str(e))
        
        logger.info(f"Generated multifamily growth report: {report_paths}")
        return report_paths
    
    def generate_retail_gap_report(self, model_results, visualization_paths=None):
        """
        Generate retail gap report.
        
        Args:
            model_results (dict): Model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Paths to generated reports
        """
        logger.info("Generating retail gap report")
        
        # Validate model results
        if not self._validate_model_results(model_results, 'retail_gap'):
            logger.warning("Invalid or empty retail gap model results, skipping report generation")
            return {}
        
        # Prepare context for template
        context = self._prepare_retail_gap_context(model_results, visualization_paths)
        
        # Generate reports
        report_paths = {}
        
        # Generate markdown report
        try:
            markdown_path = self._generate_markdown_report('retail_gap_template.md', context, 'retail_gap')
            if markdown_path:
                report_paths['markdown'] = markdown_path
        except Exception as e:
            logger.error(f"Error generating markdown report: retail_gap_template.md")
            logger.error(str(e))
        
        logger.info(f"Generated retail gap report: {report_paths}")
        return report_paths
    
    def generate_retail_void_report(self, model_results, visualization_paths=None):
        """
        Generate retail void report.
        
        Args:
            model_results (dict): Model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Paths to generated reports
        """
        logger.info("Generating retail void report")
        
        # Validate model results
        if not self._validate_model_results(model_results, 'retail_void'):
            logger.warning("Invalid or empty retail void model results, skipping report generation")
            return {}
        
        # Prepare context for template
        context = self._prepare_retail_void_context(model_results, visualization_paths)
        
        # Generate reports
        report_paths = {}
        
        # Generate markdown report
        try:
            markdown_path = self._generate_markdown_report('retail_void_template.md', context, 'retail_void')
            if markdown_path:
                report_paths['markdown'] = markdown_path
        except Exception as e:
            logger.error(f"Error generating markdown report: retail_void_template.md")
            logger.error(str(e))
        
        logger.info(f"Generated retail void report: {report_paths}")
        return report_paths
    
    def generate_summary_report(self, multifamily_results=None, retail_gap_results=None, retail_void_results=None, visualization_paths=None):
        """
        Generate summary report.
        
        Args:
            multifamily_results (dict, optional): Multifamily growth model results
            retail_gap_results (dict, optional): Retail gap model results
            retail_void_results (dict, optional): Retail void model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Paths to generated reports
        """
        logger.info("Generating summary report")
        
        # Validate that at least one model has valid results
        if not (self._validate_model_results(multifamily_results, 'multifamily_growth', strict=False) or
                self._validate_model_results(retail_gap_results, 'retail_gap', strict=False) or
                self._validate_model_results(retail_void_results, 'retail_void', strict=False)):
            logger.warning("No valid model results available, skipping summary report generation")
            return {}
        
        # Prepare context for template
        context = self._prepare_summary_report_context(multifamily_results, retail_gap_results, retail_void_results, visualization_paths)
        
        # Generate reports
        report_paths = {}
        
        # Generate markdown report
        try:
            markdown_path = self._generate_markdown_report('summary_report_template.md', context, 'summary')
            if markdown_path:
                report_paths['markdown'] = markdown_path
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info(f"Generated summary report: {report_paths}")
        return report_paths
    
    def _validate_model_results(self, model_results, model_type, strict=True):
        """
        Validate model results to ensure they contain required data.
        
        Args:
            model_results (dict): Model results to validate
            model_type (str): Type of model ('multifamily_growth', 'retail_gap', 'retail_void')
            strict (bool): Whether to strictly enforce all requirements
            
        Returns:
            bool: True if results are valid, False otherwise
        """
        if not model_results:
            logger.warning(f"No model results provided for {model_type}")
            return False
        
        if not isinstance(model_results, dict):
            logger.warning(f"Model results for {model_type} are not a dictionary")
            return False
        
        # Check for empty results
        if len(model_results) == 0:
            logger.warning(f"Empty model results for {model_type}")
            return False
        
        # Check for default/placeholder results
        if strict:
            if model_type == 'multifamily_growth':
                # Check if top emerging zips is empty or contains only default values
                if 'top_emerging_zips' in model_results:
                    if not model_results['top_emerging_zips'] or (
                        isinstance(model_results['top_emerging_zips'], list) and 
                        len(model_results['top_emerging_zips']) == 0
                    ):
                        logger.warning(f"Empty top_emerging_zips in {model_type} results")
                        return False
                elif 'top_growth_zips' in model_results:
                    if not model_results['top_growth_zips'] or (
                        isinstance(model_results['top_growth_zips'], list) and 
                        len(model_results['top_growth_zips']) == 0
                    ):
                        logger.warning(f"Empty top_growth_zips in {model_type} results")
                        return False
                else:
                    logger.warning(f"Missing required key top_emerging_zips or top_growth_zips in {model_type} results")
                    return False
                
            elif model_type == 'retail_gap':
                # Check if opportunity zones is empty or contains only default values
                if 'opportunity_zones' in model_results:
                    if not model_results['opportunity_zones'] or (
                        isinstance(model_results['opportunity_zones'], list) and 
                        len(model_results['opportunity_zones']) == 0
                    ):
                        logger.warning(f"Empty opportunity_zones in {model_type} results")
                        return False
                else:
                    logger.warning(f"Missing required key opportunity_zones in {model_type} results")
                    return False
                
            elif model_type == 'retail_void':
                # Check if void zones is empty or contains only default values
                if 'void_zones' in model_results:
                    if not model_results['void_zones'] or (
                        isinstance(model_results['void_zones'], list) and 
                        len(model_results['void_zones']) == 0
                    ):
                        logger.warning(f"Empty void_zones in {model_type} results")
                        return False
                else:
                    logger.warning(f"Missing required key void_zones in {model_type} results")
                    return False
        
        return True
    
    def _prepare_multifamily_growth_context(self, model_results, visualization_paths=None):
        """
        Prepare context for multifamily growth template.
        
        Args:
            model_results (dict): Model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Context for template
        """
        # Initialize context with defaults
        context = {
            'title': 'Multifamily Growth Analysis Report',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': 'This report analyzes multifamily housing growth patterns in Chicago.',
            'top_emerging_zips': [],
            'growth_metrics': {},
            'permit_analysis': {},
            'methodology': 'Standard multifamily growth analysis methodology was applied.',
            'visualizations': {}
        }
        
        # Update with model results
        if model_results:
            # Add top emerging zips
            if 'top_emerging_zips' in model_results and model_results['top_emerging_zips']:
                context['top_emerging_zips'] = self._ensure_list_format(model_results['top_emerging_zips'])
            elif 'top_growth_zips' in model_results and model_results['top_growth_zips']:
                context['top_emerging_zips'] = self._ensure_list_format(model_results['top_growth_zips'])
            
            # Add growth metrics
            if 'growth_metrics' in model_results and model_results['growth_metrics']:
                context['growth_metrics'] = self._ensure_dict_format(model_results['growth_metrics'])
            
            # Add growth rates if available
            if 'growth_rates' in model_results and model_results['growth_rates']:
                growth_rates = self._ensure_dict_format(model_results['growth_rates'])
                if 'growth_metrics' not in context or not context['growth_metrics']:
                    context['growth_metrics'] = {}
                
                # Add growth rates to metrics
                for key, value in growth_rates.items():
                    context['growth_metrics'][f'Growth Rate - {key}'] = value
            
            # Add permit analysis
            if 'permit_analysis' in model_results and model_results['permit_analysis']:
                context['permit_analysis'] = self._ensure_dict_format(model_results['permit_analysis'])
            
            # Add summary
            if 'summary' in model_results and model_results['summary']:
                # Extract key metrics from summary
                summary_metrics = {}
                if isinstance(model_results['summary'], dict):
                    for key, value in model_results['summary'].items():
                        if key not in ['total_zips_analyzed', 'emerging_zips_identified', 'top_zip_code', 
                                      'top_growth_score', 'avg_permit_growth', 'avg_unit_growth']:
                            continue
                        summary_metrics[key] = value
                
                # Add summary metrics to context
                if summary_metrics:
                    if 'growth_metrics' not in context or not context['growth_metrics']:
                        context['growth_metrics'] = {}
                    
                    for key, value in summary_metrics.items():
                        context['growth_metrics'][key] = value
                
                # Add summary text
                if isinstance(model_results['summary'], dict) and 'text' in model_results['summary']:
                    context['summary'] = model_results['summary']['text']
                elif isinstance(model_results['summary'], str):
                    context['summary'] = model_results['summary']
            
            # Add analysis insights if available
            if 'analysis' in model_results and model_results['analysis']:
                if isinstance(model_results['analysis'], dict) and 'insights' in model_results['analysis']:
                    insights = model_results['analysis']['insights']
                    if insights and isinstance(insights, list) and len(insights) > 0:
                        context['summary'] += "\n\n**Key Insights:**\n\n"
                        for insight in insights:
                            context['summary'] += f"- {insight}\n"
            
            # Add methodology
            if 'methodology' in model_results and model_results['methodology']:
                context['methodology'] = model_results['methodology']
        
        # Add visualization paths
        if visualization_paths:
            context['visualizations'] = visualization_paths
        
        return context
    
    def _prepare_retail_gap_context(self, model_results, visualization_paths=None):
        """
        Prepare context for retail gap template.
        
        Args:
            model_results (dict): Model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Context for template
        """
        # Initialize context with defaults
        context = {
            'title': 'Retail Gap Analysis Report',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': 'This report analyzes retail gaps in Chicago.',
            'opportunity_zones': [],
            'gap_stats': {},
            'retail_clusters': [],
            'category_gaps': {},
            'methodology': 'Standard retail gap analysis methodology was applied.',
            'visualizations': {}
        }
        
        # Extract data from model results
        if 'opportunity_zones' in model_results and 'opportunity_scores' in model_results:
            # Transform opportunity zones into expected format
            opportunity_zones = []
            for zip_code in model_results['opportunity_zones']:
                score = model_results['opportunity_scores'].get(zip_code, 0)
                opportunity_zones.append({
                    'zip_code': zip_code,
                    'retail_gap_score': score,
                    'gap_score': score
                })
            context['opportunity_zones'] = opportunity_zones
        
        # Extract gap statistics from leakage zones or cluster data
        if 'cluster_insights' in model_results:
            clusters = model_results['cluster_insights']
            if clusters:
                # Calculate statistics from clusters
                all_gaps = []
                for cluster in clusters:
                    all_gaps.extend([cluster.get('avg_gap', 0)] * cluster.get('zip_count', 0))
                
                if all_gaps:
                    context['gap_stats'] = {
                        'mean_gap': sum(all_gaps) / len(all_gaps),
                        'median_gap': sorted(all_gaps)[len(all_gaps)//2],
                        'min_gap': min(all_gaps),
                        'max_gap': max(all_gaps),
                        'std_gap': (sum((x - sum(all_gaps)/len(all_gaps))**2 for x in all_gaps) / len(all_gaps))**0.5,
                        'opportunity_count': len(model_results.get('opportunity_zones', [])),
                        'saturated_count': sum(1 for cluster in clusters if cluster.get('avg_gap', 0) < 0)
                    }
                
                # Format retail clusters
                context['retail_clusters'] = []
                for cluster in clusters:
                    context['retail_clusters'].append({
                        'retail_cluster': cluster.get('cluster_id', 0),
                        'zip_count': cluster.get('zip_count', 0),
                        'retail_per_capita': abs(cluster.get('avg_gap', 0)) * 10,  # Scale for display
                        'population': cluster.get('zip_count', 0) * 25000  # Estimate
                    })
        
        # Add visualization paths
        if visualization_paths:
            context['visualizations'] = visualization_paths
        elif 'visualizations' in model_results and 'paths' in model_results['visualizations']:
            context['visualizations'] = model_results['visualizations']['paths']
        
        # Update with model results
        if model_results:
            # Add opportunity zones
            if 'opportunity_zones' in model_results and model_results['opportunity_zones']:
                context['opportunity_zones'] = self._ensure_list_format(model_results['opportunity_zones'])
            
            # Add gap metrics
            if 'gap_metrics' in model_results and model_results['gap_metrics']:
                context['gap_metrics'] = self._ensure_dict_format(model_results['gap_metrics'])
            
            # Add gap analysis if available
            if 'gap_analysis' in model_results and model_results['gap_analysis']:
                gap_analysis = self._ensure_list_format(model_results['gap_analysis'])
                if gap_analysis:
                    context['opportunity_zones'] = gap_analysis
            
            # Add summary
            if 'summary' in model_results and model_results['summary']:
                if isinstance(model_results['summary'], dict) and 'text' in model_results['summary']:
                    context['summary'] = model_results['summary']['text']
                elif isinstance(model_results['summary'], str):
                    context['summary'] = model_results['summary']
            
            # Add methodology
            if 'methodology' in model_results and model_results['methodology']:
                context['methodology'] = model_results['methodology']
        
        # Add visualization paths
        if visualization_paths:
            context['visualizations'] = visualization_paths
        
        return context
    
    def _prepare_retail_void_context(self, model_results, visualization_paths=None):
        """
        Prepare context for retail void template.
        
        Args:
            model_results (dict): Model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Context for template
        """
        # Initialize context with defaults
        context = {
            'title': 'Retail Void Analysis Report',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': 'This report analyzes retail voids in Chicago.',
            'void_zones': [],
            'void_stats': {},
            'category_voids': {},
            'leakage_patterns': {},
            'methodology': 'Standard retail void analysis methodology was applied.',
            'visualizations': {}
        }
        
        # Extract void zones from model results
        if 'void_zones' in model_results and model_results['void_zones']:
            context['void_zones'] = model_results['void_zones']
        
        # Extract void statistics from leakage zones
        if 'leakage_zones' in model_results:
            leakage_data = model_results['leakage_zones']
            context['void_stats'] = {
                'mean_leakage': leakage_data.get('avg_leakage', 0),
                'median_leakage': leakage_data.get('avg_leakage', 0),  # Use avg as approximation
                'min_leakage': leakage_data.get('min_leakage', 0),
                'max_leakage': leakage_data.get('max_leakage', 0),
                'std_leakage': abs(leakage_data.get('max_leakage', 0) - leakage_data.get('min_leakage', 0)) / 4,  # Approximation
                'void_count': len(model_results.get('void_zones', [])),
                'total_zips': len(leakage_data.get('low_leakage_zips', [])) + len(leakage_data.get('high_leakage_zips', [])),
                'void_percentage': (len(model_results.get('void_zones', [])) / max(1, len(leakage_data.get('low_leakage_zips', [])) + len(leakage_data.get('high_leakage_zips', [])))) * 100
            }
        
        # Extract category voids
        if 'category_voids' in model_results:
            context['category_voids'] = model_results['category_voids']
        
        # Add visualization paths
        if visualization_paths:
            context['visualizations'] = visualization_paths
        elif 'visualizations' in model_results and 'paths' in model_results['visualizations']:
            context['visualizations'] = model_results['visualizations']['paths']
        
        # Update with model results
        if model_results:
            # Add void zones (now properly structured)
            if 'void_zones' in model_results and model_results['void_zones']:
                context['void_zones'] = model_results['void_zones']
            
            # Add void stats from leakage_zones
            if 'leakage_zones' in model_results and model_results['leakage_zones']:
                leakage_data = model_results['leakage_zones']
                context['void_stats'] = {
                    'mean_leakage': leakage_data.get('avg_leakage', 0),
                    'median_leakage': leakage_data.get('avg_leakage', 0),  # Approximation
                    'min_leakage': leakage_data.get('min_leakage', 0),
                    'max_leakage': leakage_data.get('max_leakage', 0),
                    'std_leakage': 0.1,  # Default value
                    'void_count': len(model_results.get('void_zones', [])),
                    'total_zips': len(leakage_data.get('low_leakage_zips', [])),
                    'void_percentage': (len(model_results.get('void_zones', [])) / max(1, len(leakage_data.get('low_leakage_zips', [])))) * 100
                }
            
            # Add category voids
            if 'category_voids' in model_results and model_results['category_voids']:
                context['category_voids'] = model_results['category_voids']
            
            # Add leakage patterns
            if 'leakage_zones' in model_results and model_results['leakage_zones']:
                context['leakage_patterns'] = model_results['leakage_zones']
            
            # Add summary from void_analysis
            if 'void_analysis' in model_results and model_results['void_analysis']:
                analysis = model_results['void_analysis']
                if 'insights' in analysis and analysis['insights']:
                    context['summary'] = ' '.join(analysis['insights'][:2])  # Use first 2 insights as summary
            
            # Add methodology
            if 'methodology' in model_results and model_results['methodology']:
                context['methodology'] = model_results['methodology']
        
        # Add visualization paths
        if visualization_paths:
            context['visualizations'] = visualization_paths
        elif 'visualizations' in model_results and 'paths' in model_results['visualizations']:
            context['visualizations'] = model_results['visualizations']['paths']
        
        return context
    
    def _prepare_summary_report_context(self, multifamily_results=None, retail_gap_results=None, retail_void_results=None, visualization_paths=None):
        """
        Prepare context for summary report template.
        
        Args:
            multifamily_results (dict, optional): Multifamily growth model results
            retail_gap_results (dict, optional): Retail gap model results
            retail_void_results (dict, optional): Retail void model results
            visualization_paths (dict, optional): Paths to visualizations
            
        Returns:
            dict: Context for template
        """
        # Initialize context with defaults
        context = {
            'title': 'Chicago Housing Pipeline & Population Shift Project Summary Report',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'pipeline_summary': 'This report summarizes the findings from the Chicago Housing Pipeline & Population Shift Project analysis.',
            'key_findings': [],
            'top_emerging_zips': [],
            'opportunity_zones': [],
            'void_zones': [],
            'model_summaries': {
                'multifamily_growth': 'Multifamily growth analysis identified emerging areas for residential development.',
                'retail_gap': 'Retail gap analysis identified areas with retail development potential.',
                'retail_void': 'Retail void analysis identified specific retail categories that are underrepresented in different areas.'
            },
            'methodology': 'Standard methodology was applied across all analyses.',
            'visualizations': {}
        }
        
        # Update with multifamily growth results
        if multifamily_results:
            # Add top emerging zips
            if 'top_emerging_zips' in multifamily_results and multifamily_results['top_emerging_zips']:
                context['top_emerging_zips'] = self._ensure_list_format(multifamily_results['top_emerging_zips'])
            elif 'top_growth_zips' in multifamily_results and multifamily_results['top_growth_zips']:
                context['top_emerging_zips'] = self._ensure_list_format(multifamily_results['top_growth_zips'])
            
            # Add multifamily growth summary
            if 'summary' in multifamily_results and multifamily_results['summary']:
                if isinstance(multifamily_results['summary'], dict) and 'text' in multifamily_results['summary']:
                    context['model_summaries']['multifamily_growth'] = multifamily_results['summary']['text']
                elif isinstance(multifamily_results['summary'], str):
                    context['model_summaries']['multifamily_growth'] = multifamily_results['summary']
            
            # Add key findings from multifamily growth
            if 'analysis' in multifamily_results and multifamily_results['analysis']:
                if isinstance(multifamily_results['analysis'], dict) and 'insights' in multifamily_results['analysis']:
                    insights = multifamily_results['analysis']['insights']
                    if insights and isinstance(insights, list) and len(insights) > 0:
                        context['key_findings'].extend(insights)
        
        # Update with retail gap results
        if retail_gap_results:
            # Add opportunity zones with proper structure
            if 'opportunity_zones' in retail_gap_results and 'opportunity_scores' in retail_gap_results:
                opportunity_zones = []
                for zip_code in retail_gap_results['opportunity_zones'][:5]:  # Top 5
                    score = retail_gap_results['opportunity_scores'].get(zip_code, 0)
                    opportunity_zones.append({
                        'zip_code': zip_code,
                        'opportunity_score': score,
                        'retail_gap_score': score
                    })
                context['opportunity_zones'] = opportunity_zones
            elif 'opportunity_zones' in retail_gap_results and retail_gap_results['opportunity_zones']:
                context['opportunity_zones'] = self._ensure_list_format(retail_gap_results['opportunity_zones'])
            
            # Add retail gap summary
            if 'analysis_summary' in retail_gap_results:
                context['model_summaries']['retail_gap'] = retail_gap_results['analysis_summary']
            elif 'summary' in retail_gap_results and retail_gap_results['summary']:
                if isinstance(retail_gap_results['summary'], dict) and 'text' in retail_gap_results['summary']:
                    context['model_summaries']['retail_gap'] = retail_gap_results['summary']['text']
                elif isinstance(retail_gap_results['summary'], str):
                    context['model_summaries']['retail_gap'] = retail_gap_results['summary']
            
            # Add key findings from retail gap
            if 'analysis' in retail_gap_results and retail_gap_results['analysis']:
                if isinstance(retail_gap_results['analysis'], dict) and 'insights' in retail_gap_results['analysis']:
                    insights = retail_gap_results['analysis']['insights']
                    if insights and isinstance(insights, list) and len(insights) > 0:
                        context['key_findings'].extend(insights)
        
        # Update with retail void results
        if retail_void_results:
            # Add void zones with proper structure
            if 'void_zones' in retail_void_results and retail_void_results['void_zones']:
                # Ensure void zones have proper structure for template
                void_zones = []
                for zone in retail_void_results['void_zones'][:3]:  # Top 3
                    if isinstance(zone, dict):
                        void_zones.append({
                            'zip_code': zone.get('zip_code', 'N/A'),
                            'leakage_score': zone.get('leakage_ratio', 0),
                            'leakage_ratio': zone.get('leakage_ratio', 0),
                            'retail_per_capita': zone.get('retail_per_capita', 0),
                            'population': zone.get('population', 0)
                        })
                    else:
                        # Handle case where zone is just a string
                        void_zones.append({
                            'zip_code': str(zone),
                            'leakage_score': 'N/A',
                            'leakage_ratio': 0,
                            'retail_per_capita': 0,
                            'population': 0
                        })
                context['void_zones'] = void_zones
            
            # Add retail void summary
            if 'void_analysis' in retail_void_results and 'insights' in retail_void_results['void_analysis']:
                insights = retail_void_results['void_analysis']['insights']
                if isinstance(insights, list):
                    context['model_summaries']['retail_void'] = '. '.join(insights)
                else:
                    context['model_summaries']['retail_void'] = str(insights)
            elif 'summary' in retail_void_results and retail_void_results['summary']:
                if isinstance(retail_void_results['summary'], dict) and 'text' in retail_void_results['summary']:
                    context['model_summaries']['retail_void'] = retail_void_results['summary']['text']
                elif isinstance(retail_void_results['summary'], str):
                    context['model_summaries']['retail_void'] = retail_void_results['summary']
            
            # Add key findings from retail void
            if 'void_analysis' in retail_void_results and retail_void_results['void_analysis']:
                if isinstance(retail_void_results['void_analysis'], dict) and 'insights' in retail_void_results['void_analysis']:
                    insights = retail_void_results['void_analysis']['insights']
                    if insights and isinstance(insights, list) and len(insights) > 0:
                        context['key_findings'].extend(insights)
        
        # Add visualization paths
        if visualization_paths:
            context['visualizations'] = visualization_paths
        
        return context
    
    def _generate_markdown_report(self, template_name, context, report_type):
        """
        Generate markdown report.
        
        Args:
            template_name (str): Name of template file
            context (dict): Context for template
            report_type (str): Type of report
            
        Returns:
            str: Path to generated report
        """
        try:
            # Get template
            template = self.env.get_template(template_name)
            
            # Render template
            rendered = template.render(**context)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{report_type}_report_{timestamp}.md"
            
            # Save to file
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                f.write(rendered)
            
            logger.info(f"Generated markdown report: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _ensure_list_format(self, data):
        """
        Ensure data is in list format.
        
        Args:
            data: Data to convert to list
            
        Returns:
            list: Data in list format
        """
        if data is None:
            return []
        
        if isinstance(data, list):
            return data
        
        if isinstance(data, dict):
            return [data]
        
        return [data]
    
    def _ensure_dict_format(self, data):
        """
        Ensure data is in dict format.
        
        Args:
            data: Data to convert to dict
            
        Returns:
            dict: Data in dict format
        """
        if data is None:
            return {}
        
        if isinstance(data, dict):
            return data
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Convert list of dicts to single dict
            result = {}
            for i, item in enumerate(data):
                for key, value in item.items():
                    result[f"{key}_{i+1}"] = value
            return result
        
        return {'value': data}
