"""
Base class for report generation in Markdown format.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import os
import json

logger = logging.getLogger(__name__)

class BaseReport:
    """Base class for generating reports in Markdown format."""
    
    def __init__(self, report_name, output_dir=None, template_dir=None):
        """
        Initialize the base report.
        
        Args:
            report_name (str): Name of the report
            output_dir (Path, optional): Directory to save report outputs
            template_dir (Path, optional): Directory containing report templates
        """
        self.report_name = report_name
        
        # Set output directory
        if output_dir is None:
            from src.config import settings
            self.output_dir = settings.REPORTS_DIR
        else:
            self.output_dir = Path(output_dir)
        
        # Set template directory
        if template_dir is None:
            from src.config import settings
            # Use the PROJECT_ROOT from settings to locate templates
            self.template_dir = settings.PROJECT_ROOT / "src" / "templates" / "reports"
        else:
            self.template_dir = Path(template_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figures directory
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        
        # Initialize data and context
        self.data = None
        self.context = {}
        self.model_results = None
        self.error_messages = []
    
    def generate(self, data=None, model_results=None, output_filename=None):
        """
        Generate report from data and model results.
        
        Args:
            data (pd.DataFrame, optional): Input data for report
            model_results (dict, optional): Model analysis results
            output_filename (str, optional): Output filename
            
        Returns:
            bool: True if report generation successful, False otherwise
        """
        try:
            logger.info(f"Generating {self.report_name} report...")
            
            # Store data and model results
            self.data = data
            self.model_results = model_results
            
            # Validate inputs
            if not self._validate_inputs():
                logger.error(f"Invalid inputs for {self.report_name} report")
                return False
            
            # Filter to valid ZIP codes if data is available
            if self.data is not None:
                self.data = self._filter_valid_zips(self.data)
                
                # Flag insufficient data
                self.data = self._flag_insufficient_data(self.data)
            
            # Prepare data for report
            logger.info(f"Data prepared for {self.report_name} report")
            
            # Prepare report context
            self._prepare_report_context()
            
            # Load template
            template_content = self._load_template()
            if template_content is None:
                logger.error(f"Failed to load template for {self.report_name} report")
                return False
            
            # Render template
            report_content = self._render_template(template_content)
            if report_content is None:
                logger.error(f"Failed to render template for {self.report_name} report")
                return False
            
            # Save report
            if output_filename is None:
                output_filename = f"{self.report_name.lower().replace(' ', '_')}_report.md"
            
            output_path = self.output_dir / output_filename
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"{self.report_name} report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating {self.report_name} report: {str(e)}")
            self.error_messages.append(str(e))
            return False
    
    def _validate_inputs(self):
        """
        Validate input data and model results.
        
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        try:
            # Check if either data or model results are provided
            if self.data is None and self.model_results is None:
                logger.warning(f"Both data and model results are missing for {self.report_name} report")
                self.error_messages.append("Missing both data and model results")
                # We'll still try to generate a report with error messages
                return True
            
            # Check data if provided
            if self.data is not None:
                if not isinstance(self.data, pd.DataFrame):
                    logger.error(f"Invalid data type for {self.report_name} report: {type(self.data)}")
                    self.error_messages.append(f"Invalid data type: {type(self.data)}")
                    self.data = None
                elif self.data.empty:
                    logger.warning(f"Empty data for {self.report_name} report")
                    self.error_messages.append("Empty data")
            
            # Check model results if provided
            if self.model_results is not None:
                if not isinstance(self.model_results, dict):
                    logger.error(f"Invalid model results type for {self.report_name} report: {type(self.model_results)}")
                    self.error_messages.append(f"Invalid model results type: {type(self.model_results)}")
                    self.model_results = None
                elif not self.model_results:
                    logger.warning(f"Empty model results for {self.report_name} report")
                    self.error_messages.append("Empty model results")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating inputs: {str(e)}")
            self.error_messages.append(f"Input validation error: {str(e)}")
            return False
    
    def _filter_valid_zips(self, df):
        """
        Filter dataframe to include only valid ZIP codes.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        try:
            # This is a placeholder - subclasses should implement specific filtering
            if df is None:
                return None
                
            # Basic validation: ZIP codes should be strings of 5 digits
            if 'zip_code' in df.columns:
                # Convert to string if not already
                df['zip_code'] = df['zip_code'].astype(str)
                
                # Filter to valid ZIP codes (5 digits)
                valid_mask = df['zip_code'].str.match(r'^\d{5}$')
                
                # Log invalid ZIP codes
                invalid_zips = df.loc[~valid_mask, 'zip_code'].unique()
                if len(invalid_zips) > 0:
                    logger.warning(f"Found {len(invalid_zips)} invalid ZIP codes: {', '.join(invalid_zips)}")
                
                # Return filtered dataframe
                return df[valid_mask].copy()
            
            return df
            
        except Exception as e:
            logger.error(f"Error filtering valid ZIP codes: {str(e)}")
            self.error_messages.append(f"ZIP code filtering error: {str(e)}")
            return df
    
    def _flag_insufficient_data(self, df):
        """
        Flag rows with insufficient data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with data_status column
        """
        try:
            # This is a placeholder - subclasses should implement specific flagging
            if df is None:
                return None
                
            if 'data_status' not in df.columns:
                df['data_status'] = 'ok'
                
            return df
            
        except Exception as e:
            logger.error(f"Error flagging insufficient data: {str(e)}")
            self.error_messages.append(f"Data flagging error: {str(e)}")
            return df
    
    def _prepare_report_context(self):
        """
        Prepare context dictionary for report template.
        To be implemented by subclasses.
        """
        self.context = {
            'report_name': self.report_name,
            'generation_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'data_count': len(self.data) if self.data is not None else 0,
            'error_messages': self.error_messages,
            'has_errors': len(self.error_messages) > 0,
            'has_data': self.data is not None and not self.data.empty,
            'has_model_results': self.model_results is not None and bool(self.model_results)
        }
    
    def _load_template(self):
        """
        Load report template.
        
        Returns:
            str: Template content
        """
        try:
            # Create a normalized template name from the report name
            template_name = f"{self.report_name.lower().replace(' ', '_')}_template.md"
            template_path = self.template_dir / template_name
            
            # Check if template exists
            if not template_path.exists():
                logger.warning(f"Template file not found: {template_path}")
                # Fall back to a default template
                default_template = """# {{ report_name }} Report

**Generated on:** {{ generation_date }}

{% if has_errors %}
## Errors

The following errors occurred during report generation:

{% for error in error_messages %}
- {{ error }}
{% endfor %}

{% endif %}

## Summary

{% if has_data %}
This report contains analysis results for {{ data_count }} records.
{% else %}
No data was available for this report.
{% endif %}

{% if has_model_results %}
## Key Findings

{% for key, value in model_results.items() if key != 'visualizations' %}
- {{ key }}: {{ value }}
{% endfor %}
{% else %}
No model results were available for this report.
{% endif %}

## Methodology

This analysis was conducted using standard statistical methods.

---

*This report was generated by the Chicago Population Analysis Project*
"""
                logger.warning(f"Using default template for {self.report_name} report")
                return default_template
            
            # Read template content
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            return template_content
            
        except Exception as e:
            logger.error(f"Error loading template for {self.report_name} report: {str(e)}")
            self.error_messages.append(f"Template loading error: {str(e)}")
            
            # Return a minimal error template
            error_template = """# {{ report_name }} Report - ERROR

**Generated on:** {{ generation_date }}

## Errors

The following errors occurred during report generation:

{% for error in error_messages %}
- {{ error }}
{% endfor %}

---

*This report was generated by the Chicago Population Analysis Project*
"""
            return error_template
    
    def _render_template(self, template_content):
        """
        Render template with context.
        
        Args:
            template_content (str): Template content
            
        Returns:
            str: Rendered content
        """
        try:
            template = Template(template_content)
            rendered = template.render(**self.context)
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering template for {self.report_name} report: {str(e)}")
            self.error_messages.append(f"Template rendering error: {str(e)}")
            
            # Return a simple error message
            error_message = f"""# {self.report_name} Report - RENDERING ERROR

**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Error

Failed to render report template: {str(e)}

---

*This report was generated by the Chicago Population Analysis Project*
"""
            return error_message
    
    def create_visualization(self, data, chart_type, x=None, y=None, title=None, filename=None):
        """
        Create visualization for the report.
        
        Args:
            data (pd.DataFrame): Data for visualization
            chart_type (str): Type of chart ('bar', 'line', 'scatter', etc.)
            x (str): Column name for x-axis
            y (str): Column name for y-axis
            title (str): Chart title
            filename (str): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Validate inputs
            if data is None or data.empty:
                logger.error("Cannot create visualization: data is None or empty")
                return None
                
            if x is not None and x not in data.columns:
                logger.error(f"Column '{x}' not found in data")
                return None
                
            if y is not None and y not in data.columns:
                logger.error(f"Column '{y}' not found in data")
                return None
            
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'bar':
                sns.barplot(data=data, x=x, y=y)
            elif chart_type == 'line':
                sns.lineplot(data=data, x=x, y=y)
            elif chart_type == 'scatter':
                sns.scatterplot(data=data, x=x, y=y)
            elif chart_type == 'heatmap':
                sns.heatmap(data=data, cmap='viridis', annot=True)
            elif chart_type == 'pie':
                plt.pie(data[y], labels=data[x], autopct='%1.1f%%')
            else:
                logger.error(f"Unsupported chart type: {chart_type}")
                return None
            
            if title:
                plt.title(title)
            
            plt.tight_layout()
            
            if filename is None:
                filename = f"{chart_type}_{x}_{y}.png"
            
            output_path = self.output_dir / "figures" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_path)
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
