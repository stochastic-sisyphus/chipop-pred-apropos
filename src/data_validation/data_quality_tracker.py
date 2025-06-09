"""
Data quality tracking for the Chicago Housing Pipeline project.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class DataQualityTracker:
    """Tracks data quality, sources, and imputation methods."""
    
    def __init__(self):
        """Initialize the data quality tracker."""
        self.quality_log = []
        self.imputation_log = []
        self.source_log = []
        self.data_freshness = {}
    
    def log_data_source(self, dataset_name, source_type, record_count, freshness=None):
        """
        Log data source information.
        
        Args:
            dataset_name (str): Name of the dataset
            source_type (str): Type of source ('api', 'cache', 'sample', 'calculated')
            record_count (int): Number of records
            freshness (str, optional): When data was collected
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'source_type': source_type,
            'record_count': record_count,
            'freshness': freshness
        }
        self.source_log.append(entry)
        
        if source_type == 'api':
            logger.info(f"✅ REAL DATA: {dataset_name} - {record_count} records from API")
        elif source_type == 'cache':
            logger.warning(f"📦 CACHED DATA: {dataset_name} - {record_count} records from cache (age: {freshness})")
        elif source_type == 'sample':
            logger.warning(f"🎭 SAMPLE DATA: {dataset_name} - {record_count} synthetic records")
        elif source_type == 'calculated':
            logger.warning(f"🧮 CALCULATED DATA: {dataset_name} - {record_count} derived records")
    
    def log_imputation(self, dataset_name, column_name, method, affected_records, total_records):
        """
        Log data imputation/filling operations.
        
        Args:
            dataset_name (str): Name of the dataset
            column_name (str): Column that was imputed
            method (str): Imputation method used
            affected_records (int): Number of records that were imputed
            total_records (int): Total number of records in dataset
        """
        imputation_pct = (affected_records / total_records * 100) if total_records > 0 else 0
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'column': column_name,
            'method': method,
            'affected_records': affected_records,
            'total_records': total_records,
            'imputation_percentage': imputation_pct
        }
        self.imputation_log.append(entry)
        
        if imputation_pct > 50:
            logger.error(f"❌ HIGH IMPUTATION: {dataset_name}.{column_name} - {imputation_pct:.1f}% imputed with {method}")
        elif imputation_pct > 20:
            logger.warning(f"⚠️ MODERATE IMPUTATION: {dataset_name}.{column_name} - {imputation_pct:.1f}% imputed with {method}")
        else:
            logger.info(f"✨ LOW IMPUTATION: {dataset_name}.{column_name} - {imputation_pct:.1f}% imputed with {method}")
    
    def log_quality_issue(self, dataset_name, issue_type, description, severity='warning'):
        """
        Log data quality issues.
        
        Args:
            dataset_name (str): Name of the dataset
            issue_type (str): Type of quality issue
            description (str): Description of the issue
            severity (str): Severity level ('info', 'warning', 'error')
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'issue_type': issue_type,
            'description': description,
            'severity': severity
        }
        self.quality_log.append(entry)
        
        if severity == 'error':
            logger.error(f"🚨 DATA QUALITY ERROR: {dataset_name} - {description}")
        elif severity == 'warning':
            logger.warning(f"⚠️ DATA QUALITY WARNING: {dataset_name} - {description}")
        else:
            logger.info(f"ℹ️ DATA QUALITY INFO: {dataset_name} - {description}")
    
    def get_quality_summary(self):
        """
        Get a summary of data quality for the current pipeline run.
        
        Returns:
            dict: Quality summary statistics
        """
        total_datasets = len(set(entry['dataset'] for entry in self.source_log))
        
        # Count data sources by type
        source_counts = {}
        for entry in self.source_log:
            source_type = entry['source_type']
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        # Count imputation severity
        high_imputation = sum(1 for entry in self.imputation_log if entry['imputation_percentage'] > 50)
        moderate_imputation = sum(1 for entry in self.imputation_log if 20 < entry['imputation_percentage'] <= 50)
        low_imputation = sum(1 for entry in self.imputation_log if entry['imputation_percentage'] <= 20)
        
        # Count quality issues by severity
        quality_issues = {}
        for entry in self.quality_log:
            severity = entry['severity']
            quality_issues[severity] = quality_issues.get(severity, 0) + 1
        
        return {
            'total_datasets': total_datasets,
            'data_sources': source_counts,
            'imputation_summary': {
                'high_imputation_columns': high_imputation,
                'moderate_imputation_columns': moderate_imputation,
                'low_imputation_columns': low_imputation,
                'total_imputed_columns': len(self.imputation_log)
            },
            'quality_issues': quality_issues,
            'data_reliability_score': self._calculate_reliability_score()
        }
    
    def _calculate_reliability_score(self):
        """
        Calculate a data reliability score (0-100).
        
        Returns:
            float: Reliability score
        """
        if not self.source_log:
            return 0
        
        score = 100
        
        # Penalize non-API data sources
        for entry in self.source_log:
            if entry['source_type'] == 'sample':
                score -= 30
            elif entry['source_type'] == 'cache':
                score -= 10
            elif entry['source_type'] == 'calculated':
                score -= 5
        
        # Penalize high imputation
        for entry in self.imputation_log:
            if entry['imputation_percentage'] > 50:
                score -= 20
            elif entry['imputation_percentage'] > 20:
                score -= 10
        
        # Penalize quality issues
        for entry in self.quality_log:
            if entry['severity'] == 'error':
                score -= 15
            elif entry['severity'] == 'warning':
                score -= 5
        
        return max(0, min(100, score))
    
    def generate_quality_report(self, output_dir):
        """
        Generate a comprehensive data quality report.
        
        Args:
            output_dir (str or Path): Directory to save the report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate summary
        summary = self.get_quality_summary()
        
        # Create detailed report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'detailed_logs': {
                'data_sources': self.source_log,
                'imputations': self.imputation_log,
                'quality_issues': self.quality_log
            }
        }
        
        # Save JSON report
        report_path = output_dir / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated data quality report: {report_path}")
        
        # Create human-readable summary
        summary_path = output_dir / f"data_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self._generate_markdown_summary(summary, summary_path)
        
        return {
            'json_report': str(report_path),
            'markdown_summary': str(summary_path),
            'reliability_score': summary['data_reliability_score']
        }
    
    def _generate_markdown_summary(self, summary, output_path):
        """Generate a markdown summary of data quality."""
        with open(output_path, 'w') as f:
            f.write("# Data Quality Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Reliability Score:** {summary['data_reliability_score']:.0f}/100\n\n")
            
            f.write("## Data Sources\n\n")
            for source_type, count in summary['data_sources'].items():
                f.write(f"- **{source_type.title()}:** {count} datasets\n")
            f.write("\n")
            
            f.write("## Imputation Summary\n\n")
            imp = summary['imputation_summary']
            f.write(f"- **High imputation (>50%):** {imp['high_imputation_columns']} columns\n")
            f.write(f"- **Moderate imputation (20-50%):** {imp['moderate_imputation_columns']} columns\n")
            f.write(f"- **Low imputation (<20%):** {imp['low_imputation_columns']} columns\n")
            f.write(f"- **Total imputed columns:** {imp['total_imputed_columns']}\n\n")
            
            if summary['quality_issues']:
                f.write("## Quality Issues\n\n")
                for severity, count in summary['quality_issues'].items():
                    f.write(f"- **{severity.title()}:** {count} issues\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if summary['data_reliability_score'] < 70:
                f.write("⚠️ **Low reliability score.** Consider:\n")
                f.write("- Refreshing cached data\n")
                f.write("- Checking API connectivity\n")
                f.write("- Reviewing high imputation columns\n\n")
            elif summary['data_reliability_score'] < 90:
                f.write("✅ **Good reliability score.** Minor improvements:\n")
                f.write("- Monitor imputation rates\n")
                f.write("- Address quality warnings\n\n")
            else:
                f.write("🎉 **Excellent reliability score!** Data quality is high.\n\n")
        
        logger.info(f"Generated data quality markdown summary: {output_path}")

# Global instance for easy access
quality_tracker = DataQualityTracker() 