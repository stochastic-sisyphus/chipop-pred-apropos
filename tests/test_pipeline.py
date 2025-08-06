"""
Test suite for the Chicago Population Analysis project.
"""

import unittest
import logging
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.pipeline.pipeline import Pipeline
from src.models.multifamily_growth_model import MultifamilyGrowthModel
from src.models.retail_gap_model import RetailGapModel
from src.models.retail_void_model import RetailVoidModel
from src.reports.report_generator import ReportGenerator

class TestChicagoPopulationAnalysis(unittest.TestCase):
    """Test suite for the Chicago Population Analysis project."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test output directory
        cls.test_output_dir = Path("test_output")
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        cls.test_data = cls._create_test_data()

        # Initialize test results
        cls.test_results = {}

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test output directory
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)

    @classmethod
    def _create_test_data(cls):
        """Create test data for analysis."""
        # Create synthetic data for testing
        zip_codes = [f'606{i:02d}' for i in range(1, 50)]
        years = list(range(2010, 2024))

        data = []
        permit_number_counter = 1
        license_number_counter = 1

        for zip_code in zip_codes:
            # Base values
            base_population = np.random.randint(10000, 50000)
            base_housing = np.random.randint(3000, 15000)
            base_retail = np.random.randint(50, 500)

            # Growth rates (some ZIPs have high housing growth but low retail growth)
            if np.random.random() < 0.3:  # 30% of ZIPs have housing growth > retail growth
                housing_growth = np.random.uniform(0.02, 0.05)  # 2-5% annual growth
                retail_growth = np.random.uniform(0.005, 0.015)  # 0.5-1.5% annual growth
            else:
                housing_growth = np.random.uniform(0.005, 0.02)  # 0.5-2% annual growth
                retail_growth = np.random.uniform(0.01, 0.03)  # 1-3% annual growth

            # Population growth typically follows housing
            pop_growth = housing_growth * np.random.uniform(0.8, 1.2)

            # Permit data
            permit_years = list(range(2008, 2024))

            # Some ZIPs have low historical permits but high recent permits
            if np.random.random() < 0.2:  # 20% of ZIPs have this pattern
                for year in permit_years:
                    if year < 2018:  # Historical period
                        permit_count = np.random.randint(0, 3)  # 0-2 permits per year
                    else:  # Recent period
                        permit_count = np.random.randint(5, 15)  # 5-15 permits per year

                    for _ in range(permit_count):
                        data.append({
                            'zip_code': zip_code,
                            'permit_year': year,
                            'permit_type': 'multifamily',
                            'unit_count': np.random.randint(5, 100),
                            'project_status': np.random.choice(['completed', 'in_progress', 'planned']),
                            'permit_number': f'P{permit_number_counter}',
                            'issue_date': f'{year}-01-15T00:00:00.000'
                        })
                        permit_number_counter += 1
            else:
                for year in permit_years:
                    permit_count = np.random.randint(0, 5)  # 0-5 permits per year

                    for _ in range(permit_count):
                        data.append({
                            'zip_code': zip_code,
                            'permit_year': year,
                            'permit_type': 'multifamily',
                            'unit_count': np.random.randint(5, 50),
                            'project_status': np.random.choice(['completed', 'in_progress', 'planned']),
                            'permit_number': f'P{permit_number_counter}',
                            'issue_date': f'{year}-01-15T00:00:00.000'
                        })
                        permit_number_counter += 1

            # Year-by-year data
            for year in years:
                year_idx = year - 2010
                population = int(base_population * (1 + pop_growth) ** year_idx)
                housing = int(base_housing * (1 + housing_growth) ** year_idx)
                retail = int(base_retail * (1 + retail_growth) ** year_idx)

                # Add random noise
                population = int(population * np.random.uniform(0.95, 1.05))
                housing = int(housing * np.random.uniform(0.95, 1.05))
                retail = int(retail * np.random.uniform(0.9, 1.1))

                data.append({
                    'zip_code': zip_code,
                    'year': year,
                    'population': population,
                    'housing_units': housing,
                    'retail_businesses': retail,
                    'median_income': np.random.randint(30000, 120000),
                    'employment_rate': np.random.uniform(0.85, 0.98),
                    'retail_sales': retail * 10000,
                    'consumer_spending': retail * 12000,
                    'license_number': f'L{license_number_counter}',
                    'business_activity': 'RETAIL',
                    'license_start_date': f'{year}-01-15T00:00:00.000',
                    'date': f'{year}-01-15T00:00:00.000',
                    'series_id': 'TEST',
                    'value': 1,
                    'data_source': 'test'
                })
                license_number_counter += 1

        df = pd.DataFrame(data)
        return {'census': df, 'permits': df, 'retail': df, 'economic': df, 'licenses': df}

    def test_directory_structure(self):
        """Test that the directory structure is correct."""
        try:
            # Check that required directories exist
            required_dirs = [
                settings.DATA_DIR,
                settings.OUTPUT_DIR,
                settings.REPORTS_DIR,
                settings.VISUALIZATIONS_DIR,
                settings.MODELS_DIR
            ]

            for dir_path in required_dirs:
                self.assertTrue(dir_path.exists(), f"Directory {dir_path} does not exist")

            logger.info("Directory structure test passed")

        except Exception as e:
            logger.error(f"Directory structure test failed: {str(e)}")
            self.fail(f"Directory structure test failed: {e}")

    def test_multifamily_growth_model(self):
        """Test the multifamily growth model."""
        try:
            # Initialize model
            model = MultifamilyGrowthModel(output_dir=self.test_output_dir)

            # Run analysis
            results = model.run(self.test_data['permits'])

            # Check if successful
            self.assertIsNotNone(results)

            # Check for output files
            self.assertTrue((self.test_output_dir / 'visualizations' / 'multifamily_growth' / 'top_emerging_zips.png').exists())
            self.assertTrue((self.test_output_dir / 'visualizations' / 'multifamily_growth' / 'growth_comparison.png').exists())
            self.assertTrue((self.test_output_dir / 'visualizations' / 'multifamily_growth' / 'units_comparison.png').exists())
            
            # Check for results
            self.assertIsNotNone(model.top_emerging_zips)
            self.assertGreater(len(model.top_emerging_zips), 0)
            
            logger.info("Multifamily growth model test passed")
            
        except Exception as e:
            logger.error(f"Multifamily growth model test failed: {str(e)}")
            self.fail(f"Multifamily growth model test failed: {e}")

    def test_retail_gap_model(self):
        """Test the retail gap model."""
        try:
            # Initialize model
            model = RetailGapModel(output_dir=self.test_output_dir)

            # Run analysis
            results = model.run(self.test_data['census'])

            # Check if successful
            self.assertIsNotNone(results)

            # Check for output files
            self.assertTrue((self.test_output_dir / 'visualizations' / 'retail_gap' / 'retail_gap_score.png').exists())
            self.assertTrue((self.test_output_dir / 'visualizations' / 'retail_gap' / 'retail_housing_comparison.png').exists())
            self.assertTrue((self.test_output_dir / 'visualizations' / 'retail_gap' / 'cluster_analysis.png').exists())
            
            # Check for results
            self.assertIsNotNone(model.opportunity_zones)
            self.assertGreater(len(model.opportunity_zones), 0)
            
            logger.info("Retail gap model test passed")
            
        except Exception as e:
            logger.error(f"Retail gap model test failed: {str(e)}")
            self.fail(f"Retail gap model test failed: {e}")

    def test_retail_void_model(self):
        """Test the retail void model."""
        try:
            # Initialize model
            model = RetailVoidModel(output_dir=self.test_output_dir)

            # Run analysis
            results = model.run(self.test_data['retail'])

            # Check if successful
            self.assertIsNotNone(results)

            # Check for output files
            self.assertTrue((self.test_output_dir / 'visualizations' / 'retail_void' / 'void_count.png').exists())
            
            # Check for results
            self.assertIsNotNone(model.void_zones)
            self.assertGreater(len(model.void_zones), 0)
            
            logger.info("Retail void model test passed")
            
        except Exception as e:
            logger.error(f"Retail void model test failed: {str(e)}")
            self.fail(f"Retail void model test failed: {e}")

    def test_markdown_reports(self):
        """Test Markdown report generation."""
        try:
            # Create test output directories
            reports_dir = self.test_output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Run multifamily growth model and report
            multifamily_model = MultifamilyGrowthModel(output_dir=self.test_output_dir / "multifamily")
            multifamily_model.run(self.test_data['permits'])
            
            report_generator = ReportGenerator(output_dir=reports_dir)
            multifamily_report_paths = report_generator.generate_multifamily_growth_report(
                model_results=multifamily_model.results
            )
            
            # Check if multifamily report exists
            self.assertTrue(Path(multifamily_report_paths['markdown']).exists())
            
            # Run retail gap model and report
            retail_gap_model = RetailGapModel(output_dir=self.test_output_dir / "retail_gap")
            retail_gap_model.run(self.test_data['census'])
            
            retail_gap_report_paths = report_generator.generate_retail_gap_report(
                model_results=retail_gap_model.results
            )
            
            # Check if retail gap report exists
            self.assertTrue(Path(retail_gap_report_paths['markdown']).exists())
            
            # Run retail void model and report
            retail_void_model = RetailVoidModel(output_dir=self.test_output_dir / "retail_void")
            retail_void_model.run(self.test_data['retail'])
            
            retail_void_report_paths = report_generator.generate_retail_void_report(
                model_results=retail_void_model.results
            )
            
            # Check if retail void report exists
            self.assertTrue(Path(retail_void_report_paths['markdown']).exists())
            
            # Check for summary report
            summary_report_paths = report_generator.generate_summary_report(
                multifamily_results=multifamily_model.results,
                retail_gap_results=retail_gap_model.results,
                retail_void_results=retail_void_model.results
            )

            # Check if summary report exists
            self.assertTrue(Path(summary_report_paths['markdown']).exists())
            
            logger.info("Markdown reports test passed")
            
        except Exception as e:
            logger.error(f"Markdown reports test failed: {str(e)}")
            self.fail(f"Markdown reports test failed: {e}")

    def test_pipeline(self):
        """Test the full pipeline."""
        try:
            # Initialize pipeline with test output directory
            pipeline = Pipeline(output_dir=self.test_output_dir, use_sample_data=True)

            # Run pipeline with test data
            results = pipeline.run(use_sample_data=True)

            # Check if successful
            self.assertEqual(results['status'], 'completed')
            
            # Check that output files were created
            self.assertGreater(results['outputs_generated'], 0)
            
            # Check that reports were created
            self.assertGreater(results['reports_generated'], 0)
            
            logger.info("Pipeline test passed")
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {str(e)}")
            self.fail(f"Pipeline test failed: {e}")

if __name__ == '__main__':
    unittest.main()
