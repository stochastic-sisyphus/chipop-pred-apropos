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
from src.pipeline import Pipeline
from src.models.multifamily_growth_model import MultifamilyGrowthModel
from src.models.retail_gap_model import RetailGapModel
from src.models.retail_void_model import RetailVoidModel
from src.reporting.multifamily_growth_report import MultifamilyGrowthReport
from src.reporting.retail_gap_report import RetailGapReport
from src.reporting.retail_void_report import RetailVoidReport

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
                            'project_status': np.random.choice(['completed', 'in_progress', 'planned'])
                        })
            else:
                for year in permit_years:
                    permit_count = np.random.randint(0, 5)  # 0-5 permits per year
                    
                    for _ in range(permit_count):
                        data.append({
                            'zip_code': zip_code,
                            'permit_year': year,
                            'permit_type': 'multifamily',
                            'unit_count': np.random.randint(5, 50),
                            'project_status': np.random.choice(['completed', 'in_progress', 'planned'])
                        })
            
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
                    'employment_rate': np.random.uniform(0.85, 0.98)
                })
        
        return pd.DataFrame(data)
    
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
            self.__class__.test_results['test_directory_structure'] = "PASSED"
            
        except Exception as e:
            logger.error(f"Directory structure test failed: {str(e)}")
            self.__class__.test_results['test_directory_structure'] = "FAILED"
            raise
    
    def test_multifamily_growth_model(self):
        """Test the multifamily growth model."""
        try:
            # Initialize model
            model = MultifamilyGrowthModel(output_dir=self.test_output_dir)
            
            # Run analysis
            success = model.run_analysis(self.test_data)
            self.assertTrue(success, "Multifamily growth model analysis failed")
            
            # Check results
            self.assertIsNotNone(model.top_emerging_zips, "Top emerging ZIPs not found")
            self.assertGreater(len(model.top_emerging_zips), 0, "No top emerging ZIPs found")
            self.assertLessEqual(len(model.top_emerging_zips), 5, "Too many top emerging ZIPs found")
            
            # Check that growth scores are realistic (not all 1.0)
            growth_scores = model.top_emerging_zips['growth_score'].values
            self.assertFalse(np.all(growth_scores == 1.0), "All growth scores are 1.0, which is unrealistic")
            
            # Check that visualizations were created
            figures_dir = self.test_output_dir / "figures"
            self.assertTrue((figures_dir / "top_emerging_multifamily_zips.png").exists(), "Top emerging ZIPs visualization not found")
            
            logger.info("Multifamily growth model test passed")
            self.__class__.test_results['test_multifamily_growth_model'] = "PASSED"
            
        except Exception as e:
            logger.error(f"Multifamily growth model test failed: {str(e)}")
            self.__class__.test_results['test_multifamily_growth_model'] = "FAILED"
            raise
    
    def test_retail_gap_model(self):
        """Test the retail gap model."""
        try:
            # Initialize model
            model = RetailGapModel(output_dir=self.test_output_dir)
            
            # Run analysis
            success = model.run_analysis(self.test_data)
            self.assertTrue(success, "Retail gap model analysis failed")
            
            # Check results
            self.assertIsNotNone(model.retail_gap_zips, "Retail gap ZIPs not found")
            self.assertGreater(len(model.retail_gap_zips), 0, "No retail gap ZIPs found")
            
            # Check that retail deficits are realistic (not all 1.0)
            retail_deficits = model.retail_gap_zips['retail_deficit'].values
            self.assertFalse(np.all(retail_deficits == 1.0), "All retail deficits are 1.0, which is unrealistic")
            
            # Check that visualizations were created
            figures_dir = self.test_output_dir / "figures"
            self.assertTrue((figures_dir / "retail_gap_map.png").exists(), "Retail gap map visualization not found")
            
            logger.info("Retail gap model test passed")
            self.__class__.test_results['test_retail_gap_model'] = "PASSED"
            
        except Exception as e:
            logger.error(f"Retail gap model test failed: {str(e)}")
            self.__class__.test_results['test_retail_gap_model'] = "FAILED"
            raise
    
    def test_retail_void_model(self):
        """Test the retail void model."""
        try:
            # Initialize model
            model = RetailVoidModel(output_dir=self.test_output_dir)
            
            # Run analysis
            success = model.run_analysis(self.test_data)
            self.assertTrue(success, "Retail void model analysis failed")
            
            # Check results
            self.assertIsNotNone(model.retail_voids, "Retail voids not found")
            self.assertGreater(len(model.retail_voids), 0, "No retail voids found")
            
            # Check that leakage ratios are realistic (not all 1.0)
            leakage_ratios = model.retail_voids['leakage_ratio'].values
            self.assertFalse(np.all(leakage_ratios == 1.0), "All leakage ratios are 1.0, which is unrealistic")
            
            # Check that visualizations were created
            figures_dir = self.test_output_dir / "figures"
            self.assertTrue((figures_dir / "spending_leakage_by_zip.png").exists(), "Spending leakage visualization not found")
            
            logger.info("Retail void model test passed")
            self.__class__.test_results['test_retail_void_model'] = "PASSED"
            
        except Exception as e:
            logger.error(f"Retail void model test failed: {str(e)}")
            self.__class__.test_results['test_retail_void_model'] = "FAILED"
            raise
    
    def test_markdown_reports(self):
        """Test Markdown report generation."""
        try:
            # Create test output directories
            reports_dir = self.test_output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Run multifamily growth model and report
            multifamily_model = MultifamilyGrowthModel(output_dir=self.test_output_dir / "multifamily")
            multifamily_model.run_analysis(self.test_data)
            
            multifamily_report = MultifamilyGrowthReport(output_dir=reports_dir)
            success = multifamily_report.generate(self.test_data, multifamily_model.results)
            self.assertTrue(success, "Multifamily growth report generation failed")
            
            # Check that Markdown report was created
            self.assertTrue((reports_dir / "multifamily_growth_report.md").exists(), "Multifamily growth Markdown report not found")
            
            # Run retail gap model and report
            retail_gap_model = RetailGapModel(output_dir=self.test_output_dir / "retail_gap")
            retail_gap_model.run_analysis(self.test_data)
            
            retail_gap_report = RetailGapReport(output_dir=reports_dir)
            success = retail_gap_report.generate(self.test_data, retail_gap_model.results)
            self.assertTrue(success, "Retail gap report generation failed")
            
            # Check that Markdown report was created
            self.assertTrue((reports_dir / "retail_gap_report.md").exists(), "Retail gap Markdown report not found")
            
            # Run retail void model and report
            retail_void_model = RetailVoidModel(output_dir=self.test_output_dir / "retail_void")
            retail_void_model.run_analysis(self.test_data)
            
            retail_void_report = RetailVoidReport(output_dir=reports_dir)
            success = retail_void_report.generate(self.test_data, retail_void_model.results)
            self.assertTrue(success, "Retail void report generation failed")
            
            # Check that Markdown report was created
            self.assertTrue((reports_dir / "retail_void_report.md").exists(), "Retail void Markdown report not found")
            
            # Verify reports are in Markdown format (not HTML)
            with open(reports_dir / "multifamily_growth_report.md", 'r') as f:
                content = f.read()
                self.assertFalse(content.strip().startswith('<!DOCTYPE html>'), "Report is in HTML format, not Markdown")
                self.assertFalse('<html' in content.lower(), "Report contains HTML tags, not pure Markdown")
            
            logger.info("Markdown reports test passed")
            self.__class__.test_results['test_markdown_reports'] = "PASSED"
            
        except Exception as e:
            logger.error(f"Markdown reports test failed: {str(e)}")
            self.__class__.test_results['test_markdown_reports'] = "FAILED"
            raise
    
    def test_pipeline(self):
        """Test the full pipeline."""
        try:
            # Initialize pipeline with test output directory
            pipeline = Pipeline(output_dir=self.test_output_dir)
            
            # Run pipeline with test data
            success = pipeline.run_with_data(self.test_data)
            self.assertTrue(success, "Pipeline execution failed")
            
            # Check that output files were created
            self.assertTrue((self.test_output_dir / "pipeline_summary.json").exists(), "Pipeline summary not found")
            
            # Check that reports were created in Markdown format
            reports_dir = self.test_output_dir / "reports"
            markdown_reports = list(reports_dir.glob("*.md"))
            self.assertGreater(len(markdown_reports), 0, "No Markdown reports found")
            
            # Check that visualizations were created
            visualizations_dir = self.test_output_dir / "visualizations"
            visualizations = list(visualizations_dir.glob("*.png"))
            self.assertGreater(len(visualizations), 0, "No visualizations found")
            
            logger.info("Pipeline test passed")
            self.__class__.test_results['test_pipeline'] = "PASSED"
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {str(e)}")
            self.__class__.test_results['test_pipeline'] = "FAILED"
            raise

def run_tests():
    """Run all tests and return results."""
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestChicagoPopulationAnalysis('test_directory_structure'))
    suite.addTest(TestChicagoPopulationAnalysis('test_multifamily_growth_model'))
    suite.addTest(TestChicagoPopulationAnalysis('test_retail_gap_model'))
    suite.addTest(TestChicagoPopulationAnalysis('test_retail_void_model'))
    suite.addTest(TestChicagoPopulationAnalysis('test_markdown_reports'))
    suite.addTest(TestChicagoPopulationAnalysis('test_pipeline'))
    
    # Run tests
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    
    # Log results
    logger.info("Test results summary:")
    for test_name, test_result in TestChicagoPopulationAnalysis.test_results.items():
        logger.info(f"{test_name}: {test_result}")
    
    if result.wasSuccessful():
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
