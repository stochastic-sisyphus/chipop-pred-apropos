import unittest
import pandas as pd
import os
from pathlib import Path
from src.data.data_collector import DataCollector

class TestDataCollector(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.collector = DataCollector()
        self.test_data_dir = Path('test_data')
        self.test_data_dir.mkdir(exist_ok=True)
        
    def test_initialization(self):
        """Test DataCollector initialization"""
        self.assertIsNotNone(self.collector)
        self.assertIsNotNone(self.collector.data_dir)
        
    def test_census_api_key(self):
        """Test Census API key is available"""
        self.assertIsNotNone(self.collector.census_api_key)
        
    def test_fred_api_key(self):
        """Test FRED API key is available"""
        self.assertIsNotNone(self.collector.fred_api_key)
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

if __name__ == '__main__':
    unittest.main()