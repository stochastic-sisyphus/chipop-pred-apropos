"""
Zillow Home Value Index (ZHVI) Data Collector

Collects ZIP-level home value data from Zillow's public research datasets.
Data source: https://www.zillow.com/research/data/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import requests

from src.config import settings

logger = logging.getLogger(__name__)

# Zillow data URLs
ZILLOW_URLS = {
    'zhvi_zip': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'zhvi_zip_sfr': 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv',
    'zori_zip': 'https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_sm_sa_month.csv',
}


class ZillowCollector:
    """
    Collector for Zillow Home Value Index data at ZIP code level.

    Provides real housing market data including:
    - Monthly home values (ZHVI) from 2000-present
    - Rental prices (ZORI) where available
    - Calculated growth rates and trends
    """

    def __init__(self, cache_dir=None):
        """Initialize the Zillow collector."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.DATA_DIR) / "external"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def collect(self, force_refresh=False) -> pd.DataFrame:
        """
        Collect Zillow ZHVI data for Chicago ZIP codes.

        Args:
            force_refresh: If True, download fresh data even if cached

        Returns:
            pd.DataFrame with home value data for Chicago ZIPs
        """
        cache_path = self.cache_dir / "zillow_zhvi_zip.csv"

        # Check cache
        if not force_refresh and cache_path.exists():
            cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if cache_age < 86400 * 7:  # Cache valid for 7 days
                logger.info("Using cached Zillow data")
                return self._load_and_filter(cache_path)

        # Download fresh data
        logger.info("Downloading Zillow ZHVI data...")
        try:
            response = requests.get(ZILLOW_URLS['zhvi_zip'], timeout=60)
            response.raise_for_status()

            with open(cache_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded Zillow data to {cache_path}")
            return self._load_and_filter(cache_path)

        except Exception as e:
            logger.error(f"Failed to download Zillow data: {e}")
            if cache_path.exists():
                logger.info("Using stale cache as fallback")
                return self._load_and_filter(cache_path)
            return None

    def _load_and_filter(self, path: Path) -> pd.DataFrame:
        """Load CSV and filter to Chicago ZIP codes."""
        df = pd.read_csv(path, dtype={'RegionName': str})

        # Filter to Chicago ZIP codes (60601-60661, 60706-60707, 60827)
        chicago_zips = [str(z) for z in settings.CHICAGO_ZIP_CODES]
        df_chicago = df[df['RegionName'].isin(chicago_zips)].copy()

        logger.info(f"Filtered to {len(df_chicago)} Chicago ZIP codes")

        self.data = df_chicago
        return df_chicago

    def get_time_series(self, zip_code: str = None) -> pd.DataFrame:
        """
        Get home value time series data.

        Args:
            zip_code: Specific ZIP code, or None for all Chicago ZIPs

        Returns:
            DataFrame with columns: zip_code, date, home_value
        """
        if self.data is None:
            self.collect()

        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()

        # Identify date columns (YYYY-MM-DD format)
        date_cols = [c for c in self.data.columns if c.startswith('20')]

        # Melt to long format
        id_vars = ['RegionName', 'City', 'Metro', 'CountyName']
        id_vars = [c for c in id_vars if c in self.data.columns]

        df_long = self.data.melt(
            id_vars=id_vars,
            value_vars=date_cols,
            var_name='date',
            value_name='home_value'
        )

        df_long = df_long.rename(columns={'RegionName': 'zip_code'})
        df_long['date'] = pd.to_datetime(df_long['date'])

        # Filter to specific ZIP if requested
        if zip_code:
            df_long = df_long[df_long['zip_code'] == str(zip_code)]

        return df_long.sort_values(['zip_code', 'date'])

    def calculate_growth_metrics(self, years_back: int = 10) -> pd.DataFrame:
        """
        Calculate home value growth metrics for each ZIP code.

        Args:
            years_back: Number of years to calculate growth over

        Returns:
            DataFrame with growth metrics per ZIP code
        """
        if self.data is None:
            self.collect()

        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()

        # Get date columns
        date_cols = sorted([c for c in self.data.columns if c.startswith('20')])

        if len(date_cols) < 12:
            logger.warning("Insufficient date columns for growth calculation")
            return pd.DataFrame()

        # Calculate metrics
        metrics = []

        for _, row in self.data.iterrows():
            zip_code = row['RegionName']

            # Get values at different time points
            values = row[date_cols].values.astype(float)

            # Remove NaN values
            valid_mask = ~np.isnan(values)
            valid_values = values[valid_mask]
            valid_dates = np.array(date_cols)[valid_mask]

            if len(valid_values) < 24:  # Need at least 2 years
                continue

            current_value = valid_values[-1]

            # 1-year growth
            if len(valid_values) >= 12:
                value_1y_ago = valid_values[-12]
                growth_1y = (current_value - value_1y_ago) / value_1y_ago if value_1y_ago > 0 else 0
            else:
                growth_1y = 0

            # 5-year growth
            if len(valid_values) >= 60:
                value_5y_ago = valid_values[-60]
                growth_5y = (current_value - value_5y_ago) / value_5y_ago if value_5y_ago > 0 else 0
                cagr_5y = ((current_value / value_5y_ago) ** (1/5) - 1) if value_5y_ago > 0 else 0
            else:
                growth_5y = 0
                cagr_5y = 0

            # 10-year growth
            if len(valid_values) >= 120:
                value_10y_ago = valid_values[-120]
                growth_10y = (current_value - value_10y_ago) / value_10y_ago if value_10y_ago > 0 else 0
                cagr_10y = ((current_value / value_10y_ago) ** (1/10) - 1) if value_10y_ago > 0 else 0
            else:
                growth_10y = 0
                cagr_10y = 0

            # Volatility (standard deviation of monthly returns)
            if len(valid_values) >= 24:
                monthly_returns = np.diff(valid_values[-24:]) / valid_values[-25:-1]
                volatility = np.std(monthly_returns) * np.sqrt(12)  # Annualized
            else:
                volatility = 0

            # Peak and trough analysis
            peak_value = np.max(valid_values)
            trough_value = np.min(valid_values[-120:]) if len(valid_values) >= 120 else np.min(valid_values)
            pct_from_peak = (current_value - peak_value) / peak_value if peak_value > 0 else 0

            metrics.append({
                'zip_code': zip_code,
                'current_home_value': current_value,
                'home_value_1y_ago': value_1y_ago if len(valid_values) >= 12 else None,
                'growth_1y': growth_1y,
                'growth_5y': growth_5y,
                'growth_10y': growth_10y,
                'cagr_5y': cagr_5y,
                'cagr_10y': cagr_10y,
                'volatility': volatility,
                'peak_value': peak_value,
                'pct_from_peak': pct_from_peak,
                'data_points': len(valid_values),
                'city': row.get('City', 'Chicago'),
                'metro': row.get('Metro', 'Chicago-Naperville-Elgin'),
            })

        df_metrics = pd.DataFrame(metrics)

        # Add affordability tier
        if len(df_metrics) > 0:
            df_metrics['affordability_tier'] = pd.qcut(
                df_metrics['current_home_value'],
                q=5,
                labels=['Very Affordable', 'Affordable', 'Moderate', 'Expensive', 'Very Expensive']
            )

        logger.info(f"Calculated growth metrics for {len(df_metrics)} ZIP codes")
        return df_metrics

    def get_gentrification_indicators(self) -> pd.DataFrame:
        """
        Calculate gentrification indicators based on home value trends.

        High growth in previously affordable areas suggests gentrification.

        Returns:
            DataFrame with gentrification risk indicators
        """
        metrics = self.calculate_growth_metrics()

        if len(metrics) == 0:
            return pd.DataFrame()

        # Calculate gentrification score
        # High score = rapid appreciation in previously affordable area

        # Normalize current value (inverted - lower value = higher score)
        value_min = metrics['current_home_value'].min()
        value_max = metrics['current_home_value'].max()
        metrics['value_score'] = 1 - (metrics['current_home_value'] - value_min) / (value_max - value_min)

        # Normalize growth (higher growth = higher score)
        growth_min = metrics['cagr_5y'].min()
        growth_max = metrics['cagr_5y'].max()
        if growth_max > growth_min:
            metrics['growth_score'] = (metrics['cagr_5y'] - growth_min) / (growth_max - growth_min)
        else:
            metrics['growth_score'] = 0.5

        # Combined gentrification score
        # Weight: 40% affordability, 60% growth
        metrics['gentrification_risk'] = (
            0.4 * metrics['value_score'] +
            0.6 * metrics['growth_score']
        )

        # Categorize
        metrics['gentrification_category'] = pd.cut(
            metrics['gentrification_risk'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Moderate', 'Elevated', 'High']
        )

        return metrics[[
            'zip_code', 'current_home_value', 'cagr_5y', 'growth_5y',
            'gentrification_risk', 'gentrification_category', 'affordability_tier'
        ]]


def integrate_with_pipeline_data(zillow_metrics: pd.DataFrame, pipeline_data: dict) -> dict:
    """
    Integrate Zillow metrics with existing pipeline data.

    Args:
        zillow_metrics: DataFrame from ZillowCollector.calculate_growth_metrics()
        pipeline_data: Existing pipeline data dict

    Returns:
        Enhanced pipeline data dict
    """
    if 'census' in pipeline_data and isinstance(pipeline_data['census'], pd.DataFrame):
        census = pipeline_data['census'].copy()

        # Merge Zillow data
        zillow_subset = zillow_metrics[[
            'zip_code', 'current_home_value', 'growth_1y', 'cagr_5y',
            'volatility', 'gentrification_risk'
        ]].copy()

        census = census.merge(zillow_subset, on='zip_code', how='left')
        pipeline_data['census'] = census

        logger.info("Integrated Zillow home value data with census data")

    # Add as separate data source too
    pipeline_data['zillow'] = zillow_metrics

    return pipeline_data
