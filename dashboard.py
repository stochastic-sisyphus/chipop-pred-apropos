import pandas as pd
from pathlib import Path
from src.config import settings

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    raise SystemExit("Streamlit is required to run the dashboard. Install with 'pip install streamlit'.")

OUTPUT_DIR = settings.OUTPUT_DIR

@st.cache_data
def load_data(mtimes):
    """Load pipeline output data for the dashboard."""
    data_files = {
        "multifamily": settings.MULTIFAMILY_ZIPS_PATH,
        "retail_gap": settings.RETAIL_GAP_ZIPS_PATH,
        "forecast": settings.POPULATION_FORECAST_PATH,
    }
    loaded = {}
    for key, path in data_files.items():
        if path.exists():
            loaded[key] = pd.read_csv(path)
    return loaded

def main():
    st.title("Chicago Housing Dashboard")

    data_files = {
        "multifamily": settings.MULTIFAMILY_ZIPS_PATH,
        "retail_gap": settings.RETAIL_GAP_ZIPS_PATH,
        "forecast": settings.POPULATION_FORECAST_PATH,
    }
    mtimes = tuple(
        path.stat().st_mtime for path in data_files.values() if path.exists()
    )
    if st.button("Refresh Data"):
        load_data.clear()

    data = load_data(mtimes)
    if not data:
        st.error("No output data found. Run the pipeline first.")
        return

    if "multifamily" in data:
        st.header("Multifamily Growth")
        st.dataframe(data["multifamily"])

    if "retail_gap" in data:
        st.header("Retail Gap Analysis")
        st.dataframe(data["retail_gap"])

    if "forecast" in data:
        st.header("Population Forecast")
        forecasts = data["forecast"].set_index("year")
        st.line_chart(forecasts)

if __name__ == "__main__":
    main()
