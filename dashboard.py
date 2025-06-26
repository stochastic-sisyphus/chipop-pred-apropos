import pandas as pd
from pathlib import Path

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    raise SystemExit("Streamlit is required to run the dashboard. Install with 'pip install streamlit'.")

OUTPUT_DIR = Path("output")

@st.cache_data
def load_data():
    """Load pipeline output data for the dashboard."""
    data_files = {
        "multifamily": OUTPUT_DIR / "data" / "top_multifamily_zips.csv",
        "retail_gap": OUTPUT_DIR / "data" / "retail_lag_zips.csv",
        "forecast": OUTPUT_DIR / "forecasts" / "population_forecast.csv",
    }
    return {
        key: pd.read_csv(path)
        for key, path in data_files.items()
        if path.exists()
    }

def main():
    st.title("Chicago Housing Dashboard")

    data = load_data()
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
