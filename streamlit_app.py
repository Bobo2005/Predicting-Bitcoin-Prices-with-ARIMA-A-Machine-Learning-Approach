import streamlit as st
import pandas as pd
import os
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="ARIMA Forecast Dashboard", layout="wide")

st.title("Bitcoin Forecasting — Quick Dashboard")

output_dir = Path("outputs")

st.sidebar.header("Data sources")
use_default = st.sidebar.checkbox("Use outputs/ files if available", value=True)

features_file = None
models_file = None

if use_default and (output_dir / "features.csv").exists():
    features_file = output_dir / "features.csv"
else:
    uploaded = st.sidebar.file_uploader("Upload features CSV (features.csv)", type=["csv"])
    if uploaded is not None:
        features_file = uploaded

if use_default and (output_dir / "model_comparison_metrics.csv").exists():
    models_file = output_dir / "model_comparison_metrics.csv"
elif use_default and (output_dir / "model_comparison_summary.csv").exists():
    models_file = output_dir / "model_comparison_summary.csv"
else:
    uploaded_models = st.sidebar.file_uploader("Upload model comparison CSV", type=["csv"], key="models")
    if uploaded_models is not None:
        models_file = uploaded_models

st.sidebar.markdown("---")
st.sidebar.markdown("Tips: run `python main.py --preprocess --use-sample --output-dir outputs` to generate example files.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Features")
    if features_file is None:
        st.info("No features file provided. Use the sidebar to upload or generate outputs/features.csv")
    else:
        try:
            feats = pd.read_csv(features_file, index_col=0, parse_dates=True)
            st.subheader("Features preview")
            st.dataframe(feats.head())

            st.subheader("Feature distributions")
            select_col = st.selectbox("Select feature to plot", options=list(feats.columns), index=0)
            fig = px.histogram(feats, x=select_col, nbins=50, title=f"Distribution of {select_col}")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Time series view")
            ts_cols = st.multiselect("Choose columns to plot", options=list(feats.columns), default=[feats.columns[0]])
            if ts_cols:
                fig_ts = px.line(feats[ts_cols])
                st.plotly_chart(fig_ts, use_container_width=True)

            st.subheader("Correlation matrix")
            corr = feats.corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to read features file: {exc}")

with col2:
    st.header("Model comparison & metrics")
    if models_file is None:
        st.info("No model comparison file provided. Run model comparison via `python main.py --compare-models` to generate one.")
    else:
        try:
            mdf = pd.read_csv(models_file)
            st.subheader("Model comparison")
            st.dataframe(mdf)

            st.subheader("Best model by MAE")
            if "mae" in mdf.columns:
                best = mdf.loc[mdf["mae"].idxmin()]
                st.write(best.to_dict())

            if "rmse" in mdf.columns:
                st.subheader("RMSE chart")
                fig_rmse = px.bar(mdf, x="model", y="rmse", title="RMSE by model")
                st.plotly_chart(fig_rmse, use_container_width=True)

            if "mape" in mdf.columns:
                st.subheader("MAPE chart")
                fig_mape = px.bar(mdf, x="model", y="mape", title="MAPE by model")
                st.plotly_chart(fig_mape, use_container_width=True)
        except Exception as exc:
            st.error(f"Unable to read model comparison file: {exc}")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with Copilot CLI runtime in VS Code.")
