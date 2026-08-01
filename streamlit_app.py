import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ARIMA Forecast Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RUN_LOG = OUTPUT_DIR / "dashboard_run.log"
MODEL_FILE = OUTPUT_DIR / "rf_model.joblib"
FEATURES_FILE = OUTPUT_DIR / "features.csv"
COMPARISON_FILE = OUTPUT_DIR / "model_comparison_metrics.csv"
COMPARISON_SUMMARY_FILE = OUTPUT_DIR / "model_comparison_summary.csv"

st.title("Bitcoin Forecasting Dashboard")

if "initialized" not in st.session_state:
    st.session_state.update(
        {
            "use_default": True,
            "ticker": "BTC-USD",
            "start_date": "2017-04-01",
            "end_date": "2025-04-05",
            "forecast_steps": 30,
            "output_dir": "outputs",
            "use_sample": True,
            "candidate_orders": "1,1,1;2,1,2;1,1,2;2,1,1;0,1,1",
            "compare_models": True,
            "seasonal_period": 7,
            "rf_lags": 7,
            "ma_window": 7,
            "run_process": None,
            "run_status": "idle",
            # UI preferences persisted in session state
            "show_ci": True,
            "visible_traces": ["Historical Price", "ARIMA (test)", "Future Forecast"],
            "initialized": True,
        }
    )


@st.cache_data(show_spinner=False)
def load_csv(path: Path, index_col: int = 0, parse_dates: bool = True) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)


@st.cache_resource(show_spinner=False)
def load_rf_model(path: Path) -> Any:
    return joblib.load(path)


def get_feature_dataset(source: Any) -> pd.DataFrame:
    df = pd.read_csv(source, index_col=0, parse_dates=True)
    return df


def find_feature_file() -> Path | None:
    if st.session_state.use_default and FEATURES_FILE.exists():
        return FEATURES_FILE
    return None


def find_model_file() -> Path | None:
    if MODEL_FILE.exists():
        return MODEL_FILE
    return None


def get_lag_feature_matrix(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    expected = [f"lag_{i}" for i in range(1, lags + 1)]
    if all(col in df.columns for col in expected):
        return df[expected].dropna()
    if "close" in df.columns:
        lagged = pd.DataFrame(
            {f"lag_{i}": df["close"].shift(i) for i in range(1, lags + 1)}
        )
        return lagged.dropna()
    raise ValueError(
        "Unable to derive lag features for SHAP explainability. Generate outputs/features.csv or include close/lags in your uploaded file."
    )


def tail_text(path: Path, num_lines: int = 20) -> str:
    if not path.exists():
        return "No logs available yet."
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    return "".join(lines[-num_lines:])


def run_main_process(args: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(BASE_DIR / "main.py"),
        "--ticker",
        args["ticker"],
        "--start-date",
        args["start_date"],
        "--end-date",
        args["end_date"],
        "--forecast-steps",
        str(args["forecast_steps"]),
        "--output-dir",
        args["output_dir"],
    ]
    if args["use_sample"]:
        command.append("--use-sample")
    if args["compare_models"]:
        command.append("--compare-models")
    if args["candidate_orders"]:
        command.extend(["--candidate-orders", args["candidate_orders"]])
    if args["seasonal_period"] is not None:
        command.extend(["--seasonal-period", str(args["seasonal_period"])])
    if args["rf_lags"] is not None:
        command.extend(["--rf-lags", str(args["rf_lags"])])
    if args["ma_window"] is not None:
        command.extend(["--ma-window", str(args["ma_window"])])

    with open(RUN_LOG, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    st.session_state.run_process = process
    st.session_state.run_status = "running"
    st.session_state.run_command = " ".join(command)


def show_download_buttons() -> None:
    if OUTPUT_DIR.exists():
        col_a, col_b = st.columns(2)
        with col_a:
            if FEATURES_FILE.exists():
                st.download_button(
                    "Download features.csv",
                    FEATURES_FILE.read_bytes(),
                    file_name="features.csv",
                    mime="text/csv",
                )
            if COMPARISON_FILE.exists():
                st.download_button(
                    "Download model_comparison_metrics.csv",
                    COMPARISON_FILE.read_bytes(),
                    file_name="model_comparison_metrics.csv",
                    mime="text/csv",
                )
            # additional forecast-related downloads
            arima_csv = OUTPUT_DIR / "arima_forecast.csv"
            future_csv = OUTPUT_DIR / "future_forecast.csv"
            combined_csv = OUTPUT_DIR / "combined_forecast.csv"
            prices_csv = OUTPUT_DIR / "prices.csv"
            if arima_csv.exists():
                st.download_button(
                    "Download arima_forecast.csv",
                    arima_csv.read_bytes(),
                    file_name="arima_forecast.csv",
                    mime="text/csv",
                )
            if future_csv.exists():
                st.download_button(
                    "Download future_forecast.csv",
                    future_csv.read_bytes(),
                    file_name="future_forecast.csv",
                    mime="text/csv",
                )
            if combined_csv.exists():
                st.download_button(
                    "Download combined_forecast.csv",
                    combined_csv.read_bytes(),
                    file_name="combined_forecast.csv",
                    mime="text/csv",
                )
        with col_b:
            if (OUTPUT_DIR / "metrics.csv").exists():
                st.download_button(
                    "Download metrics.csv",
                    (OUTPUT_DIR / "metrics.csv").read_bytes(),
                    file_name="metrics.csv",
                    mime="text/csv",
                )
            model_file = find_model_file()
            if model_file is not None:
                st.download_button(
                    "Download rf_model.joblib",
                    model_file.read_bytes(),
                    file_name="rf_model.joblib",
                    mime="application/octet-stream",
                )
            if prices_csv.exists():
                st.download_button(
                    "Download prices.csv",
                    prices_csv.read_bytes(),
                    file_name="prices.csv",
                    mime="text/csv",
                )


def explain_rf_model() -> None:
    shap_available = True
    try:
        shap = importlib.import_module("shap")
    except ImportError:
        shap_available = False
        shap = None

    model_path = find_model_file()
    if model_path is None:
        st.warning(
            "No saved RandomForest model found in outputs/rf_model.joblib. Run `python main.py --compare-models` first."
        )
        return

    feature_source = find_feature_file()
    if feature_source is None:
        st.warning(
            "No features file found in outputs/features.csv. Run `python main.py --preprocess` before explainability."
        )
        return

    try:
        df = get_feature_dataset(feature_source)
        X = get_lag_feature_matrix(df, st.session_state.rf_lags)
        model = load_rf_model(model_path)
    except Exception as exc:
        st.error(f"Unable to prepare explainability data: {exc}")
        return

    st.subheader("RandomForest baseline explainability")
    st.markdown(
        "Use SHAP to inspect feature importance for the trained RandomForest baseline. "
        "This uses the saved `outputs/rf_model.joblib` model and lag features from `outputs/features.csv`."
    )

    if not shap_available or shap is None:
        st.warning(
            "SHAP is not installed. Install it with `pip install shap` and restart the dashboard."
        )
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_abs = pd.DataFrame(abs(shap_values.values), columns=X.columns, index=X.index)
    feature_importance = shap_abs.mean(axis=0).sort_values(ascending=False)
    st.markdown("### Global SHAP feature importance")
    fig = px.bar(
        feature_importance.reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        ),
        x="importance",
        y="feature",
        orientation="h",
        title="Average absolute SHAP value by feature",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Local explanation for a selected sample")
    sample_index = st.selectbox(
        "Choose a row to explain",
        options=list(X.index[-20:]),
        format_func=lambda idx: str(idx),
    )
    sample_X = X.loc[[sample_index]]
    sample_shap = explainer(sample_X)
    local_df = pd.DataFrame(
        {
            "feature": X.columns,
            "shap_value": sample_shap.values[0],
            "abs_value": abs(sample_shap.values[0]),
            "feature_value": sample_X.iloc[0].values,
        }
    ).sort_values("abs_value", ascending=False)
    st.dataframe(local_df.reset_index(drop=True).head(20))
    fig_local = px.bar(
        local_df.head(20),
        x="shap_value",
        y="feature",
        orientation="h",
        title=f"SHAP contribution for row {sample_index}",
    )
    st.plotly_chart(fig_local, use_container_width=True)


def run_page() -> None:
    st.header("Run forecasts from the dashboard")
    with st.form("forecast_form"):
        ticker = st.text_input("Ticker", value=st.session_state.ticker)
        start_date = st.text_input("Start date", value=st.session_state.start_date)
        end_date = st.text_input("End date", value=st.session_state.end_date)
        forecast_steps = st.number_input(
            "Forecast steps",
            min_value=1,
            max_value=365,
            value=st.session_state.forecast_steps,
        )
        output_dir = st.text_input(
            "Output directory", value=st.session_state.output_dir
        )
        use_sample = st.checkbox(
            "Use sample data (no network)", value=st.session_state.use_sample
        )
        compare_models = st.checkbox(
            "Compare models after forecasting", value=st.session_state.compare_models
        )
        candidate_orders = st.text_input(
            "ARIMA candidate orders",
            value=st.session_state.candidate_orders,
        )
        seasonal_period = st.number_input(
            "Seasonal period for SARIMAX",
            min_value=1,
            max_value=30,
            value=st.session_state.seasonal_period,
        )
        rf_lags = st.number_input(
            "RandomForest lag features",
            min_value=1,
            max_value=30,
            value=st.session_state.rf_lags,
        )
        ma_window = st.number_input(
            "Moving average window",
            min_value=1,
            max_value=60,
            value=st.session_state.ma_window,
        )
        submit_button = st.form_submit_button("Start forecast run")

    if submit_button:
        st.session_state.update(
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "forecast_steps": forecast_steps,
                "output_dir": output_dir,
                "use_sample": use_sample,
                "candidate_orders": candidate_orders,
                "compare_models": compare_models,
                "seasonal_period": seasonal_period,
                "rf_lags": rf_lags,
                "ma_window": ma_window,
            }
        )
        run_args = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "forecast_steps": forecast_steps,
            "output_dir": output_dir,
            "use_sample": use_sample,
            "compare_models": compare_models,
            "candidate_orders": candidate_orders,
            "seasonal_period": seasonal_period,
            "rf_lags": rf_lags,
            "ma_window": ma_window,
        }
        run_main_process(run_args)

    if st.session_state.run_process is not None:
        process = st.session_state.run_process
        return_code = process.poll()
        if return_code is None:
            st.info("Forecast run is currently executing in the background.")
        else:
            st.success(f"Forecast run finished with exit code {return_code}.")
            st.session_state.run_status = "done"
            st.session_state.run_process = None
    st.markdown("#### Output logs")
    st.text_area("Forecast process log", value=tail_text(RUN_LOG), height=240)


tab_overview, tab_explain, tab_run = st.tabs(
    [
        "Overview",
        "Explainability",
        "Run Forecast",
    ]
)

with tab_overview:
    st.header("Overview")
    st.sidebar.header("Data sources")
    st.session_state.use_default = st.sidebar.checkbox(
        "Use outputs/ files if available", value=st.session_state.use_default
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Tips: run `python main.py --preprocess --use-sample --output-dir outputs` to generate example files."
    )
    if st.session_state.use_default:
        feat_source = FEATURES_FILE if FEATURES_FILE.exists() else None
        model_source = (
            COMPARISON_FILE
            if COMPARISON_FILE.exists()
            else (COMPARISON_SUMMARY_FILE if COMPARISON_SUMMARY_FILE.exists() else None)
        )
    else:
        feat_source = None
        model_source = None

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Features")
        if feat_source is None:
            st.info(
                "No features file found in outputs. Upload a features CSV or generate outputs/features.csv."
            )
        else:
            try:
                feats = load_csv(feat_source)
                st.dataframe(feats.head())
                st.markdown("### Feature distributions")
                select_col = st.selectbox(
                    "Select feature to plot", options=list(feats.columns), index=0
                )
                fig = px.histogram(
                    feats, x=select_col, nbins=50, title=f"Distribution of {select_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Time series view")
                ts_cols = st.multiselect(
                    "Choose columns to plot",
                    options=list(feats.columns),
                    default=[feats.columns[0]],
                )
                if ts_cols:
                    fig_ts = px.line(feats[ts_cols])
                    st.plotly_chart(fig_ts, use_container_width=True)
                st.markdown("### Correlation matrix")
                corr = feats.corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as exc:
                st.error(f"Unable to read features file: {exc}")
    with col2:
        st.subheader("Model comparison & metrics")
        if model_source is None:
            st.info(
                "No model comparison file found in outputs. Generate model_comparison_metrics.csv by running `python main.py --compare-models`."
            )
        else:
            try:
                mdf = load_csv(model_source, index_col=None)
                st.dataframe(mdf)
                if "mae" in mdf.columns:
                    best = mdf.loc[mdf["mae"].idxmin()]
                    st.markdown("### Best model by MAE")
                    st.write(best.to_dict())
                if "rmse" in mdf.columns:
                    fig_rmse = px.bar(mdf, x="model", y="rmse", title="RMSE by model")
                    st.plotly_chart(fig_rmse, use_container_width=True)
                if "mape" in mdf.columns:
                    fig_mape = px.bar(mdf, x="model", y="mape", title="MAPE by model")
                    st.plotly_chart(fig_mape, use_container_width=True)
            except Exception as exc:
                st.error(f"Unable to read model comparison file: {exc}")

        st.markdown("### Forecast visualizations")
        arima_img = OUTPUT_DIR / "arima_forecast.png"
        future_img = OUTPUT_DIR / "future_forecast.png"
        arima_csv = OUTPUT_DIR / "arima_forecast.csv"
        future_csv = OUTPUT_DIR / "future_forecast.csv"

        # Prefer combined interactive chart when available
        combined_csv = OUTPUT_DIR / "combined_forecast.csv"
        if combined_csv.exists():
            try:
                cdf = pd.read_csv(combined_csv, index_col=0, parse_dates=True)

                # UI control: show/hide confidence intervals (persisted in session_state)
                show_ci = st.checkbox(
                    "Show confidence intervals",
                    value=st.session_state.get("show_ci", True),
                    key="show_ci",
                )

                # Trace visibility controls (persisted)
                trace_options = ["Historical Price", "ARIMA (test)", "Future Forecast"]
                visible = st.multiselect(
                    "Show traces",
                    options=trace_options,
                    default=st.session_state.get("visible_traces", trace_options),
                    key="visible_traces",
                )

                # build interactive Plotly figure with improved hover and date formatting
                fig = go.Figure()
                hover_template = "%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>"

                if "price" in cdf.columns and "Historical Price" in visible:
                    fig.add_trace(
                        go.Scatter(
                            x=cdf.index,
                            y=cdf["price"],
                            mode="lines",
                            name="Historical Price",
                            line=dict(color="black"),
                            hovertemplate=hover_template,
                        )
                    )
                if "arima_forecast" in cdf.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=cdf.index,
                            y=cdf["arima_forecast"],
                            mode="lines",
                            name="ARIMA (test)",
                            line=dict(color="red", dash="dash"),
                            hovertemplate=hover_template,
                        )
                    )
                    # CI for ARIMA (test)
                    if (
                        show_ci
                        and "arima_upper" in cdf.columns
                        and "arima_lower" in cdf.columns
                    ):
                        # add upper then lower with fill to create shaded band
                        fig.add_trace(
                            go.Scatter(
                                x=cdf.index,
                                y=cdf.get("arima_upper"),
                                line=dict(width=0),
                                hoverinfo="skip",
                                showlegend=False,
                                name="ARIMA CI Upper",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=cdf.index,
                                y=cdf.get("arima_lower"),
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(255,0,0,0.12)",
                                hoverinfo="skip",
                                showlegend=False,
                                name="ARIMA CI Lower",
                            )
                        )
                if "future_forecast" in cdf.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=cdf.index,
                            y=cdf["future_forecast"],
                            mode="lines",
                            name="Future Forecast",
                            line=dict(color="green"),
                            hovertemplate=hover_template,
                        )
                    )
                    # CI for future
                    if (
                        show_ci
                        and "future_upper" in cdf.columns
                        and "future_lower" in cdf.columns
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=cdf.index,
                                y=cdf.get("future_upper"),
                                line=dict(width=0),
                                hoverinfo="skip",
                                showlegend=False,
                                name="Future CI Upper",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=cdf.index,
                                y=cdf.get("future_lower"),
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(0,128,0,0.12)",
                                hoverinfo="skip",
                                showlegend=False,
                                name="Future CI Lower",
                            )
                        )

                fig.update_layout(
                    title="Combined Forecast: Historical + ARIMA + Future",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    yaxis=dict(tickprefix="$"),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Percent-change summary between last historical and final forecast
                try:
                    last_hist = cdf["price"].dropna().iloc[-1]
                    final_pred = cdf["future_forecast"].dropna().iloc[-1]
                    pct_change = (final_pred - last_hist) / last_hist * 100
                    st.markdown("### Forecast summary")
                    final_date = cdf["future_forecast"].dropna().index[-1].date()
                    st.metric(
                        label=f"Predicted price on {final_date}",
                        value=f"{final_pred:.2f}",
                        delta=f"{pct_change:.2f}%",
                    )
                except Exception:
                    # fall back to earlier simple metric if combined fails
                    pass

            except Exception as exc:
                st.error(f"Unable to load combined forecast CSV: {exc}")

        else:
            if arima_img.exists():
                st.markdown("#### ARIMA forecast (plot)")
                st.image(str(arima_img), use_column_width=True)
            elif arima_csv.exists():
                try:
                    af = pd.read_csv(
                        arima_csv, index_col=0, parse_dates=True, squeeze=True
                    )
                    st.line_chart(af)
                except Exception as exc:
                    st.error(f"Unable to load ARIMA forecast CSV: {exc}")
            else:
                st.info(
                    "No ARIMA forecast image or CSV found. Run a forecast to generate outputs/arima_forecast.png or arima_forecast.csv"
                )

            if future_img.exists():
                st.markdown("#### Future forecast (plot)")
                st.image(str(future_img), use_column_width=True)
            elif future_csv.exists():
                try:
                    ff = pd.read_csv(
                        future_csv, index_col=0, parse_dates=True, squeeze=True
                    )
                    st.line_chart(ff)
                except Exception as exc:
                    st.error(f"Unable to load future forecast CSV: {exc}")
            else:
                st.info(
                    "No future forecast image or CSV found. Run a forecast to generate outputs/future_forecast.png or future_forecast.csv"
                )

            # Show numeric future predictions (most recent forecasted price(s))
            if future_csv.exists():
                try:
                    ff = pd.read_csv(future_csv, index_col=0, parse_dates=True)
                    # support single-column CSVs where the forecast values are in column 0
                    val_col = ff.columns[0] if len(ff.columns) >= 1 else None
                    if val_col is not None:
                        latest = ff[val_col].iloc[-1]
                        st.markdown("#### Latest future-predicted price")
                        st.metric(
                            label="Predicted price (last forecasted date)",
                            value=f"{latest:.2f}",
                        )
                        # show table of all future predictions
                        st.markdown("#### Future forecast table")
                        st.dataframe(ff)
                except Exception as exc:
                    st.error(f"Unable to display numeric future forecast: {exc}")

        st.markdown("### Downloads")
        show_download_buttons()

with tab_explain:
    explain_rf_model()

with tab_run:
    run_page()
