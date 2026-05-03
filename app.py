import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
from xgboost import XGBRegressor

from modules import (
    load_and_clean, prepare_xy, temporal_split,
    log_transform, build_preprocessor, fit_transform_preprocessor,
    predict, evaluate, merge_train_val,
    validate_train_columns, validate_predict_columns, validate_batch_columns,
    REQUIRED_TRAIN_COLS, REQUIRED_PREDICT_COLS,
    STOREY_ORDER, FLAT_TYPES, REGION_URA, TRANSPORT_TYPES, LINE_COLORS,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HDB Resale Price Analyser",
    page_icon="🏠",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

XGB_BEST_PARAMS = {
    'colsample_bytree': 0.6905983100791752,
    'learning_rate':    0.19710010921874047,
    'max_depth':        6,
    'min_child_weight': 9,
    'n_estimators':     227,
    'reg_alpha':        0.5177513505274801,
    'reg_lambda':       2.529359307131251,
    'subsample':        0.8702760468157122,
    'random_state':     42,
    'n_jobs':           -1,
}

# No hardcoded benchmark — performance page shows live results only

# Column descriptions from metadata (used as tooltips and in overview)
COL_DESCRIPTIONS = {
    'month':                        'Month of flat transaction (YYYY-MM).',
    'town':                         'HDB-defined town where the flat is located.',
    'flat_type':                    'Category of the flat (e.g. 3 ROOM, 4 ROOM, EXECUTIVE).',
    'flat_model':                   'Architectural model of the flat.',
    'storey_range':                 'Storey band the flat is located on (e.g. 07 TO 09).',
    'lease_commence_date':          'Year the flat\'s 99-year lease commenced.',
    'remaining_lease_years':        'Full years of lease remaining at time of purchase.',
    'remaining_lease_months':       'Additional months of lease remaining (added to remaining_lease_years).',
    'floor_area_sqft':              'Size of the flat in square feet.',
    'planning_area_ura':            'URA planning area where the flat is sited.',
    'region_ura':                   'Cardinal region of Singapore (Central, East, West, North, North-East).',
    'latitude':                     'Latitude coordinate of the flat.',
    'longitude':                    'Longitude coordinate of the flat.',
    'closest_mrt_station':          'Name of the closest MRT/LRT station to the flat.',
    'distance_to_mrt_meters':       'Straight-line displacement (not road distance) to closest MRT/LRT station, in metres.',
    'transport_type':               'Type of rail service at the closest station: MRT or LRT.',
    'line_color':                   'Train line of the closest station (e.g. Red = North-South Line, Green = East-West Line).',
    'distance_to_cbd':              'Straight-line displacement to Raffles Place MRT, used as proxy for the CBD centre, in metres.',
    'closest_pri_school':           'Name of the closest primary school to the flat.',
    'distance_to_pri_school_meters':'Straight-line displacement to the closest primary school, in metres.',
    'resale_price':                 'Transaction price of the resale flat in SGD.',
    'price_per_sqft':               'Derived column: resale_price / floor_area_sqft. Dropped before modelling (target leakage).',
    'floor_area_sqm':               'Size of the flat in square metres. Dropped (redundant with floor_area_sqft).',
    'blk_no':                       'Block number — address subcomponent. Dropped (not predictive at scale).',
    'road_name':                    'Road name — address subcomponent. Dropped.',
    'building':                     'Building name — address subcomponent. Dropped.',
    'postal':                       'Postal code — address subcomponent. Dropped.',
    'x':                            'SVY21 X coordinate. Dropped (redundant with lat/lng).',
    'y':                            'SVY21 Y coordinate. Dropped (redundant with lat/lng).',
}

# ── Session State Init ────────────────────────────────────────────────────────

for key in ['df', 'preprocessor', 'model', 'metrics', 'y_test', 'y_pred', 'feature_names', 'X_test_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏠 HDB Resale\nPrice Analyser")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📂 Upload & Overview",
        "📊 Price Distribution",
        "🗓️ Temporal & Geospatial",
        "🤖 Model Training",
        "📈 Model Performance",
        "🔮 Prediction",
    ])

    st.markdown("---")

    if st.session_state["df"] is not None:
        st.success(f"✅ Dataset: {len(st.session_state['df']):,} rows")
    else:
        st.warning("⬜ No dataset loaded")

    if st.session_state["model"] is not None:
        st.success("✅ Model trained")
    else:
        st.info("⬜ Model not trained")

    st.markdown("---")
    if st.button("🔄 Reset All", use_container_width=True):
        for key in ['df', 'preprocessor', 'model', 'metrics', 'y_test', 'y_pred', 'feature_names', 'X_test_raw']:
            st.session_state[key] = None
        st.rerun()

# ── Helpers ───────────────────────────────────────────────────────────────────

def require_data():
    if st.session_state["df"] is None:
        st.warning("Please upload a CSV dataset on the **Upload & Overview** page first.")
        st.stop()
    return st.session_state["df"]

def require_model():
    if st.session_state["model"] is None:
        st.warning("Please train the model on the **Model Training** page first.")
        st.stop()
    return st.session_state["model"], st.session_state["preprocessor"]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload & Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "📂 Upload & Overview":
    st.title("Upload & Dataset Overview")

    uploaded = st.file_uploader("Upload HDB resale CSV", type=["csv"])
    if uploaded:
        with st.spinner("Loading and cleaning..."):
            df_raw = load_and_clean(uploaded)

        errors = validate_train_columns(df_raw)
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        df_raw['month'] = pd.to_datetime(df_raw['month'])
        st.session_state["df"] = df_raw
        st.success(f"Loaded **{len(df_raw):,}** rows × **{df_raw.shape[1]}** columns")

    if st.session_state["df"] is None:
        st.info("Upload a CSV file to begin.")
        st.stop()

    df = st.session_state["df"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",      f"{len(df):,}")
    c2.metric("Features",           f"{df.shape[1]}")
    c3.metric("Missing Values",     f"{df.isnull().sum().sum()}")
    c4.metric("Duplicates Removed", f"{df.duplicated().sum()}")

    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    num_cols = ['resale_price', 'floor_area_sqft', 'remaining_lease_years',
                'distance_to_mrt_meters', 'distance_to_cbd', 'distance_to_pri_school_meters']
    existing_num = [c for c in num_cols if c in df.columns]
    st.dataframe(df[existing_num].describe().round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Categorical Cardinality")
    cat_cols = ['town', 'flat_type', 'flat_model', 'storey_range',
                'planning_area_ura', 'region_ura', 'closest_mrt_station']
    existing_cat = [c for c in cat_cols if c in df.columns]
    card_df = pd.DataFrame({
        'Column':        existing_cat,
        'Unique Values': [df[c].nunique() for c in existing_cat],
        'Sample Values': [', '.join(df[c].dropna().unique()[:3].astype(str)) for c in existing_cat],
    })
    st.dataframe(card_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Dataset Description")
    st.markdown("""
**HDB Resale Flat Price (Jan 2017 – Jul 2025)**

This dataset contains records of Singapore public housing (HDB) resale flat transactions.
Each row represents one transaction. The dataset was sourced from Singapore's
[National Data Repository](https://data.gov.sg) and enriched with five categories of
additional features: geospatial coordinates, closest MRT station and distance,
closest primary school and distance, distance to the CBD, and URA planning area.

**Source:** Housing and Development Board (HDB), Singapore ·
**License:** [Singapore Open Data Licence v1.0](https://data.gov.sg/open-data-licence)
""")

    col_ref = pd.DataFrame([
        {'Column': col, 'Description': desc}
        for col, desc in COL_DESCRIPTIONS.items()
        if col in df.columns
    ])
    st.dataframe(col_ref, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Price Distribution
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Price Distribution":
    st.title("Price Distribution")
    df = require_data()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Scale")
        fig = px.histogram(df, x='resale_price', nbins=80,
                           labels={'resale_price': 'Resale Price (SGD)'},
                           color_discrete_sequence=['steelblue'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Log Scale")
        fig = px.histogram(df, x=np.log(df['resale_price']), nbins=80,
                           labels={'x': 'log(Resale Price)'},
                           color_discrete_sequence=['coral'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Price by Flat Type")
    order = df.groupby('flat_type')['resale_price'].median().sort_values().index.tolist()
    fig = px.box(df, x='flat_type', y='resale_price',
                 category_orders={'flat_type': order},
                 labels={'resale_price': 'Resale Price (SGD)', 'flat_type': 'Flat Type'},
                 color='flat_type')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Price by Storey Range")
    existing_storeys = [s for s in STOREY_ORDER if s in df['storey_range'].unique()]
    fig = px.box(df, x='storey_range', y='resale_price',
                 category_orders={'storey_range': existing_storeys},
                 labels={'resale_price': 'Resale Price (SGD)', 'storey_range': 'Storey Range'},
                 color='storey_range')
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Median Price by Town (Top 15)")
    town_price = (df.groupby('town')['resale_price']
                    .median().sort_values(ascending=False)
                    .head(15).reset_index())
    fig = px.bar(town_price, x='town', y='resale_price',
                 labels={'resale_price': 'Median Resale Price (SGD)', 'town': 'Town'},
                 color='resale_price', color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Temporal & Geospatial
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗓️ Temporal & Geospatial":
    st.title("Temporal & Geospatial Analysis")
    df = require_data()

    st.subheader("Monthly Median Resale Price")
    monthly = (df.groupby('month')['resale_price']
                 .agg(['median', 'count']).reset_index()
                 .rename(columns={'median': 'Median Price', 'count': 'Transactions'}))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['Median Price'],
                             name='Median Price', line=dict(color='steelblue', width=2)),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=monthly['month'], y=monthly['Transactions'],
                         name='Transactions', marker_color='lightgrey', opacity=0.5),
                  secondary_y=True)
    fig.update_yaxes(title_text="Median Price (SGD)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Transactions", secondary_y=True)
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Geospatial Price Distribution")
    sample_size = st.slider("Sample size for map", 5000, min(50000, len(df)), 20000, step=5000)
    df_sample = df.sample(n=sample_size, random_state=42)
    fig = px.scatter(df_sample, x='longitude', y='latitude',
                     color='resale_price', color_continuous_scale='RdYlGn', opacity=0.5,
                     labels={'resale_price': 'Resale Price (SGD)',
                             'longitude': 'Longitude', 'latitude': 'Latitude'},
                     hover_data={'town': True, 'flat_type': True,
                                 'resale_price': ':,.0f',
                                 'longitude': False, 'latitude': False})
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(coloraxis_colorbar=dict(title="Price (SGD)"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation with Resale Price")
    num_features = ['floor_area_sqft', 'remaining_lease_years',
                    'distance_to_mrt_meters', 'distance_to_cbd',
                    'distance_to_pri_school_meters']
    existing_feat = [f for f in num_features if f in df.columns]
    corr = df[existing_feat + ['resale_price']].corr()['resale_price'].drop('resale_price')
    corr_df = (corr.reset_index()
                   .rename(columns={'index': 'Feature', 'resale_price': 'Correlation'})
                   .sort_values('Correlation'))
    fig = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                 color='Correlation', color_continuous_scale='RdBu', range_color=[-1, 1])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Training
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Training":
    st.title("Model Training")
    df_raw = require_data()

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown(
            "Runs the full pipeline: temporal split (70/15/15) → log transform → "
            "feature encoding (target encoding with 5-fold CV, ordinal, one-hot, standard scaling) "
            "→ train+val merge → XGBoost training with tuned hyperparameters."
        )
        with st.expander("XGBoost hyperparameters (tuned)"):
            st.json(XGB_BEST_PARAMS)
    with col_r:
        st.metric("Train", "70%")
        st.metric("Validation", "15%")
        st.metric("Test", "15%")

    st.markdown("---")

    if st.button("▶ Run Pipeline & Train", type="primary", use_container_width=True):
        for key in ['preprocessor', 'model', 'metrics', 'y_test', 'y_pred', 'feature_names']:
            st.session_state[key] = None

        with st.status("Running pipeline...", expanded=True) as status:
            st.write("Preparing features and target...")
            X, y = prepare_xy(df_raw)

            st.write("Temporal split (70 / 15 / 15)...")
            X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X, y)
            st.write(f"→ Train: **{len(X_train):,}** | Val: **{len(X_val):,}** | Test: **{len(X_test):,}**")

            st.write("Log-transforming target...")
            y_train_log, y_val_log, y_test_log = log_transform(y_train, y_val, y_test)

            st.write("Fitting preprocessor on training data (target encoding with CV)...")
            preprocessor = build_preprocessor()
            preprocessor, X_tr, X_val_p, X_te = fit_transform_preprocessor(
                preprocessor, X_train, y_train_log, X_val, X_test
            )
            st.write(f"→ Feature matrix: **{X_tr.shape[1]}** features")

            st.write("Merging train + val for final model fit...")
            X_final, y_final = merge_train_val(X_tr, X_val_p, y_train_log, y_val_log)

            st.write("Training XGBoost (tuned)...")
            model = XGBRegressor(**XGB_BEST_PARAMS)
            model.fit(X_final, y_final)

            y_pred = predict(model, X_te)
            metrics = evaluate(y_test.values, y_pred, X_te.shape[1])

            try:
                feature_names = preprocessor.get_feature_names_out().tolist()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_te.shape[1])]

            st.session_state.update({
                "preprocessor":  preprocessor,
                "model":         model,
                "X_test_raw":    X_test.copy(),   # raw (unprocessed) for residual diagnostics
                "y_test":        y_test.values,
                "y_pred":        y_pred,
                "metrics":       metrics,
                "feature_names": feature_names,
            })

            status.update(label="Training complete! ✅", state="complete")

        m = st.session_state["metrics"]
        st.success(
            f"Test R²: **{m['R2']:.4f}** | "
            f"RMSE: **${m['RMSE']:,.0f}** | "
            f"MAE: **${m['MAE']:,.0f}**"
        )

    st.markdown("---")
    st.subheader("Save Trained Model")

    if st.session_state["model"] is None:
        st.info("Train the model first to enable saving.")
    else:
        st.caption("Your browser's save dialog lets you choose the download location.")
        col_a, col_b = st.columns(2)
        with col_a:
            buf = io.BytesIO()
            joblib.dump(st.session_state["model"], buf)
            buf.seek(0)
            st.download_button("💾 Download Model (xgb_model.pkl)", data=buf,
                               file_name="xgb_model.pkl", mime="application/octet-stream",
                               use_container_width=True)
        with col_b:
            buf = io.BytesIO()
            joblib.dump(st.session_state["preprocessor"], buf)
            buf.seek(0)
            st.download_button("💾 Download Preprocessor (preprocessor.pkl)", data=buf,
                               file_name="preprocessor.pkl", mime="application/octet-stream",
                               use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.title("Model Performance — XGBoost (Tuned)")

    if st.session_state["metrics"] is None:
        st.info("Train the model on the **Model Training** page first.")
        st.stop()

    m          = st.session_state["metrics"]
    y_test     = st.session_state["y_test"]
    y_pred     = st.session_state["y_pred"]
    feat_names = st.session_state["feature_names"]
    model      = st.session_state["model"]
    X_test_raw = st.session_state["X_test_raw"]

    # ── Metrics ───────────────────────────────────────────────────────────────

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",      f"{m['R2']:.4f}")
    c2.metric("Adj. R²", f"{m['Adj_R2']:.4f}")
    c3.metric("RMSE",    f"${m['RMSE']:,.0f}")
    c4.metric("MAE",     f"${m['MAE']:,.0f}")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4 = st.tabs([
        "Actual vs Predicted",
        "Residuals",
        "Residual Diagnostics",
        "Feature Importance",
    ])

    with tab1:
        fig = px.scatter(x=y_test, y=y_pred,
                         labels={'x': 'Actual Price (SGD)', 'y': 'Predicted Price (SGD)'},
                         opacity=0.3, color_discrete_sequence=['steelblue'])
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        fig.add_shape(type='line', x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
                      line=dict(color='red', dash='dash'))
        fig.update_layout(title="Actual vs Predicted Resale Price")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        residuals = y_test - y_pred
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(x=y_pred, y=residuals,
                             labels={'x': 'Predicted Price (SGD)', 'y': 'Residual (SGD)'},
                             opacity=0.3, color_discrete_sequence=['tomato'])
            fig.add_hline(y=0, line_dash='dash', line_color='black')
            fig.update_layout(title="Residuals vs Predicted")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(x=residuals, nbins=60,
                               labels={'x': 'Residual (SGD)', 'y': 'Frequency'},
                               color_discrete_sequence=['mediumpurple'])
            fig.add_vline(x=0, line_dash='dash', line_color='black')
            fig.update_layout(title="Residual Distribution", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown(
            "These plots break down mean residuals by category, revealing where the model "
            "is systematically over- or under-predicting. Positive residual = model underestimates; "
            "negative = overestimates."
        )

        residuals = y_test - y_pred
        diag_df   = X_test_raw.copy()
        diag_df['residual'] = residuals

        # Plot 1: by flat type
        st.subheader("Mean Residual by Flat Type")
        ft_resid = (diag_df.groupby('flat_type')['residual']
                           .mean().sort_values().reset_index()
                           .rename(columns={'residual': 'Mean Residual (SGD)'}))
        fig = px.bar(ft_resid, x='Mean Residual (SGD)', y='flat_type', orientation='h',
                     color='Mean Residual (SGD)', color_continuous_scale='RdBu',
                     labels={'flat_type': 'Flat Type'})
        fig.add_vline(x=0, line_dash='dash', line_color='black')
        fig.update_layout(coloraxis_showscale=False,
                          title="Systematic bias by flat type")
        st.plotly_chart(fig, use_container_width=True)

        # Plot 2: by town (10 most biased)
        st.subheader("Mean Residual by Town (10 Most Biased)")
        town_resid = (diag_df.groupby('town')['residual']
                             .mean().sort_values().head(10).reset_index()
                             .rename(columns={'residual': 'Mean Residual (SGD)'}))
        fig = px.bar(town_resid, x='Mean Residual (SGD)', y='town', orientation='h',
                     color='Mean Residual (SGD)', color_continuous_scale='RdBu',
                     labels={'town': 'Town'})
        fig.add_vline(x=0, line_dash='dash', line_color='black')
        fig.update_layout(coloraxis_showscale=False,
                          title="Towns with the largest systematic underestimation")
        st.plotly_chart(fig, use_container_width=True)

        # Plot 3: over time
        st.subheader("Mean Residual Over Time")
        diag_df['month_parsed'] = pd.to_datetime(diag_df['month'])
        time_resid = (diag_df.groupby('month_parsed')['residual']
                             .mean().reset_index()
                             .rename(columns={'residual': 'Mean Residual (SGD)'}))
        fig = px.line(time_resid, x='month_parsed', y='Mean Residual (SGD)',
                      labels={'month_parsed': 'Month'},
                      color_discrete_sequence=['steelblue'])
        fig.add_hline(y=0, line_dash='dash', line_color='black')
        fig.update_layout(title="Temporal drift in model residuals")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "A rising trend near the end of the test period suggests the model lags "
            "behind recent price movements — a signal that periodic retraining would be needed "
            "in a production setting."
        )

    with tab4:
        importances = model.feature_importances_
        top_n = st.slider("Show top N features", 5, min(30, len(importances)), 15)
        imp_df = (pd.Series(importances, index=feat_names)
                    .nlargest(top_n).sort_values().reset_index()
                    .rename(columns={'index': 'Feature', 0: 'Importance'}))
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Teal')
        fig.update_layout(coloraxis_showscale=False,
                          title=f"Top {top_n} Feature Importances (XGBoost)")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Prediction":
    st.title("Prediction")
    model, preprocessor = require_model()

    # Get known categorical values from training data if available
    df_ref = st.session_state["df"]

    def get_opts(col, fallback):
        if df_ref is not None and col in df_ref.columns:
            return sorted(df_ref[col].dropna().unique().tolist())
        return fallback

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    # ── Single Prediction ──────────────────────────────────────────────────────

    with tab_single:
        st.markdown("Fill in the flat details below to get an estimated resale price.")

        with st.form("prediction_form"):
            st.markdown("**Transaction & Flat Identity**")
            c1, c2, c3 = st.columns(3)
            with c1:
                month = st.text_input("Month (YYYY-MM)",
                    value="2024-01",
                    help=COL_DESCRIPTIONS['month'])
            with c2:
                town = st.selectbox("Town", get_opts('town', ['ANG MO KIO', 'BEDOK', 'BISHAN']),
                    help=COL_DESCRIPTIONS['town'])
            with c3:
                flat_type = st.selectbox("Flat Type", get_opts('flat_type', FLAT_TYPES),
                    help=COL_DESCRIPTIONS['flat_type'])

            c1, c2, c3 = st.columns(3)
            with c1:
                flat_model = st.selectbox("Flat Model", get_opts('flat_model', ['Model A', 'Improved', 'New Generation']),
                    help=COL_DESCRIPTIONS['flat_model'])
            with c2:
                storey_range = st.selectbox("Storey Range", STOREY_ORDER,
                    index=2, help=COL_DESCRIPTIONS['storey_range'])
            with c3:
                floor_area_sqft = st.number_input("Floor Area (sqft)",
                    min_value=200, max_value=3000, value=1000, step=10,
                    help=COL_DESCRIPTIONS['floor_area_sqft'])

            st.markdown("**Lease**")
            c1, c2, c3 = st.columns(3)
            with c1:
                lease_commence_date = st.number_input("Lease Commence Year",
                    min_value=1960, max_value=2025, value=1990, step=1,
                    help=COL_DESCRIPTIONS['lease_commence_date'])
            with c2:
                remaining_lease_years = st.number_input("Remaining Lease (Years)",
                    min_value=0, max_value=99, value=70, step=1,
                    help=COL_DESCRIPTIONS['remaining_lease_years'])
            with c3:
                remaining_lease_months = st.number_input("Remaining Lease (Months)",
                    min_value=0, max_value=11, value=0, step=1,
                    help=COL_DESCRIPTIONS['remaining_lease_months'])

            st.markdown("**Location**")
            c1, c2, c3 = st.columns(3)
            with c1:
                planning_area_ura = st.selectbox("URA Planning Area",
                    get_opts('planning_area_ura', ['Ang Mo Kio', 'Bedok', 'Bishan']),
                    help=COL_DESCRIPTIONS['planning_area_ura'])
            with c2:
                region_ura = st.selectbox("Region", get_opts('region_ura', REGION_URA),
                    help=COL_DESCRIPTIONS['region_ura'])
            with c3:
                pass

            c1, c2 = st.columns(2)
            with c1:
                latitude = st.number_input("Latitude", min_value=1.2, max_value=1.5,
                    value=1.35, format="%.6f", help=COL_DESCRIPTIONS['latitude'])
            with c2:
                longitude = st.number_input("Longitude", min_value=103.6, max_value=104.0,
                    value=103.82, format="%.6f", help=COL_DESCRIPTIONS['longitude'])

            st.markdown("**Transport**")
            c1, c2, c3 = st.columns(3)
            with c1:
                closest_mrt_station = st.selectbox("Closest MRT Station",
                    get_opts('closest_mrt_station', ['ANG MO KIO', 'BEDOK', 'BISHAN']),
                    help=COL_DESCRIPTIONS['closest_mrt_station'])
            with c2:
                distance_to_mrt_meters = st.number_input("Distance to MRT (m)",
                    min_value=0, max_value=5000, value=500, step=10,
                    help=COL_DESCRIPTIONS['distance_to_mrt_meters'])
            with c3:
                transport_type = st.selectbox("Transport Type",
                    get_opts('transport_type', TRANSPORT_TYPES),
                    help=COL_DESCRIPTIONS['transport_type'])

            c1, c2 = st.columns(2)
            with c1:
                line_color = st.selectbox("MRT Line Color",
                    get_opts('line_color', LINE_COLORS),
                    help=COL_DESCRIPTIONS['line_color'])
            with c2:
                distance_to_cbd = st.number_input("Distance to CBD (m)",
                    min_value=0, max_value=30000, value=8000, step=100,
                    help=COL_DESCRIPTIONS['distance_to_cbd'])

            st.markdown("**School**")
            c1, c2 = st.columns(2)
            with c1:
                closest_pri_school = st.selectbox("Closest Primary School",
                    get_opts('closest_pri_school', ['ANG MO KIO PRIMARY SCHOOL']),
                    help=COL_DESCRIPTIONS['closest_pri_school'])
            with c2:
                distance_to_pri_school_meters = st.number_input("Distance to Primary School (m)",
                    min_value=0, max_value=5000, value=400, step=10,
                    help=COL_DESCRIPTIONS['distance_to_pri_school_meters'])

            submitted = st.form_submit_button("🔮 Estimate Price", type="primary",
                                              use_container_width=True)

        if submitted:
            row = pd.DataFrame([{
                'month':                         month,
                'town':                          town,
                'flat_type':                     flat_type,
                'flat_model':                    flat_model,
                'storey_range':                  storey_range,
                'floor_area_sqft':               floor_area_sqft,
                'lease_commence_date':           lease_commence_date,
                'remaining_lease_years':         remaining_lease_years,
                'remaining_lease_months':        remaining_lease_months,
                'planning_area_ura':             planning_area_ura,
                'region_ura':                    region_ura,
                'latitude':                      latitude,
                'longitude':                     longitude,
                'closest_mrt_station':           closest_mrt_station,
                'distance_to_mrt_meters':        distance_to_mrt_meters,
                'transport_type':                transport_type,
                'line_color':                    line_color,
                'distance_to_cbd':               distance_to_cbd,
                'closest_pri_school':            closest_pri_school,
                'distance_to_pri_school_meters': distance_to_pri_school_meters,
            }])

            errors = validate_predict_columns(row)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    X_input = preprocessor.transform(row)
                    price   = predict(model, X_input)[0]
                    st.success(f"### Estimated Resale Price: **SGD {price:,.0f}**")
                    st.caption(
                        f"Model MAE ≈ $29,484 — actual price likely within "
                        f"**SGD {price - 29484:,.0f} – {price + 29484:,.0f}**"
                    )
                except Exception as ex:
                    st.error(f"Prediction failed: {ex}")

    # ── Batch Prediction ───────────────────────────────────────────────────────

    with tab_batch:
        st.markdown(
            "Upload a CSV without the `resale_price` column. "
            "The app will append a `predicted_resale_price` column and let you download the result."
        )

        with st.expander("Required columns for batch prediction"):
            req_df = pd.DataFrame([
                {'Column': col, 'Description': COL_DESCRIPTIONS.get(col, '')}
                for col in REQUIRED_PREDICT_COLS
            ])
            st.dataframe(req_df, use_container_width=True, hide_index=True)

        batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"],
                                       key="batch_uploader")

        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.write(f"Uploaded: **{len(batch_df):,}** rows × **{batch_df.shape[1]}** columns")

            errors = validate_batch_columns(batch_df)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                if st.button("▶ Run Batch Prediction", type="primary"):
                    with st.spinner(f"Predicting {len(batch_df):,} rows..."):
                        try:
                            X_batch = preprocessor.transform(batch_df)
                            prices  = predict(model, X_batch)
                            batch_df.insert(0, 'predicted_resale_price', prices.round(2))

                            st.success(f"Done. Preview:")
                            st.dataframe(batch_df[['predicted_resale_price'] +
                                                   [c for c in batch_df.columns
                                                    if c != 'predicted_resale_price']].head(10),
                                         use_container_width=True)

                            csv_buf = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv_buf,
                                file_name="predictions.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        except Exception as ex:
                            st.error(f"Batch prediction failed: {ex}")