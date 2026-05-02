import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DataSmart",
    page_icon="assets/favicon.png" if False else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #f7f6f3;
        color: #1a1a1a;
    }

    section[data-testid="stSidebar"] {
        background-color: #1c1c1e;
        border-right: 1px solid #2c2c2e;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e5e5 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #a0a0a0 !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }

    .brand {
        font-family: 'DM Mono', monospace;
        font-size: 1.25rem;
        font-weight: 500;
        color: #ffffff;
        letter-spacing: -0.02em;
        padding: 1.5rem 0 0.25rem 0;
        display: block;
    }

    .brand-sub {
        font-size: 0.72rem;
        color: #6b6b6b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: block;
        margin-bottom: 1.5rem;
    }

    .page-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }

    .page-sub {
        font-size: 0.88rem;
        color: #6b6b6b;
        margin-bottom: 2rem;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9a9a9a;
        margin: 2rem 0 0.75rem 0;
        display: block;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e8e6e1;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9a9a9a;
        margin-bottom: 0.3rem;
    }

    .stat-value {
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: #1a1a1a;
        line-height: 1;
    }

    .result-good {
        background: #f0faf4;
        border-left: 3px solid #2d9e5f;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.875rem;
        color: #1a4a2e;
        margin: 0.4rem 0;
    }

    .result-warn {
        background: #fdf8f0;
        border-left: 3px solid #c97c2e;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.875rem;
        color: #5c3a10;
        margin: 0.4rem 0;
    }

    .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: none;
        border-radius: 7px;
        padding: 0.55rem 1.4rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: background 0.15s ease;
    }

    .stButton > button:hover {
        background-color: #333333;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #e8e6e1;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9a9a9a;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        border-radius: 0;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #1a1a1a;
        border-bottom: 2px solid #1a1a1a;
        background: transparent;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8e6e1;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }

    div[data-testid="stMetric"] label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9a9a9a;
        font-weight: 600;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'DM Mono', monospace;
        font-size: 1.5rem;
        color: #1a1a1a;
    }

    .stDataFrame {
        border: 1px solid #e8e6e1;
        border-radius: 8px;
        overflow: hidden;
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #ffffff;
        border-color: #e8e6e1;
        border-radius: 7px;
        font-size: 0.875rem;
    }

    .stFileUploader > div {
        background: #ffffff;
        border: 1px dashed #d0cdc8;
        border-radius: 10px;
    }

    .upload-hint {
        font-size: 0.78rem;
        color: #9a9a9a;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    hr {
        border: none;
        border-top: 1px solid #2c2c2e;
        margin: 1.25rem 0;
    }

    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4a4a4a;
        margin: 1.25rem 0 0.6rem 0;
        display: block;
    }

    .empty-state {
        background: #ffffff;
        border: 1px solid #e8e6e1;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }

    .empty-state h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }

    .empty-state p {
        font-size: 0.875rem;
        color: #9a9a9a;
        margin: 0;
    }

    .chart-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b6b6b;
        margin-bottom: 0.5rem;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


# ── Chart style ───────────────────────────────────────────────────────────────

CHART_BG   = "#ffffff"
CHART_FACE = "#ffffff"
BAR_COLOR  = "#1a1a1a"
LINE_COLOR = "#2d9e5f"
TICK_COLOR = "#9a9a9a"
SPINE_COLOR = "#e8e6e1"
TEXT_COLOR = "#1a1a1a"

def style_axes(ax, fig):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_FACE)
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_column_types(df):
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def clean_data(df, strategy="mean"):
    df_clean = df.copy()
    report   = []

    dupes = df_clean.duplicated().sum()
    if dupes > 0:
        df_clean = df_clean.drop_duplicates()
        report.append(("good", f"Removed {dupes} duplicate rows"))

    numeric_cols, categorical_cols = detect_column_types(df_clean)
    missing_before = df_clean.isnull().sum().sum()

    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if strategy == "mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == "median":
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(0, inplace=True)

    for col in categorical_cols:
        if df_clean[col].isnull().any():
            fill = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
            df_clean[col].fillna(fill, inplace=True)

    missing_after = df_clean.isnull().sum().sum()
    if missing_before > 0:
        filled = int(missing_before - missing_after)
        report.append(("good", f"Filled {filled} missing values using {strategy} strategy"))

    outlier_count = 0
    for col in numeric_cols:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_count += int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())

    if outlier_count > 0:
        report.append(("warn", f"Detected {outlier_count} potential outliers — values kept, review manually"))

    if not report:
        report.append(("good", "No issues found. Data is already clean."))

    return df_clean, report


def encode_categoricals(df):
    df_enc = df.copy()
    le     = LabelEncoder()
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc


def train_model(X_train, X_test, y_train, y_test, model_name, task):
    models = {
        "Random Forest":    RandomForestRegressor(n_estimators=100, random_state=42)  if task == "regression" else RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":    DecisionTreeRegressor(random_state=42)                    if task == "regression" else DecisionTreeClassifier(random_state=42),
        "Linear / Logistic": LinearRegression()                                        if task == "regression" else LogisticRegression(max_iter=500),
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task == "regression":
        return model, preds, {
            "MAE":      round(mean_absolute_error(y_test, preds), 4),
            "R2 Score": round(r2_score(y_test, preds), 4),
        }
    else:
        return model, preds, {
            "Accuracy": f"{round(accuracy_score(y_test, preds) * 100, 2)}%"
        }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<span class='brand'>DataSmart</span>", unsafe_allow_html=True)
    st.markdown("<span class='brand-sub'>Data analysis platform</span>", unsafe_allow_html=True)

    st.markdown("<span class='sidebar-section'>Dataset</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        st.markdown(f"<span style='font-size:0.8rem;color:#6b6b6b;'>{uploaded_file.name}</span>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-section'>Configuration</span>", unsafe_allow_html=True)
    fill_strategy = st.selectbox("Missing value strategy", ["mean", "median", "zero"])
    test_size     = st.slider("Test split ratio", 0.1, 0.4, 0.2, 0.05)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:0.75rem;color:#4a4a4a;line-height:1.7;display:block'>"
        "Upload any CSV to clean, explore, and train predictive models — no coding needed."
        "</span>",
        unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("<div class='page-title'>DataSmart</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Upload a CSV file to get started — cleaning, analysis, and model training in one place.</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
    <div class='empty-state'>
        <h3>No file selected</h3>
        <p>Use the sidebar to upload a CSV file and the app will handle the rest.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded_file)
numeric_cols, categorical_cols = detect_column_types(df)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Cleaning", "Analysis", "Model"])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("<span class='section-label'>Dataset summary</span>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           f"{df.shape[0]:,}")
    c2.metric("Columns",        df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate rows", int(df.duplicated().sum()))

    st.markdown("<span class='section-label'>Preview</span>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='section-label'>Column types</span>", unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Column":    df.columns,
            "Type":      df.dtypes.astype(str).values,
            "Missing":   df.isnull().sum().values,
            "Missing %": (df.isnull().sum().values / len(df) * 100).round(1),
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with col2:
        if numeric_cols:
            st.markdown("<span class='section-label'>Numeric summary</span>", unsafe_allow_html=True)
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)


# ── Tab 2: Cleaning ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("<span class='section-label'>Automatic cleaning</span>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:0.875rem;color:#6b6b6b;display:block;margin-bottom:1rem'>"
        "Removes duplicates, fills missing values, and flags outliers based on the strategy selected in the sidebar."
        "</span>",
        unsafe_allow_html=True
    )

    if st.button("Run cleaning"):
        with st.spinner("Processing..."):
            df_clean, report = clean_data(df, fill_strategy)
            st.session_state["df_clean"]     = df_clean
            st.session_state["clean_report"] = report

    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"]
        report   = st.session_state["clean_report"]

        st.markdown("<span class='section-label'>Cleaning report</span>", unsafe_allow_html=True)
        for kind, msg in report:
            css_class = "result-good" if kind == "good" else "result-warn"
            st.markdown(f"<div class='{css_class}'>{msg}</div>", unsafe_allow_html=True)

        st.markdown("<span class='section-label'>Before vs after</span>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Rows before",    df.shape[0])
        col1.metric("Missing before", int(df.isnull().sum().sum()))
        col2.metric("Rows after",     df_clean.shape[0])
        col2.metric("Missing after",  int(df_clean.isnull().sum().sum()))

        st.markdown("<span class='section-label'>Cleaned data preview</span>", unsafe_allow_html=True)
        st.dataframe(df_clean.head(10), use_container_width=True)

        csv_bytes = df_clean.to_csv(index=False).encode()
        st.download_button("Download cleaned CSV", csv_bytes, "cleaned_data.csv", "text/csv")
    else:
        st.markdown(
            "<span style='font-size:0.875rem;color:#9a9a9a'>Click 'Run cleaning' to process the dataset.</span>",
            unsafe_allow_html=True
        )


# ── Tab 3: Analysis ───────────────────────────────────────────────────────────
with tab3:
    if not numeric_cols:
        st.warning("No numeric columns found in this dataset.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<span class='chart-title'>Distribution</span>", unsafe_allow_html=True)
            sel_col = st.selectbox("Column", numeric_cols, key="dist_col", label_visibility="collapsed")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            style_axes(ax, fig)
            ax.hist(df[sel_col].dropna(), bins=28, color=BAR_COLOR, edgecolor=CHART_BG, alpha=0.85, linewidth=0.4)
            ax.set_xlabel(sel_col, fontsize=9, color=TICK_COLOR)
            ax.set_ylabel("Count",  fontsize=9, color=TICK_COLOR)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("<span class='chart-title'>Outlier detection</span>", unsafe_allow_html=True)
            sel_col2 = st.selectbox("Column", numeric_cols, key="box_col", label_visibility="collapsed")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            style_axes(ax2, fig2)
            ax2.boxplot(
                df[sel_col2].dropna(), patch_artist=True,
                boxprops=dict(facecolor="#e8e6e1", color=BAR_COLOR),
                medianprops=dict(color=LINE_COLOR, linewidth=1.5),
                whiskerprops=dict(color=TICK_COLOR),
                capprops=dict(color=TICK_COLOR),
                flierprops=dict(markerfacecolor="#c0392b", marker=".", markersize=4, alpha=0.6),
            )
            ax2.set_ylabel(sel_col2, fontsize=9, color=TICK_COLOR)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        if len(numeric_cols) > 1:
            st.markdown("<span class='section-label'>Correlation matrix</span>", unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(10, 4.5))
            style_axes(ax3, fig3)
            corr = df[numeric_cols].corr()
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax3, linewidths=0.4, linecolor="#f0ede8",
                annot_kws={"size": 8}, vmin=-1, vmax=1,
            )
            ax3.tick_params(labelsize=8, colors=TICK_COLOR)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        if categorical_cols:
            st.markdown("<span class='section-label'>Category breakdown</span>", unsafe_allow_html=True)
            sel_cat    = st.selectbox("Column", categorical_cols, label_visibility="collapsed")
            val_counts = df[sel_cat].value_counts().head(10)
            fig4, ax4  = plt.subplots(figsize=(8, 3.5))
            style_axes(ax4, fig4)
            shades = [BAR_COLOR if i == 0 else f"#{hex(int(26 + i * 22))[2:].zfill(2)}{hex(int(26 + i * 22))[2:].zfill(2)}{hex(int(26 + i * 22))[2:].zfill(2)}" for i in range(len(val_counts))]
            ax4.barh(val_counts.index.astype(str)[::-1], val_counts.values[::-1], color=shades[::-1], height=0.6)
            ax4.set_xlabel("Count", fontsize=9, color=TICK_COLOR)
            ax4.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()


# ── Tab 4: Model ──────────────────────────────────────────────────────────────
with tab4:
    work_df = st.session_state.get("df_clean", df)

    st.markdown("<span class='section-label'>Model configuration</span>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        target_col  = st.selectbox("Target column",  work_df.columns)
    with col2:
        task_type   = st.selectbox("Task type",      ["regression", "classification"])
    with col3:
        model_choice = st.selectbox("Algorithm",     ["Random Forest", "Decision Tree", "Linear / Logistic"])

    feature_cols      = [c for c in work_df.columns if c != target_col]
    selected_features = st.multiselect(
        "Feature columns",
        feature_cols,
        default=feature_cols[:min(6, len(feature_cols))]
    )

    if st.button("Train model"):
        if not selected_features:
            st.error("Select at least one feature column.")
        else:
            with st.spinner("Training..."):
                try:
                    model_df = encode_categoricals(work_df[selected_features + [target_col]].dropna())
                    X = model_df[selected_features]
                    y = model_df[target_col]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    model, preds, metrics = train_model(
                        X_train, X_test, y_train, y_test, model_choice, task_type
                    )

                    st.session_state["trained_model"]  = model
                    st.session_state["model_preds"]    = preds
                    st.session_state["model_metrics"]  = metrics
                    st.session_state["y_test"]         = y_test
                    st.session_state["feature_cols"]   = selected_features
                    st.success("Model trained.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if "trained_model" in st.session_state:
        metrics = st.session_state["model_metrics"]
        preds   = st.session_state["model_preds"]
        y_test  = st.session_state["y_test"]

        st.markdown("<span class='section-label'>Performance metrics</span>", unsafe_allow_html=True)
        cols = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(k, v)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<span class='section-label'>Actual vs predicted (first 20)</span>", unsafe_allow_html=True)
            compare_df = pd.DataFrame({
                "Actual":    y_test.values[:20],
                "Predicted": preds[:20],
            })
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

        with col2:
            if task_type == "regression":
                st.markdown("<span class='chart-title'>Prediction accuracy</span>", unsafe_allow_html=True)
                fig5, ax5 = plt.subplots(figsize=(6, 3.5))
                style_axes(ax5, fig5)
                ax5.scatter(y_test.values[:60], preds[:60], color=BAR_COLOR, alpha=0.55, s=28, linewidths=0)
                lims = [
                    min(float(y_test.min()), float(preds.min())),
                    max(float(y_test.max()), float(preds.max())),
                ]
                ax5.plot(lims, lims, color=LINE_COLOR, linewidth=1.2, linestyle="--")
                ax5.set_xlabel("Actual",    fontsize=9, color=TICK_COLOR)
                ax5.set_ylabel("Predicted", fontsize=9, color=TICK_COLOR)
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close()

        model = st.session_state["trained_model"]
        if hasattr(model, "feature_importances_"):
            st.markdown("<span class='section-label'>Feature importance</span>", unsafe_allow_html=True)
            feat_imp = pd.Series(
                model.feature_importances_,
                index=st.session_state["feature_cols"]
            ).sort_values(ascending=True)

            fig6, ax6 = plt.subplots(figsize=(8, max(2.5, len(feat_imp) * 0.38)))
            style_axes(ax6, fig6)
            bar_colors = [BAR_COLOR if v == feat_imp.max() else "#d0cdc8" for v in feat_imp.values]
            ax6.barh(feat_imp.index, feat_imp.values, color=bar_colors, height=0.55)
            ax6.set_xlabel("Importance", fontsize=9, color=TICK_COLOR)
            ax6.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()
