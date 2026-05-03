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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@300;400;500&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
    }

    .stApp {
        background-color: #FAFAF8;
        color: #1a1a1a;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #1A1A2E;
        padding: 0;
    }

    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }

    /* Force ALL sidebar text to be white */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] small {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #CBD5E1 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }

    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #2D2D4E !important;
        border-color: #4A4A6A !important;
        border-radius: 6px !important;
    }

    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #2D2D4E !important;
        border: 1px dashed #4A4A6A !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #4F46E5 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #94A3B8 !important;
    }

    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #94A3B8 !important;
    }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        font-family: 'EB Garamond', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFFFFF !important;
        letter-spacing: -0.01em;
        line-height: 1.1;
        margin-bottom: 0.2rem;
        display: block;
    }

    .sidebar-tagline {
        font-size: 0.72rem;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 400;
        display: block;
        margin-bottom: 2rem;
    }

    .sidebar-section {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #64748B !important;
        margin: 1.8rem 0 0.8rem 0;
        display: block;
        border-top: 1px solid #2D2D4E;
        padding-top: 1.2rem;
    }

    .sidebar-desc {
        font-size: 0.8rem;
        color: #94A3B8 !important;
        line-height: 1.7;
        display: block;
        margin-top: 1.5rem;
    }

    /* ── Page headings ── */
    .page-title {
        font-family: 'EB Garamond', Georgia, serif;
        font-size: 2.4rem;
        font-weight: 500;
        color: #1a1a1a;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.3rem;
    }

    .page-sub {
        font-size: 0.92rem;
        color: #6B7280;
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    /* ── Section labels ── */
    .section-label {
        font-family: 'EB Garamond', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 500;
        color: #1a1a1a;
        margin: 2rem 0 0.6rem 0;
        display: block;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0.4rem;
    }

    .chart-label {
        font-family: 'EB Garamond', Georgia, serif;
        font-size: 1rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.5rem;
        display: block;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #9CA3AF !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'EB Garamond', Georgia, serif !important;
        font-size: 1.8rem !important;
        color: #111827 !important;
        font-weight: 500 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #E5E7EB;
        gap: 0;
        margin-bottom: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        padding: 0.65rem 1.3rem;
        border-bottom: 2px solid transparent;
        border-radius: 0;
    }

    .stTabs [aria-selected="true"] {
        color: #111827 !important;
        border-bottom: 2px solid #111827 !important;
        background: transparent !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #111827;
        color: #FFFFFF;
        border: none;
        border-radius: 7px;
        padding: 0.55rem 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        transition: background 0.15s;
    }

    .stButton > button:hover {
        background-color: #374151;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background-color: transparent;
        color: #111827;
        border: 1px solid #D1D5DB;
        border-radius: 7px;
        padding: 0.5rem 1.3rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .stDownloadButton > button:hover {
        background-color: #F9FAFB;
        border-color: #9CA3AF;
    }

    /* ── Result boxes ── */
    .result-good {
        background: #F0FDF4;
        border-left: 3px solid #16A34A;
        padding: 0.75rem 1rem;
        border-radius: 0 7px 7px 0;
        font-size: 0.875rem;
        color: #14532D;
        margin: 0.4rem 0;
        font-family: 'Inter', sans-serif;
    }

    .result-warn {
        background: #FFFBEB;
        border-left: 3px solid #D97706;
        padding: 0.75rem 1rem;
        border-radius: 0 7px 7px 0;
        font-size: 0.875rem;
        color: #78350F;
        margin: 0.4rem 0;
        font-family: 'Inter', sans-serif;
    }

    /* ── Empty state ── */
    .empty-state {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 4rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .empty-title {
        font-family: 'EB Garamond', Georgia, serif;
        font-size: 1.4rem;
        font-weight: 500;
        color: #111827;
        margin-bottom: 0.5rem;
    }

    .empty-sub {
        font-size: 0.875rem;
        color: #9CA3AF;
        font-family: 'Inter', sans-serif;
    }

    /* ── Table / dataframe ── */
    .stDataFrame {
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #FFFFFF;
        border-color: #D1D5DB;
        border-radius: 7px;
        font-size: 0.875rem;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #2D2D4E; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Chart helpers ─────────────────────────────────────────────────────────────
BG       = "#FFFFFF"
BAR      = "#1A1A2E"
ACCENT   = "#4F46E5"
LINE_CLR = "#16A34A"
TICK     = "#9CA3AF"
SPINE    = "#E5E7EB"
TXT      = "#111827"

def style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK, labelsize=9)
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    ax.xaxis.label.set_color(TICK)
    ax.yaxis.label.set_color(TICK)


# ── Data helpers ──────────────────────────────────────────────────────────────
def col_types(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object","category"]).columns.tolist()
    return num, cat

def clean_data(df, strategy):
    d = df.copy(); report = []
    dup = d.duplicated().sum()
    if dup:
        d = d.drop_duplicates()
        report.append(("good", f"Removed {dup} duplicate rows"))
    num, cat = col_types(d)
    mb = d.isnull().sum().sum()
    for c in num:
        if d[c].isnull().any():
            d[c].fillna(d[c].mean() if strategy=="mean" else d[c].median() if strategy=="median" else 0, inplace=True)
    for c in cat:
        if d[c].isnull().any():
            d[c].fillna(d[c].mode()[0] if not d[c].mode().empty else "Unknown", inplace=True)
    filled = int(mb - d.isnull().sum().sum())
    if filled:
        report.append(("good", f"Filled {filled} missing values using {strategy}"))
    oc = sum(int(((d[c]<d[c].quantile(.25)-1.5*(d[c].quantile(.75)-d[c].quantile(.25)))|(d[c]>d[c].quantile(.75)+1.5*(d[c].quantile(.75)-d[c].quantile(.25)))).sum()) for c in num)
    if oc:
        report.append(("warn", f"Detected {oc} potential outliers — review manually"))
    if not report:
        report.append(("good","No issues found. Data is already clean."))
    return d, report

def encode_cats(df):
    d = df.copy(); le = LabelEncoder()
    for c in d.select_dtypes(include=["object","category"]).columns:
        d[c] = le.fit_transform(d[c].astype(str))
    return d

def run_model(Xtr, Xte, ytr, yte, name, task):
    m = {
        "Random Forest":     RandomForestRegressor(100,random_state=42)  if task=="regression" else RandomForestClassifier(100,random_state=42),
        "Decision Tree":     DecisionTreeRegressor(random_state=42)       if task=="regression" else DecisionTreeClassifier(random_state=42),
        "Linear / Logistic": LinearRegression()                            if task=="regression" else LogisticRegression(max_iter=500),
    }[name]
    m.fit(Xtr, ytr); p = m.predict(Xte)
    if task=="regression":
        return m, p, {"MAE": round(mean_absolute_error(yte,p),4), "R² Score": round(r2_score(yte,p),4)}
    return m, p, {"Accuracy": f"{round(accuracy_score(yte,p)*100,2)}%"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<span class='sidebar-brand'>DataSmart</span>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-tagline'>Data Analysis Platform</span>", unsafe_allow_html=True)

    st.markdown("<span class='sidebar-section'>Dataset</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file:
        st.markdown(f"<span style='font-size:0.8rem;color:#94A3B8;margin-top:0.4rem;display:block'>{uploaded_file.name}</span>", unsafe_allow_html=True)

    st.markdown("<span class='sidebar-section'>Configuration</span>", unsafe_allow_html=True)
    fill_strategy = st.selectbox("Missing Value Strategy", ["mean","median","zero"])
    test_size     = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)

    st.markdown(
        "<span class='sidebar-desc'>Upload any CSV file to clean, analyze, and train predictive models — no coding needed.</span>",
        unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>DataSmart</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Upload a CSV file to get started — cleaning, analysis, and model training in one place.</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
    <div class='empty-state'>
        <div class='empty-title'>No file selected</div>
        <div class='empty-sub'>Use the sidebar to upload a CSV file and the app will handle the rest.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded_file)
num_cols, cat_cols = col_types(df)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Cleaning", "Analysis", "Model"])


# ── Overview ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<span class='section-label'>Dataset Summary</span>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Rows",      f"{df.shape[0]:,}")
    c2.metric("Total Columns",   df.shape[1])
    c3.metric("Missing Values",  int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows",  int(df.duplicated().sum()))

    st.markdown("<span class='section-label'>Data Preview</span>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='section-label'>Column Information</span>", unsafe_allow_html=True)
        info = pd.DataFrame({
            "Column":    df.columns,
            "Type":      df.dtypes.astype(str).values,
            "Missing":   df.isnull().sum().values,
            "Missing %": (df.isnull().sum().values / len(df) * 100).round(1),
        })
        st.dataframe(info, use_container_width=True, hide_index=True)
    with col2:
        if num_cols:
            st.markdown("<span class='section-label'>Statistical Summary</span>", unsafe_allow_html=True)
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)


# ── Cleaning ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<span class='section-label'>Automatic Data Cleaning</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.875rem;color:#6B7280;display:block;margin-bottom:1.2rem'>Removes duplicates, fills missing values, and flags outliers based on the strategy selected in the sidebar.</span>", unsafe_allow_html=True)

    if st.button("Run Cleaning"):
        with st.spinner("Processing..."):
            df_c, rep = clean_data(df, fill_strategy)
            st.session_state["df_clean"] = df_c
            st.session_state["report"]   = rep

    if "df_clean" in st.session_state:
        df_c = st.session_state["df_clean"]
        rep  = st.session_state["report"]

        st.markdown("<span class='section-label'>Cleaning Report</span>", unsafe_allow_html=True)
        for kind, msg in rep:
            st.markdown(f"<div class='result-{'good' if kind=='good' else 'warn'}'>{msg}</div>", unsafe_allow_html=True)

        st.markdown("<span class='section-label'>Before vs After</span>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        b1.metric("Rows Before",    df.shape[0])
        b1.metric("Missing Before", int(df.isnull().sum().sum()))
        b2.metric("Rows After",     df_c.shape[0])
        b2.metric("Missing After",  int(df_c.isnull().sum().sum()))

        st.markdown("<span class='section-label'>Cleaned Data Preview</span>", unsafe_allow_html=True)
        st.dataframe(df_c.head(10), use_container_width=True)
        st.download_button("Download Cleaned CSV", df_c.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")
    else:
        st.markdown("<span style='font-size:0.875rem;color:#9CA3AF'>Click Run Cleaning to process your dataset.</span>", unsafe_allow_html=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
with tab3:
    if not num_cols:
        st.warning("No numeric columns found in this dataset.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<span class='chart-label'>Distribution</span>", unsafe_allow_html=True)
            s1 = st.selectbox("Select column", num_cols, key="dc", label_visibility="collapsed")
            fig, ax = plt.subplots(figsize=(6,3.5))
            style_ax(ax, fig)
            ax.hist(df[s1].dropna(), bins=28, color=BAR, edgecolor=BG, alpha=0.88, linewidth=0.3)
            ax.set_xlabel(s1, fontsize=9); ax.set_ylabel("Count", fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("<span class='chart-label'>Outlier Detection</span>", unsafe_allow_html=True)
            s2 = st.selectbox("Select column", num_cols, key="bc", label_visibility="collapsed")
            fig2, ax2 = plt.subplots(figsize=(6,3.5))
            style_ax(ax2, fig2)
            ax2.boxplot(df[s2].dropna(), patch_artist=True,
                boxprops=dict(facecolor="#E0E7FF", color=BAR),
                medianprops=dict(color=LINE_CLR, linewidth=2),
                whiskerprops=dict(color=TICK), capprops=dict(color=TICK),
                flierprops=dict(markerfacecolor="#EF4444", marker=".", markersize=5, alpha=0.6))
            ax2.set_ylabel(s2, fontsize=9)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        if len(num_cols) > 1:
            st.markdown("<span class='section-label'>Correlation Matrix</span>", unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(10,4.5))
            style_ax(ax3, fig3)
            sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax3, linewidths=0.4, linecolor="#F3F4F6",
                        annot_kws={"size":8}, vmin=-1, vmax=1)
            ax3.tick_params(labelsize=8, colors=TICK)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        if cat_cols:
            st.markdown("<span class='section-label'>Category Breakdown</span>", unsafe_allow_html=True)
            sc = st.selectbox("Select column", cat_cols, label_visibility="collapsed")
            vc = df[sc].value_counts().head(10)
            fig4, ax4 = plt.subplots(figsize=(8,3.5))
            style_ax(ax4, fig4)
            colors = [BAR if i==0 else "#CBD5E1" for i in range(len(vc))]
            ax4.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=colors[::-1], height=0.55)
            ax4.set_xlabel("Count", fontsize=9)
            plt.tight_layout(); st.pyplot(fig4); plt.close()


# ── Model ─────────────────────────────────────────────────────────────────────
with tab4:
    wdf = st.session_state.get("df_clean", df)

    st.markdown("<span class='section-label'>Model Configuration</span>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: target = st.selectbox("Target Column",  wdf.columns)
    with c2: task   = st.selectbox("Task Type",      ["regression","classification"])
    with c3: algo   = st.selectbox("Algorithm",      ["Random Forest","Decision Tree","Linear / Logistic"])

    feats = [c for c in wdf.columns if c != target]
    sel   = st.multiselect("Feature Columns", feats, default=feats[:min(6,len(feats))])

    if st.button("Train Model"):
        if not sel:
            st.error("Select at least one feature column.")
        else:
            with st.spinner("Training..."):
                try:
                    mdf = encode_cats(wdf[sel+[target]].dropna())
                    X, y = mdf[sel], mdf[target]
                    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=test_size, random_state=42)
                    model, preds, metrics = run_model(Xtr, Xte, ytr, yte, algo, task)
                    st.session_state.update({"model":model,"preds":preds,"metrics":metrics,"yte":yte,"sel":sel})
                    st.success("Model trained successfully.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if "model" in st.session_state:
        metrics = st.session_state["metrics"]
        preds   = st.session_state["preds"]
        yte     = st.session_state["yte"]

        st.markdown("<span class='section-label'>Performance Metrics</span>", unsafe_allow_html=True)
        mc = st.columns(len(metrics))
        for i,(k,v) in enumerate(metrics.items()):
            mc[i].metric(k, v)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<span class='section-label'>Actual vs Predicted</span>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Actual": yte.values[:20], "Predicted": preds[:20]}),
                         use_container_width=True, hide_index=True)
        with col2:
            if task == "regression":
                st.markdown("<span class='chart-label'>Prediction Accuracy</span>", unsafe_allow_html=True)
                fig5, ax5 = plt.subplots(figsize=(6,3.5))
                style_ax(ax5, fig5)
                ax5.scatter(yte.values[:60], preds[:60], color=BAR, alpha=0.55, s=28, linewidths=0)
                lm = [min(float(yte.min()),float(preds.min())), max(float(yte.max()),float(preds.max()))]
                ax5.plot(lm, lm, color=LINE_CLR, linewidth=1.3, linestyle="--")
                ax5.set_xlabel("Actual", fontsize=9); ax5.set_ylabel("Predicted", fontsize=9)
                plt.tight_layout(); st.pyplot(fig5); plt.close()

        model = st.session_state["model"]
        if hasattr(model, "feature_importances_"):
            st.markdown("<span class='section-label'>Feature Importance</span>", unsafe_allow_html=True)
            fi = pd.Series(model.feature_importances_, index=st.session_state["sel"]).sort_values(ascending=True)
            fig6, ax6 = plt.subplots(figsize=(8, max(2.5, len(fi)*0.4)))
            style_ax(ax6, fig6)
            ax6.barh(fi.index, fi.values,
                     color=[BAR if v==fi.max() else "#CBD5E1" for v in fi.values], height=0.55)
            ax6.set_xlabel("Importance", fontsize=9)
            plt.tight_layout(); st.pyplot(fig6); plt.close()
