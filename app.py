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
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ════════════════════════════════════════
   GLOBAL
════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #F5F4F0;
    color: #111827;
}

/* ════════════════════════════════════════
   SIDEBAR — full nuclear override
════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    min-width: 260px;
}
section[data-testid="stSidebar"] > div {
    padding: 2rem 1.4rem 2rem 1.4rem;
}

/* Every possible text node in sidebar → white */
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] *::before,
section[data-testid="stSidebar"] *::after {
    color: #F1F5F9 !important;
}

/* Sidebar labels (small caps style) */
section[data-testid="stSidebar"] label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #94A3B8 !important;
}

/* Sidebar select box */
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 7px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-baseweb="select"] input {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
}

/* Sidebar file uploader box */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: #1E293B !important;
    border: 1.5px dashed #475569 !important;
    border-radius: 10px !important;
    padding: 1.1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
}

/* Slider track & thumb */
section[data-testid="stSidebar"] [data-testid="stSlider"] * {
    color: #94A3B8 !important;
}

/* Sidebar divider lines */
section[data-testid="stSidebar"] hr {
    border-color: #1E293B !important;
    margin: 1.2rem 0 !important;
}

/* ════════════════════════════════════════
   SIDEBAR TYPOGRAPHY CLASSES
════════════════════════════════════════ */
.sb-brand {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 2rem;
    font-weight: 600;
    color: #FFFFFF !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
    display: block;
    margin-bottom: 0.15rem;
}
.sb-tagline {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #64748B !important;
    display: block;
    margin-bottom: 1.8rem;
}
.sb-section {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #475569 !important;
    display: block;
    margin: 1.6rem 0 0.7rem 0;
    padding-top: 1.2rem;
    border-top: 1px solid #1E293B;
}
.sb-hint {
    font-size: 0.78rem;
    color: #64748B !important;
    line-height: 1.7;
    display: block;
    margin-top: 1.4rem;
}

/* ════════════════════════════════════════
   MAIN AREA TYPOGRAPHY
════════════════════════════════════════ */
.pg-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 500;
    color: #0F172A;
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.pg-sub {
    font-size: 0.95rem;
    color: #64748B;
    font-weight: 400;
    line-height: 1.65;
    margin-bottom: 2rem;
}
.sec-head {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 500;
    color: #0F172A;
    border-bottom: 1.5px solid #E2E8F0;
    padding-bottom: 0.4rem;
    margin: 2rem 0 0.8rem 0;
    display: block;
}
.chart-head {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 500;
    color: #1E293B;
    margin-bottom: 0.5rem;
    display: block;
}

/* ════════════════════════════════════════
   METRIC CARDS
════════════════════════════════════════ */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
div[data-testid="stMetric"] label {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #94A3B8 !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 2rem !important;
    color: #0F172A !important;
    font-weight: 500 !important;
    line-height: 1.1 !important;
}

/* ════════════════════════════════════════
   TABS
════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1.5px solid #E2E8F0;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94A3B8;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.7rem 1.4rem;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    font-family: 'Inter', sans-serif;
}
.stTabs [aria-selected="true"] {
    color: #0F172A !important;
    border-bottom: 2px solid #0F172A !important;
    background: transparent !important;
}

/* ════════════════════════════════════════
   BUTTONS
════════════════════════════════════════ */
.stButton > button {
    background-color: #0F172A;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.6rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    transition: background 0.15s ease;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background-color: #1E293B;
}
.stDownloadButton > button {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1.5px solid #CBD5E1;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
}
.stDownloadButton > button:hover {
    background-color: #F8FAFC;
    border-color: #94A3B8;
}

/* ════════════════════════════════════════
   MAIN AREA INPUTS
════════════════════════════════════════ */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #0F172A !important;
}
.stSelectbox [data-baseweb="select"] *,
.stMultiSelect [data-baseweb="select"] * {
    color: #0F172A !important;
    background: #FFFFFF !important;
}
.stSelectbox label,
.stMultiSelect label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    letter-spacing: 0.01em !important;
}

/* ════════════════════════════════════════
   RESULT BOXES
════════════════════════════════════════ */
.r-good {
    background: #F0FDF4;
    border-left: 3px solid #16A34A;
    padding: 0.8rem 1.1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.875rem;
    color: #14532D;
    margin: 0.45rem 0;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
}
.r-warn {
    background: #FFFBEB;
    border-left: 3px solid #D97706;
    padding: 0.8rem 1.1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.875rem;
    color: #78350F;
    margin: 0.45rem 0;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
}

/* ════════════════════════════════════════
   EMPTY STATE
════════════════════════════════════════ */
.empty-wrap {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 16px;
    padding: 5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.empty-head {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 500;
    color: #0F172A;
    margin-bottom: 0.5rem;
}
.empty-body {
    font-size: 0.9rem;
    color: #94A3B8;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}

/* ════════════════════════════════════════
   DATAFRAME
════════════════════════════════════════ */
.stDataFrame {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #FFFFFF !important;
}
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
    st.markdown("<span class='sb-brand'>DataSmart</span>", unsafe_allow_html=True)
    st.markdown("<span class='sb-tagline'>Know your data better</span>", unsafe_allow_html=True)

    st.markdown("<span class='sb-section'>Dataset</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file:
        st.markdown(f"<span style='font-size:0.8rem;color:#94A3B8;margin-top:0.4rem;display:block'>{uploaded_file.name}</span>", unsafe_allow_html=True)

    st.markdown("<span class='sb-section'>Configuration</span>", unsafe_allow_html=True)
    fill_strategy = st.selectbox("Missing Value Strategy", ["mean","median","zero"])
    test_size     = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)

    st.markdown(
        "<span class='sb-hint'>Works with any CSV — sales data, student records, survey results, you name it. No coding required.</span>",
        unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='pg-title'>DataSmart</div>", unsafe_allow_html=True)
st.markdown("<div class='pg-sub'>Drop in any CSV and see what your data is really telling you — clean it, explore it, and build a working model in minutes.</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
    <div class='empty-wrap'>
        <div class='empty-head'>No file uploaded yet</div>
        <div class='empty-body'>Pick a CSV from the sidebar — the app takes it from there.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded_file)
num_cols, cat_cols = col_types(df)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Cleaning", "Analysis", "Model"])


# ── Overview ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<span class='sec-head'>What is in this file</span>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",            f"{df.shape[0]:,}")
    c2.metric("Columns",         df.shape[1])
    c3.metric("Missing values",  int(df.isnull().sum().sum()))
    c4.metric("Duplicate rows",  int(df.duplicated().sum()))

    st.markdown("<span class='sec-head'>First look at the data</span>", unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='sec-head'>Column breakdown</span>", unsafe_allow_html=True)
        info = pd.DataFrame({
            "Column":    df.columns,
            "Type":      df.dtypes.astype(str).values,
            "Missing":   df.isnull().sum().values,
            "Missing %": (df.isnull().sum().values / len(df) * 100).round(1),
        })
        st.dataframe(info, use_container_width=True, hide_index=True)
    with col2:
        if num_cols:
            st.markdown("<span class='sec-head'>Numbers at a glance</span>", unsafe_allow_html=True)
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)


# ── Cleaning ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<span class='sec-head'>Clean up your data</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.875rem;color:#6B7280;display:block;margin-bottom:1.2rem'>Finds and fixes the common problems — missing values, duplicate rows, and unusual outliers. Choose your preferred fill method in the sidebar before running.</span>", unsafe_allow_html=True)

    if st.button("Run cleaning"):
        with st.spinner("Processing..."):
            df_c, rep = clean_data(df, fill_strategy)
            st.session_state["df_clean"] = df_c
            st.session_state["report"]   = rep

    if "df_clean" in st.session_state:
        df_c = st.session_state["df_clean"]
        rep  = st.session_state["report"]

        st.markdown("<span class='sec-head'>Here is what changed</span>", unsafe_allow_html=True)
        for kind, msg in rep:
            css = "r-good" if kind == "good" else "r-warn"
            st.markdown(f"<div class='{css}'>{msg}</div>", unsafe_allow_html=True)

        st.markdown("<span class='sec-head'>Before and after</span>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        b1.metric("Rows Before",    df.shape[0])
        b1.metric("Missing Before", int(df.isnull().sum().sum()))
        b2.metric("Rows After",     df_c.shape[0])
        b2.metric("Missing After",  int(df_c.isnull().sum().sum()))

        st.markdown("<span class='sec-head'>Your cleaned data</span>", unsafe_allow_html=True)
        st.dataframe(df_c.head(10), use_container_width=True)
        st.download_button("Save cleaned CSV", df_c.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")
    else:
        st.markdown("<span style='font-size:0.875rem;color:#9CA3AF'>Hit the button above when you are ready to clean.</span>", unsafe_allow_html=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
with tab3:
    if not num_cols:
        st.warning("No numeric columns found in this dataset.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<span class='chart-head'>How is this column distributed?</span>", unsafe_allow_html=True)
            s1 = st.selectbox("Select column", num_cols, key="dc", label_visibility="collapsed")
            fig, ax = plt.subplots(figsize=(6,3.5))
            style_ax(ax, fig)
            ax.hist(df[s1].dropna(), bins=28, color=BAR, edgecolor=BG, alpha=0.88, linewidth=0.3)
            ax.set_xlabel(s1, fontsize=9); ax.set_ylabel("Count", fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("<span class='chart-head'>Any unusual values?</span>", unsafe_allow_html=True)
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
            st.markdown("<span class='sec-head'>How do columns relate to each other?</span>", unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(10,4.5))
            style_ax(ax3, fig3)
            sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax3, linewidths=0.4, linecolor="#F3F4F6",
                        annot_kws={"size":8}, vmin=-1, vmax=1)
            ax3.tick_params(labelsize=8, colors=TICK)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        if cat_cols:
            st.markdown("<span class='sec-head'>What are the most common values?</span>", unsafe_allow_html=True)
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

    st.markdown("<span class='sec-head'>Set up your model</span>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: target = st.selectbox("What do you want to predict?",  wdf.columns)
    with c2: task   = st.selectbox("What kind of prediction?",      ["regression","classification"])
    with c3: algo   = st.selectbox("Which algorithm to use?",       ["Random Forest","Decision Tree","Linear / Logistic"])

    feats = [c for c in wdf.columns if c != target]
    sel   = st.multiselect("Which columns should the model learn from?", feats, default=feats[:min(6,len(feats))])

    if st.button("Train model"):
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

        st.markdown("<span class='sec-head'>How well did it do?</span>", unsafe_allow_html=True)
        mc = st.columns(len(metrics))
        for i,(k,v) in enumerate(metrics.items()):
            mc[i].metric(k, v)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<span class='sec-head'>Predicted vs what actually happened</span>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Actual": yte.values[:20], "Predicted": preds[:20]}),
                         use_container_width=True, hide_index=True)
        with col2:
            if task == "regression":
                st.markdown("<span class='chart-head'>How close were the predictions?</span>", unsafe_allow_html=True)
                fig5, ax5 = plt.subplots(figsize=(6,3.5))
                style_ax(ax5, fig5)
                ax5.scatter(yte.values[:60], preds[:60], color=BAR, alpha=0.55, s=28, linewidths=0)
                lm = [min(float(yte.min()),float(preds.min())), max(float(yte.max()),float(preds.max()))]
                ax5.plot(lm, lm, color=LINE_CLR, linewidth=1.3, linestyle="--")
                ax5.set_xlabel("Actual", fontsize=9); ax5.set_ylabel("Predicted", fontsize=9)
                plt.tight_layout(); st.pyplot(fig5); plt.close()

        model = st.session_state["model"]
        if hasattr(model, "feature_importances_"):
            st.markdown("<span class='sec-head'>Which columns mattered most?</span>", unsafe_allow_html=True)
            fi = pd.Series(model.feature_importances_, index=st.session_state["sel"]).sort_values(ascending=True)
            fig6, ax6 = plt.subplots(figsize=(8, max(2.5, len(fi)*0.4)))
            style_ax(ax6, fig6)
            ax6.barh(fi.index, fi.values,
                     color=[BAR if v==fi.max() else "#CBD5E1" for v in fi.values], height=0.55)
            ax6.set_xlabel("Importance", fontsize=9)
            plt.tight_layout(); st.pyplot(fig6); plt.close()
