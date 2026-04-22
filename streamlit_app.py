"""
Intelligent App Testing System — Streamlit Dashboard
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent App Testing System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 5px;
    }
    .sub-header {
        text-align: center; color: #888; font-size: 1rem; margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea20, #764ba220);
        border-radius: 12px; padding: 18px; border: 1px solid #667eea40;
        text-align: center;
    }
    .risk-high   { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low    { color: #27ae60; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_preprocess(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/bug_dataset.csv")

    # Fill missing values
    df["Severity"].fillna("Medium", inplace=True)
    df["Time_to_Fix_Days"].fillna(df["Time_to_Fix_Days"].median(), inplace=True)

    # Standardize text
    df["Severity"] = df["Severity"].str.strip().str.capitalize()
    df["Status"]   = df["Status"].str.strip().str.capitalize()
    df["Module"]   = df["Module"].str.strip()

    # Convert dates
    df["Release_Date"]  = pd.to_datetime(df["Release_Date"])
    df["Release_Year"]  = df["Release_Date"].dt.year
    df["Release_Month"] = df["Release_Date"].dt.month

    # Encode mappings
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    status_map   = {"Open": 1, "Fixed": 0, "Reopened": 2}
    df["Severity_Score"] = df["Severity"].map(severity_map)
    df["Status_Score"]   = df["Status"].map(status_map)

    return df


@st.cache_data
def compute_module_risk(df):
    risk = df.groupby("Module").agg(
        Total_Bugs      = ("Bug_ID",           "count"),
        High_Severity   = ("Severity",         lambda x: (x == "High").sum()),
        Reopened_Bugs   = ("Status",           lambda x: (x == "Reopened").sum()),
        Avg_Occurrences = ("Occurrences",      "mean"),
        Avg_Time_to_Fix = ("Time_to_Fix_Days", "mean"),
    ).reset_index()

    risk["Risk_Score"] = (
        risk["High_Severity"]   * 3 +
        risk["Reopened_Bugs"]   * 2 +
        risk["Avg_Occurrences"] * 1
    ).round(2)

    min_r, max_r = risk["Risk_Score"].min(), risk["Risk_Score"].max()
    risk["Risk_Score_Normalized"] = (
        (risk["Risk_Score"] - min_r) / (max_r - min_r) * 100
    ).round(2)

    risk["Risk_Level"] = risk["Risk_Score_Normalized"].apply(
        lambda x: "🔴 High" if x >= 70 else ("🟠 Medium" if x >= 40 else "🟢 Low")
    )

    return risk.sort_values("Risk_Score_Normalized", ascending=False)


@st.cache_data
def train_model(df):
    le_module   = LabelEncoder()
    le_severity = LabelEncoder()
    le_version  = LabelEncoder()

    df = df.copy()
    df["Module_Enc"]   = le_module.fit_transform(df["Module"])
    df["Severity_Enc"] = le_severity.fit_transform(df["Severity"])
    df["Version_Enc"]  = le_version.fit_transform(df["App_Version"])
    df["Target"]       = (df["Status"] == "Reopened").astype(int)

    FEATURES = ["Module_Enc", "Severity_Enc", "Version_Enc",
                "Occurrences", "Time_to_Fix_Days"]
    X = df[FEATURES]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)
    feat_imp = pd.Series(model.feature_importances_, index=FEATURES)

    return model, report, cm, feat_imp, le_module, le_severity, le_version


def validate_fix(row):
    score = 0
    if row["Severity"]        == "High":   score += 3
    elif row["Severity"]      == "Medium":  score += 2
    else:                                   score += 1
    if row["Occurrences"]      > 20:        score += 3
    elif row["Occurrences"]    > 10:        score += 2
    else:                                   score += 1
    if row["Time_to_Fix_Days"] > 15:        score += 2
    elif row["Time_to_Fix_Days"] > 7:       score += 1
    return "⚠️ At Risk" if score >= 6 else "✅ Likely Resolved"


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bug.png", width=64)
    st.title("🧠 Testing System")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload your CSV (optional)", type=["csv"])
    st.markdown("*Leave empty to use the default synthetic dataset.*")
    st.markdown("---")

    st.markdown("### 🔎 Filters")

df = load_and_preprocess(uploaded)

with st.sidebar:
    selected_versions = st.multiselect(
        "App Versions", options=sorted(df["App_Version"].unique()),
        default=sorted(df["App_Version"].unique())
    )
    selected_modules = st.multiselect(
        "Modules", options=sorted(df["Module"].unique()),
        default=sorted(df["Module"].unique())
    )
    selected_severity = st.multiselect(
        "Severity", options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )

# Apply filters
df_filtered = df[
    df["App_Version"].isin(selected_versions) &
    df["Module"].isin(selected_modules) &
    df["Severity"].isin(selected_severity)
]

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">🧠 Intelligent App Testing System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data Science in SDLC — Bug Risk Analysis, Prediction & Prioritization</p>', unsafe_allow_html=True)

# ─── KPI Metrics ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Total Bugs",       len(df_filtered))
c2.metric("🔴 High Severity",    (df_filtered["Severity"] == "High").sum())
c3.metric("🔁 Reopened",         (df_filtered["Status"]   == "Reopened").sum())
c4.metric("✅ Fixed",             (df_filtered["Status"]   == "Fixed").sum())
c5.metric("⏱️ Avg Fix (days)",   f"{df_filtered['Time_to_Fix_Days'].mean():.1f}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA",
    "⚠️ Risk Scoring",
    "🤖 ML Model",
    "📝 NLP Analysis",
    "✅ Fix Validation",
    "💡 Insights"
])


# ──────────────────────────────────────────────────────────────────
# TAB 1: EDA
# ──────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_filtered["Module"].value_counts().reset_index(),
            x="Module", y="count", color="count",
            color_continuous_scale="Reds",
            title="🔴 Bug Count per Module",
            labels={"count": "Bug Count"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sev_counts = df_filtered["Severity"].value_counts()
        fig2 = px.pie(
            names=sev_counts.index, values=sev_counts.values,
            color=sev_counts.index,
            color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"},
            title="🟠 Severity Distribution", hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        version_order = sorted(df_filtered["App_Version"].unique())
        vc = df_filtered["App_Version"].value_counts().reindex(version_order).reset_index()
        vc.columns = ["App_Version", "count"]
        fig3 = px.line(vc, x="App_Version", y="count", markers=True,
                       title="📈 Bugs per App Version",
                       color_discrete_sequence=["#764ba2"])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        status_counts = df_filtered["Status"].value_counts()
        fig4 = px.bar(
            x=status_counts.index, y=status_counts.values,
            color=status_counts.index,
            color_discrete_map={"Fixed": "#27ae60", "Open": "#3498db", "Reopened": "#e74c3c"},
            title="📋 Status Distribution",
            labels={"x": "Status", "y": "Count"}
        )
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🔥 Module × Severity Heatmap")
    heatmap_data = pd.crosstab(df_filtered["Module"], df_filtered["Severity"])
    for col in ["Low", "Medium", "High"]:
        if col not in heatmap_data.columns:
            heatmap_data[col] = 0
    heatmap_data = heatmap_data[["Low", "Medium", "High"]]

    fig5 = px.imshow(
        heatmap_data, text_auto=True, color_continuous_scale="YlOrRd",
        title="Module vs Severity Heatmap"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df_filtered.head(50), use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 2: RISK SCORING
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("⚠️ Module Risk Scoring")

    module_risk = compute_module_risk(df_filtered)

    colors = module_risk["Risk_Score_Normalized"].apply(
        lambda x: "#e74c3c" if x >= 70 else ("#f39c12" if x >= 40 else "#27ae60")
    )

    fig6 = go.Figure(go.Bar(
        x=module_risk["Module"],
        y=module_risk["Risk_Score_Normalized"],
        marker_color=colors.tolist(),
        text=module_risk["Risk_Score_Normalized"].round(1),
        textposition="outside"
    ))
    fig6.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="High Risk ≥70")
    fig6.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk ≥40")
    fig6.update_layout(title="⚠️ Module Risk Score (0–100)", xaxis_tickangle=-30)
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("📋 Module Risk Table")
    display_cols = ["Module", "Total_Bugs", "High_Severity", "Reopened_Bugs",
                    "Avg_Occurrences", "Avg_Time_to_Fix", "Risk_Score_Normalized", "Risk_Level"]
    st.dataframe(
        module_risk[display_cols].rename(columns={
            "Risk_Score_Normalized": "Risk Score (0–100)"
        }),
        use_container_width=True
    )


# ──────────────────────────────────────────────────────────────────
# TAB 3: ML MODEL
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 Bug Reopen Prediction — Random Forest")

    if len(df_filtered) < 50:
        st.warning("⚠️ Not enough data to train. Please adjust filters.")
    else:
        with st.spinner("Training model..."):
            model, report, cm, feat_imp, le_mod, le_sev, le_ver = train_model(df_filtered)

        col1, col2 = st.columns(2)
        with col1:
            acc  = round(report["accuracy"] * 100, 1)
            prec = round(report["weighted avg"]["precision"] * 100, 1)
            rec  = round(report["weighted avg"]["recall"] * 100, 1)
            f1   = round(report["weighted avg"]["f1-score"] * 100, 1)
            st.metric("✅ Accuracy",  f"{acc}%")
            st.metric("🎯 Precision", f"{prec}%")
            st.metric("🔍 Recall",    f"{rec}%")
            st.metric("📊 F1-Score",  f"{f1}%")

        with col2:
            fig7, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Not Reopened", "Reopened"],
                        yticklabels=["Not Reopened", "Reopened"])
            ax.set_title("Confusion Matrix")
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            st.pyplot(fig7)

        st.subheader("🔍 Feature Importance")
        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fig8 = px.bar(fi_df.sort_values("Importance", ascending=True),
                      x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Reds")
        st.plotly_chart(fig8, use_container_width=True)

        st.subheader("🔮 Predict a Single Bug")
        with st.form("predict_form"):
            p1 = st.selectbox("Module",      sorted(df["Module"].unique()))
            p2 = st.selectbox("Severity",    ["Low", "Medium", "High"])
            p3 = st.selectbox("App Version", sorted(df["App_Version"].unique()))
            p4 = st.slider("Occurrences",    1, 100, 10)
            p5 = st.slider("Time to Fix (days)", 0.5, 30.0, 5.0)
            submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            try:
                m_enc = le_mod.transform([p1])[0]
                s_enc = le_sev.transform([p2])[0]
                v_enc = le_ver.transform([p3])[0]
                pred  = model.predict([[m_enc, s_enc, v_enc, p4, p5]])[0]
                prob  = model.predict_proba([[m_enc, s_enc, v_enc, p4, p5]])[0][1]
                if pred == 1:
                    st.error(f"⚠️ This bug is likely to REOPEN. Confidence: {prob*100:.1f}%")
                else:
                    st.success(f"✅ This bug is likely RESOLVED. Reopen risk: {prob*100:.1f}%")
            except Exception as e:
                st.warning(f"Could not predict: {e}")


# ──────────────────────────────────────────────────────────────────
# TAB 4: NLP
# ──────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📝 NLP — Recurring Bug Detection")
    st.info("Using TF-IDF + Cosine Similarity to detect duplicate/recurring bug descriptions.")

    threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8, 0.05)

    with st.spinner("Analyzing bug descriptions..."):
        tfidf  = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(df_filtered["Bug_Description"])
        sim    = cosine_similarity(matrix)

        pairs = []
        for i in range(len(df_filtered)):
            for j in range(i + 1, len(df_filtered)):
                if sim[i][j] >= threshold:
                    pairs.append({
                        "Bug 1": df_filtered.iloc[i]["Bug_ID"],
                        "Bug 2": df_filtered.iloc[j]["Bug_ID"],
                        "Description 1": df_filtered.iloc[i]["Bug_Description"],
                        "Description 2": df_filtered.iloc[j]["Bug_Description"],
                        "Similarity":    round(sim[i][j], 3)
                    })

    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values("Similarity", ascending=False)
        st.success(f"✅ Found {len(pairs_df)} recurring bug pairs (similarity ≥ {threshold})")
        st.dataframe(pairs_df, use_container_width=True)
    else:
        st.warning("No recurring bug pairs found at this threshold. Try lowering it.")

    st.subheader("🔤 Top Terms in Bug Descriptions")
    tfidf2  = TfidfVectorizer(stop_words="english", max_features=20)
    tfidf2.fit(df_filtered["Bug_Description"])
    terms   = pd.Series(
        tfidf2.idf_, index=tfidf2.get_feature_names_out()
    ).sort_values().head(20)
    fig9 = px.bar(x=terms.values, y=terms.index, orientation="h",
                  title="Most Frequent Terms in Bug Descriptions",
                  color=terms.values, color_continuous_scale="Blues")
    st.plotly_chart(fig9, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 5: FIX VALIDATION
# ──────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("✅ Bug Fix Validation")
    st.info("Assesses whether 'Fixed' bugs are truly resolved or at risk of reappearing.")

    fixed = df_filtered[df_filtered["Status"] == "Fixed"].copy()
    fixed["Validation"] = fixed.apply(validate_fix, axis=1)

    c1, c2 = st.columns(2)
    resolved = (fixed["Validation"] == "✅ Likely Resolved").sum()
    at_risk  = (fixed["Validation"] == "⚠️ At Risk").sum()
    c1.metric("✅ Likely Resolved", resolved)
    c2.metric("⚠️ At Risk of Reappearing", at_risk)

    fig10 = px.pie(
        names=fixed["Validation"].value_counts().index,
        values=fixed["Validation"].value_counts().values,
        color_discrete_map={"✅ Likely Resolved": "#27ae60", "⚠️ At Risk": "#e74c3c"},
        title="Fix Validation Summary", hole=0.35
    )
    st.plotly_chart(fig10, use_container_width=True)

    st.subheader("📋 At-Risk Fixed Bugs")
    at_risk_df = fixed[fixed["Validation"] == "⚠️ At Risk"][
        ["Bug_ID", "Module", "Severity", "Occurrences", "Time_to_Fix_Days", "Validation"]
    ]
    st.dataframe(at_risk_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 6: INSIGHTS
# ──────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("💡 Actionable Insights & Recommendations")

    module_risk2    = compute_module_risk(df_filtered)
    top_module      = module_risk2.iloc[0]["Module"]
    top_risk        = module_risk2.iloc[0]["Risk_Score_Normalized"]
    reopen_rate     = round((df_filtered["Status"] == "Reopened").mean() * 100, 1)
    high_sev_pct    = round((df_filtered["Severity"] == "High").mean() * 100, 1)
    avg_ttf         = round(df_filtered["Time_to_Fix_Days"].mean(), 1)
    versions        = sorted(df_filtered["App_Version"].unique())
    latest, prev    = versions[-1], versions[-2] if len(versions) > 1 else (versions[-1], versions[-1])
    latest_bugs     = df_filtered[df_filtered["App_Version"] == latest].shape[0]
    prev_bugs       = df_filtered[df_filtered["App_Version"] == prev].shape[0]
    trend_icon      = "✅ Improving" if latest_bugs < prev_bugs else "⚠️ Worsening"

    insights = [
        ("🔴", "Highest Risk Module",
         f"**{top_module}** has a risk score of **{top_risk:.1f}/100**. Focus testing efforts here first."),
        ("🔁", "High Reopen Rate",
         f"**{reopen_rate}%** of bugs are reopened. Improve root cause analysis before marking bugs as fixed."),
        ("⚠️", "High Severity Concentration",
         f"**{high_sev_pct}%** of bugs are High severity. Assign senior testers to critical modules."),
        ("⏱️", "Fix Time SLA",
         f"Average fix time is **{avg_ttf} days**. Set SLA: High severity bugs must be fixed within 5 days."),
        ("📈", "Version Trend",
         f"Latest version **{latest}** is **{trend_icon}**: {latest_bugs} bugs vs {prev_bugs} in {prev}."),
        ("🔬", "Duplicate Bug Reports",
         "NLP analysis reveals duplicate bug descriptions. Merge them to reduce redundant testing."),
        ("🧪", "Regression Test Gaps",
         "Payment and Auth modules show high reopen rates — add dedicated regression test suites."),
        ("🎯", "Priority-Based Testing",
         "Use the Priority Score system to plan sprints. Test Top 20 bugs every release cycle."),
        ("🧹", "Premature Bug Closure",
         "Validation logic flags bugs with high risk scores. Extend monitoring before closure."),
        ("🤖", "Model Retraining",
         "Retrain the ML model with each new release to continuously improve prediction accuracy."),
    ]

    for i, (icon, title, desc) in enumerate(insights, 1):
        with st.expander(f"{icon} Insight {i}: {title}", expanded=(i <= 3)):
            st.markdown(desc)

    # Testing priority table
    st.subheader("🚨 Top 20 Testing Priorities for Next Release")
    df_pri = df_filtered.merge(module_risk2[["Module", "Risk_Score_Normalized"]], on="Module")
    df_pri["Priority_Score"] = (
        df_pri["Risk_Score_Normalized"] * 0.5 +
        df_pri["Severity_Score"]        * 15  +
        df_pri["Occurrences"]           * 0.5
    ).round(2)

    top20 = (
        df_pri[df_pri["Status"] != "Fixed"]
        .sort_values("Priority_Score", ascending=False)
        [["Bug_ID", "Module", "Severity", "Status", "Occurrences",
          "Risk_Score_Normalized", "Priority_Score"]]
        .head(20)
        .rename(columns={"Risk_Score_Normalized": "Module Risk"})
    )
    st.dataframe(top20, use_container_width=True)

    csv = top20.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Priority List as CSV",
        data=csv,
        file_name="testing_priority.csv",
        mime="text/csv"
    )

# ─── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using <b>Python + Streamlit</b> | "
    "Intelligent App Testing System — Data Science in SDLC</center>",
    unsafe_allow_html=True
)
