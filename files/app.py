# ============================================================
# app.py — Bank Customer Churn Predictor (Complete & Fixed)
# European Central Bank · Retail Analytics
# ============================================================
import subprocess
subprocess.run(["pip", "install", "xgboost", "--break-system-packages"], capture_output=True)

import matplotlib
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import joblib
import pickle
import pandas as pd
import seaborn as sns

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

df = pd.read_csv("European_Bank.csv")
# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2a4a 50%, #1a3a6b 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '🏦';
        position: absolute;
        font-size: 8rem;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        opacity: 0.08;
    }
    .main-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 0.3rem 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 0.95rem; opacity: 0.75; margin: 0; }
    .main-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        margin-top: 0.6rem;
        letter-spacing: 0.5px;
    }

    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1rem 0; }
    .metric-card {
        background: white;
        border: 1px solid #e8edf2;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #1a3a6b; }
    .metric-card .lbl { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.2rem; }
    .metric-card .tag { font-size: 0.65rem; color: #9ca3af; }

    .risk-high  { background:#fff1f2; border:1.5px solid #fecdd3; border-left:5px solid #dc2626; border-radius:12px; padding:1.6rem; }
    .risk-med   { background:#fffbeb; border:1.5px solid #fde68a; border-left:5px solid #d97706; border-radius:12px; padding:1.6rem; }
    .risk-low   { background:#f0fdf4; border:1.5px solid #bbf7d0; border-left:5px solid #16a34a; border-radius:12px; padding:1.6rem; }
    .risk-high h2 { color:#dc2626; margin: 0 0 0.3rem 0; }
    .risk-med  h2 { color:#d97706; margin: 0 0 0.3rem 0; }
    .risk-low  h2 { color:#16a34a; margin: 0 0 0.3rem 0; }
    .risk-high p, .risk-med p, .risk-low p { margin: 0.2rem 0; font-size: 0.9rem; color: #374151; }
    .risk-score { font-size: 3.2rem; font-weight: 800; margin: 0.4rem 0; }

    .insight-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        margin: 0.35rem 0;
        font-size: 0.87rem;
        color: #1e40af;
    }

    .model-row {
        display: flex; align-items: center; gap: 0.6rem;
        background: white; border: 1px solid #e5e7eb;
        border-radius: 8px; padding: 0.75rem 1rem; margin: 0.4rem 0;
    }
    .model-row.best { border-color: #fbbf24; background: #fffbeb; }
    .model-name { font-weight: 600; font-size: 0.9rem; flex: 1; color: #111827; }
    .model-badge { background: #fef3c7; color: #92400e; font-size: 0.7rem; font-weight: 600;
                   padding: 0.15rem 0.5rem; border-radius: 10px; }
    .bar-wrap { flex: 2; background: #f3f4f6; border-radius: 4px; height: 8px; }
    .bar-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #1a3a6b, #2563eb); }
    .bar-val { font-size: 0.82rem; font-weight: 600; color: #374151; min-width: 48px; text-align: right; }

    .fi-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.35rem 0; }
    .fi-name { font-size: 0.82rem; color: #374151; flex: 1; }
    .fi-bar-wrap { flex: 2; background: #f3f4f6; border-radius: 4px; height: 7px; }
    .fi-bar { height: 7px; border-radius: 4px; background: linear-gradient(90deg, #7c3aed, #a78bfa); }
    .fi-val { font-size: 0.75rem; color: #6b7280; min-width: 38px; text-align: right; }

    .scenario-card {
        background: white; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 1rem; margin: 0.4rem 0;
        display: flex; align-items: center; gap: 0.8rem;
    }
    .scenario-icon { font-size: 1.5rem; }
    .scenario-text { flex: 1; }
    .scenario-title { font-weight: 600; font-size: 0.88rem; color: #111827; margin: 0; }
    .scenario-sub   { font-size: 0.78rem; color: #6b7280; margin: 0; }
    .scenario-prob  { font-size: 1.1rem; font-weight: 700; min-width: 48px; text-align: right; }

    .stButton>button {
        background: linear-gradient(135deg, #1a3a6b, #2563eb);
        color: white; border: none; border-radius: 8px;
        padding: 0.65rem 1.5rem; font-weight: 600; width: 100%;
        font-size: 0.95rem; letter-spacing: 0.3px;
        transition: all 0.2s;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }

    .section-title { font-size: 1rem; font-weight: 700; color: #111827;
                     margin: 1.2rem 0 0.6rem; border-bottom: 2px solid #e5e7eb;
                     padding-bottom: 0.35rem; }
    .tip { background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 6px;
           padding: 0.5rem 0.8rem; font-size: 0.8rem; color: #64748b; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model      = joblib.load('best_churn_model.pkl')
    scaler     = joblib.load('scaler.pkl')
    meta       = joblib.load('model_meta.pkl')
    return model, scaler, meta

try:
    best_model, scaler, meta = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Model files not found. Please run the training script first.\n\nError: {e}")
    st.stop()

RESULTS       = meta['results']
ALL_MODELS    = meta['all_models']
BEST_NAME     = meta['best_name']
BEST_THRESH   = meta['best_threshold']
FEATURE_COLS  = meta['feature_cols']
FEAT_IMP      = meta.get('feature_importance', {})


# ─────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────
def build_features(credit_score, geography, gender, age, tenure,
                   balance, salary, num_products, has_cr_card, is_active):
    bal_sal_ratio = balance / (salary + 1)
    prod_density  = num_products / (tenure + 1)
    engagement    = int(is_active) * num_products
    age_tenure    = age * tenure

    data = {
        'Year'               : 2025,
        'CreditScore'        : credit_score,
        'Age'                : age,
        'Tenure'             : tenure,
        'Balance'            : balance,
        'NumOfProducts'      : num_products,
        'HasCrCard'          : int(has_cr_card),
        'IsActiveMember'     : int(is_active),
        'EstimatedSalary'    : salary,
        'BalanceSalaryRatio' : bal_sal_ratio,
        'ProductDensity'     : prod_density,
        'EngagementProduct'  : engagement,
        'AgeTenure'          : age_tenure,
        'Geography_Germany'  : int(geography == "Germany"),
        'Geography_Spain'    : int(geography == "Spain"),
        'Gender_Male'        : int(gender == "Male"),
    }
    return pd.DataFrame([data])[FEATURE_COLS]

def predict_prob(X_df, model_name=None):
    model = ALL_MODELS.get(model_name, best_model) if model_name else best_model
    X_scaled = scaler.transform(X_df)
    return float(model.predict_proba(X_scaled)[0][1])


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Customer Churn Intelligence System</h1>
    <p>ML-powered churn risk scoring · Predictive retention analytics</p>
    <span class="badge">EUROPEAN CENTRAL BANK · RETAIL ANALYTICS DIVISION</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — CUSTOMER PROFILE
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # import streamlit as st
    # st.sidebar.image("bank.avif", width=250)
    st.markdown("## 🧾 Customer Profile")
    st.markdown("---")
    st.markdown("**🆔 Demographics**")
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender    = st.selectbox("Gender", ["Male", "Female"])
    age       = st.slider("Age", 18, 80, 38)

    st.markdown("**💳 Financial**")
    credit_score = st.slider("Credit Score", 300, 900, 650, 10)
    balance      = st.number_input("Account Balance (€)", 0.0, 300_000.0, 60_000.0, 1_000.0)
    salary       = st.number_input("Estimated Salary (€)", 10_000.0, 200_000.0, 80_000.0, 1_000.0)

    st.markdown("**🔗 Engagement**")
    tenure       = st.slider("Tenure (years)", 0, 10, 5)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card  = st.checkbox("Has Credit Card", value=True)
    is_active    = st.checkbox("Is Active Member", value=True)

    st.markdown("---")
    predict_btn  = st.button("🔮 Analyse Churn Risk", use_container_width=True)
    st.markdown(f'<div class="tip">Using <strong>{BEST_NAME}</strong> · Threshold: {BEST_THRESH:.2f}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk Calculator",
    "📊 Model Performance",
    "🔍 Feature Importance",
    "🎮 Scenario Simulator",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — RISK CALCULATOR
# ════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        X_inp = build_features(credit_score, geography, gender, age, tenure,
                               balance, salary, num_products, has_cr_card, is_active)
        prob     = predict_prob(X_inp)
        prob_pct = prob * 100
        risk_score = round(prob_pct)

        col_risk, col_info = st.columns([1.4, 1])

        with col_risk:
            if prob_pct >= 60:
                st.markdown(f"""<div class="risk-high">
                    <h2>🔴 HIGH CHURN RISK</h2>
                    <div class="risk-score">{risk_score}%</div>
                    <p>Churn probability: <strong>{prob:.4f}</strong></p>
                    <p>⚠️ Immediate retention action required</p>
                    <p style="font-size:0.8rem;margin-top:0.5rem;opacity:0.7">Threshold: {BEST_THRESH:.2f}</p>
                </div>""", unsafe_allow_html=True)
            elif prob_pct >= 35:
                st.markdown(f"""<div class="risk-med">
                    <h2>🟡 MEDIUM CHURN RISK</h2>
                    <div class="risk-score">{risk_score}%</div>
                    <p>Churn probability: <strong>{prob:.4f}</strong></p>
                    <p>📋 Proactive engagement recommended</p>
                    <p style="font-size:0.8rem;margin-top:0.5rem;opacity:0.7">Threshold: {BEST_THRESH:.2f}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="risk-low">
                    <h2>🟢 LOW CHURN RISK</h2>
                    <div class="risk-score">{risk_score}%</div>
                    <p>Churn probability: <strong>{prob:.4f}</strong></p>
                    <p>✅ Customer likely to be retained</p>
                    <p style="font-size:0.8rem;margin-top:0.5rem;opacity:0.7">Threshold: {BEST_THRESH:.2f}</p>
                </div>""", unsafe_allow_html=True)

        with col_info:
            st.markdown('<div class="section-title">📋 Customer Summary</div>', unsafe_allow_html=True)
            data_rows = {
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Credit Score": credit_score,
                "Balance": f"€{balance:,.0f}",
                "Salary": f"€{salary:,.0f}",
                "Tenure": f"{tenure} yrs",
                "Products": num_products,
                "Credit Card": "Yes" if has_cr_card else "No",
                "Active": "Yes" if is_active else "No",
            }
            for k, v in data_rows.items():
                st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.markdown('<div class="section-title">💡 Retention Insights</div>', unsafe_allow_html=True)

        insights = []
        if age > 45:
            insights.append("🎂 Older customers (45+) have higher churn tendency — consider premium loyalty rewards.")
        if num_products == 1:
            insights.append("📦 Single-product customers churn 3× more — cross-sell a savings or investment product.")
        if balance == 0:
            insights.append("💰 Zero-balance account signals disengagement — activate balance-building incentives.")
        if not is_active:
            insights.append("😴 Inactive members are at significantly higher risk — trigger a re-engagement campaign.")
        if geography == "Germany":
            insights.append("🇩🇪 German customers show higher historical churn — apply region-specific retention offers.")
        if credit_score < 500:
            insights.append("📉 Low credit score may indicate financial stress — offer financial wellness support.")
        if not insights:
            insights.append("✅ No critical churn indicators — maintain standard engagement and monitor quarterly.")

        for tip in insights:
            st.markdown(f'<div class="insight-box">{tip}</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Fill in the customer profile in the sidebar and click **Analyse Churn Risk** to get a prediction.")
        st.markdown("""
        **How it works:**
        1. Enter customer details in the sidebar
        2. Click the predict button
        3. Get an instant churn risk score with actionable insights

        The model uses **Gradient Boosting** (best model by ROC-AUC = **86.96%**) trained on 10,000 European bank customers.
        """)


# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🏆 Model Comparison</div>', unsafe_allow_html=True)

    best_res = RESULTS[BEST_NAME]
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="val">{best_res['accuracy']}%</div>
            <div class="lbl">Accuracy</div>
            <div class="tag">{BEST_NAME}</div>
        </div>
        <div class="metric-card">
            <div class="val">{best_res['roc_auc']}%</div>
            <div class="lbl">ROC-AUC</div>
            <div class="tag">Best score</div>
        </div>
        <div class="metric-card">
            <div class="val">{best_res['f1']}%</div>
            <div class="lbl">F1-Score</div>
            <div class="tag">Balanced</div>
        </div>
        <div class="metric-card">
            <div class="val">{best_res['recall']}%</div>
            <div class="lbl">Recall</div>
            <div class="tag">Churner capture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 All Models — ROC-AUC</div>', unsafe_allow_html=True)
    for name, res in sorted(RESULTS.items(), key=lambda x: x[1]['roc_auc'], reverse=True):
        is_best = name == BEST_NAME
        badge   = '<span class="model-badge">⭐ BEST</span>' if is_best else ''
        pct     = res['roc_auc']
        row_cls = "model-row best" if is_best else "model-row"
        st.markdown(f"""
        <div class="{row_cls}">
            <span class="model-name">{name} {badge}</span>
            <div class="bar-wrap"><div class="bar-fill" style="width:{pct}%"></div></div>
            <span class="bar-val">{pct}%</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Detailed Metrics Table</div>', unsafe_allow_html=True)
    df_res = pd.DataFrame({
        k: {m: f"{v}%" for m, v in r.items()}
        for k, r in RESULTS.items()
    }).T
    df_res.index.name = "Model"
    st.dataframe(df_res, use_container_width=True)

    st.markdown("""
    <div class="tip">
    <strong>Why Gradient Boosting?</strong> It achieves the highest ROC-AUC (86.96%) meaning it best discriminates
    between churners and retained customers. Combined with threshold tuning, recall is maximised to catch actual churners.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔍 Top Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
    
    if FEAT_IMP:
        max_fi = max(FEAT_IMP.values())
        feature_labels = {
            'Age': '🎂 Age',
            'NumOfProducts': '📦 Num of Products',
            'EngagementProduct': '🤝 Engagement × Product',
            'Balance': '💰 Balance',
            'Geography_Germany': '🇩🇪 Germany',
            'CreditScore': '📊 Credit Score',
            'EstimatedSalary': '💼 Estimated Salary',
            'IsActiveMember': '✅ Is Active Member',
            'AgeTenure': '⏳ Age × Tenure',
            'BalanceSalaryRatio': '⚖️ Balance/Salary Ratio',
            'Tenure': '📅 Tenure',
            'HasCrCard': '💳 Has Credit Card',
            'ProductDensity': '📐 Product Density',
            'Geography_Spain': '🇪🇸 Spain',
            'Gender_Male': '👤 Male',
            'Year': '📆 Year',
        }
        for feat, imp in list(FEAT_IMP.items())[:10]:
            label = feature_labels.get(feat, feat)
            bar_w = (imp / max_fi) * 100
            st.markdown(f"""
            <div class="fi-row">
                <span class="fi-name">{label}</span>
                <div class="fi-bar-wrap"><div class="fi-bar" style="width:{bar_w}%"></div></div>
                <span class="fi-val">{imp*100:.1f}%</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">💡 Key Churn Drivers — Business Interpretation</div>', unsafe_allow_html=True)
    interpretations = [
        ("🎂", "Age is the #1 driver", "Customers aged 45–60 have 2× higher churn. Target them with premium loyalty programmes."),
        ("📦", "Product count matters", "Single-product users churn 3× more than multi-product users. Cross-sell is the most effective retention lever."),
        ("🤝", "Engagement × Products", "Inactive customers with few products are highest-risk. Activation campaigns save the most."),
        ("💰", "Zero balance = danger", "Customers with near-zero balances are disengaged. Balance-building incentives (bonus interest) reduce churn."),
        ("🇩🇪", "Germany effect", "German customers churn at a higher rate than France/Spain — apply region-specific retention budgets."),
    ]
    for icon, title, desc in interpretations:
        st.markdown(f"""
        <div class="insight-box">
            <strong>{icon} {title}</strong><br>{desc}
        </div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Churn Rate by Age Group</div>', unsafe_allow_html=True
         )

        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0,30,40,50,60,100],
        labels=['< 30','30–40','40–50','50–60','60+'])

        age_churn = (
        df.groupby('AgeGroup')['Exited'].mean().mul(100).round(1))
        
        age_groups = age_churn.index.tolist()
        churn_rates = age_churn.values.tolist()

        def bar_color(v):
            if v >= 50:
               return '#E24B4A'
            elif v >= 25:
               return '#BA7517'
            return '#1D9E75'

        colors = [bar_color(v) for v in churn_rates]

        fig, ax = plt.subplots(figsize=(8,5))

        bars = ax.bar(
            age_groups,
            churn_rates,
            color=colors,
            edgecolor='white',
            linewidth=1.5,
            width=0.55
        )

        for bar, val in zip(bars, churn_rates):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{val:.1f}%',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_title("Churn Rate by Age Group", fontweight='bold')
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Churn Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 70)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
    with col2:
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Churn Rate by Number of Products</div>', unsafe_allow_html=True)

        prod_churn = (
        df.groupby('NumOfProducts')['Exited'].mean().mul(100).round(1))
        
        products = prod_churn.index.astype(str).tolist()
        churn_rates = prod_churn.values.tolist()

        colors = [bar_color(v) for v in churn_rates]

        fig, ax = plt.subplots(figsize=(8,5))
        
        bars = ax.bar(
            products,
            churn_rates,
            color=colors,
            edgecolor='white',
            linewidth=1.5,
            width=0.55
        )

        for bar, val in zip(bars, churn_rates):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{val:.1f}%',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_title("Churn Rate by Number of Products", fontweight='bold')
        ax.set_xlabel("Number of Products")
        ax.set_ylabel("Churn Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
 
   

# ════════════════════════════════════════════════════════════
# TAB 4 — SCENARIO SIMULATOR
# ════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🎮 What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.write("See how changing engagement variables affects churn probability for the current customer profile.")

    base_X   = build_features(credit_score, geography, gender, age, tenure,
                               balance, salary, num_products, has_cr_card, is_active)
    base_prob = predict_prob(base_X)

    scenarios = [
        ("🔄", "Add 1 More Product",
         build_features(credit_score, geography, gender, age, tenure,
                        balance, salary, min(num_products+1,4), has_cr_card, is_active)),
        ("⚡", "Activate Membership",
         build_features(credit_score, geography, gender, age, tenure,
                        balance, salary, num_products, has_cr_card, True)),
        ("💰", "Add €50k to Balance",
         build_features(credit_score, geography, gender, age, tenure,
                        balance+50000, salary, num_products, has_cr_card, is_active)),
        ("📈", "Improve Credit Score +100",
         build_features(min(credit_score+100,900), geography, gender, age, tenure,
                        balance, salary, num_products, has_cr_card, is_active)),
        ("🎁", "Add Credit Card",
         build_features(credit_score, geography, gender, age, tenure,
                        balance, salary, num_products, True, is_active)),
        ("🚀", "Best Case (Active + 2 Products + Balance)",
         build_features(credit_score, geography, gender, age, tenure,
                        balance+30000, salary, min(num_products+1,4), True, True)),
    ]

    st.markdown(f'<div class="scenario-card"><span class="scenario-icon">📍</span>'
                f'<div class="scenario-text"><p class="scenario-title">Current Profile (Baseline)</p>'
                f'<p class="scenario-sub">Your entered customer</p></div>'
                f'<span class="scenario-prob" style="color:#374151">{base_prob*100:.1f}%</span></div>',
                unsafe_allow_html=True)

    for icon, label, X_scen in scenarios:
        scen_prob = predict_prob(X_scen)
        delta     = scen_prob - base_prob
        delta_str = f"{'▼' if delta < 0 else '▲'} {abs(delta*100):.1f}pp"
        col_prob  = "#16a34a" if delta < 0 else "#dc2626" if delta > 0.01 else "#374151"
        sub       = f"{delta_str} vs baseline"
        st.markdown(f"""
        <div class="scenario-card">
            <span class="scenario-icon">{icon}</span>
            <div class="scenario-text">
                <p class="scenario-title">{label}</p>
                <p class="scenario-sub">{sub}</p>
            </div>
            <span class="scenario-prob" style="color:{col_prob}">{scen_prob*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🎛️ Custom Scenario Builder</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        custom_products = st.selectbox("Products", [1,2,3,4], index=[1,2,3,4].index(num_products), key="cs_prod")
        custom_active   = st.checkbox("Active Member", value=is_active, key="cs_act")
    with c2:
        custom_balance  = st.number_input("Balance (€)", 0.0, 300_000.0, float(balance), 5_000.0, key="cs_bal")
        custom_credit   = st.slider("Credit Score", 300, 900, credit_score, 10, key="cs_cred")
    with c3:
        custom_tenure   = st.slider("Tenure (yrs)", 0, 10, tenure, key="cs_ten")
        custom_cc       = st.checkbox("Has Credit Card", value=has_cr_card, key="cs_cc")

    X_custom   = build_features(custom_credit, geography, gender, age, custom_tenure,
                                custom_balance, salary, custom_products, custom_cc, custom_active)
    prob_custom = predict_prob(X_custom)
    delta_custom = prob_custom - base_prob
    color_custom = "#16a34a" if prob_custom < 0.35 else "#d97706" if prob_custom < 0.6 else "#dc2626"

    st.markdown(f"""
    <div style="background:white;border:1.5px solid {color_custom};border-radius:10px;padding:1rem;margin-top:0.5rem;text-align:center;">
        <div style="font-size:0.8rem;color:#6b7280;margin-bottom:0.3rem;">Custom Scenario Churn Risk</div>
        <div style="font-size:2.5rem;font-weight:800;color:{color_custom}">{prob_custom*100:.1f}%</div>
        <div style="font-size:0.85rem;color:#6b7280;">
            {'▼' if delta_custom<0 else '▲'} {abs(delta_custom*100):.1f}pp vs baseline
        </div>
    </div>
    """, unsafe_allow_html=True)
