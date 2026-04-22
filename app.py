"""
🧟 Human-Zombie Score Predictor  — Streamlit App
=================================================
Run:  streamlit run app.py
Requires: streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn, plotly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧟 Zombie Screening — Earth Junior",
    page_icon="🧟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Creepster&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    background-color: #0a0d0f;
    color: #d4e8c2;
  }

  h1, h2, h3 { font-family: 'Creepster', cursive !important; letter-spacing: 2px; }
  h1 { font-size: 3rem !important; color: #6dff6d !important; text-shadow: 0 0 20px #3aff3a88; }
  h2 { color: #a8ff78 !important; }
  h3 { color: #78ffd6 !important; }

  p, li, label, div { font-family: 'Exo 2', sans-serif !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a0d 0%, #0a130a 100%);
    border-right: 1px solid #2d5a2d;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1a0d, #162616);
    border: 1px solid #3d7a3d;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 16px #00ff0022;
  }
  div[data-testid="metric-container"] > label {
    color: #78ffd6 !important; font-size: 0.8rem !important;
  }
  div[data-testid="metric-container"] > div {
    color: #6dff6d !important; font-size: 1.8rem !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(90deg, #1a3d1a, #0d260d);
    color: #6dff6d;
    border: 1px solid #3d7a3d;
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
    transition: all 0.3s;
  }
  .stButton > button:hover {
    background: linear-gradient(90deg, #2a5d2a, #1a3d1a);
    box-shadow: 0 0 12px #6dff6d55;
    color: #afffaf;
  }

  /* Sliders */
  .stSlider > div > div > div > div { background: #6dff6d !important; }

  /* Score gauge number */
  .zombie-score {
    font-family: 'Creepster', cursive;
    font-size: 5rem;
    text-align: center;
    padding: 0.5rem;
    border-radius: 16px;
    transition: all 0.4s;
  }

  .info-box {
    background: #0d1a0d;
    border-left: 4px solid #6dff6d;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.9rem;
    color: #b0d8a4;
  }

  /* Tab active */
  button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #6dff6d !important;
    color: #6dff6d !important;
  }
  button[data-baseweb="tab"] { font-family: 'Exo 2', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ─── Custom Linear Regression ────────────────────────────────────────────────
class LinearRegressionCustom:
    def __init__(self, learning_rate=0.1, num_iterations=500, cost_function='mse'):
        self.learning_rate  = learning_rate
        self.num_iterations = num_iterations
        self.cost_function  = cost_function
        self.cost_history   = []
        self.theta          = None
        self.theta_history  = []
        self.X_mean = self.X_std = None

    def _add_bias(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def _normalize(self, X, fit=False):
        X = np.array(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std  = X.std(axis=0, ddof=0)
            self.X_std[self.X_std == 0] = 1.0
        return (X - self.X_mean) / self.X_std

    def _cost(self, pred, y):
        m, e = len(y), pred - y
        if self.cost_function == 'mse':  return (1/(2*m)) * np.sum(e**2)
        if self.cost_function == 'mae':  return (1/m)    * np.sum(np.abs(e))
        return np.sqrt((1/m) * np.sum(e**2))

    def _grad(self, Xb, pred, y):
        m, e = len(y), pred - y
        if self.cost_function == 'mae':  return (1/m) * (Xb.T @ np.sign(e))
        if self.cost_function == 'rmse':
            rmse = np.sqrt((1/m)*np.sum(e**2)); denom = (m*rmse) if rmse else m
            return (1/denom) * (Xb.T @ e)
        return (1/m) * (Xb.T @ e)

    def fit(self, X, y):
        Xn = self._normalize(X, fit=True)
        Xb = self._add_bias(Xn)
        y  = np.array(y, dtype=float).flatten()
        self.theta = np.zeros(Xb.shape[1])
        self.cost_history = []; self.theta_history = []
        for _ in range(self.num_iterations):
            pred = Xb @ self.theta
            self.cost_history.append(self._cost(pred, y))
            self.theta_history.append(self.theta.copy())
            self.theta -= self.learning_rate * self._grad(Xb, pred, y)
        return self.cost_history

    def predict(self, X):
        return self._add_bias(self._normalize(X)) @ self.theta


# ─── Data loading & caching ──────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        # Generate synthetic data if no file uploaded
        rng = np.random.default_rng(42)
        n = 1000
        score = rng.uniform(0, 100, n)
        df = pd.DataFrame({
            "Height (cm)":               170 - 0.3*score + rng.normal(0, 5, n),
            "Weight (kg)":               75  - 0.2*score + rng.normal(0, 5, n),
            "Screen Time (hrs)":         2   + 0.1*score + rng.normal(0, 1, n),
            "Junk Food (days/week)":     1   + 0.06*score + rng.normal(0, 0.5, n),
            "Physical Activity (hrs/week)": 10 - 0.09*score + rng.normal(0, 1, n),
            "Task Completion (scale)":   5   - 0.04*score + rng.normal(0, 0.8, n),
            "Human-Zombie Score":        score,
        })
    df.drop_duplicates(inplace=True)
    df = df.clip(lower=0)
    return df


@st.cache_data
def prepare_models(df_hash: int, df: pd.DataFrame):
    X = df.drop(columns=["Human-Zombie Score"])
    y = df["Human-Zombie Score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    sk_model = LinearRegression()
    sk_model.fit(Xtr_s, y_train)

    custom = LinearRegressionCustom(0.1, 500, 'mse')
    custom.fit(X_train, y_train)

    ridge_cv   = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5).fit(Xtr_s, y_train)
    lasso_cv   = LassoCV(alphas=np.logspace(-2, 2, 50), cv=5, max_iter=10000).fit(Xtr_s, y_train)
    elastic_cv = ElasticNetCV(alphas=np.logspace(-2, 2, 50), cv=5, max_iter=10000).fit(Xtr_s, y_train)

    return (X, y, X_train, X_test, y_train, y_test,
            scaler, sk_model, custom, ridge_cv, lasso_cv, elastic_cv,
            Xtr_s, Xte_s)


# ─── Helpers ──────────────────────────────────────────────────────────────────
ZOMBIE_PALETTE = ["#6dff6d", "#a8ff78", "#78ffd6", "#f8ff6d", "#ff9a3c", "#ff4e4e"]

def score_color(score: float) -> str:
    if score < 20:   return "#6dff6d"
    elif score < 40: return "#a8ff78"
    elif score < 60: return "#f8ff6d"
    elif score < 80: return "#ff9a3c"
    return "#ff4e4e"

def score_label(score: float) -> str:
    if score < 20:   return "🧑 Fully Human"
    elif score < 40: return "😟 Slightly Undead"
    elif score < 60: return "🤢 Concerning"
    elif score < 80: return "🧟 Mostly Zombie"
    return "☠️ Full Zombie — DENIED!"

def mpl_dark():
    plt.rcParams.update({
        "figure.facecolor": "#0a0d0f", "axes.facecolor": "#0d1a0d",
        "axes.edgecolor": "#2d5a2d",   "axes.labelcolor": "#b0d8a4",
        "xtick.color": "#b0d8a4",      "ytick.color": "#b0d8a4",
        "text.color": "#d4e8c2",       "grid.color": "#1a3d1a",
        "grid.alpha": 0.5,
    })


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Earth Junior Mission")
    st.markdown('<div class="info-box">Upload your dataset or use synthetic data to begin the screening protocol.</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("📂 Upload CSV", type="csv")
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    lr        = st.select_slider("Learning Rate", [0.001, 0.01, 0.05, 0.1, 0.3, 0.5], value=0.1)
    n_iters   = st.slider("Gradient Descent Iterations", 100, 1000, 500, 50)
    cost_fn   = st.radio("Cost Function", ["mse", "mae", "rmse"], horizontal=True)
    st.markdown("---")
    st.markdown('<div class="info-box">Year 3050 — The last ML engineer stands between humanity and zombie apocalypse.</div>',
                unsafe_allow_html=True)


# ─── Load data ────────────────────────────────────────────────────────────────
df = load_data(uploaded)
(X, y, X_train, X_test, y_train, y_test,
 scaler, sk_model, custom, ridge_cv, lasso_cv, elastic_cv,
 Xtr_s, Xte_s) = prepare_models(id(df), df)

FEATURES = list(X.columns)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<h1>🧟 Zombie Screening Station</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#78ffd6;font-family:\'Share Tech Mono\',monospace;">EARTH JUNIOR MIGRATION AUTHORITY  |  SCREENING PROTOCOL v7.3</p>',
            unsafe_allow_html=True)
st.markdown("---")


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 Predict Score",
    "📊 Data Explorer",
    "📉 Model Training",
    "🏆 Model Comparison",
    "🔬 Feature Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🧬 Passenger Screening Interface")
    st.markdown('<div class="info-box">Adjust the biometric sliders to evaluate a passenger\'s zombie-ness score.</div>',
                unsafe_allow_html=True)

    col_inputs, col_result = st.columns([3, 2])

    with col_inputs:
        st.markdown("### 📋 Biometric Input")
        c1, c2 = st.columns(2)
        with c1:
            height   = st.slider("📏 Height (cm)",
                                 float(df["Height (cm)"].min()),
                                 float(df["Height (cm)"].max()),
                                 float(df["Height (cm)"].mean()), step=0.5)
            weight   = st.slider("⚖️ Weight (kg)",
                                 float(df["Weight (kg)"].min()),
                                 float(df["Weight (kg)"].max()),
                                 float(df["Weight (kg)"].mean()), step=0.5)
            screen   = st.slider("📱 Screen Time (hrs)",
                                 float(df["Screen Time (hrs)"].min()),
                                 float(df["Screen Time (hrs)"].max()),
                                 float(df["Screen Time (hrs)"].mean()), step=0.1)
        with c2:
            junk     = st.slider("🍔 Junk Food (days/wk)",
                                 float(df["Junk Food (days/week)"].min()),
                                 float(df["Junk Food (days/week)"].max()),
                                 float(df["Junk Food (days/week)"].mean()), step=0.1)
            activity = st.slider("🏃 Physical Activity (hrs/wk)",
                                 float(df["Physical Activity (hrs/week)"].min()),
                                 float(df["Physical Activity (hrs/week)"].max()),
                                 float(df["Physical Activity (hrs/week)"].mean()), step=0.1)
            task     = st.slider("✅ Task Completion (0-10)",
                                 float(df["Task Completion (scale)"].min()),
                                 float(df["Task Completion (scale)"].max()),
                                 float(df["Task Completion (scale)"].mean()), step=0.1)

    # Predict
    inp         = np.array([[height, weight, screen, junk, activity, task]])
    inp_scaled  = scaler.transform(inp)
    sk_score    = float(np.clip(sk_model.predict(inp_scaled)[0], 0, 100))
    cust_score  = float(np.clip(custom.predict(inp)[0], 0, 100))
    avg_score   = (sk_score + cust_score) / 2

    with col_result:
        st.markdown("### 🎯 Screening Result")
        colour = score_color(avg_score)
        label  = score_label(avg_score)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1a0d,#162616);
                    border:2px solid {colour};border-radius:16px;
                    padding:1.5rem;text-align:center;
                    box-shadow:0 0 24px {colour}44;">
          <div class="zombie-score" style="color:{colour};">{avg_score:.1f}</div>
          <div style="font-family:\'Exo 2\',sans-serif;font-size:1.3rem;
                      color:{colour};margin-top:0.5rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#b0d8a4"},
                "bar": {"color": colour},
                "bgcolor": "#0d1a0d",
                "bordercolor": "#2d5a2d",
                "steps": [
                    {"range": [0,  20], "color": "#0d2b0d"},
                    {"range": [20, 40], "color": "#1a3d1a"},
                    {"range": [40, 60], "color": "#3d3d0d"},
                    {"range": [60, 80], "color": "#3d1a0d"},
                    {"range": [80, 100],"color": "#3d0d0d"},
                ],
                "threshold": {"line": {"color": "#ff4e4e", "width": 3},
                              "thickness": 0.75, "value": avg_score},
            },
            number={"font": {"color": colour, "family": "Share Tech Mono"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0a0d0f", font_color="#d4e8c2", height=260,
            margin=dict(l=20, r=20, t=10, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.metric("sklearn LR",       f"{sk_score:.2f}")
        st.metric("Custom GD Model",  f"{cust_score:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Dataset Explorer")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",    f"{len(df):,}")
    c2.metric("Features",         str(len(FEATURES)))
    c3.metric("Avg Zombie Score", f"{y.mean():.1f}")
    c4.metric("Max Zombie Score", f"{y.max():.1f}")

    st.markdown("### 🔥 Correlation Heatmap")
    corr = df.corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
        font_color="#d4e8c2", height=450,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### 📈 Feature Distributions")
    fig_dist = make_subplots(rows=2, cols=3,
        subplot_titles=FEATURES,
        vertical_spacing=0.12, horizontal_spacing=0.08)
    for idx, col in enumerate(FEATURES):
        r, c = divmod(idx, 3)
        fig_dist.add_trace(
            go.Histogram(x=df[col], nbinsx=30,
                         marker_color=ZOMBIE_PALETTE[idx], opacity=0.75,
                         name=col, showlegend=False),
            row=r+1, col=c+1,
        )
    fig_dist.update_layout(
        paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
        font_color="#d4e8c2", height=500,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### 🎯 Features vs Human-Zombie Score")
    selected_feat = st.selectbox("Select feature", FEATURES)
    fig_scatter = px.scatter(
        df, x=selected_feat, y="Human-Zombie Score",
        color="Human-Zombie Score",
        color_continuous_scale="RdYlGn_r",
        opacity=0.65,
        trendline="ols",
        trendline_color_override="#6dff6d",
    )
    fig_scatter.update_layout(
        paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
        font_color="#d4e8c2", height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### 🗃️ Raw Data Sample")
    st.dataframe(df.sample(min(20, len(df))).style.background_gradient(
        cmap="YlGn", subset=["Human-Zombie Score"]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📉 Custom Gradient Descent Training")
    st.markdown('<div class="info-box">Train the custom from-scratch model with the settings in the sidebar.</div>',
                unsafe_allow_html=True)

    if st.button("🚀 Train Custom Model Now"):
        with st.spinner("Training in progress…"):
            prog_model = LinearRegressionCustom(lr, n_iters, cost_fn)
            cost_hist  = prog_model.fit(X_train, y_train)
            preds_test = prog_model.predict(X_test)
            mse_val    = mean_squared_error(y_test, preds_test)
            r2_val     = r2_score(y_test, preds_test)

        st.success("Training complete!")
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Final Cost",  f"{cost_hist[-1]:.4f}")
        cm2.metric("Test MSE",    f"{mse_val:.2f}")
        cm3.metric("Test R²",     f"{r2_val:.4f}")

        # Cost history
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Scatter(
            y=cost_hist, mode='lines',
            line=dict(color='#6dff6d', width=2), name='Cost',
        ))
        fig_ch.update_layout(
            title="Cost History During Gradient Descent",
            xaxis_title="Iteration", yaxis_title="Cost",
            paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
            font_color="#d4e8c2", height=380,
        )
        st.plotly_chart(fig_ch, use_container_width=True)

        # Actual vs predicted
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_test.values, y=preds_test, mode='markers',
            marker=dict(color='#78ffd6', opacity=0.55, size=5),
            name='Predictions',
        ))
        rng = [y_test.min(), y_test.max()]
        fig_avp.add_trace(go.Scatter(
            x=rng, y=rng, mode='lines',
            line=dict(color='#ff4e4e', dash='dash'), name='Perfect Fit',
        ))
        fig_avp.update_layout(
            title="Actual vs Predicted Scores",
            xaxis_title="Actual", yaxis_title="Predicted",
            paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
            font_color="#d4e8c2", height=380,
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        # Feature weights bar
        theta_names = ["bias"] + list(X_train.columns)
        colours = ['#ff9a3c' if w < 0 else '#6dff6d'
                   for w in prog_model.theta]
        fig_w = go.Figure(go.Bar(
            x=theta_names, y=prog_model.theta,
            marker_color=colours,
        ))
        fig_w.update_layout(
            title="Learned θ (Weights)",
            paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
            font_color="#d4e8c2", height=360,
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("👆 Click **Train Custom Model Now** to start training with the sidebar settings.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🏆 Model Leaderboard")

    results = {}
    for name, mdl, Xte in [
        ("sklearn LR",   sk_model,   Xte_s),
        ("Ridge (CV)",   ridge_cv,   Xte_s),
        ("Lasso (CV)",   lasso_cv,   Xte_s),
        ("ElasticNet (CV)", elastic_cv, Xte_s),
    ]:
        yp = mdl.predict(Xte)
        results[name] = {
            "MSE":  mean_squared_error(y_test, yp),
            "RMSE": np.sqrt(mean_squared_error(y_test, yp)),
            "R²":   r2_score(y_test, yp),
        }
    # Custom model
    yp_cust = custom.predict(X_test)
    results["Custom GD"] = {
        "MSE":  mean_squared_error(y_test, yp_cust),
        "RMSE": np.sqrt(mean_squared_error(y_test, yp_cust)),
        "R²":   r2_score(y_test, yp_cust),
    }

    res_df = pd.DataFrame(results).T.sort_values("R²", ascending=False)
    st.dataframe(res_df.style.format("{:.4f}").background_gradient(
        cmap="YlGn", subset=["R²"]).background_gradient(
        cmap="YlOrRd_r", subset=["MSE", "RMSE"]),
        use_container_width=True)

    # Bar chart comparison
    fig_bar = go.Figure()
    metrics = ["MSE", "RMSE", "R²"]
    for i, metric in enumerate(metrics):
        fig_bar.add_trace(go.Bar(
            name=metric,
            x=list(results.keys()),
            y=[results[m][metric] for m in results],
            marker_color=ZOMBIE_PALETTE[i],
        ))
    fig_bar.update_layout(
        barmode='group', title="Model Metrics Comparison",
        paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
        font_color="#d4e8c2", height=400,
        legend=dict(bgcolor="#0d1a0d", bordercolor="#2d5a2d"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Regularisation paths
    st.markdown("### 🔬 Regularisation — Ridge vs Lasso Coefficients")
    alphas = np.logspace(-3, 2, 60)
    ridge_coefs = np.array([
        Ridge(alpha=a).fit(Xtr_s, y_train).coef_ for a in alphas])
    lasso_coefs = np.array([
        Lasso(alpha=a, max_iter=5000).fit(Xtr_s, y_train).coef_ for a in alphas])

    fig_reg = make_subplots(rows=1, cols=2,
        subplot_titles=["Ridge Coefficient Path", "Lasso Coefficient Path"])
    for i, fname in enumerate(FEATURES):
        fig_reg.add_trace(go.Scatter(
            x=np.log10(alphas), y=ridge_coefs[:, i],
            mode='lines', name=fname,
            line=dict(color=ZOMBIE_PALETTE[i % len(ZOMBIE_PALETTE)]),
            showlegend=True,
        ), row=1, col=1)
        fig_reg.add_trace(go.Scatter(
            x=np.log10(alphas), y=lasso_coefs[:, i],
            mode='lines', name=fname,
            line=dict(color=ZOMBIE_PALETTE[i % len(ZOMBIE_PALETTE)]),
            showlegend=False,
        ), row=1, col=2)
    fig_reg.update_layout(
        paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
        font_color="#d4e8c2", height=430,
        legend=dict(bgcolor="#0d1a0d", bordercolor="#2d5a2d"),
    )
    fig_reg.update_xaxes(title_text="log₁₀(α)")
    fig_reg.update_yaxes(title_text="Coefficient Value")
    st.plotly_chart(fig_reg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔬 Feature Analysis")

    # Feature importance
    st.markdown("### 📌 Feature Importance (sklearn LR coefficients)")
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": sk_model.coef_,
        "Abs": np.abs(sk_model.coef_),
    }).sort_values("Abs", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        orientation='h',
        marker_color=['#ff4e4e' if c < 0 else '#6dff6d'
                      for c in coef_df["Coefficient"]],
    ))
    fig_imp.update_layout(
        title="Standardised Feature Coefficients",
        paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
        font_color="#d4e8c2", height=380,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Pairplot alternative: scatter matrix
    st.markdown("### 🌐 Scatter Matrix")
    sel = st.multiselect("Choose features", FEATURES,
                         default=FEATURES[:4])
    if sel:
        fig_pm = px.scatter_matrix(
            df, dimensions=sel + ["Human-Zombie Score"],
            color="Human-Zombie Score",
            color_continuous_scale="RdYlGn_r",
            opacity=0.5,
        )
        fig_pm.update_traces(diagonal_visible=False)
        fig_pm.update_layout(
            paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
            font_color="#d4e8c2", height=600,
        )
        st.plotly_chart(fig_pm, use_container_width=True)

    # Score distribution
    st.markdown("### 🧟 Score Distribution")
    fig_yd = px.histogram(
        df, x="Human-Zombie Score", nbins=40,
        color_discrete_sequence=["#6dff6d"],
    )
    fig_yd.add_vline(x=50, line_dash="dash", line_color="#ff4e4e",
                     annotation_text="Danger Threshold",
                     annotation_font_color="#ff4e4e")
    fig_yd.update_layout(
        paper_bgcolor="#0a0d0f", plot_bgcolor="#0d1a0d",
        font_color="#d4e8c2", height=350,
    )
    st.plotly_chart(fig_yd, use_container_width=True)


# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-family:\'Share Tech Mono\',monospace;'
    'color:#3d7a3d;font-size:0.8rem;">EARTH JUNIOR MIGRATION AUTHORITY '
    '· SCREENING PROTOCOL v7.3 · YEAR 3050</p>',
    unsafe_allow_html=True,
)
