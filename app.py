import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from backend.config import SENSOR_CSV, OPENAI_API_KEY
from backend.retriever import retrieve
from backend.models import train_anomaly_model, score_anomalies
from backend.llm import llm_summarize
import subprocess, sys
st.set_page_config(
    page_title="🏆 Smart Building RAG",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
:root {
  --card-bg: rgba(255,255,255,0.06);
  --muted: #a9b1bd;
  --accent1: #00ffb4;
  --accent2: #00aaff;
  --danger: #ff4757;
  --warn: #ffbe76;
  --ok: #2ed573;
}
html, body, [class*="css"] {
  font-family: 'Montserrat', sans-serif;
}
body {
  background: linear-gradient(135deg, #0f172a, #0b2447, #19376d, #1e293b);
  background-size: 400% 400%;
  animation: gradientFlow 18s ease infinite;
}
.header-glow {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    padding: 1rem;
    background: linear-gradient(90deg, #00ffb4, #00aaff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: textShimmer 3s ease-in-out infinite;
}
@keyframes gradientFlow {
  0% { background-position: 0% 50% }
  50% { background-position: 100% 50% }
  100% { background-position: 0% 50% }
}
.topbar {
  position: fixed; top:0; left:0; width:100%; z-index:1000;
  background: rgba(0,0,0,0.55); backdrop-filter: blur(8px);
  display:flex; align-items:center; justify-content:space-between;
  padding: 0.75rem 1.25rem; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.brand { font-weight:800; letter-spacing:0.3px; background: linear-gradient(90deg, var(--accent1), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size:1.25rem; }
.badge { color: var(--muted); font-size:0.85rem; }
.spacer { height:58px; } /* push content below fixed bar */

.section-card {
  background: var(--card-bg);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 1.25rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  transition: transform 0.25s ease, box-shadow 0.3s ease;
}
.section-card:hover { transform: translateY(-3px); box-shadow: 0 16px 38px rgba(0,255,180,0.12); }

.kpi {
  background: linear-gradient(135deg, rgba(0,255,180,0.08), rgba(0,170,255,0.08));
  border: 1px solid rgba(0,255,180,0.35);
  border-radius: 16px; padding: 1rem 1.1rem;
}
.kpi .label { color: var(--muted); font-size:0.9rem; }
.kpi .value { font-size: 1.9rem; font-weight:800; color: var(--accent1); line-height:1.1; }
.kpi .sub { font-size:0.8rem; color:#9aa4b2; }

.ai-summary {
  background: rgba(0, 200, 100, 0.12);
  border: 1px solid rgba(0, 200, 100, 0.32);
  padding: 1rem; border-radius: 12px;
}

.alert-card {
  background: rgba(255,71,87,0.12);
  border: 1px solid rgba(255,71,87,0.45);
  border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
  animation: alertPulse 1.6s infinite alternate;
}
@keyframes alertPulse { from { box-shadow: 0 0 8px rgba(255,71,87,0.25); } to { box-shadow: 0 0 22px rgba(255,71,87,0.6); } }

.severity {
  display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; font-weight: 700;
  border: 1px solid rgba(255,255,255,0.15);
}
.sev-high { background: rgba(255,71,87,0.18); color: #ff6b81; }
.sev-med  { background: rgba(255,190,118,0.18); color: #ffd28a; }
.sev-low  { background: rgba(46,213,115,0.18); color: #7bed9f; }

.footer {
  color: var(--muted); font-size: 0.85rem; text-align:center; padding: 0.8rem 0; opacity:0.8;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header-glow">📡 Smart Building IoT Data RAG</div>', unsafe_allow_html=True)
st.write("")
st.markdown(f"""
<div class="topbar">
  <div class="brand">🏢 Smart Building RAG</div>
  <div class="badge">v1.2 • {datetime.now().strftime('%b %d, %Y %H:%M')}</div>
</div>
<div class="spacer"></div>
""", unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def load_data(path: str):
    if Path(path).exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp")
        return df
    return None

def format_num(x, unit=""):
    try:
        return f"{x:,.2f}{unit}"
    except Exception:
        return f"{x}{unit}"
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    with st.expander("📦 Data"):
        if st.button("Generate Sample Data", use_container_width=True):
            with st.spinner("⏳ Generating sample data..."):
                subprocess.run([sys.executable, "backend/data_simulator.py", "--rows", "1500", "--out", str(SENSOR_CSV)], check=True)
            st.success("✅ Sample data generated!")
    with st.expander("🏭 Filters", expanded=True):
        equipment_filter = st.selectbox("Equipment", ["", "hvac", "chiller", "building"])
    with st.expander("🤖 AI Settings", expanded=True):
        use_llm = st.toggle("Use AI Summaries", value=bool(OPENAI_API_KEY))
        st.caption("AI summaries use your configured OpenAI key via backend.llm.")
    with st.expander("ℹ️ About"):
        st.write("Interactive IoT/RAG dashboard with anomaly detection and retrieval‑augmented Q&A.")
        st.caption("Design: glassmorphism, animated background, and Plotly interactivity.")
df = load_data(SENSOR_CSV)
if df is not None and len(df) > 0:
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">🌡 Temperature</div>
          <div class="value">{latest['temp_c']:.1f}°C</div>
          <div class="sub">Last update: {latest['timestamp']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">💨 Vibration</div>
          <div class="value">{latest['vibration']:.2f}</div>
          <div class="sub">Device: {latest.get('device_id','—')}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi">
          <div class="label">⚡ Power</div>
          <div class="value">{latest['power_kw']:.2f} kW</div>
          <div class="sub">Phase: {latest.get('phase','—')}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        win = df.tail(200)
        pct = (win['temp_c'].pct_change().fillna(0).tail(1).values[0]) * 100
        arrow = "▲" if pct >= 0 else "▼"
        color = "#7bed9f" if pct >= 0 else "#ff6b81"
        st.markdown(f"""
        <div class="kpi">
          <div class="label">📈 5‑min Temp Δ</div>
          <div class="value" style="color:{color}">{arrow} {pct:.2f}%</div>
          <div class="sub">vs last window</div>
        </div>""", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Live Data", "💬 Q&A", "🚨 Alerts"])
with tab1:
    if df is None or len(df) == 0:
        st.info("No sensor data found. Use **Generate Sample Data** from the sidebar.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        df_viz = df.copy()
        if equipment_filter:
            if 'equipment' in df_viz.columns:
                df_viz = df_viz[df_viz['equipment'].str.lower() == equipment_filter.lower()]
        fig = px.line(
        df_viz.tail(800),
        x="timestamp",
        y=["temp_c", "vibration", "power_kw"],
        markers=False,
        title="Sensor Trends (last 800 rows)",
        color_discrete_map={
        "temp_c": "#00B5FF",
        "vibration":"#7FFF00",
        "power_kw": "#BC13FE"
    }
)
        fig.update_layout(
            template="plotly_dark",
            height=420,
            legend_title_text="Signals",
            margin=dict(l=10,r=10,t=50,b=10)
        )
        fig.update_traces(hovertemplate='%{y:.3f}<extra>%{fullData.name}</extra>')
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hist1 = px.histogram(df_viz.tail(1000), x="temp_c", nbins=30, title="Temperature Distribution")
            hist1.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(hist1, use_container_width=True)
        with c2:
            hist2 = px.histogram(df_viz.tail(1000), x="vibration", nbins=30, title="Vibration Distribution")
            hist2.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(hist2, use_container_width=True)
        with c3:
            hist3 = px.histogram(df_viz.tail(1000), x="power_kw", nbins=30, title="Power Distribution")
            hist3.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(hist3, use_container_width=True)

        st.dataframe(df_viz.tail(300), use_container_width=True, height=280)
        st.markdown('</div>', unsafe_allow_html=True)
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    q = st.text_input("Ask about maintenance, SOPs, or specs", placeholder="e.g., Recommended chiller maintenance interval?")
    colq1, colq2 = st.columns([1,1])
    with colq1:
        run_retrieve = st.button("🔍 Retrieve Answer", use_container_width=True)
    with colq2:
        clear_q = st.button("🧹 Clear", use_container_width=True)

    if clear_q:
        st.experimental_rerun()

    if run_retrieve:
        if q.strip():
            with st.spinner("🔎 Searching knowledge base..."):
                hits = retrieve(q, equipment=equipment_filter or None, k=4)
            if hits:
                for h in hits:
                    src = h['metadata'].get('source', 'Unknown')
                    st.markdown(f"<div class='section-card' style='margin-top:0.6rem'><b>📄 Source:</b> {src}<br>{h['text']}</div>", unsafe_allow_html=True)
                if use_llm:
                    with st.spinner("🤖 Drafting AI summary..."):
                        summary = llm_summarize(q, [h['text'] for h in hits]) or "(No summary)"
                    st.markdown('<div class="ai-summary"><b>💡 AI Summary</b></div>', unsafe_allow_html=True)
                    st.write(summary)
            else:
                st.warning("No matches found in the retriever.")
        else:
            st.warning("Please type a question first.")
    st.markdown('</div>', unsafe_allow_html=True)
with tab3:
    if df is None or len(df) < 80:
        st.info("Generate data first to see alerts and anomaly scoring.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        df_train = df.tail(900)
        df_score = df.tail(500).copy().reset_index(drop=True)
        with st.spinner("🧠 Training anomaly model & scoring..."):
            model = train_anomaly_model(df_train)
            scores = score_anomalies(model, df_score)
        df_score["anomaly_score"] = scores
        threshold = float(np.percentile(scores, 98))
        figA = go.Figure()
        figA.add_trace(go.Scatter(x=df_score["timestamp"], y=df_score["anomaly_score"],
                                  mode="lines", name="Anomaly Score"))
        figA.add_hline(y=threshold, line_dash="dash", annotation_text="Threshold (98th pct)", annotation_position="top left")
        figA.update_layout(template="plotly_dark", height=360, title="Anomaly Scores (last 500 rows)",
                           margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(figA, use_container_width=True)
        alerts = df_score[df_score["anomaly_score"] >= threshold].tail(12)
        if len(alerts) == 0:
            st.success("No anomalies above threshold. System looks healthy ✅")
        else:
            st.subheader("Recent Alerts")
            for _, row in alerts.iterrows():
                score = row["anomaly_score"]
                if score >= threshold * 1.25:
                    sev = "sev-high"; label = "🔴 High"
                elif score >= threshold * 1.05:
                    sev = "sev-med"; label = "🟡 Medium"
                else:
                    sev = "sev-low"; label = "🟢 Low"
                st.markdown(
                    f"<div class='alert-card'>"
                    f"⚠ <b>{row['timestamp']}</b> — {row.get('device_id','device')} "
                    f"| Score: <b>{score:.2f}</b> "
                    f"| <span class='severity {sev}'>{label}</span>"
                    f"</div>", unsafe_allow_html=True
                )

            st.markdown("##### Alert Details")
            st.dataframe(
                alerts[["timestamp","device_id","anomaly_score"]].sort_values("anomaly_score", ascending=False),
                use_container_width=True, height=250
            )
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  Built Using Streamlit • © Nervesparks
</div>
""", unsafe_allow_html=True)