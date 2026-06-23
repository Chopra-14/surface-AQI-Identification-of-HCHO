import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="India AQI Monitor",
    page_icon="🛰️",
    layout="wide"
)

# ============================================
# GLOBAL STYLING
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        color: #666;
        font-size: 0.95rem;
        margin-top: -8px;
    }
    .disclaimer-box {
        background: #fff8e1;
        border-left: 4px solid #f5a623;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #6b5b00;
        margin: 14px 0 20px 0;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-label {
        color: #777;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 8px;
    }
    .hotspot-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #e31a1c;
        margin-bottom: 12px;
    }
    .hotspot-num {
        background: #1a1a2e;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 13px;
        margin-right: 10px;
    }
    .hotspot-title {
        font-size: 15px;
        font-weight: 600;
        color: #1a1a2e;
    }
    .hotspot-stat {
        color: #555;
        font-size: 12.5px;
        margin-top: 6px;
    }
    .footer-box {
        text-align: center;
        color: #888;
        font-size: 0.82rem;
        padding: 20px 0 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('india_aqi_final.csv')
    hotspots = pd.read_csv('hotspots_summary.csv')
    return df, hotspots

df, hotspots = load_data()

# ============================================
# HEADER
# ============================================
st.markdown('<p class="main-title">🛰️ India Air Quality Monitor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Satellite-Powered Air Quality Intelligence — Bharatiya Antariksh Hackathon 2026</p>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer-box">
⚠️ Values shown are a <b>Satellite Pollution Index (SPI, 0–500 scale)</b> derived from Sentinel-5P data,
validated against CPCB ground stations (r = 0.45). This is not an official CPCB AQI value.
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("Filters")
category_filter = st.sidebar.multiselect(
    "Pollution Category",
    df['category'].unique().tolist(),
    default=df['category'].unique().tolist()
)
filtered = df[df['category'].isin(category_filter)]

st.sidebar.markdown("---")
st.sidebar.caption("Data: Sentinel-5P TROPOMI (2024 annual average)")
st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} points")

# ============================================
# TOP METRICS
# ============================================
col1, col2, col3, col4 = st.columns(4)
metrics = [
    ("Average SPI (India)", f"{filtered['aqi'].mean():.0f}" if len(filtered) else "—"),
    ("Worst SPI Recorded", f"{filtered['aqi'].max():.0f}" if len(filtered) else "—"),
    ("Hotspot Zones Found", f"{len(hotspots)}"),
    ("Points Analyzed", f"{len(filtered):,}")
]
for col, (label, value) in zip([col1, col2, col3, col4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# POLLUTION HEATMAP — CONTINUOUS GRADIENT
# ============================================
st.markdown('<p class="section-header">📍 Pollution Heatmap of India</p>', unsafe_allow_html=True)
st.caption("Satellite Pollution Index (SPI) — continuous surface interpolated from filtered sample points")

if len(filtered) > 0:
    fig = go.Figure()

    fig.add_trace(go.Densitymap(
        lat=filtered['latitude'],
        lon=filtered['longitude'],
        z=filtered['aqi'],
        radius=35,
        colorscale=[
            [0.0, '#2166ac'],
            [0.2, '#67a9cf'],
            [0.4, '#ffffbf'],
            [0.6, '#fdae61'],
            [0.8, '#e31a1c'],
            [1.0, '#7f0000']
        ],
        zmin=0,
        zmax=500,
        opacity=0.75,
        showscale=True,
        colorbar=dict(title="SPI", thickness=15, len=0.7),
        hovertemplate="SPI: %{z:.0f}<extra></extra>"
    ))

    for idx, row in hotspots.iterrows():
        fig.add_trace(go.Scattermap(
            lat=[row['lat']], lon=[row['lon']],
            mode='markers',
            marker=dict(size=34, color='#d62728', opacity=0.45),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scattermap(
            lat=[row['lat']], lon=[row['lon']],
            mode='markers+text',
            marker=dict(size=26, color='white', opacity=0.95),
            text=[str(idx + 1)],
            textfont=dict(size=14, color='#1a1a2e'),
            hovertemplate=f"<b>{row['region_name']}</b><br>Avg SPI: {row['avg_aqi']:.0f}<br>Max SPI: {row['max_aqi']:.0f}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=22.5, lon=80.5),
            zoom=3.8
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data points match the selected filters. Try selecting at least one category.")

# ============================================
# HOTSPOT CARDS
# ============================================
st.markdown("---")
st.markdown('<p class="section-header">🔥 Detected Pollution Hotspots — Details</p>', unsafe_allow_html=True)

cols = st.columns(3)
for idx, row in hotspots.iterrows():
    with cols[idx % 3]:
        severity_color = "#7f0000" if row['avg_aqi'] > 220 else "#e31a1c" if row['avg_aqi'] > 150 else "#fdae61"
        st.markdown(f"""
        <div class="hotspot-card" style="border-left-color:{severity_color}">
            <span class="hotspot-num">{idx+1}</span>
            <span class="hotspot-title">{row['region_name']}</span>
            <div class="hotspot-stat">Avg SPI: <b>{row['avg_aqi']:.0f}</b> &nbsp;|&nbsp; Max SPI: <b>{row['max_aqi']:.0f}</b></div>
            <div class="hotspot-stat">{int(row['point_count'])} satellite data points</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# HOTSPOT SUMMARY TABLE
# ============================================
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 View full hotspot data table"):
    st.dataframe(
        hotspots[['region_name', 'avg_aqi', 'max_aqi', 'point_count']].rename(
            columns={'region_name': 'Region', 'avg_aqi': 'Avg SPI', 'max_aqi': 'Max SPI', 'point_count': 'Data Points'}
        ),
        use_container_width=True,
        hide_index=True
    )

# ============================================
# SEASONAL TREND CHART (ILLUSTRATIVE)
# ============================================
st.markdown("---")
st.markdown('<p class="section-header">📅 Seasonal Pollution Trend (2024)</p>', unsafe_allow_html=True)
st.caption("⚠️ Illustrative — based on known seasonal pollution patterns in India (e.g. crop-burning season). Not derived from monthly satellite readings.")

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
np.random.seed(42)
base = df['aqi'].mean()
seasonal_factor = [1.3, 1.2, 1.0, 0.85, 0.8, 0.75, 0.7, 0.75, 0.85, 1.4, 1.6, 1.35]
monthly_spi = [round(base * f + np.random.normal(0, 5), 1) for f in seasonal_factor]

def spi_category(v):
    if v <= 50: return 'Good'
    elif v <= 100: return 'Moderate'
    elif v <= 200: return 'Unhealthy'
    elif v <= 300: return 'Very Unhealthy'
    else: return 'Hazardous'

monthly_cat = [spi_category(v) for v in monthly_spi]

fig2 = go.Figure()

fig2.add_vrect(
    x0=8.5, x1=11.5,
    fillcolor="red", opacity=0.08, line_width=0,
    annotation_text="Crop Burning Season", annotation_position="top",
    annotation_font=dict(color="darkred", size=11)
)

fig2.add_trace(go.Scatter(
    x=months,
    y=monthly_spi,
    mode='lines+markers',
    line=dict(color='#e8743b', width=3),
    marker=dict(size=9, color='#e8743b', line=dict(width=1.5, color='white')),
    fill='tozeroy',
    fillcolor='rgba(232,116,59,0.12)',
    customdata=monthly_cat,
    hovertemplate="<b>%{x}</b><br>SPI: %{y}<br>Category: %{customdata}<extra></extra>"
))

fig2.update_layout(
    title=dict(text="Monthly Pollution Trend — India (Illustrative)", font=dict(size=15)),
    yaxis_title="Average SPI",
    height=400,
    margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor='white',
    hovermode='x unified',
    yaxis=dict(gridcolor='#eee', zeroline=False),
    xaxis=dict(gridcolor='#fff')
)

st.plotly_chart(fig2, use_container_width=True)

# ============================================
# CPCB VALIDATION CHART
# ============================================
st.markdown("---")
st.markdown('<p class="section-header">✅ Validation: Satellite vs CPCB Ground Truth</p>', unsafe_allow_html=True)
st.image('comparison_chart_final.png', use_container_width=True)
st.caption("Our satellite pollution index shows moderate positive correlation (r = 0.45) with CPCB ground station readings across 10 major Indian cities — correctly identifying highly polluted cities like Delhi and Kolkata as severe, and comparatively cleaner cities like Chennai and Hyderabad as lower.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer-box">
Built for Bharatiya Antariksh Hackathon 2026 &nbsp;|&nbsp; Data: Sentinel-5P TROPOMI, CPCB &nbsp;|&nbsp; Team Code Vipers
</div>
""", unsafe_allow_html=True)