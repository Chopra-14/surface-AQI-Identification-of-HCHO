"""
UPDATED MAP + HOTSPOT SECTION for app.py
Replace your existing Folium heatmap section with this code block.
Professional continuous-gradient look, matching reference design.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ============================================
# Pollution Heatmap — Continuous Gradient
# ============================================

st.subheader("📍 Pollution Heatmap of India")
st.caption("Satellite Pollution Index (SPI) — continuous surface interpolated from 2,993 sample points")

df = pd.read_csv('india_aqi_final.csv')
hotspots = pd.read_csv('hotspots_summary.csv')

fig = go.Figure()

# Smooth gradient density layer
fig.add_trace(go.Densitymap(
    lat=df['latitude'],
    lon=df['longitude'],
    z=df['aqi'],
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

# Numbered hotspot markers
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

# ============================================
# Hotspot Cards — Clean Grid Layout
# ============================================

st.markdown("---")
st.subheader("🔥 Detected Pollution Hotspots — Details")

st.markdown("""
<style>
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
    font-size: 14px;
    margin-right: 10px;
}
.hotspot-title {
    font-size: 16px;
    font-weight: 600;
    color: #1a1a2e;
}
.hotspot-stat {
    color: #555;
    font-size: 13px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

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