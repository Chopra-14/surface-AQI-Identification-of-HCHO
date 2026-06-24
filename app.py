import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="India AQI Monitor", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .subtitle { color: #666; font-size: 0.95rem; margin-top: -8px; }
    .disclaimer-box { background: #fff8e1; border-left: 4px solid #f5a623; padding: 10px 16px;
        border-radius: 6px; font-size: 0.85rem; color: #6b5b00; margin: 14px 0 20px 0; }
    .metric-card { background: white; border-radius: 12px; padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }
    .metric-label { color: #777; font-size: 0.82rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-top: 4px; }
    .section-header { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; margin-top: 8px; }
    .hotspot-card { background: white; border-radius: 12px; padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #e31a1c; margin-bottom: 12px; }
    .hotspot-title { font-size: 15px; font-weight: 600; color: #1a1a2e; }
    .hotspot-stat { color: #555; font-size: 12.5px; margin-top: 6px; }
    .footer-box { text-align: center; color: #888; font-size: 0.82rem; padding: 20px 0 6px 0; }
</style>
""", unsafe_allow_html=True)

# ── COLOR SCALE ──
COLORSCALE = [
    [0.00, '#006837'], [0.15, '#31a354'], [0.30, '#78c679'],
    [0.42, '#c2e699'], [0.50, '#ffffb2'], [0.60, '#fecc5c'],
    [0.70, '#fd8d3c'], [0.80, '#f03b20'], [0.88, '#9e0142'],
    [1.00, '#5e0035'],
]

# ── LOAD DATA ──
@st.cache_data
def load_data():
    df = pd.read_csv('india_aqi_final.csv')
    hotspots = pd.read_csv('hotspots_summary.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    hotspots.columns = [c.strip().lower() for c in hotspots.columns]
    return df, hotspots

df, hotspots = load_data()

# ── HEADER ──
st.markdown('<p class="main-title">🛰️ India Air Quality Monitor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Satellite-Powered Air Quality Intelligence — Bharatiya Antariksh Hackathon 2026</p>', unsafe_allow_html=True)
st.markdown("""<div class="disclaimer-box">
⚠️ Values shown are a <b>Satellite Pollution Index (SPI, 0–500 scale)</b> derived from Sentinel-5P data,
validated against CPCB ground stations (r = 0.45). This is not an official CPCB AQI value.
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ──
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

# ── METRICS ──
col1, col2, col3, col4 = st.columns(4)
metrics = [
    ("Average SPI (India)", f"{filtered['aqi'].mean():.0f}" if len(filtered) else "—"),
    ("Worst SPI Recorded",  f"{filtered['aqi'].max():.0f}"  if len(filtered) else "—"),
    ("Hotspot Zones Found", f"{len(hotspots)}"),
    ("Points Analyzed",     f"{len(filtered):,}")
]
for col, (label, value) in zip([col1, col2, col3, col4], metrics):
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# POLLUTION HEATMAP — using Densitymapbox (no shapely needed)
# ============================================================
st.markdown('<p class="section-header">📍 Pollution Heatmap of India</p>', unsafe_allow_html=True)
st.caption("Satellite Pollution Index (SPI) — density map from satellite data points")

if len(filtered) > 0:
    from streamlit_plotly_events import plotly_events

    # ── Auto-detect lat/lon columns ──
    lat_col = 'latitude' if 'latitude' in filtered.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in filtered.columns else 'lon'

    lat_vals = filtered[lat_col].values.astype(float)
    lon_vals = filtered[lon_col].values.astype(float)
    aqi_vals = filtered['aqi'].values.astype(float)

    # Auto-swap if lat/lon are reversed
    if lat_vals.mean() > 50 or lat_vals.mean() < 0:
        lat_vals, lon_vals = lon_vals, lat_vals

    vmin = float(np.percentile(aqi_vals, 5))
    vmax = float(np.percentile(aqi_vals, 95))

    map_fig = go.Figure()

    map_fig.add_trace(go.Densitymapbox(
        lat=lat_vals,
        lon=lon_vals,
        z=aqi_vals,
        radius=25,
        colorscale=COLORSCALE,
        zmin=vmin, zmax=vmax,
        colorbar=dict(
            title=dict(text='SPI', side='right'),
            thickness=15,
            tickfont=dict(size=10),
        ),
        showscale=True,
        hoverinfo='skip',
        name='heatmap',
        opacity=0.85,
    ))

    # ── 2. Pin colors by pollution level ──
    pin_colors = []
    for _, row in hotspots.iterrows():
        avg = row['avg_aqi']
        if avg > 280:   pin_colors.append('#5e0035')
        elif avg > 220: pin_colors.append('#9e0142')
        elif avg > 160: pin_colors.append('#fd8d3c')
        elif avg > 100: pin_colors.append('#fecc5c')
        else:           pin_colors.append('#31a354')

    # ── 3. Hover text ──
    hover_texts = []
    for _, row in hotspots.iterrows():
        avg = row['avg_aqi']
        if avg > 280:   cat = "🟣 Hazardous"
        elif avg > 220: cat = "🔴 Very Unhealthy"
        elif avg > 160: cat = "🟠 Unhealthy"
        elif avg > 100: cat = "🟡 Moderate"
        else:           cat = "🟢 Good"
        hover_texts.append(
            f"<b>📍 {row['region_name']}</b><br>"
            f"━━━━━━━━━━━━━━━<br>"
            f"📊 Avg SPI : <b>{row['avg_aqi']:.0f}</b><br>"
            f"⚠️ Max SPI : <b>{row['max_aqi']:.0f}</b><br>"
            f"🗂️ Points  : {int(row['point_count'])}<br>"
            f"🏷️ {cat}<br>"
            f"<i>Click to see charts ↓</i>"
        )

    # ── 4. Hotspot pins on map ──
    map_fig.add_trace(go.Scattermapbox(
        lat=hotspots['lat'],
        lon=hotspots['lon'],
        mode='markers+text',
        marker=dict(
            size=22,
            color=pin_colors,
            opacity=1.0,
        ),
        text=[str(i + 1) for i in range(len(hotspots))],
        textfont=dict(size=10, color='white'),
        hovertext=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='#1a1a2e',
            bordercolor='#555',
            font=dict(color='white', size=12),
            align='left'
        ),
        name='hotspots',
        showlegend=False
    ))

    map_fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=22, lon=82),
            zoom=3.8,
        ),
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.caption("📍 Hover over a pin to see stats. Click any pin to open detailed charts below.")
    clicked = plotly_events(
        map_fig, click_event=True, hover_event=False,
        override_height=620, key="hotspot_map"
    )

    # ── Session state ──
    if 'selected_hotspot_idx' not in st.session_state:
        st.session_state.selected_hotspot_idx = 0

    if clicked:
        point = clicked[0]
        cx = point.get('lon') or point.get('x')
        cy = point.get('lat') or point.get('y')
        if cx is not None and cy is not None:
            distances = np.sqrt((hotspots['lon'] - cx)**2 + (hotspots['lat'] - cy)**2)
            nearest_idx = distances.idxmin()
            if distances[nearest_idx] < 2.0:
                st.session_state.selected_hotspot_idx = nearest_idx

    sel_idx         = st.session_state.selected_hotspot_idx
    sel_row         = hotspots.iloc[sel_idx]
    selected_region = sel_row['region_name']
    national_avg    = df['aqi'].mean()

    # ── 4-PANEL CHARTS ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-header">📊 Detailed Analysis — 📍 {selected_region}</p>', unsafe_allow_html=True)

    detail_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"📊 {selected_region} vs National Average",
            "⚠️ Avg vs Peak SPI",
            "📍 All Hotspots Comparison",
            "📅 Seasonal Trend (India)"
        ),
        specs=[[{"type":"bar"},{"type":"scatter"}],[{"type":"bar"},{"type":"scatter"}]],
        vertical_spacing=0.22, horizontal_spacing=0.12
    )

    detail_fig.add_trace(go.Bar(
        x=[selected_region, 'India Avg'],
        y=[sel_row['avg_aqi'], national_avg],
        marker_color=['#9e0142', '#31a354'],
        marker_line=dict(width=0),
        text=[f"{sel_row['avg_aqi']:.0f}", f"{national_avg:.0f}"],
        textposition='outside', showlegend=False, width=0.4
    ), row=1, col=1)

    detail_fig.add_trace(go.Scatter(
        x=['Average SPI', 'Peak SPI'],
        y=[sel_row['avg_aqi'], sel_row['max_aqi']],
        mode='lines+markers',
        line=dict(color='#fd8d3c', width=3),
        marker=dict(size=14, color=['#fecc5c','#5e0035'], line=dict(width=2, color='white')),
        text=[f"{sel_row['avg_aqi']:.0f}", f"{sel_row['max_aqi']:.0f}"],
        textposition='top center', showlegend=False
    ), row=1, col=2)

    bar_colors = ['#9e0142' if r == selected_region else '#c7c7c7' for r in hotspots['region_name']]
    detail_fig.add_trace(go.Bar(
        x=hotspots['region_name'], y=hotspots['avg_aqi'],
        marker_color=bar_colors, marker_line=dict(width=0),
        text=[f"{v:.0f}" for v in hotspots['avg_aqi']],
        textposition='outside', showlegend=False
    ), row=2, col=1)

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    sf = [1.3,1.2,1.0,0.85,0.8,0.75,0.7,0.75,0.85,1.4,1.6,1.35]
    mv = [national_avg * f for f in sf]
    detail_fig.add_trace(go.Scatter(
        x=months, y=mv, mode='lines+markers',
        line=dict(color='#fd8d3c', width=2.5),
        marker=dict(size=8, color='#fd8d3c', line=dict(width=1.5, color='white')),
        fill='tozeroy', fillcolor='rgba(253,141,60,0.15)',
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>SPI: %{y:.0f}<extra></extra>"
    ), row=2, col=2)

    detail_fig.add_vrect(x0=8.5, x1=10.5, fillcolor='rgba(158,1,66,0.12)', line_width=0, row=2, col=2)
    detail_fig.add_annotation(x=9.5, y=max(mv)*1.08, text="🔥 Crop Burning",
        showarrow=False, font=dict(size=10, color='#9e0142'), row=2, col=2)

    detail_fig.update_layout(
        height=680, showlegend=False,
        margin=dict(t=80, l=20, r=20, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Arial', color='#1a1a2e')
    )
    for ax in ['xaxis','yaxis','xaxis2','yaxis2','xaxis3','yaxis3','xaxis4','yaxis4']:
        detail_fig.update_layout(**{ax: dict(gridcolor='#f0f0f0', zeroline=False)})

    st.plotly_chart(detail_fig, use_container_width=True)
    st.caption("⚠️ Seasonal panel shows illustrative national pattern based on known India pollution cycles.")

else:
    st.warning("No data points match the selected filters.")

# ── HOTSPOT CARDS ──
st.markdown("---")
st.markdown('<p class="section-header">🔥 Detected Pollution Hotspots</p>', unsafe_allow_html=True)
cols = st.columns(3)
for idx, row in hotspots.iterrows():
    with cols[idx % 3]:
        avg = row['avg_aqi']
        if avg > 280:   sc, badge = '#5e0035', '🟣 Hazardous'
        elif avg > 220: sc, badge = '#9e0142', '🔴 Very Unhealthy'
        elif avg > 160: sc, badge = '#fd8d3c', '🟠 Unhealthy'
        else:           sc, badge = '#fecc5c', '🟡 Moderate'
        st.markdown(f"""<div class="hotspot-card" style="border-left-color:{sc}">
            <span class="hotspot-title">📍 {row['region_name']}</span>
            <span style="float:right;font-size:11px;color:{sc};font-weight:600">{badge}</span>
            <div class="hotspot-stat">Avg SPI: <b>{row['avg_aqi']:.0f}</b> &nbsp;|&nbsp; Max SPI: <b>{row['max_aqi']:.0f}</b></div>
            <div class="hotspot-stat">{int(row['point_count'])} satellite data points</div>
        </div>""", unsafe_allow_html=True)

# ── TABLE ──
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 View full hotspot data table"):
    st.dataframe(
        hotspots[['region_name','avg_aqi','max_aqi','point_count']].rename(
            columns={'region_name':'Region','avg_aqi':'Avg SPI','max_aqi':'Max SPI','point_count':'Data Points'}
        ), use_container_width=True, hide_index=True
    )

# ── SEASONAL TREND ──
st.markdown("---")
st.markdown('<p class="section-header">📅 Seasonal Pollution Trend (2024)</p>', unsafe_allow_html=True)
st.caption("⚠️ Illustrative — based on known seasonal pollution patterns in India.")
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
np.random.seed(42)
base = df['aqi'].mean()
sf2 = [1.3,1.2,1.0,0.85,0.8,0.75,0.7,0.75,0.85,1.4,1.6,1.35]
monthly_spi = [round(base*f + np.random.normal(0,5),1) for f in sf2]
def spi_cat(v):
    if v<=50: return 'Good'
    elif v<=100: return 'Moderate'
    elif v<=200: return 'Unhealthy'
    elif v<=300: return 'Very Unhealthy'
    else: return 'Hazardous'
fig2 = go.Figure()
fig2.add_vrect(x0=8.5, x1=11.5, fillcolor="rgba(158,1,66,0.10)", line_width=0,
    annotation_text="🔥 Crop Burning Season", annotation_position="top",
    annotation_font=dict(color="#9e0142", size=11))
fig2.add_trace(go.Scatter(
    x=months, y=monthly_spi, mode='lines+markers',
    line=dict(color='#fd8d3c', width=3),
    marker=dict(size=10, color='#fd8d3c', line=dict(width=1.5, color='white')),
    fill='tozeroy', fillcolor='rgba(253,141,60,0.12)',
    customdata=[spi_cat(v) for v in monthly_spi],
    hovertemplate="<b>%{x}</b><br>SPI: %{y}<br>Category: %{customdata}<extra></extra>"
))
fig2.update_layout(
    title=dict(text="Monthly Pollution Trend — India (Illustrative)", font=dict(size=15)),
    yaxis_title="Average SPI", height=400,
    margin=dict(l=10,r=10,t=50,b=10),
    plot_bgcolor='white', paper_bgcolor='white',
    hovermode='x unified',
    yaxis=dict(gridcolor='#eee', zeroline=False),
    xaxis=dict(gridcolor='#fff')
)
st.plotly_chart(fig2, use_container_width=True)

# ── CPCB VALIDATION ──
st.markdown("---")
st.markdown('<p class="section-header">✅ Validation: Satellite vs CPCB Ground Truth</p>', unsafe_allow_html=True)
st.image('comparison_chart_final.png', use_container_width=True)
st.caption("Satellite pollution index shows moderate positive correlation (r = 0.45) with CPCB ground station readings across 10 major Indian cities.")

# ── FOOTER ──
st.markdown("---")
st.markdown("""<div class="footer-box">
Built for Bharatiya Antariksh Hackathon 2026 &nbsp;|&nbsp; Data: Sentinel-5P TROPOMI, CPCB &nbsp;|&nbsp; Team Code Vipers
</div>""", unsafe_allow_html=True)