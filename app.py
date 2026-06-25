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

if len(filtered) >= 5:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union
    from plotly.subplots import make_subplots

    @st.cache_data
    def get_india_geometry():
        gdf = gpd.read_file('india_boundary_simplified.geojson')
        return gdf, unary_union(gdf.geometry)

    india_gdf, india_shape = get_india_geometry()

    @st.cache_data
    def build_heatmap_grid(lat_vals, lon_vals, aqi_vals):
        grid_lon = np.linspace(68, 97, 250)
        grid_lat = np.linspace(8, 37, 250)
        gx, gy = np.meshgrid(grid_lon, grid_lat)
        gz = griddata((lon_vals, lat_vals), aqi_vals, (gx, gy), method='linear')
        gz = gaussian_filter(np.nan_to_num(gz, nan=np.nanmean(gz)), sigma=2)
        mask = shapely.contains_xy(india_shape, gx, gy)
        gz_masked = np.where(mask, gz, np.nan)
        return grid_lon, grid_lat, gz_masked

    grid_lon, grid_lat, gz_masked = build_heatmap_grid(
        filtered['latitude'].values, filtered['longitude'].values, filtered['aqi'].values
    )

    vmin = np.nanpercentile(gz_masked, 5)
    vmax = np.nanpercentile(gz_masked, 95)

    map_fig = go.Figure()

    map_fig.add_trace(go.Contour(
        x=grid_lon, y=grid_lat, z=gz_masked,
        colorscale=[[0, '#1a9850'], [0.1, '#66bd63'], [0.2, '#a6d96a'], [0.3, '#d9ef8b'],
                    [0.4, '#fee08b'], [0.5, '#fdae61'], [0.6, '#fc4e2a'], [0.75, '#e31a1c'],
                    [0.85, '#bd0026'], [1.0, '#800026']],
        contours=dict(coloring='fill', showlines=False),
        line_smoothing=0.85,
        colorbar=dict(title='SPI', thickness=15),
        zmin=vmin, zmax=vmax,
        hoverinfo='skip'
    ))

    # State border outlines
    for geom in india_gdf.geometry:
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            if poly.area < 0.01:
                continue
            x, y = poly.exterior.xy
            map_fig.add_trace(go.Scatter(
                x=list(x), y=list(y), mode='lines',
                line=dict(color='#444444', width=0.7),
                showlegend=False, hoverinfo='skip'
            ))

    # Hotspot pins using the 📍 emoji, with hover tooltip on the same trace
    map_fig.add_trace(go.Scatter(
        x=hotspots['lon'], y=hotspots['lat'],
        mode='text',
        text=['📍'] * len(hotspots),
        textfont=dict(size=26),
        customdata=hotspots[['region_name', 'avg_aqi', 'max_aqi', 'point_count']].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Avg SPI: %{customdata[1]:.0f}<br>"
            "Max SPI: %{customdata[2]:.0f}<br>"
            "Data points: %{customdata[3]:.0f}"
            "<extra></extra>"
        ),
        name='hotspots',
        showlegend=False
    ))

    map_fig.update_layout(
        xaxis=dict(range=[67.5, 97.5], visible=False),
        yaxis=dict(range=[7.5, 37.5], visible=False, scaleanchor='x', scaleratio=1),
        height=600, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white', paper_bgcolor='white'
    )

    st.caption("📍 Hover a pin to see its stats. Use the dropdown below to see a detailed breakdown for any hotspot.")
    st.plotly_chart(map_fig, use_container_width=True, key="hotspot_map")

    # Dropdown-based selection - reliable across all environments, unlike click-to-select
    # on third-party map components which can break depending on the browser/frontend bundle
    region_options = [f"{row['region_name']} (#{i+1})" for i, row in hotspots.iterrows()]
    selected_label = st.selectbox("🔍 Select a hotspot for detailed breakdown:", region_options, key="hotspot_selector")
    st.session_state.selected_hotspot_idx = region_options.index(selected_label)

    selected_region = hotspots.iloc[st.session_state.selected_hotspot_idx]['region_name']
    sel_row = hotspots.iloc[st.session_state.selected_hotspot_idx]
    national_avg = df['aqi'].mean()

    # ============================================
    # 4-Panel Detail Charts for the Clicked Hotspot
    # ============================================
    st.markdown("<br>", unsafe_allow_html=True)

    detail_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Avg SPI: This Region vs National Average",
            "This Region: Average vs Peak SPI",
            "Sample Size vs Other Hotspots",
            "National Seasonal Pattern (Illustrative)"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.18, horizontal_spacing=0.1
    )

    detail_fig.add_trace(go.Bar(
        x=['This Region', 'India Average'], y=[sel_row['avg_aqi'], national_avg],
        marker_color=['#bd0026', '#66bd63'], showlegend=False,
        text=[f"{sel_row['avg_aqi']:.0f}", f"{national_avg:.0f}"], textposition='outside'
    ), row=1, col=1)

    detail_fig.add_trace(go.Bar(
        x=['Average', 'Peak (Max)'], y=[sel_row['avg_aqi'], sel_row['max_aqi']],
        marker_color=['#fdae61', '#800026'], showlegend=False,
        text=[f"{sel_row['avg_aqi']:.0f}", f"{sel_row['max_aqi']:.0f}"], textposition='outside'
    ), row=1, col=2)

    bar_colors = ['#1a1a2e' if r == selected_region else '#cbd5e1' for r in hotspots['region_name']]
    detail_fig.add_trace(go.Bar(
        x=hotspots['region_name'], y=hotspots['point_count'],
        marker_color=bar_colors, showlegend=False
    ), row=2, col=1)

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    seasonal_factor = [1.3, 1.2, 1.0, 0.85, 0.8, 0.75, 0.7, 0.75, 0.85, 1.4, 1.6, 1.35]
    monthly_vals = [national_avg * f for f in seasonal_factor]
    detail_fig.add_trace(go.Scatter(
        x=months, y=monthly_vals, mode='lines+markers',
        line=dict(color='#e8743b', width=2.5), showlegend=False
    ), row=2, col=2)

    detail_fig.update_layout(
        height=620,
        title_text=f"📊 Detailed Breakdown — {selected_region}",
        title_font_size=18,
        showlegend=False,
        margin=dict(t=80)
    )
    st.plotly_chart(detail_fig, use_container_width=True)
    st.caption("⚠️ The seasonal panel reuses the national illustrative pattern — region-specific monthly data was not collected in this version.")
else:
    st.warning(f"Only {len(filtered)} point(s) match the selected filter(s) — need at least 5 to build a reliable map surface. Try selecting more categories.")

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
            <span class="hotspot-num">📍</span>
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