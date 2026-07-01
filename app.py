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
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    from plotly.subplots import make_subplots
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import io, base64

    @st.cache_data
    def get_india_geometry():
        gdf = gpd.read_file('india_boundary_simplified.geojson')
        return gdf, unary_union(gdf.geometry)

    india_gdf, india_shape = get_india_geometry()

    @st.cache_data
    def build_folium_map(lat_vals, lon_vals, aqi_vals, hotspots_data):
        # Build interpolated grid - same verified approach as the matplotlib version
        grid_lon = np.linspace(68, 97, 300)
        grid_lat = np.linspace(8, 37, 300)
        gx, gy = np.meshgrid(grid_lon, grid_lat)
        gz = griddata((lon_vals, lat_vals), aqi_vals, (gx, gy), method='linear')
        gz = gaussian_filter(np.nan_to_num(gz, nan=np.nanmean(gz)), sigma=2)
        mask = shapely.contains_xy(india_shape, gx, gy)
        gz_masked = np.where(mask, gz, np.nan)

        vmin = np.nanpercentile(gz_masked, 5)
        vmax = np.nanpercentile(gz_masked, 95)

        colors_list = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b',
                       '#fdae61', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
        cmap = mcolors.LinearSegmentedColormap.from_list('aqi', colors_list)

        # Render to transparent PNG (outside India = transparent pixels)
        rgba = cmap((gz_masked - vmin) / (vmax - vmin))
        rgba[np.isnan(gz_masked)] = [0, 0, 0, 0]

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(rgba, origin='lower', extent=[68, 97, 8, 37], aspect='auto')
        ax.axis('off')
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    pad_inches=0, transparent=True)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)

        # Build Folium map with the image overlaid
        m = folium.Map(location=[22.5, 80], zoom_start=5,
                       tiles='CartoDB positron', min_zoom=4, max_zoom=10)

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[8, 68], [37, 97]],
            opacity=0.75,
            interactive=False,
            cross_origin=False
        ).add_to(m)

        # Clickable pins on top of the overlay
        for _, row in hotspots_data.iterrows():
            popup_html = f"""
            <div style="font-family:Arial; width:190px;">
                <b>📍 {row['region_name']}</b><br>
                Avg SPI: <b>{row['avg_aqi']:.0f}</b><br>
                Max SPI: <b>{row['max_aqi']:.0f}</b><br>
                Data points: {int(row['point_count'])}
            </div>
            """
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"📍 {row['region_name']} — click for details",
                icon=folium.Icon(color='darkred', icon='map-pin', prefix='fa')
            ).add_to(m)

        return m

    folium_map = build_folium_map(
        filtered['latitude'].values, filtered['longitude'].values,
        filtered['aqi'].values, hotspots
    )

    st.caption("🗺️ Fully interactive — zoom, pan, and click any pin for a popup with its stats.")
    map_output = st_folium(folium_map, width=None, height=550,
                           returned_objects=["last_object_clicked_tooltip"], key="hotspot_map")

    # Dropdown-based selection for the detailed 4-chart breakdown below
    st.markdown("""
    <div style="background:#f0f2f6; border-radius:10px; padding:12px 16px; margin-top:8px; margin-bottom:4px;">
    <b>👇 Want the full breakdown for a hotspot?</b> Pick it below — same pins shown on the map above.
    </div>
    """, unsafe_allow_html=True)
    region_options = [f"📍 {row['region_name']} (#{i+1})" for i, row in hotspots.iterrows()]

    # If the user clicked a pin on the live map, auto-select it in the dropdown too
    default_idx = st.session_state.get('selected_hotspot_idx', 0)
    clicked_tooltip = map_output.get("last_object_clicked_tooltip") if map_output else None
    if clicked_tooltip:
        for i, row in hotspots.iterrows():
            if row['region_name'] in clicked_tooltip:
                default_idx = i
                break

    selected_label = st.selectbox(
        "Select a hotspot:", region_options, index=default_idx,
        key="hotspot_selector", label_visibility="collapsed"
    )
    st.session_state.selected_hotspot_idx = region_options.index(selected_label)

    selected_region = hotspots.iloc[st.session_state.selected_hotspot_idx]['region_name']
    sel_row = hotspots.iloc[st.session_state.selected_hotspot_idx]
    national_avg = df['aqi'].mean()

    # ============================================
    # 4-Panel Detail Charts for the Selected Hotspot
    # ============================================
    st.markdown("<br>", unsafe_allow_html=True)

    detail_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Avg SPI — This Region vs National Average",
            "This Region — Average vs Peak SPI",
            "Sample Size vs Other Hotspots",
            "National Seasonal Pattern (Illustrative)"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.22, horizontal_spacing=0.12
    )

    detail_fig.add_trace(go.Bar(
        x=['This Region', 'India Average'], y=[sel_row['avg_aqi'], national_avg],
        marker_color=['#bd0026', '#66bd63'], showlegend=False,
        text=[f"{sel_row['avg_aqi']:.0f}", f"{national_avg:.0f}"], textposition='outside',
        textfont=dict(size=13, color='#1a1a2e'),
        hovertemplate="<b>%{x}</b><br>Avg SPI: %{y:.0f}<extra></extra>"
    ), row=1, col=1)

    detail_fig.add_trace(go.Bar(
        x=['Average', 'Peak (Max)'], y=[sel_row['avg_aqi'], sel_row['max_aqi']],
        marker_color=['#fdae61', '#800026'], showlegend=False,
        text=[f"{sel_row['avg_aqi']:.0f}", f"{sel_row['max_aqi']:.0f}"], textposition='outside',
        textfont=dict(size=13, color='#1a1a2e'),
        hovertemplate="<b>%{x}</b><br>SPI: %{y:.0f}<extra></extra>"
    ), row=1, col=2)

    bar_colors = ['#1a1a2e' if r == selected_region else '#cbd5e1' for r in hotspots['region_name']]
    detail_fig.add_trace(go.Bar(
        x=hotspots['region_name'], y=hotspots['point_count'],
        marker_color=bar_colors, showlegend=False,
        text=hotspots['point_count'], textposition='outside',
        textfont=dict(size=12, color='#1a1a2e'),
        hovertemplate="<b>%{x}</b><br>%{y} data points<extra></extra>"
    ), row=2, col=1)

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    seasonal_factor = [1.3, 1.2, 1.0, 0.85, 0.8, 0.75, 0.7, 0.75, 0.85, 1.4, 1.6, 1.35]
    monthly_vals = [national_avg * f for f in seasonal_factor]
    detail_fig.add_trace(go.Scatter(
        x=months, y=monthly_vals, mode='lines+markers',
        line=dict(color='#e8743b', width=3),
        marker=dict(size=7, color='#e8743b', line=dict(width=1, color='white')),
        fill='tozeroy', fillcolor='rgba(232,116,59,0.1)',
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>SPI: %{y:.0f}<extra></extra>"
    ), row=2, col=2)

    detail_fig.update_layout(
        height=640,
        title_text=f"📊 Detailed Breakdown — {selected_region}",
        title_font=dict(size=19, color='#1a1a2e', family='Arial Black'),
        showlegend=False,
        margin=dict(t=90, b=30, l=50, r=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#444'),
        hoverlabel=dict(bgcolor='white', font_size=13, bordercolor='#ccc')
    )
    detail_fig.update_yaxes(gridcolor='#eee', zeroline=False)
    detail_fig.update_xaxes(showgrid=False)
    detail_fig.update_annotations(font=dict(size=13, color='#555'))

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