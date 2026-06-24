import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    .chart-panel {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin-top: 16px;
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
# POLLUTION HEATMAP — UPDATED COLORS + PIN ICONS
# ============================================
st.markdown('<p class="section-header">📍 Pollution Heatmap of India</p>', unsafe_allow_html=True)
st.caption("Satellite Pollution Index (SPI) — continuous surface interpolated from filtered sample points")

if len(filtered) > 0:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union
    from streamlit_plotly_events import plotly_events

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

    # ── UPDATED COLOR SCALE: teal → yellow → orange → purple (no red) ──
    COLORSCALE = [
        [0.00, '#006837'],   # deep green  — Good
        [0.15, '#31a354'],   # green
        [0.30, '#78c679'],   # light green
        [0.42, '#c2e699'],   # pale green
        [0.50, '#ffffb2'],   # yellow       — Moderate
        [0.60, '#fecc5c'],   # amber
        [0.70, '#fd8d3c'],   # orange       — Unhealthy
        [0.80, '#f03b20'],   # orange-red
        [0.88, '#9e0142'],   # deep magenta — Very Unhealthy
        [1.00, '#5e0035'],   # dark purple  — Hazardous
    ]

    map_fig.add_trace(go.Contour(
        x=grid_lon, y=grid_lat, z=gz_masked,
        colorscale=COLORSCALE,
        contours=dict(coloring='fill', showlines=False),
        line_smoothing=0.85,
        colorbar=dict(
            title='SPI',
            thickness=15,
            tickvals=[50, 100, 150, 200, 250, 300, 350],
            ticktext=['50\nGood', '100\nMod', '150', '200\nUnhealthy', '250', '300\nV.Bad', '350\nHazard'],
            tickfont=dict(size=10)
        ),
        zmin=vmin, zmax=vmax,
        hoverinfo='skip'
    ))

    # State borders
    for geom in india_gdf.geometry:
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            if poly.area < 0.01:
                continue
            x, y = poly.exterior.xy
            map_fig.add_trace(go.Scatter(
                x=list(x), y=list(y), mode='lines',
                line=dict(color='#333333', width=0.8),
                showlegend=False, hoverinfo='skip'
            ))

    # ── LOCATION PIN MARKERS (circle + triangle-down to simulate pin) ──
    # Outer shadow (dark circle behind)
    map_fig.add_trace(go.Scatter(
        x=hotspots['lon'],
        y=hotspots['lat'],
        mode='markers',
        marker=dict(
            symbol='circle',
            size=30,
            color='#1a1a2e',
            line=dict(width=0)
        ),
        showlegend=False,
        hoverinfo='skip',
        name='pin_shadow'
    ))

    # Inner colored pin (based on pollution level)
    pin_colors = []
    for _, row in hotspots.iterrows():
        avg = row['avg_aqi']
        if avg > 280:
            pin_colors.append('#5e0035')   # dark purple
        elif avg > 220:
            pin_colors.append('#9e0142')   # magenta
        elif avg > 160:
            pin_colors.append('#fd8d3c')   # orange
        elif avg > 100:
            pin_colors.append('#fecc5c')   # amber
        else:
            pin_colors.append('#31a354')   # green

    # Build rich hover text for each hotspot
    hover_texts = []
    for _, row in hotspots.iterrows():
        avg = row['avg_aqi']
        if avg > 280:
            cat = "🟣 Hazardous"
        elif avg > 220:
            cat = "🔴 Very Unhealthy"
        elif avg > 160:
            cat = "🟠 Unhealthy"
        elif avg > 100:
            cat = "🟡 Moderate"
        else:
            cat = "🟢 Good"

        hover_texts.append(
            f"<b>📍 {row['region_name']}</b><br>"
            f"━━━━━━━━━━━━━━━━━<br>"
            f"📊 Avg SPI : <b>{row['avg_aqi']:.0f}</b><br>"
            f"⚠️ Max SPI : <b>{row['max_aqi']:.0f}</b><br>"
            f"🗂️ Data Points : {int(row['point_count'])}<br>"
            f"🏷️ Category : {cat}<br>"
            f"<i>Click to see detailed charts ↓</i>"
        )

    map_fig.add_trace(go.Scatter(
        x=hotspots['lon'],
        y=hotspots['lat'],
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=22,
            color=pin_colors,
            line=dict(width=2.5, color='white'),
        ),
        text=[str(i + 1) for i in range(len(hotspots))],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial Black'),
        customdata=hotspots['region_name'],
        hovertext=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='#1a1a2e',
            bordercolor='#444',
            font=dict(color='white', size=12),
            align='left'
        ),
        name='hotspots',
        showlegend=False
    ))

    map_fig.update_xaxes(range=[67.5, 97.5], visible=False)
    map_fig.update_yaxes(range=[7.5, 37.5], scaleanchor='x', scaleratio=1, visible=False)
    map_fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.caption("📍 Hover over a pin to see location stats. Click any pin to open detailed charts below.")
    clicked = plotly_events(map_fig, click_event=True, hover_event=False, override_height=620, key="hotspot_map")

    # Session state for selected hotspot
    if 'selected_hotspot_idx' not in st.session_state:
        st.session_state.selected_hotspot_idx = 0

    if clicked:
        point = clicked[0]
        clicked_x, clicked_y = point.get('x'), point.get('y')
        if clicked_x is not None and clicked_y is not None:
            distances = np.sqrt((hotspots['lon'] - clicked_x) ** 2 + (hotspots['lat'] - clicked_y) ** 2)
            nearest_idx = distances.idxmin()
            if distances[nearest_idx] < 1.5:
                st.session_state.selected_hotspot_idx = nearest_idx

    selected_idx   = st.session_state.selected_hotspot_idx
    sel_row        = hotspots.iloc[selected_idx]
    selected_region = sel_row['region_name']
    national_avg   = df['aqi'].mean()

    # ============================================
    # 4-PANEL DETAIL CHARTS (on click)
    # ============================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-header">📊 Detailed Analysis — 📍 {selected_region}</p>', unsafe_allow_html=True)

    detail_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"📊 {selected_region} vs National Average",
            f"⚠️ Avg vs Peak SPI — {selected_region}",
            "📍 All Hotspots — SPI Comparison",
            "📅 Seasonal Pollution Trend (India)"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.22,
        horizontal_spacing=0.12
    )

    # Chart 1 — This region vs national avg (bar)
    detail_fig.add_trace(go.Bar(
        x=[selected_region, 'India Average'],
        y=[sel_row['avg_aqi'], national_avg],
        marker_color=['#9e0142', '#31a354'],
        marker_line=dict(width=0),
        text=[f"{sel_row['avg_aqi']:.0f}", f"{national_avg:.0f}"],
        textposition='outside',
        textfont=dict(size=13, color='#1a1a2e'),
        showlegend=False,
        width=0.5
    ), row=1, col=1)

    # Chart 2 — Avg vs Peak (line+marker to look different from chart 1)
    detail_fig.add_trace(go.Scatter(
        x=['Average SPI', 'Peak SPI'],
        y=[sel_row['avg_aqi'], sel_row['max_aqi']],
        mode='lines+markers',
        line=dict(color='#fd8d3c', width=3),
        marker=dict(size=14, color=['#fecc5c', '#5e0035'],
                    line=dict(width=2, color='white')),
        text=[f"{sel_row['avg_aqi']:.0f}", f"{sel_row['max_aqi']:.0f}"],
        textposition='top center',
        textfont=dict(size=13, color='#1a1a2e'),
        showlegend=False
    ), row=1, col=2)

    # Chart 3 — All hotspots bar comparison (highlight selected)
    bar_colors = []
    for r in hotspots['region_name']:
        if r == selected_region:
            bar_colors.append('#9e0142')
        else:
            bar_colors.append('#c7c7c7')

    detail_fig.add_trace(go.Bar(
        x=hotspots['region_name'],
        y=hotspots['avg_aqi'],
        marker_color=bar_colors,
        marker_line=dict(width=0),
        text=[f"{v:.0f}" for v in hotspots['avg_aqi']],
        textposition='outside',
        textfont=dict(size=10),
        showlegend=False
    ), row=2, col=1)

    # Chart 4 — Seasonal trend line with crop burning highlight
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    seasonal_factor = [1.3, 1.2, 1.0, 0.85, 0.8, 0.75, 0.7, 0.75, 0.85, 1.4, 1.6, 1.35]
    monthly_vals = [national_avg * f for f in seasonal_factor]

    detail_fig.add_trace(go.Scatter(
        x=months,
        y=monthly_vals,
        mode='lines+markers',
        line=dict(color='#fd8d3c', width=2.5),
        marker=dict(
            size=9,
            color=monthly_vals,
            colorscale=COLORSCALE,
            cmin=vmin, cmax=vmax,
            line=dict(width=1.5, color='white')
        ),
        fill='tozeroy',
        fillcolor='rgba(253,141,60,0.15)',
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>SPI: %{y:.0f}<extra></extra>"
    ), row=2, col=2)

    # Crop burning annotation on chart 4
    detail_fig.add_vrect(
        x0=8.5, x1=10.5,
        fillcolor='rgba(158,1,66,0.12)',
        line_width=0,
        row=2, col=2
    )
    detail_fig.add_annotation(
        x=9.5, y=max(monthly_vals) * 1.05,
        text="🔥 Crop Burning",
        showarrow=False,
        font=dict(size=10, color='#9e0142'),
        row=2, col=2
    )

    detail_fig.update_layout(
        height=680,
        showlegend=False,
        margin=dict(t=80, l=20, r=20, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', color='#1a1a2e')
    )

    # Clean gridlines on all subplots
    for axis in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2', 'xaxis3', 'yaxis3', 'xaxis4', 'yaxis4']:
        detail_fig.update_layout(**{
            axis: dict(gridcolor='#f0f0f0', zeroline=False,
                       showline=True, linecolor='#e0e0e0')
        })

    st.plotly_chart(detail_fig, use_container_width=True)
    st.caption("⚠️ Seasonal panel shows national illustrative pattern based on known India pollution cycles.")

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
        if row['avg_aqi'] > 280:
            severity_color = "#5e0035"
            badge = "🟣 Hazardous"
        elif row['avg_aqi'] > 220:
            severity_color = "#9e0142"
            badge = "🔴 Very Unhealthy"
        elif row['avg_aqi'] > 160:
            severity_color = "#fd8d3c"
            badge = "🟠 Unhealthy"
        else:
            severity_color = "#fecc5c"
            badge = "🟡 Moderate"

        st.markdown(f"""
        <div class="hotspot-card" style="border-left-color:{severity_color}">
            <span class="hotspot-title">📍 {row['region_name']}</span>
            <span style="float:right;font-size:11px;color:{severity_color};font-weight:600">{badge}</span>
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
            columns={'region_name': 'Region', 'avg_aqi': 'Avg SPI',
                     'max_aqi': 'Max SPI', 'point_count': 'Data Points'}
        ),
        use_container_width=True,
        hide_index=True
    )

# ============================================
# SEASONAL TREND CHART
# ============================================
st.markdown("---")
st.markdown('<p class="section-header">📅 Seasonal Pollution Trend (2024)</p>', unsafe_allow_html=True)
st.caption("⚠️ Illustrative — based on known seasonal pollution patterns in India. Not derived from monthly satellite readings.")

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
np.random.seed(42)
base = df['aqi'].mean()
seasonal_factor = [1.3, 1.2, 1.0, 0.85, 0.8, 0.75, 0.7, 0.75, 0.85, 1.4, 1.6, 1.35]
monthly_spi = [round(base * f + np.random.normal(0, 5), 1) for f in seasonal_factor]

def spi_category(v):
    if v <= 50:   return 'Good'
    elif v <= 100: return 'Moderate'
    elif v <= 200: return 'Unhealthy'
    elif v <= 300: return 'Very Unhealthy'
    else:          return 'Hazardous'

monthly_cat = [spi_category(v) for v in monthly_spi]

fig2 = go.Figure()

fig2.add_vrect(
    x0=8.5, x1=11.5,
    fillcolor="rgba(158,1,66,0.10)", opacity=1, line_width=0,
    annotation_text="🔥 Crop Burning Season",
    annotation_position="top",
    annotation_font=dict(color="#9e0142", size=11)
)

fig2.add_trace(go.Scatter(
    x=months,
    y=monthly_spi,
    mode='lines+markers',
    line=dict(color='#fd8d3c', width=3),
    marker=dict(
        size=10,
        color=monthly_spi,
        colorscale=COLORSCALE,
        cmin=vmin, cmax=vmax,
        line=dict(width=1.5, color='white')
    ),
    fill='tozeroy',
    fillcolor='rgba(253,141,60,0.12)',
    customdata=monthly_cat,
    hovertemplate="<b>%{x}</b><br>SPI: %{y}<br>Category: %{customdata}<extra></extra>"
))

fig2.update_layout(
    title=dict(text="Monthly Pollution Trend — India (Illustrative)", font=dict(size=15)),
    yaxis_title="Average SPI",
    height=400,
    margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor='white',
    paper_bgcolor='white',
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
st.caption("Our satellite pollution index shows moderate positive correlation (r = 0.45) with CPCB ground station readings across 10 major Indian cities.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer-box">
Built for Bharatiya Antariksh Hackathon 2026 &nbsp;|&nbsp; Data: Sentinel-5P TROPOMI, CPCB &nbsp;|&nbsp; Team Code Vipers
</div>
""", unsafe_allow_html=True)