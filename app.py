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
</style>
""", unsafe_allow_html=True)

# ============================================
# COLOR SCALE (shared across all charts)
# ============================================
COLORSCALE = [
    [0.00, '#006837'],
    [0.15, '#31a354'],
    [0.30, '#78c679'],
    [0.42, '#c2e699'],
    [0.50, '#ffffb2'],
    [0.60, '#fecc5c'],
    [0.70, '#fd8d3c'],
    [0.80, '#f03b20'],
    [0.88, '#9e0142'],
    [1.00, '#5e0035'],
]

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('india_aqi_final.csv')
    hotspots = pd.read_csv('hotspots_summary.csv')
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    hotspots.columns = [c.strip().lower() for c in hotspots.columns]
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
    ("Worst SPI Recorded",  f"{filtered['aqi'].max():.0f}"  if len(filtered) else "—"),
    ("Hotspot Zones Found", f"{len(hotspots)}"),
    ("Points Analyzed",     f"{len(filtered):,}")
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
# POLLUTION HEATMAP
# ============================================
st.markdown('<p class="section-header">📍 Pollution Heatmap of India</p>', unsafe_allow_html=True)
st.caption("Satellite Pollution Index (SPI) — continuous surface interpolated from filtered sample points")

if len(filtered) > 0:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import geopandas as gpd
    from shapely.ops import unary_union
    from streamlit_plotly_events import plotly_events

    # ── Load India boundary ──
    @st.cache_data
    def get_india_geometry():
        gdf = gpd.read_file('india_boundary_simplified.geojson')
        return gdf, unary_union(gdf.geometry)

    india_gdf, india_shape = get_india_geometry()

    # ── Build interpolated grid ──
    # NOTE: No @st.cache_data here — numpy arrays cause cache hash issues
    def build_heatmap_grid(lat_vals, lon_vals, aqi_vals, _india_shape):
        grid_lon = np.linspace(68, 97, 300)
        grid_lat = np.linspace(8, 37, 300)
        gx, gy = np.meshgrid(grid_lon, grid_lat)

        # Interpolate: cubic for smooth surface, linear as fallback
        try:
            gz = griddata((lon_vals, lat_vals), aqi_vals, (gx, gy), method='cubic')
            gz_lin = griddata((lon_vals, lat_vals), aqi_vals, (gx, gy), method='linear')
            gz = np.where(np.isnan(gz), gz_lin, gz)
        except Exception:
            gz = griddata((lon_vals, lat_vals), aqi_vals, (gx, gy), method='linear')

        mean_val = float(np.nanmean(gz)) if not np.all(np.isnan(gz)) else 150.0
        gz = np.nan_to_num(gz, nan=mean_val)
        gz = gaussian_filter(gz, sigma=3)

        # Mask outside India — try all available shapely APIs
        try:
            from shapely.vectorized import contains
            mask = contains(_india_shape, gx.ravel(), gy.ravel()).reshape(gx.shape)
        except Exception:
            try:
                import shapely
                mask = shapely.contains_xy(_india_shape, gx, gy)
            except Exception:
                try:
                    from shapely.geometry import Point
                    flat_x = gx.ravel()
                    flat_y = gy.ravel()
                    mask = np.array([
                        _india_shape.contains(Point(float(flat_x[i]), float(flat_y[i])))
                        for i in range(len(flat_x))
                    ]).reshape(gx.shape)
                except Exception:
                    # Final fallback: rough bounding box of India
                    mask = (
                        (gx >= 68) & (gx <= 97) &
                        (gy >= 8)  & (gy <= 37)
                    )

        gz_masked = np.where(mask, gz, np.nan)
        return grid_lon, grid_lat, gz_masked

    grid_lon, grid_lat, gz_masked = build_heatmap_grid(
        filtered['latitude'].values,
        filtered['longitude'].values,
        filtered['aqi'].values,
        india_shape
    )

    vmin = float(np.nanpercentile(gz_masked, 5))  if not np.all(np.isnan(gz_masked)) else 50
    vmax = float(np.nanpercentile(gz_masked, 95)) if not np.all(np.isnan(gz_masked)) else 400

    # ── Build map figure ──
    map_fig = go.Figure()

    # 1. Heatmap layer (reliable — no NaN masking issues like Contour)
    map_fig.add_trace(go.Heatmap(
        x=grid_lon,
        y=grid_lat,
        z=gz_masked,
        colorscale=COLORSCALE,
        zmin=vmin, zmax=vmax,
        zsmooth='best',
        opacity=0.85,
        colorbar=dict(
            title=dict(text='SPI', side='right'),
            thickness=15,
            tickfont=dict(size=10),
            tickvals=[50, 100, 150, 200, 250, 300, 350],
            ticktext=['50 Good', '100 Mod', '150', '200 Bad', '250', '300 V.Bad', '350 Haz'],
        ),
        hoverinfo='skip',
        showscale=True,
    ))

    # 2. State borders
    for geom in india_gdf.geometry:
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for poly in polys:
            if poly.area < 0.01:
                continue
            x, y = poly.exterior.xy
            map_fig.add_trace(go.Scatter(
                x=list(x), y=list(y), mode='lines',
                line=dict(color='#222222', width=1.0),
                showlegend=False, hoverinfo='skip'
            ))

    # 3. Pin color by pollution level
    pin_colors = []
    for _, row in hotspots.iterrows():
        avg = row['avg_aqi']
        if avg > 280:   pin_colors.append('#5e0035')
        elif avg > 220: pin_colors.append('#9e0142')
        elif avg > 160: pin_colors.append('#fd8d3c')
        elif avg > 100: pin_colors.append('#fecc5c')
        else:           pin_colors.append('#31a354')

    # 4. Hover text for each pin
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

    # 5. Shadow circle (dark border effect)
    map_fig.add_trace(go.Scatter(
        x=hotspots['lon'], y=hotspots['lat'],
        mode='markers',
        marker=dict(symbol='circle', size=30, color='#1a1a2e', line=dict(width=0)),
        showlegend=False, hoverinfo='skip', name='pin_bg'
    ))

    # 6. Colored pin with number + hover
    map_fig.add_trace(go.Scatter(
        x=hotspots['lon'], y=hotspots['lat'],
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

    map_fig.update_xaxes(range=[67.5, 97.5], visible=False)
    map_fig.update_yaxes(range=[7.5, 37.5], scaleanchor='x', scaleratio=1, visible=False)
    map_fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.caption("📍 Hover over a pin to see stats. Click any pin to open detailed charts below.")
    clicked = plotly_events(
        map_fig, click_event=True, hover_event=False,
        override_height=620, key="hotspot_map"
    )

    # ── Session state for selected hotspot ──
    if 'selected_hotspot_idx' not in st.session_state:
        st.session_state.selected_hotspot_idx = 0

    if clicked:
        point = clicked[0]
        cx, cy = point.get('x'), point.get('y')
        if cx is not None and cy is not None:
            distances = np.sqrt((hotspots['lon'] - cx) ** 2 + (hotspots['lat'] - cy) ** 2)
            nearest_idx = distances.idxmin()
            if distances[nearest_idx] < 1.5:
                st.session_state.selected_hotspot_idx = nearest_idx

    sel_idx         = st.session_state.selected_hotspot_idx
    sel_row         = hotspots.iloc[sel_idx]
    selected_region = sel_row['region_name']
    national_avg    = df['aqi'].mean()

    # ============================================
    # 4-PANEL DETAIL CHARTS
    # ============================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-header">📊 Detailed Analysis — 📍 {selected_region}</p>', unsafe_allow_html=True)

    detail_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"📊 {selected_region} vs National Average",
            f"⚠️ Avg vs Peak SPI",
            "📍 All Hotspots Comparison",
            "📅 Seasonal Trend (India)"
        ),
        specs=[
            [{"type": "bar"},     {"type": "scatter"}],
            [{"type": "bar"},     {"type": "scatter"}]
        ],
        vertical_spacing=0.22,
        horizontal_spacing=0.12
    )

    # Chart 1 — Region vs National (bar)
    detail_fig.add_trace(go.Bar(
        x=[selected_region, 'India Avg'],
        y=[sel_row['avg_aqi'], national_avg],
        marker_color=['#9e0142', '#31a354'],
        marker_line=dict(width=0),
        text=[f"{sel_row['avg_aqi']:.0f}", f"{national_avg:.0f}"],
        textposition='outside',
        textfont=dict(size=13),
        showlegend=False, width=0.4
    ), row=1, col=1)

    # Chart 2 — Avg vs Peak (line)
    detail_fig.add_trace(go.Scatter(
        x=['Average SPI', 'Peak SPI'],
        y=[sel_row['avg_aqi'], sel_row['max_aqi']],
        mode='lines+markers',
        line=dict(color='#fd8d3c', width=3),
        marker=dict(size=14, color=['#fecc5c', '#5e0035'],
                    line=dict(width=2, color='white')),
        text=[f"{sel_row['avg_aqi']:.0f}", f"{sel_row['max_aqi']:.0f}"],
        textposition='top center',
        textfont=dict(size=13),
        showlegend=False
    ), row=1, col=2)

    # Chart 3 — All hotspots bar (highlight selected)
    bar_colors = [
        '#9e0142' if r == selected_region else '#c7c7c7'
        for r in hotspots['region_name']
    ]
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

    # Chart 4 — Seasonal line with crop burning zone
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    seasonal_factor = [1.3,1.2,1.0,0.85,0.8,0.75,0.7,0.75,0.85,1.4,1.6,1.35]
    monthly_vals = [national_avg * f for f in seasonal_factor]

    detail_fig.add_trace(go.Scatter(
        x=months, y=monthly_vals,
        mode='lines+markers',
        line=dict(color='#fd8d3c', width=2.5),
        marker=dict(size=8, color='#fd8d3c', line=dict(width=1.5, color='white')),
        fill='tozeroy',
        fillcolor='rgba(253,141,60,0.15)',
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>SPI: %{y:.0f}<extra></extra>"
    ), row=2, col=2)

    detail_fig.add_vrect(
        x0=8.5, x1=10.5,
        fillcolor='rgba(158,1,66,0.12)',
        line_width=0, row=2, col=2
    )
    detail_fig.add_annotation(
        x=9.5, y=max(monthly_vals) * 1.08,
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
    for ax in ['xaxis','yaxis','xaxis2','yaxis2','xaxis3','yaxis3','xaxis4','yaxis4']:
        detail_fig.update_layout(**{ax: dict(gridcolor='#f0f0f0', zeroline=False)})

    st.plotly_chart(detail_fig, use_container_width=True)
    st.caption("⚠️ Seasonal panel shows national illustrative pattern based on known India pollution cycles.")

else:
    st.warning("No data points match the selected filters.")

# ============================================
# HOTSPOT CARDS
# ============================================
st.markdown("---")
st.markdown('<p class="section-header">🔥 Detected Pollution Hotspots</p>', unsafe_allow_html=True)

cols = st.columns(3)
for idx, row in hotspots.iterrows():
    with cols[idx % 3]:
        avg = row['avg_aqi']
        if avg > 280:
            sc, badge = '#5e0035', '🟣 Hazardous'
        elif avg > 220:
            sc, badge = '#9e0142', '🔴 Very Unhealthy'
        elif avg > 160:
            sc, badge = '#fd8d3c', '🟠 Unhealthy'
        else:
            sc, badge = '#fecc5c', '🟡 Moderate'

        st.markdown(f"""
        <div class="hotspot-card" style="border-left-color:{sc}">
            <span class="hotspot-title">📍 {row['region_name']}</span>
            <span style="float:right;font-size:11px;color:{sc};font-weight:600">{badge}</span>
            <div class="hotspot-stat">Avg SPI: <b>{row['avg_aqi']:.0f}</b> &nbsp;|&nbsp; Max SPI: <b>{row['max_aqi']:.0f}</b></div>
            <div class="hotspot-stat">{int(row['point_count'])} satellite data points</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# HOTSPOT TABLE
# ============================================
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 View full hotspot data table"):
    st.dataframe(
        hotspots[['region_name','avg_aqi','max_aqi','point_count']].rename(
            columns={'region_name':'Region','avg_aqi':'Avg SPI',
                     'max_aqi':'Max SPI','point_count':'Data Points'}
        ),
        use_container_width=True, hide_index=True
    )

# ============================================
# SEASONAL TREND
# ============================================
st.markdown("---")
st.markdown('<p class="section-header">📅 Seasonal Pollution Trend (2024)</p>', unsafe_allow_html=True)
st.caption("⚠️ Illustrative — based on known seasonal pollution patterns in India.")

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
np.random.seed(42)
base = df['aqi'].mean()
seasonal_factor = [1.3,1.2,1.0,0.85,0.8,0.75,0.7,0.75,0.85,1.4,1.6,1.35]
monthly_spi = [round(base * f + np.random.normal(0, 5), 1) for f in seasonal_factor]

def spi_category(v):
    if v <= 50:    return 'Good'
    elif v <= 100: return 'Moderate'
    elif v <= 200: return 'Unhealthy'
    elif v <= 300: return 'Very Unhealthy'
    else:          return 'Hazardous'

monthly_cat = [spi_category(v) for v in monthly_spi]

fig2 = go.Figure()
fig2.add_vrect(
    x0=8.5, x1=11.5,
    fillcolor="rgba(158,1,66,0.10)", line_width=0,
    annotation_text="🔥 Crop Burning Season",
    annotation_position="top",
    annotation_font=dict(color="#9e0142", size=11)
)
fig2.add_trace(go.Scatter(
    x=months, y=monthly_spi,
    mode='lines+markers',
    line=dict(color='#fd8d3c', width=3),
    marker=dict(size=10, color='#fd8d3c', line=dict(width=1.5, color='white')),
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
# CPCB VALIDATION
# ============================================
st.markdown("---")
st.markdown('<p class="section-header">✅ Validation: Satellite vs CPCB Ground Truth</p>', unsafe_allow_html=True)
st.image('comparison_chart_final.png', use_container_width=True)
st.caption("Satellite pollution index shows moderate positive correlation (r = 0.45) with CPCB ground station readings across 10 major Indian cities.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer-box">
Built for Bharatiya Antariksh Hackathon 2026 &nbsp;|&nbsp; Data: Sentinel-5P TROPOMI, CPCB &nbsp;|&nbsp; Team Code Vipers
</div>
""", unsafe_allow_html=True)