import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import requests

st.set_page_config(page_title="India Satellite Pollution Index Monitor", layout="wide")

st.title("🛰️ India Air Quality Monitor")
st.caption("Satellite-Powered Air Quality Intelligence — Bharatiya Antariksh Hackathon 2026")
st.caption("⚠️ AQI shown is a Satellite Pollution Index (0-500 scale), validated against CPCB ground stations (r=0.45). Not an official CPCB AQI value.")

# Load data
df = pd.read_csv('india_aqi_final.csv')
hotspots = pd.read_csv('hotspots_summary.csv')

# Sidebar filters
st.sidebar.header("Filters")
category_filter = st.sidebar.multiselect(
    "AQI Category",
    df['category'].unique().tolist(),
    default=df['category'].unique().tolist()
)
filtered = df[df['category'].isin(category_filter)]

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average AQI (India)", f"{df['aqi'].mean():.0f}")
col2.metric("Worst AQI Recorded", f"{df['aqi'].max():.0f}")
col3.metric("Hotspot Zones Found", len(hotspots))
col4.metric("Points Analyzed", len(df))

# Map section
st.subheader("📍 Pollution Heatmap of India")

@st.cache_data
def get_india_boundary():
    url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
    return requests.get(url).json()

india_boundary = get_india_boundary()

m = folium.Map(location=[20.5, 78.9], zoom_start=5, min_zoom=4, max_zoom=8)
heat_data = filtered[['latitude', 'longitude', 'aqi']].values.tolist()
HeatMap(heat_data, radius=15, blur=12,
        gradient={'0.2':'blue','0.4':'lime','0.6':'yellow','0.8':'orange','1.0':'red'}).add_to(m)

folium.GeoJson(
    india_boundary,
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1.5}
).add_to(m)

for _, row in hotspots.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"<b>{row['region_name']}</b><br>Avg AQI: {row['avg_aqi']:.0f}<br>Max AQI: {row['max_aqi']:.0f}",
        tooltip=f"{row['region_name']} — Hotspot",
        icon=folium.Icon(color='red', icon='exclamation-sign')
    ).add_to(m)

st_folium(m, width=1200, height=500)

# Hotspot table
st.subheader("🔥 Detected Pollution Hotspots")
st.dataframe(hotspots[['region_name','avg_aqi','max_aqi','point_count']].rename(
    columns={'region_name':'Region','avg_aqi':'Avg AQI','max_aqi':'Max AQI','point_count':'Data Points'}
), use_container_width=True)

# CPCB validation chart
st.subheader("✅ Validation: Satellite vs CPCB Ground Truth")
st.image('comparison_chart_final.png', use_container_width=True)
st.caption("Our satellite pollution index shows moderate positive correlation (r=0.45) with CPCB ground station readings across 10 major Indian cities.")

st.markdown("---")
st.caption("Built for Bharatiya Antariksh Hackathon 2026 | Data: Sentinel-5P TROPOMI, CPCB")
