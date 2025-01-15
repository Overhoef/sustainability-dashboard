import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as m
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import time

df = pd.read_csv('Sustainabilty_dashboard_2025.csv')

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.LONGITUDE_ADES, df.LATITUDE_ADES), crs="EPSG:4326"
)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.LONGITUDE_ADEP, df.LATITUDE_ADEP), crs="EPSG:4326"
)

eham = gdf[gdf['ADEP'] == 'EHAM']

eham_small = eham.sample(frac=0.2, random_state=200)

top = df.sort_values(by = ['Average_rating'], ascending = True)
ehamtop = top.head(5) 

worst = df.sort_values(by = ['Average_rating'], ascending = False)
ehamworst = worst.head(5)

# Create a Plotly figure
fig = go.Figure()

# # Add scattermapbox trace
# fig.add_trace(go.Scattermapbox(
#     lat=eham['LATITUDE_ADEP'],
#     lon=eham['LONGITUDE_ADEP'],
#     mode='markers',
#     marker=dict(size=10, color='black'),
#     text=eham['IATA_ADEP'],
#     hoverinfo='text'
# ))

# # Add scattermapbox trace
# fig.add_trace(go.Scattermapbox(
#     lat=eham['LATITUDE_ADES'],
#     lon=eham['LONGITUDE_ADES'],
#     mode='markers',
#     marker=dict(size=10, color='black'),
#     text=eham['IATA_ADES'],
#     hoverinfo='text'
# ))

# #fig = px.line_mapbox(eham, lat=eham['LATITUDE_ADEP'], lon=eham['LONGITUDE_ADEP'], color=eham[''], zoom=3, height=300)

# # Set mapbox layout
# fig.update_layout(
#     mapbox=dict(
#         style='open-street-map', 
#         center=dict(lat=52, lon=16),  # Centered around the US for better initial view
#         zoom=2
#     ),
#     width=1400,
#     height=800,
#     title='World map'
# )

for _,row in eham_small.iterrows():
    fig.add_trace(go.Scattermapbox(mode='lines',
                                   lon=[row['LONGITUDE_ADES'], row['LONGITUDE_ADEP']],
                                   lat=[row['LATITUDE_ADES'], row['LATITUDE_ADEP']],
                                   line_color='green',
                                   name=row['AIRCRAFT_ID']
                                  ))
fig.update_layout(
    height=600,
    mapbox=dict(
        style='open-street-map',
        zoom=4,
        center=dict(lat=52, lon=16))
)

# Lay-out
st.sidebar.header("Introduction")
st.sidebar.write("""
In this project, we address the pressing issue of environmental impact in aviation by creating a labeling system to evaluate the sustainability of airline routes. 
Inspired by established practices in emissions labeling, we assigned sustainability grades (from **A to G**) to various routes and airlines based on their environmental performance.
""")

# Key Metrics Section
st.sidebar.header("Key Metrics")
st.sidebar.write("""
Our labeling system is built upon three core ratings:
- **NOx Rating:** Evaluates the impact on air quality and environmental health caused by nitrogen oxide emissions.
- **CO₂ Rating:** Reflects the contribution of carbon dioxide emissions to global climate change.
- **Fuel Flow Rating:** Assesses fuel efficiency, measured as fuel flow per kilometer, to determine overall energy usage.

By combining these metrics, we calculated an average rating that determines each route’s overall sustainability grade, making it easier to compare different options.
""")

# Goal Section
st.sidebar.header("Our Goal")
st.sidebar.write("""
Offer actionable insights to **municipalities, governments, and airports** to support data-driven policymaking and promote sustainability initiatives.
""")

# Features Section
st.sidebar.header("Features of Our Tool")
st.sidebar.write("""
- **Dynamic Labels:** Each route is assigned a sustainability grade (A = Most sustainable, G = Least sustainable) based on its combined NOx, CO₂, and fuel flow ratings.
- **Comparative Insights:** Analyze how airlines perform on the same route to identify the most sustainable operators.
- **Geographical Analysis:** Discover how sustainability varies across different regions and routes.
- **Interactive Visualizations:** Explore trends, such as the influence of aircraft type, engine model, and flight distance on sustainability.
""")

# Why It Matters Section
st.sidebar.header("Why It Matters")
st.sidebar.write("""
Aviation is a significant contributor to greenhouse gas emissions, and as demand for air travel grows, addressing its environmental impact becomes increasingly important. 
By making sustainability metrics transparent, this tool aligns with global efforts to create a greener and more sustainable aviation industry.

We invite you to explore the labels, compare airlines and routes, and gain insights into the environmental impact of air travel. Together, we can work toward a more sustainable future for aviation! 🚀
""")


st.title('🛫 Sustainability Dashboard 🛬')

box1, box2 = st.columns(2)

with box1:
    st.selectbox('departure', df['IATA_ADEP'], index = 5)

with box2:
    st.selectbox('destination', df['IATA_ADES'])

filter1, filter2, filter3, filter4 = st.columns(4)

with filter1:
    st.checkbox('Ratings')

with filter2:
    st.checkbox('Airlines')

with filter3:
    st.checkbox('Occupancy rate')

with filter4:
    st.checkbox('Nog een filter bedenken??')

# Show the plot
st.plotly_chart(fig)

text1, img1 = st.columns([0.7, 0.3])

with text1:
    st.subheader('uitleg labels')
    st.write('kleine uitleg van de labels')

with img1:
    st.image('energielabels.png')

st.header('Vluchten op gekozen route:')
st.table(ehamtop)

col1, col2 = st.columns(2)

with col1:
    st.header('Top 5 beste vluchten')
    st.table(ehamtop)

with col2:
    st.header('Top 5 slechtste vluchten')
    st.table(ehamworst)
