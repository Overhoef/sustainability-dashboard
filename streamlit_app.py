import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as m
import matplotlib.cm as cm
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

top = df.sort_values(by = ['Average_rating'], ascending = True)
ehamtop = top.head(5) 

worst = df.sort_values(by = ['Average_rating'], ascending = False)
ehamworst = worst.head(5)

def assign_color(letter):
  """
  Assigns a color based on the letter.

  Args:
    letter: The letter to assign a color to.

  Returns:
    The color assigned to the letter.
  """
  colors = {
      'A': 'darkgreen',
      'B': 'green',
      'C': 'lime',
      'D': 'yellow',
      'E': 'orange',
      'F': 'purple',
      'G': 'pink'
  }
  return colors.get(letter, 'gray')  # Default to gray if letter is not found

# Create a new column with assigned colors
df['color'] = df['Overall_rating'].apply(assign_color)

# Get unique airports
unique_departure_airports = df['ADEP'].unique()
unique_destination_airports = df['ADES'].unique()

## Select boxes for departure and destination

st.title('🛫 Sustainability Dashboard 🛬')


# Select Boxes
box1, box2 = st.columns(2)

with box1:
    departure_airport = st.selectbox('Departure', ['All'] + list(unique_departure_airports))

with box2:
    destination_airport = st.selectbox('Destination', ['All'] + list(unique_destination_airports), index=1)

# Filters
filtercol1, filtercol2 = st.columns(2)

with filtercol1:
    rating_filter = st.checkbox('Ratings')
    occupancy_filter = st.checkbox('Occupancy rate')

with filtercol2:
    airline_filter = st.checkbox('Airlines')
    aircraft_filter = st.checkbox('Aircraft type')

# # Filter data based on selected airports
if departure_airport != 'All' and destination_airport != 'All':
        filtered_df = df[(df['ADEP'] == departure_airport) & (df['ADES'] == destination_airport)]
elif departure_airport == 'All' and destination_airport != 'All':
        filtered_df = df[df['ADES'] == destination_airport]
elif departure_airport != 'All' and destination_airport == 'All':
        filtered_df = df[df['ADEP'] == departure_airport]
else:
        # If nothing selected, show best 100, worst 100, and average 100
        best_100 = df.sort_values(by='Average_rating', ascending=True).head(100)
        worst_100 = df.sort_values(by='Average_rating', ascending=False).head(100)
        avg_100 = df.sample(n=100, random_state=42)  # Sample 100 random flights
        filtered_df = pd.concat([best_100, worst_100, avg_100])

    # Create a Plotly figure
fig = go.Figure()

    # Add scattermapbox traces for departure and arrival airports
fig.add_trace(go.Scattermapbox(
        lat=df['LATITUDE_ADES'],
        lon=df['LONGITUDE_ADES'],
        mode='markers',
        marker=dict(size=5, color='white'),
        text=df['IATA_ADES'],
        hoverinfo='text'
    ))
fig.add_trace(go.Scattermapbox(
        lat=df['LATITUDE_ADEP'],
        lon=df['LONGITUDE_ADEP'],
        mode='markers',
        marker=dict(size=5, color='white'),
        text=df['IATA_ADEP'],
        hoverinfo='text'
    ))

    # Add scattermapbox traces for routes using flight_id
if not filtered_df.empty:
    for _, row in filtered_df.iterrows():
        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lon=[row['LONGITUDE_ADEP'], row['LONGITUDE_ADES']],
            lat=[row['LATITUDE_ADEP'], row['LATITUDE_ADES']],
            line=dict(color='red', width=2),
            opacity=0.7,
            hoverinfo='text',
            hovertext=f"Route: {row['ADEP']} - {row['ADES']}",
            name=row['FLT_UID']  # Use flight_id for unique tracing
))

else:
    st.warning("No matching flights found for the selected criteria. Try broadening your search or selecting 'All' for departure and/or destination.")


fig.update_layout(
        showlegend=False,
        height=800,
        width=1200,
        mapbox=dict(
            style='carto-darkmatter',
            zoom=3,
            center=dict(lat=50, lon=20)  # EU centerd
        ),
    )
# Show the plot
st.plotly_chart(fig)

if st.button("Show/Hide Legend"):
    fig.update_layout(showlegend=not fig.layout.showlegend)
    st.plotly_chart(fig)

selected, export = st.columns([0.8, 0.2])

with selected:
    # Display filtered data with limited rows
    max_rows_to_show = 10  # Adjust as needed
    selected_columns = ['flight_id', 'ADEP', 'ADES', 'Average_rating', 'CO2_rating', 'NOx_rating']  # Select desired columns

    st.header('Filtered Flights:')
    if len(filtered_df) > max_rows_to_show:
        st.text(f"Showing first {max_rows_to_show} rows. Total rows: {len(filtered_df)}")
    st.table(filtered_df[selected_columns].head(max_rows_to_show)) 

with export:
    # Download button for filtered data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(filtered_df)
    
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv',
    )

text1, img1 = st.columns([0.7, 0.3])

with text1:
    st.subheader('How do the labels work?')
    st.write('Small explanation')

with img1:
    st.image('energielabels.png')

col1, col2 = st.columns(2)

with col1:
    st.header('Top 5 best flights')
    st.table(ehamtop)

with col2:
    st.header('Top 5 worst flights')
    st.table(ehamworst)

# Lay-out
st.sidebar.title("Introduction")
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
