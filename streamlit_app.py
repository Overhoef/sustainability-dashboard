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


# Config
st.set_page_config(layout="wide")  # Set wide mode as default

# Cache the data loading function
@st.cache_data
def load_data():
    df = pd.read_csv('Sustainabilty_dashboard_2025.csv')
    return df

# Load data using the cached function
df = load_data()

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
      'C': 'limegreen',
      'D': 'yellow',
      'E': 'orange',
      'F': 'purple',
      'G': 'pink'
  }
  return colors.get(letter, 'gray')  # Default to gray if letter is not found

# Create a new column with assigned colors
df['color'] = df['Overall_rating'].apply(assign_color)

# Create an airline color dictionary with unique colors
airline_colors = {
    'Air France': '#0077C0',  # Blue
    'American Airlines': '#FF0000',  # Red
    'British Airways': '#0000FF',  # Blue
    'Delta Air Lines': '#008000',  # Green
    'Emirates': '#FFA500',  # Orange
    'Etihad Airways': '#800080',  # Purple
    'Lufthansa': '#FFFF00',  # Yellow
    'Qatar Airways': '#00FFFF',  # Cyan
    'Singapore Airlines': '#FF00FF',  # Magenta
    'United Airlines': '#C0C0C0',  # Silver
    'Southwest Airlines': '#FFA07A',  # Light Salmon
    'Turkish Airlines': '#A020F0',  # Purple
    'KLM': '#00BFFF',  # Deep Sky Blue
    'Iberia': '#FFC0CB',  # Pink
    'Air Canada': '#8B0000',  # Dark Red
    'ANA All Nippon Airways': '#008B8B',  # Teal
    'Japan Airlines': '#A52A2A',  # Brown
    'Korean Air': '#FFFF00',  # Yellow
    'China Southern Airlines': '#008080',  # Teal
    'Air China': '#800080',  # Purple
    'Cathay Pacific': '#00008B',  # Dark Blue
    'Qantas': '#F0E68C',  # Khaki
    'Finnair': '#008080',  # Teal
    'SAS': '#FF4500',  # Orange Red
    'Norwegian Air': '#008B8B',  # Teal
    'Vueling': '#FFA500',  # Orange
    'Austrian Airlines': '#FFFF00',  # Yellow
    'Swiss International Air Lines': '#FF0000',  # Red
    'Brussels Airlines': '#0000FF',  # Blue
    'Czech Airlines': '#008000',  # Green
    'LOT Polish Airlines': '#FFFF00',  # Yellow
    'Aer Lingus': '#8B0000',  # Dark Red
    'Icelandair': '#0000FF',  # Blue
    'TAP Air Portugal': '#008000',  # Green
    'Scandinavian Airlines': '#FF0000',  # Red
    'Edelweiss Air': '#0000FF',  # Blue
    'Eurowings': '#008000',  # Green
    'Transavia': '#FFFF00',  # Yellow
    'easyJet': '#FFA500',  # Orange
    'Vueling': '#FF0000',  # Red
    'Ryanair': '#008000',  # Green
    'Wizz Air': '#800080',  # Purple
    # Add more airlines and their corresponding colors here
}

aircraft_colors = {
    'A319-100': '#C0C0C0', 
    'A320-200': '#FFA07A',  
    'A321-200': '#F08080', 
    # ... (Add more aircraft variants and unique colors) 
    'A330-300': '#FFC0CB', 
    'A330-900': '#8B0000', 
    'A350-900': '#008B8B', 
    'A350-1000': '#A52A2A', 
    'A380-800': '#FFFF00', 
    'B737-800': '#008080', 
    'B737-MAX 8': '#800080', 
    'B737-700': '#00008B', 
    'B737-900': '#F0E68C', 
    'B737-MAX 9': '#008080', 
    'B737-MAX 10': '#FF0000', 
    'B747-400': '#0000FF', 
    'B747-8': '#008000', 
    'B747-8F': '#FFFF00', 
    'B757-200': '#00FFFF', 
    'B757-300': '#FF00FF', 
    'B767-300': '#C0C0C0', 
    'B767-300ER': '#FFA07A', 
    'B767-400ER': '#90EE90', 
    'B777-200': '#F08080', 
    'B777-200ER': '#800080', 
    'B777-300': '#A020F0', 
    'B777-300ER': '#00BFFF', 
    'B777-9': '#FFC0CB', 
    'B777F': '#8B0000', 
    'B787-8': '#008B8B', 
    'B787-9': '#A52A2A', 
    'B787-10': '#FFFF00', 
    'E170': '#008080', 
    'E175': '#800080', 
    'E190': '#00008B', 
    'E195': '#F0E68C', 
    'CRJ-900': '#008080', 
    'CRJ-700': '#FF0000', 
    'CRJ-200': '#0000FF', 
    'ATR 72': '#008000', 
    'ATR 42': '#FFFF00', 
    # ... (Add more aircraft variants and unique colors)
}
# Get unique airports
unique_departure_airports = df['ADEP'].unique()
unique_destination_airports = df['ADES'].unique()

# Streamlit Page
st.title('🛫 Sustainability Dashboard 🛬')


# Select Boxes
box1, box2 = st.columns(2)

with box1:
    departure_airport = st.selectbox('Departure', ['All'] + list(unique_departure_airports))

with box2:
    destination_airport = st.selectbox('Destination', ['All'] + list(unique_destination_airports), )

# # Filter data based on selected airports
if departure_airport != 'All' and destination_airport != 'All':
        filtered_df = df[(df['ADEP'] == departure_airport) & (df['ADES'] == destination_airport)]
else:
    # If nothing selected, show best 100, worst 100, and average 100
    best_100 = df.sort_values(by='Average_rating', ascending=True).head(100)
    worst_100 = df.sort_values(by='Average_rating', ascending=False).head(100)
    avg_100 = df.sample(n=100, random_state=42)  # Sample 100 random flights
    filtered_df = pd.concat([best_100, worst_100, avg_100])

# Filters
filtercol1, filtercol2 = st.columns(2)

with filtercol1:
    rating_filter = st.checkbox('Ratings')
    density_filter = st.checkbox('Loadfactor Density')

with filtercol2:
    airline_filter = st.checkbox('Airlines')
    aircraft_filter = st.checkbox('Aircraft type')

# mapfilter = st.radio('Filters:', ["None", "Ratings","Loadfactor","Airlines","Aircraft type"], horizontal=True)

# Calculate route-specific average load factor
route_avg_load_factors = filtered_df.groupby(['ADEP', 'ADES'])['Loadfactor'].mean()

    # Create a Plotly figure
fig = go.Figure()

    # Add scattermapbox traces for departure and arrival airports
fig.add_trace(go.Scattermapbox(
    lat=df['LATITUDE_ADES'],
    lon=df['LONGITUDE_ADES'],
    mode='markers',
    marker=dict(size=5, color='white'),
    text=df['ADES'],
    hoverinfo='text'
))

fig.add_trace(go.Scattermapbox(
    lat=df['LATITUDE_ADEP'],
    lon=df['LONGITUDE_ADEP'],
    mode='markers',
    marker=dict(size=5, color='white'),
    text=df['ADEP'],
    hoverinfo='text'
))

line_color = 'gray' 

if rating_filter:
    # Apply rating filter and assign colors accordingly
    filtered_df['color'] = filtered_df['Overall_rating'].apply(assign_color) 
    line_color = filtered_df['color']
else:
    # Default to gray for all lines when filter is not applied
    filtered_df['color'] = 'gray'


    # Add scattermapbox traces for routes using flight_id
for _, row in filtered_df.iterrows():
    # Get route-specific average load factor
    route = (row['ADEP'], row['ADES'])
    avg_load_factor = route_avg_load_factors.get(route) 

    # DENSITY FILTER 
    if density_filter:
        # Normalize load factor to a range of 0 to 1 
        normalized_load_factor = (avg_load_factor - filtered_df['Loadfactor'].min()) / (filtered_df['Loadfactor'].max() - filtered_df['Loadfactor'].min()) 
        # Create a color based on load factor using a blue-to-yellow gradient
        line_color = plt.cm.viridis(normalized_load_factor)
        line_color = f'rgb({int(line_color[0]*255)},{int(line_color[1]*255)},{int(line_color[2]*255)})' 

    # AIRLINE FILTER
    if airline_filter and row['Airline'] in airline_colors: 
        line_color = airline_colors.get(row['Airline']) 

    # AIRCRAFT FILTER
    if aircraft_filter and row['Aircraft Variant'] in aircraft_colors: 
        line_color = aircraft_colors.get(row['Aircraft Variant']) 

    fig.add_trace(go.Scattermapbox(
        mode='lines',
        lon=[row['LONGITUDE_ADES'], row['LONGITUDE_ADEP']],
        lat=[row['LATITUDE_ADES'], row['LATITUDE_ADEP']],
        line=dict(color=line_color, width=1),
        opacity=0.4,
        text=f"Route: {row['ADEP']} - {row['ADES']}<br>Avg. Load Factor: {avg_load_factor:.2f}%.<br> {row['Airline']}.<br> {row['Aircraft Variant']} <br> {row['Overall_rating']}",
        # name=row['FLT_UID'],  # Use flight_id for unique tracing
        legendgroup=row['Overall_rating'],  # Group traces by rating for legend
        name=row['Overall_rating']  # Use rating as legend label
))

fig.update_layout(
        # showlegend=False,
        height=800,
        width=1200,
        mapbox=dict(
            style='carto-darkmatter',
            zoom=3,
            center=dict(lat=50, lon=20),  # EU centered
            # projection=dict(type='equirectangular')
    ),
    )

# Show the plot
st.plotly_chart(fig)

selected, export = st.columns([0.8, 0.2])

with selected:
    # Display filtered data with limited rows
    max_rows_to_show = 5  # Adjust as needed
    selected_columns = ['FLT_UID', 'NAME_ADEP', 'ADEP', 'NAME_ADES', 'ADES', 'Airline', 'Aircraft Variant', 'REGISTRATION', 'Flight Time', 'Distance (km)','CO2 rating', 'NOx rating', 'Fuel Flow rating', 'Overall_rating']  # Select desired columns

    st.header('Filtered Flights:')
    if len(filtered_df) > max_rows_to_show:
        st.text(f"Showing select amount of columns (most important ones) and the first {max_rows_to_show} rows. Total rows: {len(filtered_df)}")
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
    st.table(filtered_df[selected_columns].head(5))

with col2:
    st.header('Top 5 worst flights')
    filtered_df.sort_values(by = ['Average_rating'], ascending = False)
    st.table(filtered_df[selected_columns].tail(5))

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
