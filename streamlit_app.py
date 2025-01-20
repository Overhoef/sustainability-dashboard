import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as m
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import time
import sys


# Config
st.set_page_config(layout="wide")  # Set wide mode as default

# Data
@st.cache_data
def load_data():
        df = pd.read_csv('Sustainabilty_dashboard_2025.csv', usecols=[
        'ADEP', 
        'ADES', 
        'AIRCRAFT_ID',
        'Operator',
        'Aircraft Variant',
        'Average_rating',
        'Distance (km)',
        'Flight Time', 
        'FLT_UID', 
        'Overall_rating', 
        'NAME_ADEP', 
        'LONGITUDE_ADEP',
        'LATITUDE_ADEP',
        'NAME_ADES', 
        'LATITUDE_ADES',
        'LONGITUDE_ADES',
        'Loadfactor'    
    ]) # pd.read_csv('Sustainabilty_dashboard_2025.csv')
            return df

df = load_data()

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.LONGITUDE_ADES, df.LATITUDE_ADES), crs="EPSG:4326"
)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.LONGITUDE_ADEP, df.LATITUDE_ADEP), crs="EPSG:4326"
)

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
'Emirates Airline' : "#e61919",
'Delta Air Lines': "#e62519",
'KLM Royal Dutch Airlines' : "#e63119",
'Aeromexico' : "#e63c19",
'Air Transat' : "#e64819",
'Air Canada' : "#e65319",
'Transavia' : "#e65f19",
'Corendon Dutch Airlines' : "#e66a19",
'Xiamen Airlines' : "#e67619",
'China Airlines' : "#e68119",
'Georgian Airways' : "#e68d19",
'Korean Air' : "#e69819",
'Suparna Airlines' : "#e6a419",
'Cathay Pacific' : "#e6af19",
'Bulgaria Air' : "#e6bb19",
'easyJet Europe' : "#e6c619",
'Qatar Airways' : "#e6d219",
'Turkish Airlines': "#e6dd19",
'Corendon Airlines' : "#e2e619",
'airBaltic' : "#d7e619",
'Lufthansa' : "#cbe619",
'LOT Polish Airlines' : "#c0e619",
'easyJet' : "#b4e619",
'TUI Airlines Netherlands' : "#a9e619",
'Austrian Airlines' : "#9de619",
'Vueling': "#92e619",
'TAP Air Portugal' : "#86e619",
'Italia Trasporto Aereo (ITA Airways)' : "#7be619",
'BA CityFlyer' : "#6fe619",
'China Southern Airlines' : "#64e619",
'Air France' : "#58e619",
'Ryanair' : "#4de619",
'Air Europa Líneas Aéreas' : "#41e619",
'Kuwait Airways' : "#36e619",
'Finnair' : "#2ae619",
'Aer Lingus' : "#1ee619",
'SWISS' : "#19e620",
'Aegean Airlines' : "#19e62c",
'Pegasus Airlines' : "#19e637",
'Air Serbia' : "#19e643",
'EVA Air' : "#19e64e",
'Air Malta' : "#19e65a",
'TAROM' : "#19e665",
'BA Euroflyer' : "#19e671",
'British Airways' : "#19e67c",
'Etihad Airways' : "#19e688",
'Royal Air Maroc'  : "#19e693",
'SAS' : "#19e69f",
'Croatia Airlines' : "#19e6aa",
'Norwegian Air Sweden AOC' : "#19e6b6",
'Iberia Express' : "#19e6c1",
'Royal Jordanian' : "#19e6cd",
'Icelandair' : "#19e6d8",
'EgyptAir' : "#19e6e4",
'American Airlines' : "#19dce6",
'United Airlines' : "#19d0e6",
'Singapore Airlines' : "#19c5e6",
'Surinam Airways' : "#19b9e6",
'Kenya Airways' : "#19aee6",
'Garuda Indonesia' : "#19a2e6",
'China Eastern Airlines' : "#1997e6",
'Global Jet Luxembourg' : "#198be6",
'SunExpress' : "#1980e6",
'Pantanal Linhas Aéreas' : "#1974e6",
'Air Astana' : "#195de6",
'El Al' : "#1952e6",
'Sky Express' : "#1946e6",
'Saudia' : "#193ae6",
'TUI Airways Ltd' : "#192fe6",
'easyJet Switzerland' : "#1923e6",
'PLAY' : "#1b19e6",
'Air Alsie' : "#2719e6",
'Air India' : "#3219e6",
'Air Arabia Maroc' : "#3e19e6",
'Eurowings' : "#4919e6",
'Norwegian Air Shuttle AOC': "#5519e6",
'Malaysia Airlines' : "#6019e6",
'Avies Air Company' : "#6c19e6",
'Arkia Israeli Airlines': "#7719e6",
'KLM Cityhopper' : "#8319e6",
'MHS Aviation' : "#8e19e6",
'UR Airlines' : "#9a19e6",
'TUI fly Belgium': "#a519e6",
'DC Aviation GmbH': "#b119e6",
'Titan Airways': "#bc19e6",
'London Executive Aviation Ltd': "#c819e6",
'FLYONE' : "#d319e6",
'SmartLynx Airlines Estonia' : "#df19e6",
'Private Wings Flugcharter' : "#e619e1",
'AlbaStar' : "#e619d5",
'AirX Charter' : "#e619ca",
'Ryanair UK' : "#e619be",
'Arcus Air' : "#e619b3",
'EFS European Flight Service' : "#e619a7",
'RVL Group' : "#e6199c",
'DOT LT' : "#e61990",
'Gama Aviation' : "#e61985",
'Jet Story' : "#e61979",
'National Airlines (US)' : "#e6196e",
'Enter Air' : "#e61962",
'Air Hamburg' : "#e61956",
'ABS Jets' : "#e6194b",
'MJet' : "#e6193f" ,
'Corendon Airlines Europe' : "#e61934",
'Avcon Jet' : "#e61928",
'Transavia France' : "#e61954",
'Copenhagen Airtaxi' : "#e61949",
'SmartWings' : "#e6193e",
'SunClass Airlines' : "#e61933",
'European Air Charter' : "#e61927",
'FlexFlight' : "#1969e6"
}

aircraft_colors = {
    '777-300ER': '#FF5733',
    'A350-900XWB': '#33FF57',
    'A330-300': '#3357FF',
    '787-8': '#FF33A8',
    '777-200ER': '#33FFF5',
    'A330-300E': '#F5FF33',
    '787-9': '#A833FF',
    'A321neoLR': '#FF8333',
    '737-800': '#33FF83',
    '747-400F': '#5733FF',
    '737-900': '#FFA833',
    '737-700': '#83FF33',
    '787-10': '#33A8FF',
    '747-400(SF)': '#FF3357',
    '747-8F': '#F533FF',
    'A330-900neo': '#FF5733',
    'A380-800': '#33FF57',
    'A320-200': '#3357FF',
    'A319-100': '#FF33A8',
    '777F': '#33FFF5',
    '737-8': '#F5FF33',
    'A220-300': '#A833FF',
    'ERJ195 AR': '#FF8333',
    'A320neo': '#33FF83',
    'ERJ190-100 LR': '#5733FF',
    'A321-200': '#FFA833',
    'ERJ170-100 LR': '#83FF33',
    'A321neo ACF': '#33A8FF',
    'A321-100': '#FF3357',
    '767-300ER': '#F533FF',
    'A330-200': '#FF5733',
    '767-400ER': '#33FF57',
    'A350-1000XWB': '#3357FF',
    'A318-100': '#FF33A8',
    'A220-100': '#33FFF5',
    'ACJ319': '#F5FF33',
    '757-200(ETOPS)': '#A833FF',
    'ATR 42-300': '#FF8333',
    '737-8200': '#33FF83',
    'ERJ175 LR': '#5733FF',
    'ERJ175 STD': '#FFA833',
    'J32': '#83FF33',
    '777-200LR': '#33A8FF',
    '737-900ER': '#FF3357',
    'ATR 72-500': '#F533FF',
    'A321neo': '#FF5733',
    'A330-200F': '#33FF57',
    'ERJ195 LR': '#3357FF',
    'J31': '#FF33A8',
    '757-300': '#33FFF5',
    'ERJ190-100 SR': '#F5FF33',
    'DO328-110': '#A833FF',
    'ACJ319neo': '#FF8333',
    '737-300': '#33FF83',
    'ERJ190-100 IGW (AR)': '#5733FF',
    'DHC-7-102': '#FFA833',
    'A330-300P2F': '#83FF33',
    'A340-300E': '#33A8FF',
    'ERJ190-100 STD': '#FF3357',
    'EMB135BJ (Legacy 600)': '#F533FF',
    'CHALLENGER 850': '#FF5733',
    '757-200': '#33FF57',
    'REIMS-CESSNA F406': '#3357FF',
    'ERJ190-100 ECJ': '#FF33A8',
    '747-400ERF': '#33FFF5',
    'DO228-202K': '#F5FF33',
    '737 BBJ': '#A833FF',
    '747-400(BCF)': '#FF8333'
}

# Get unique airports
unique_departure_airports = df['ADEP'].unique()
unique_destination_airports = df['ADES'].unique()

# Streamlit Page
st.title('🛫 Sustainability Dashboard 🛬')

# Multiselect boxes
box1, box2 = st.columns(2)

with box1:
    departure_airport = st.multiselect('Departure', ['All'] + list(unique_departure_airports), default="EHAM")

with box2:
    destination_airport = st.multiselect('Destination', ['All'] + list(unique_destination_airports), default='All')

if ('All' not in departure_airport) and ('All' not in destination_airport):
    filtered_df = df[(df['ADEP'].isin(departure_airport)) & (df['ADES'].isin(destination_airport))]
    # hopefully never too big; just copy
    # NOTE: if data for the map_df will be changed somewhere; use deep=True here. Changes will then not be reflected in filtered_df.
    # Copy by reference (shallow) or by value (deep) https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html
    map_df = filtered_df.copy() 

# only departure seleced
elif ('All' not in departure_airport) and (len(departure_airport) > 0):
    filtered_df = df[(df['ADEP'].values == departure_airport)]
    if len(filtered_df) > 300:
        best_100 = filtered_df.sort_values(by='Average_rating', ascending=True).head(100)
        worst_100 = filtered_df.sort_values(by='Average_rating', ascending=False).head(100)
        avg_100 = filtered_df.sample(n=100, random_state=42)  # Sample 100 random flights
        map_df = pd.concat([best_100, worst_100, avg_100])
    else:
        map_df = filtered_df.copy()
# only destination seleced    
elif ('All' not in destination_airport) and (len(departure_airport) > 0):
    filtered_df = df[df['ADES'].values == destination_airport]
    if len(filtered_df) > 300:
        best_100 = filtered_df.sort_values(by='Average_rating', ascending=True).head(100)
        worst_100 = filtered_df.sort_values(by='Average_rating', ascending=False).head(100)
        avg_100 = filtered_df.sample(n=100, random_state=42)
        map_df = pd.concat([best_100, worst_100, avg_100])
    else:
        map_df = filtered_df.copy()
  # Sample 100 random flights
else: 
    # Nothing selected, show best 100, worst 100, and average 100
    filtered_df = df
    best_100 = df.sort_values(by='Average_rating', ascending=True).head(100)
    worst_100 = df.sort_values(by='Average_rating', ascending=False).head(100)
    avg_100 = df.sample(n=100, random_state=42)  # Sample 100 random flights
    map_df = pd.concat([best_100, worst_100, avg_100])

sys.stderr.write(f"Filtered: {len(filtered_df)} Map: {len(map_df)}\n")


mapfilter = st.radio('Filters:', ["None", "Ratings","Loadfactor","Airlines","Aircraft type"], horizontal=True)

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
    text=df['ADEP'],
    hoverinfo='text',
    showlegend=False,
))

fig.add_trace(go.Scattermapbox(
    lat=df['LATITUDE_ADEP'],
    lon=df['LONGITUDE_ADEP'],
    mode='markers',
    marker=dict(size=5, color='white'),
    text=df['ADEP'],
    hoverinfo='text',
    showlegend=False,
))

line_color = 'gray' 


# else:
#     # Default to gray for all lines when filter is not applied
#     filtered_df['color'] = 'gray'


    # Add scattermapbox traces for routes using flight_id
for _, row in filtered_df.iterrows():
    # Get route-specific average load factor
    route = (row['ADEP'], row['ADES'])
    avg_load_factor = route_avg_load_factors.get(route) 

    match mapfilter:
        case 'All':
            line_color = 'gray'
        case 'Ratings':
            sys.stderr.write(f"Rating: {row['Overall_rating']}\n")
            line_color = airline_colors.get(row['Operator'], 'gray')
            # Apply rating filter and assign colors accordingly
            filtered_df['color'] = filtered_df['Overall_rating'].apply(assign_color)
            line_color = row['color']
        

        # DENSITY FILTER
        case "Loadfactor Density":
            # Normalize load factor to a range of 0 to 1
            normalized_load_factor = (avg_load_factor - filtered_df['Loadfactor'].min()) / (filtered_df['Loadfactor'].max() - filtered_df['Loadfactor'].min())
            # Create a color based on load factor using a blue-to-yellow gradient
            line_color = plt.cm.viridis(normalized_load_factor)
            line_color = f'rgb({int(line_color[0]*255)},{int(line_color[1]*255)},{int(line_color[2]*255)})'

        # AIRLINE FILTER
        case 'Airline':
            line_color = airline_colors.get(row['Operator'],'gray')
        

        # AIRCRAFT FILTER
        case 'Aircraft Type':
            if row['Aircraft Variant'] in aircraft_colors:
                line_color = aircraft_colors.get(row['Aircraft Variant'])

    fig.add_trace(go.Scattermapbox(
        mode='lines',
        lon=[row['LONGITUDE_ADES'], row['LONGITUDE_ADEP']],
        lat=[row['LATITUDE_ADES'], row['LATITUDE_ADEP']],
        line=dict(color=line_color, width=1),
        opacity=0.4,
        text=f"Route: {row['ADEP']} - {row['ADES']}<br>Avg. Load Factor: {avg_load_factor:.2f}%.<br> Operator: {row['Operator']}.<br> Aircraft Variant: {row['Aircraft Variant']} <br> Overall rating:{row['Overall_rating']}",
        # name=row['FLT_UID'],  # Use flight_id for unique tracing
        legendgroup=row['Overall_rating'],  # Group traces by rating for legend
        name=f"{row['Overall_rating']} - <span style='color:#999'>{ row['AIRCRAFT_ID']}</span>"  # Use rating as legend label
))
    
fig.update_layout(
        showlegend=True,
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
    selected_columns = ['FLT_UID', 'NAME_ADEP', 'ADEP', 'NAME_ADES', 'ADES', 'Operator', 'Aircraft Variant', 'AIRCRAFT_ID', 'Flight Time', 'Distance (km)', 'Overall_rating']  # Select desired columns

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

# engine, airline, aircraft = st.tabs(['🚀 Engine 🚀', '💺 Airline 💺', '🛩️ Aircraft 🛩️'])

# with engine:
#     st.header('🚀 Engine 🚀')

#     st.subheader('Grote Plot')
#     filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#     st.table(filtered_df[selected_columns].tail(5))
#     st.write('korte uitleg')

#     engine_col1, engine_col2 = st.columns(2)

#     with engine_col1:
#         st.subheader('Kleine Plot')
#         filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#         st.table(filtered_df[selected_columns].tail(5))
#         st.write('korte uitleg')

#     with engine_col2:
#         st.subheader('Kleine Plot')
#         filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#         st.table(filtered_df[selected_columns].tail(5))
#         st.write('korte uitleg')

# with airline:
#     st.subheader('💺 Airline 💺')
#     filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#     st.table(filtered_df[selected_columns].tail(5))

#     airline_col1, airline_col2 = st.columns(2)

#     with airline_col1:
#         filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#         st.table(filtered_df[selected_columns].tail(5))

#     with airline_col2:
#         filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#         st.table(filtered_df[selected_columns].tail(5))

# with aircraft:
#     st.subheader('🛩️ Aircraft 🛩️')
#     filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#     st.table(filtered_df[selected_columns].tail(5))
    
#     aircraft_col1, aircraft_col2 = st.columns(2)

#     with aircraft_col1:
#         filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#         st.table(filtered_df[selected_columns].tail(5))

#     with aircraft_col2:
#         filtered_df.sort_values(by = ['Average_rating'], ascending = False)
#         st.table(filtered_df[selected_columns].tail(5))

# Lay-out
st.sidebar.title("📖 Introduction")
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
