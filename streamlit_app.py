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
def load_data(csvf):
#    df = pd.read_csv(
#       csvf,
#        usecols=[
#            "ADEP",
#            "ADES",
#            "AIRCRAFT_ID",
#            "Operator",
#            "Aircraft Variant",
#            "Average_rating",
#            "Distance (km)",
#            "Engine Model",
#            "Engine Manufacturer",
#            "Aircraft Manufacturer",
#            "Loadfactor (%)",
#            "Flight Time",
#            "FLT_UID",
#            "Overall_rating",
#            "NAME_ADEP",
#            "COUNTRY_CODE_ADES",
#            "COUNTRY_CODE_ADEP",
#            "LONGITUDE_ADEP",
#            "LATITUDE_ADEP",
#            "EOBT_1",
#            "CO2 rating",
#            "NOx rating",
#            "Fuel Flow rating",
#            "NAME_ADES",
#            "LATITUDE_ADES",
#            "LONGITUDE_ADES",
#            "Loadfactor",
#        ],
#    )  # 
    pd.read_csv('Sustainabilty_dashboard_2025.csv')
    return df


df = load_data("Sustainabilty_dashboard_2025.csv")


gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.LONGITUDE_ADES, df.LATITUDE_ADES),
    crs="EPSG:4326",
)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.LONGITUDE_ADEP, df.LATITUDE_ADEP),
    crs="EPSG:4326",
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
        "A": "darkgreen",
        "B": "green",
        "C": "limegreen",
        "D": "yellow",
        "E": "orange",
        "F": "purple",
        "G": "pink",
    }
    return colors.get(letter, "gray")  # Default to gray if letter is not found


# Create a new column with assigned colors
df["color"] = df["Overall_rating"].apply(assign_color)

# Create an airline color dictionary with unique colors
airline_colors = {
    "Emirates Airline": "#e61919",
    "Delta Air Lines": "#e62519",
    "KLM Royal Dutch Airlines": "#e63119",
    "Aeromexico": "#e63c19",
    "Air Transat": "#e64819",
    "Air Canada": "#e65319",
    "Transavia": "#e65f19",
    "Corendon Dutch Airlines": "#e66a19",
    "Xiamen Airlines": "#e67619",
    "China Airlines": "#e68119",
    "Georgian Airways": "#e68d19",
    "Korean Air": "#e69819",
    "Suparna Airlines": "#e6a419",
    "Cathay Pacific": "#e6af19",
    "Bulgaria Air": "#e6bb19",
    "easyJet Europe": "#e6c619",
    "Qatar Airways": "#e6d219",
    "Turkish Airlines": "#e6dd19",
    "Corendon Airlines": "#e2e619",
    "airBaltic": "#d7e619",
    "Lufthansa": "#cbe619",
    "LOT Polish Airlines": "#c0e619",
    "easyJet": "#b4e619",
    "TUI Airlines Netherlands": "#a9e619",
    "Austrian Airlines": "#9de619",
    "Vueling": "#92e619",
    "TAP Air Portugal": "#86e619",
    "Italia Trasporto Aereo (ITA Airways)": "#7be619",
    "BA CityFlyer": "#6fe619",
    "China Southern Airlines": "#64e619",
    "Air France": "#58e619",
    "Ryanair": "#4de619",
    "Air Europa Líneas Aéreas": "#41e619",
    "Kuwait Airways": "#36e619",
    "Finnair": "#2ae619",
    "Aer Lingus": "#1ee619",
    "SWISS": "#19e620",
    "Aegean Airlines": "#19e62c",
    "Pegasus Airlines": "#19e637",
    "Air Serbia": "#19e643",
    "EVA Air": "#19e64e",
    "Air Malta": "#19e65a",
    "TAROM": "#19e665",
    "BA Euroflyer": "#19e671",
    "British Airways": "#19e67c",
    "Etihad Airways": "#19e688",
    "Royal Air Maroc": "#19e693",
    "SAS": "#19e69f",
    "Croatia Airlines": "#19e6aa",
    "Norwegian Air Sweden AOC": "#19e6b6",
    "Iberia Express": "#19e6c1",
    "Royal Jordanian": "#19e6cd",
    "Icelandair": "#19e6d8",
    "EgyptAir": "#19e6e4",
    "American Airlines": "#19dce6",
    "United Airlines": "#19d0e6",
    "Singapore Airlines": "#19c5e6",
    "Surinam Airways": "#19b9e6",
    "Kenya Airways": "#19aee6",
    "Garuda Indonesia": "#19a2e6",
    "China Eastern Airlines": "#1997e6",
    "Global Jet Luxembourg": "#198be6",
    "SunExpress": "#1980e6",
    "Pantanal Linhas Aéreas": "#1974e6",
    "Air Astana": "#195de6",
    "El Al": "#1952e6",
    "Sky Express": "#1946e6",
    "Saudia": "#193ae6",
    "TUI Airways Ltd": "#192fe6",
    "easyJet Switzerland": "#1923e6",
    "PLAY": "#1b19e6",
    "Air Alsie": "#2719e6",
    "Air India": "#3219e6",
    "Air Arabia Maroc": "#3e19e6",
    "Eurowings": "#4919e6",
    "Norwegian Air Shuttle AOC": "#5519e6",
    "Malaysia Airlines": "#6019e6",
    "Avies Air Company": "#6c19e6",
    "Arkia Israeli Airlines": "#7719e6",
    "KLM Cityhopper": "#8319e6",
    "MHS Aviation": "#8e19e6",
    "UR Airlines": "#9a19e6",
    "TUI fly Belgium": "#a519e6",
    "DC Aviation GmbH": "#b119e6",
    "Titan Airways": "#bc19e6",
    "London Executive Aviation Ltd": "#c819e6",
    "FLYONE": "#d319e6",
    "SmartLynx Airlines Estonia": "#df19e6",
    "Private Wings Flugcharter": "#e619e1",
    "AlbaStar": "#e619d5",
    "AirX Charter": "#e619ca",
    "Ryanair UK": "#e619be",
    "Arcus Air": "#e619b3",
    "EFS European Flight Service": "#e619a7",
    "RVL Group": "#e6199c",
    "DOT LT": "#e61990",
    "Gama Aviation": "#e61985",
    "Jet Story": "#e61979",
    "National Airlines (US)": "#e6196e",
    "Enter Air": "#e61962",
    "Air Hamburg": "#e61956",
    "ABS Jets": "#e6194b",
    "MJet": "#e6193f",
    "Corendon Airlines Europe": "#e61934",
    "Avcon Jet": "#e61928",
    "Transavia France": "#e61954",
    "Copenhagen Airtaxi": "#e61949",
    "SmartWings": "#e6193e",
    "SunClass Airlines": "#e61933",
    "European Air Charter": "#e61927",
    "FlexFlight": "#1969e6",
}

aircraft_colors = {
    "A319-100": "#C0C0C0",  # Light Grey
    "A320-200": "#FFA07A",  # Light Salmon
    "A321-200": "#F08080",  # Light Coral
    "A330-300": "#FFC0CB",  # Pink
    "A330-900": "#8B0000",  # Dark Red
    "A350-900": "#008B8B",  # Teal
    "A350-1000": "#A52A2A",  # Brown
    "A380-800": "#FFFF00",  # Yellow
    "B737-800": "#00FFFF",  # Cyan
    "B737-MAX 8": "#FF00FF",  # Magenta
    "B737-700": "#800080",  # Purple
    "B737-900": "#A020F0",  # Purple (deeper)
    "B737-MAX 9": "#00BFFF",  # Deep Sky Blue
    "B737-MAX 10": "#FFC0CB",  # Pink
    "B747-400": "#8B0000",  # Dark Red
    "B747-8": "#008B8B",  # Teal
    "B747-8F": "#A52A2A",  # Brown
    "B757-200": "#FFFF00",  # Yellow
    "B757-300": "#008080",  # Teal
    "B767-300": "#800080",  # Purple
    "B767-300ER": "#00008B",  # Dark Blue
    "B767-400ER": "#F0E68C",  # Goldenrod
    "B777-200": "#008080",  # Teal
    "B777-200ER": "#FF0000",  # Red
    "B777-300": "#0000FF",  # Blue
    "B777-300ER": "#008000",  # Green
    "B777-9": "#FFFF00",  # Yellow
    "B777F": "#00FFFF",  # Cyan
    "B787-8": "#FF00FF",  # Magenta
    "B787-9": "#C0C0C0",  # Light Grey
    "B787-10": "#FFA07A",  # Light Salmon
    "E170": "#F08080",  # Light Coral
    "E175": "#FFC0CB",  # Pink
    "E190": "#8B0000",  # Dark Red
    "E195": "#008B8B",  # Teal
    "CRJ-900": "#A52A2A",  # Brown
    "CRJ-700": "#FFFF00",  # Yellow
    "CRJ-200": "#008080",  # Teal
    "ATR 72": "#800080",  # Purple
    "ATR 42": "#00008B",  # Dark Blue
}

# Get unique airports
tmp_uda = df["ADEP"].unique()

unique_departure_airports = {"All": list(tmp_uda)}
for k, l in zip(tmp_uda, tmp_uda):
    unique_departure_airports[k] = l

tmp_uda = df["ADES"].unique()
unique_destination_airports = {"All": list(tmp_uda)}
for k, l in zip(tmp_uda, tmp_uda):
    unique_destination_airports[k] = l

# Streamlit Page
st.title("🛫 Sustainability Dashboard 🛬")
st.write("""Note: all flights are either departing from or arriving at EHAM""")
# Multiselect boxes
box1, box2 = st.columns(2)

# options are:
# NOTE: this is NOT selections
if "deplist" not in st.session_state:
    st.session_state.deplist = unique_departure_airports
if "destlist" not in st.session_state:
    st.session_state.destlist = ["EHAM", "EHAM"]

# init session state for selected airports
if "deps" not in st.session_state:
    st.session_state.deps = []
if "dests" not in st.session_state:
    st.session_state.dests = []


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def depSelected():
    # sys.stderr.write(f"Destination changed: {st.session_state.deps}\n")
    if "All" in st.session_state.deps:
        st.session_state.deps = {"All": "All"}
        st.session_state.deplist = {"All": "All"}
        st.session_state.destlist = {"EHAM": "EHAM"}
        st.session_state.dests = {"EHAM", "EHAM"}
    else:
        st.session_state.destlist = unique_destination_airports


def destSelected():
    # sys.stderr.write(f"dest change: {st.session_state.dests}\n")
    if "All" in st.session_state.dests:
        # st.session_state.dests = unique_departure_airports
        st.session_state.dests = {"All": "All"}
        st.session_state.destlist = {"All": "All"}
        st.session_state.deplist = {"EHAM": "EHAM"}
        st.session_state.deps = {"EHAM": "EHAM"}
    else:
        st.session_state.deplist = unique_departure_airports


def format_airport(airport):
    # get the first key of the dictionary
    return airport


# sys.stderr.write(f"deps: {st.session_state.deps}\ndest: {st.session_state.dests}\n")
with box1:
    departure_airport = st.multiselect(
        "Departure",
        st.session_state.deplist,
        format_func=format_airport,
        on_change=depSelected,
        key="deps",
    )

with box2:
    destination_airport = st.multiselect(
        "Destination",
        st.session_state.destlist,
        format_func=format_airport,
        on_change=destSelected,
        key="dests",
    )

    adep = flatten(list(map(unique_departure_airports.get, st.session_state.deps)))
    ades = flatten(list(map(unique_destination_airports.get, st.session_state.dests)))
    # sys.stderr.write(f"-----------\nadep: {adep}\nades: {ades}\n---------\n")
    filtered_df = df[(df["ADEP"].isin(adep)) & (df["ADES"].isin(ades))]
    # sys.stderr.write(f"filtered_df: {len(filtered_df)}\n")
    if len(filtered_df) > 300:
        best_100 = filtered_df.sort_values(by="Average_rating", ascending=True).head(
            100
        )
        worst_100 = filtered_df.sort_values(by="Average_rating", ascending=False).head(
            100
        )
        avg_100 = filtered_df.sample(
            n=100, random_state=42
        )  # Sample 100 random flights
        map_df = pd.concat([best_100, worst_100, avg_100])
    else:
        map_df = filtered_df.copy()

sys.stderr.write(f"filtered_df: {len(filtered_df)} map_df: {len(map_df)}\n")
mapfilter = st.radio(
    "Filters:",
    ["None", "Ratings", "Loadfactor", "Airlines", "Aircraft type"],
    horizontal=True,
)

# # Calculate route-specific average load factor
route_avg_load_factors = filtered_df.groupby(["ADEP", "ADES"])["Loadfactor"].mean()

# Create a Plotly figure
fig = go.Figure()

#     # Add scattermapbox traces for departure and arrival airports
fig.add_trace(
    go.Scattermapbox(
        lat=df["LATITUDE_ADES"],
        lon=df["LONGITUDE_ADES"],
        mode="markers",
        marker=dict(size=5, color="white"),
        text=df["NAME_ADES"] + ", " + df["COUNTRY_CODE_ADES"] + "<br>" + df["ADES"],
        hoverinfo="text",
        showlegend=False,
    )
)

fig.add_trace(
    go.Scattermapbox(
        lat=df["LATITUDE_ADEP"],
        lon=df["LONGITUDE_ADEP"],
        mode="markers",
        marker=dict(size=5, color="white"),
        text=df["NAME_ADEP"] + ", " + df["COUNTRY_CODE_ADEP"] + "<br>" + df["ADEP"],
        hoverinfo="text",
        showlegend=False,
    )
)

line_color = "gray"


# else:
#     # Default to gray for all lines when filter is not applied
#     filtered_df['color'] = 'gray'


# Add scattermapbox traces for routes using flight_id
for _, row in map_df.iterrows():
    # Get route-specific average load factor
    route = (row["ADEP"], row["ADES"])
    avg_load_factor = route_avg_load_factors.get(route)

    match mapfilter:
        case "All":
            line_color = "gray"
        case "Ratings":
            # sys.stderr.write(f"Rating: {row['Overall_rating']}\n")
            line_color = airline_colors.get(row["Operator"], "gray")
            # Apply rating filter and assign colors accordingly
            filtered_df["color"] = filtered_df["Overall_rating"].apply(assign_color)
            line_color = row["color"]

        # DENSITY FILTER
        case "Loadfactor":
            # Normalize load factor to a range of 0 to 1
            normalized_load_factor = (
                avg_load_factor - filtered_df["Loadfactor"].min()
            ) / (filtered_df["Loadfactor"].max() - filtered_df["Loadfactor"].min())
            # Create a color based on load factor using a blue-to-yellow gradient
            line_color = plt.cm.viridis(normalized_load_factor)
            sys.stderr.write(f"Loadfactor: {avg_load_factor}\n")
            line_color = f"rgb({int(line_color[0]*255)},{int(line_color[1]*255)},{int(line_color[2]*255)})"

        # AIRLINE FILTER
        case "Airlines":
            line_color = airline_colors.get(row["Operator"], "gray")

        # AIRCRAFT FILTER
        case "Aircraft type":
            if row["Aircraft Variant"] in aircraft_colors:
                line_color = aircraft_colors.get(row["Aircraft Variant"])

    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            lon=[row["LONGITUDE_ADES"], row["LONGITUDE_ADEP"]],
            lat=[row["LATITUDE_ADES"], row["LATITUDE_ADEP"]],
            line=dict(color=line_color, width=1),
            opacity=0.6,
            text=f"Route: {row['ADEP']} - {row['ADES']}<br>Callsign: {row['AIRCRAFT_ID']}<br> Operator: {row['Operator']}.<br> Aircraft Variant: {row['Aircraft Variant']} <br> Avg. Load Factor: {avg_load_factor:.2f}<br> Overall rating:{row['Overall_rating']}",
            # name=row['FLT_UID'],  # Use flight_id for unique tracing
            legendgroup=row["Overall_rating"],  # Group traces by rating for legend
            name=f"{row['Overall_rating']} - <span style='color:#999'>{ row['AIRCRAFT_ID']}</span>",  # Use rating as legend label
        )
    )

fig.update_layout(
    showlegend=True,
    height=800,
    width=1200,
    mapbox=dict(
        style="carto-darkmatter",
        zoom=2.5,
        center=dict(lat=50, lon=20),  # EU centered
        # projection=dict(type='equirectangular')
    ),
)

# Show the plot
st.plotly_chart(fig)

selected, export = st.columns([0.75, 0.25])

with selected:
    # Display filtered data with limited rows
    max_rows_to_show = 5  # Adjust as needed
    selected_columns = [
        "FLT_UID",
        "EOBT_1",
        "NAME_ADEP",
        "ADEP",
        "NAME_ADES",
        "ADES",
        "Operator",
        "Aircraft Variant",
        "AIRCRAFT_ID",
        "Flight Time",
        "Distance (km)",
        "CO2 rating",
        "NOx rating",
        "Fuel Flow rating",
        "Overall_rating",
    ]  # Select desired columns

    st.header("Filtered Flights:")
    if len(filtered_df) > max_rows_to_show:
        st.text(
            f"Showing select amount of columns (most important ones) and the first {max_rows_to_show} rows. Total rows: {len(filtered_df)}"
        )
    st.table(filtered_df[selected_columns].head(max_rows_to_show))

with export:
    # Download button for filtered data
    def convert_df(df):
        return df.to_csv().encode("utf-8")

    csv = convert_df(filtered_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
    )

text1, img1 = st.columns([0.75, 0.25])

with text1:
    st.subheader("How do the labels work?")
    st.write(
        """
The A-to-G environmental label offers insights into the environmental impact of flights, making it a useful reference for airlines, airports, and governments.

Flights labeled A are the most eco-efficient, with the lowest CO₂ emissions per passenger, while G flights have the highest impact. These labels are based on data such as fuel efficiency, aircraft type, route length, and passenger load.

For airlines, the system provides a clearer picture of fleet efficiency and operational impact. Governments and policymakers can use these insights to inform decisions around greener aviation strategies and environmental goals. It’s a tool that highlights opportunities for improvement across the sector.

"""
    )

with img1:
    st.image("energielabels.png")

col1, col2 = st.columns(2)

with col1:
    st.header("Top 5 Best Flights")
    st.table(filtered_df.sort_values(by=["Average_rating"]).head(5)[selected_columns])

with col2:
    st.header("Top 5 Worst Flights")
    st.table(
        filtered_df.sort_values(by=["Average_rating"], ascending=False).head(5)[
            selected_columns
        ]
    )

engine, airline, aircraft, loadfactor = st.tabs(
    [
        "🚀 Engine 🚀",
        "💺 Airlines 💺",
        "✈️ Aircraft ✈️",
        "🛩️ Load Factor 🛩️",
    ]
)

with engine:
    st.header("🚀 Engine Insights 🚀")

    # Ensure 'Average_rating' is numeric
    df["Average_rating"] = pd.to_numeric(df["Average_rating"], errors="coerce")

    # Group data by 'Engine Model' and calculate average rating
    avg_ratings_by_engine_model = (
        df.groupby("Engine Model")["Average_rating"].mean().sort_values()
    )

    st.subheader("Average Ratings by Engine Model")

    # Create a Plotly Express figure
    fig = px.bar(
        avg_ratings_by_engine_model.reset_index(),
        x="Engine Model",
        y="Average_rating",
        color="Average_rating",
        color_continuous_scale="RdBu",  # Red-blue color scale for full range
        title="Average Ratings by Engine Model",
        labels={"Engine Model": "Engine Model", "Average_rating": "Average Rating"},
        text="Average_rating",
    )

    # Update layout for readability
    fig.update_layout(
        xaxis_tickangle=-60,
        yaxis_title="Average Rating (1=A, 7=G)",
        xaxis_title="Engine Model",
        showlegend=False,  # Color indicates rating
    )

    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True)

    engine_col1, engine_col2 = st.columns(2)

    with engine_col1:
        # Groepeer de data op 'Engine Model' en bereken de gemiddelde rating
        avg_ratings_by_engine_model = (
            df.groupby("Engine Model")["Average_rating"].mean().sort_values()
        )

        # Selecteer de beste en slechtste 20 motoren
        best_20 = avg_ratings_by_engine_model.tail(20)
        worst_20 = avg_ratings_by_engine_model.head(20)

        fig = px.bar(
            worst_20.reset_index(),
            x="Engine Model",
            y="Average_rating",
            color="Average_rating",
            color_continuous_scale="RdBu",  # Omgekeerde rood-blauw schaal
            title="Top 20 Engine Rated the Lowest on Average",
            labels={"Engine Model": "Motormodel"},
            text="Average_rating",
        )
        fig.update_layout(
            xaxis_tickangle=-60,
            yaxis_title="Average Rating",
            xaxis_title="Engine Model",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with engine_col2:
        fig = px.bar(
            best_20.reset_index(),
            x="Engine Model",
            y="Average_rating",
            color="Average_rating",
            color_continuous_scale="RdBu",
            title="Top 20 Engine Rated the Highes on Average",
            labels={
                "Engine Model": "Engine type",
            },
            text="Average_rating",
        )
        fig.update_layout(
            xaxis_tickangle=-60,
            yaxis_title="Average Rating",
            xaxis_title="Engine Model",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Ensure 'Average_rating' is numeric
    df["Average_rating"] = pd.to_numeric(df["Average_rating"], errors="coerce")

    # Group data by 'Engine Model' and calculate average rating
    avg_ratings_by_engine_model = (
        df.groupby(["Engine Manufacturer", "Engine Model"])["Average_rating"]
        .mean()
        .sort_values()
    )

    # Create a Plotly Express figure
    fig = px.bar(
        avg_ratings_by_engine_model.reset_index(),
        x="Engine Model",
        y="Average_rating",
        color="Engine Manufacturer",
        color_continuous_scale="RdBu",  # Red-blue color scale for full range
        title="Average Ratings by Engine Model",
        barmode="group",
        labels={"Engine Model": "Engine Model", "Average_rating": "Average Rating"},
        text="Average_rating",
    )

    # Update layout for readability
    fig.update_layout(
        xaxis_tickangle=-60,
        yaxis_title="Average Rating (1=A, 7=G)",
        xaxis_title="Engine Model",
        legend_title="Engine Manufacturer",
        # legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True)

with airline:
    st.header("💺 Airlines Insights 💺")

    st.subheader("Average Ratings Per Airline Across Routes")
    # Create a selectbox for airline selection
    selected_airline = st.selectbox("Select Airline", df["Operator"].unique(), index=6)

    airline_data = df[df["Operator"] == selected_airline]

    # Group by route and calculate average rating
    route_ratings = (
        airline_data.groupby(["ADEP", "ADES"])["Average_rating"]
        .mean()
        .reset_index()
        .sort_values(by="Average_rating", ascending=True)
    )

    # Create a column for route names
    route_ratings["Route"] = route_ratings["ADEP"] + " - " + route_ratings["ADES"]

    fig = px.bar(
        route_ratings,
        x="Route",
        y="Average_rating",
        color="Route",  # Use color for visual differentiation
        color_continuous_scale="RdBu",
        title=f"Average Ratings Across Routes for {selected_airline}",
        labels={"Route": "Route", "Average_rating": "Average Rating"},
        text="Average_rating",  # Show values on top of bars
    )
    fig.update_layout(
        xaxis_tickangle=-90,  # Rotate x-axis labels for better readability
        yaxis_title="Average Rating",
        xaxis_title="Route",
        showlegend=False,  # Remove legend since color is already used for differentiation
    )

    st.plotly_chart(fig, use_container_width=True)

    airline_col1, airline_col2 = st.columns(2)

    with airline_col1:
        # Visualization: Compare Airlines on the Same Route
        # Group by route (ADEP -> ADES) and airline, calculate the average rating
        route_airline_ratings = (
            filtered_df.groupby(["ADEP", "ADES", "Operator"])["Average_rating"]
            .mean()
            .reset_index()
        )
        # Calculate average rating for each airline across all selected routes
        airline_avg_ratings = (
            route_airline_ratings.groupby("Operator")["Average_rating"]
            .mean()
            .reset_index()
        )

        # Plotting comparison of airlines for the specific route (flight count)
        adep = flatten(list(map(unique_departure_airports.get, st.session_state.deps)))
        ades = flatten(
            list(map(unique_destination_airports.get, st.session_state.dests))
        )
        route_data = airline_avg_ratings[
            (route_airline_ratings["ADEP"].isin(adep))
            & (route_airline_ratings["ADES"].isin(ades))
        ]

        # Sort by average rating for clarity
        route_data = route_data.sort_values(by="Average_rating", ascending=True)

        # Plotting comparison of airlines for the specific route
        # st.subheader(f'Average Ratings by Airlines for Route {", ".join(st.session_state.deps)} -> {", ".join(st.session_state.dests)}')

        fig = px.bar(
            route_data,
            x="Operator",
            y="Average_rating",
            color="Operator",
            title=f'Average Ratings by Airlines for Route {", ".join(st.session_state.deps)} -> {", ".join(st.session_state.dests)}',
            labels={"Operator": "Airline", "Average_rating": "Average Rating"},
            text="Average_rating",  # Show values on top of bars
        )
        fig.update_layout(
            xaxis_tickangle=-60,  # Rotate x-axis labels for better readability
            yaxis_title="Average Rating (1=A, 7=G)",
            xaxis_title="Airline",
            showlegend=False,  # Remove legend since color is already used for differentiation
        )

        st.plotly_chart(fig, use_container_width=True)

    # Plotting comparison of airlines for the specific route (flight count)
    adep = flatten(list(map(unique_departure_airports.get, st.session_state.deps)))
    adests = flatten(list(map(unique_destination_airports.get, st.session_state.dests)))
    with airline_col2:

        route_data = filtered_df[
            (filtered_df["ADEP"].isin(adep)) & (filtered_df["ADES"].isin(adests))
        ]

        # Calculate total flights per airline (after filtering for the specific route)
        route_airline_stats = (
            route_data.groupby(["Operator"])
            .agg(Total_Flights=("Operator", "count"))  # Count the number of flights
            .reset_index()
        )

        # st.subheader(f'Total Flights by Airlines for Route {", ".join(st.session_state.deps)} -> {", ".join(st.session_state.dests)}')

        fig_flights = px.bar(
            route_airline_stats,
            x="Operator",
            y="Total_Flights",
            color="Operator",
            title=f'Total Flights by Airlines for Route {", ".join(st.session_state.deps)} -> {", ".join(st.session_state.dests)}',
            labels={"Operator": "Airline", "Total_Flights": "Total Flights"},
            text="Total_Flights",  # Show values on top of bars
        )
        fig_flights.update_layout(
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            yaxis_title="Total Flights",
            xaxis_title="Airline",
            showlegend=False,  # Remove legend since color is already used for differentiation
        )

        st.plotly_chart(fig_flights, use_container_width=True)

with aircraft:
    st.header("✈️ Aircraft Insights ✈️")

    ## Group by route, operator, and aircraft type, calculate average rating
    route_operator_aircraft_ratings = (
        filtered_df.groupby(["ADEP", "ADES", "Operator", "Aircraft Variant"])[
            "Average_rating"
        ]
        .mean()
        .reset_index()
    )

    if departure_airport and destination_airport:
        # Filter data for the selected route
        adep = flatten(list(map(unique_departure_airports.get, st.session_state.deps)))
        adests = flatten(
            list(map(unique_destination_airports.get, st.session_state.dests))
        )

        route_data = route_operator_aircraft_ratings[
            (route_operator_aircraft_ratings["ADEP"].isin(adep))
            & (route_operator_aircraft_ratings["ADES"].isin(adests))
        ]

        # Check if data is available for the selected route
        if not route_data.empty:
            # Calculate mean average rating for each aircraft type within each operator
            mean_ratings_by_operator_aircraft = (
                route_data.groupby(["Operator", "Aircraft Variant"])["Average_rating"]
                .mean()
                .reset_index()
            )

            # Plotting comparison of airlines and aircraft types for the specific route
            st.subheader(
                f'Mean Average Ratings by Airline and Aircraft Type for Route {", ".join(st.session_state.deps)} -> {", ".join(st.session_state.dests)}'
            )

            fig = px.bar(
                mean_ratings_by_operator_aircraft,
                x="Operator",
                y="Average_rating",
                color="Aircraft Variant",
                barmode="group",  # Use 'group' for side-by-side bars
                title=f'Mean Average Ratings by Airline and Aircraft Type for Route {", ".join(st.session_state.deps)} -> {", ".join(st.session_state.dests)}',
                labels={
                    "Operator": "Airline",
                    "Average_rating": "Mean Average Rating (1=A, 7=G)",
                },
                text="Average_rating",  # Show values on top of bars
            )
            fig.update_layout(
                xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
                yaxis_title="Mean Average Rating (1=A, 7=G)",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected route.")
    else:
        st.info(
            "Please enter departure and destination airport codes to view airline and aircraft type comparisons."
        )

    aircraft_col1, aircraft_col2 = st.columns(2)

    with aircraft_col1:
        # Ensure 'Average_rating' is numeric
        # df['Average_rating'] = pd.to_numeric(df['Average_rating'], errors='coerce')

        # Calculate average rating per variant and select top 10 (nlargest for descending order)
        top_10_variants = (
            df.groupby("Aircraft Variant")["Average_rating"].mean().nlargest(20)
        )

        # st.subheader('Top 20 Worst Aircraft Variants by Average Rating')

        fig1 = px.bar(
            top_10_variants.to_frame().reset_index(),  # Convert to DataFrame for plotting
            x="Aircraft Variant",
            y="Average_rating",
            color="Average_rating",  # Use color for visual differentiation
            title="Top 20 Aircraft Variants by Average Rating (Worst to Best)",
            labels={
                "Aircraft Variant": "Aircraft Variant",
                "Average_rating": "Average Rating",
            },
            text="Average_rating",  # Show values on top of bars
        )
        fig1.update_layout(
            xaxis_tickangle=-60,  # Rotate x-axis labels for better readability
            yaxis_title="Average Rating (1=A, 7=G)",
            xaxis_title="Aircraft Variant",
            showlegend=False,  # Remove legend since color is already used for differentiation
        )

        st.plotly_chart(fig1, use_container_width=True)

    with aircraft_col2:
        # Ensure 'Average_rating' is numeric
        df["Average_rating"] = pd.to_numeric(df["Average_rating"], errors="coerce")

        # Group data and calculate average rating
        avg_ratings = (
            df.groupby(["Aircraft Manufacturer", "Aircraft Variant"])["Average_rating"]
            .mean()
            .reset_index()
        )

        # Select top 20 variants with the best ratings
        top_20_variants = avg_ratings.nsmallest(20, "Average_rating")

        # st.subheader("Top 20 Aircraft Variants by Average Ratings")

        fig = px.bar(
            top_20_variants,
            x="Aircraft Variant",
            y="Average_rating",
            color="Aircraft Manufacturer",
            title="Top 20 Aircraft Variants with the Best Average Ratings",
            labels={
                "Aircraft Variant": "Aircraft Variant",
                "Average_rating": "Average Rating",
            },
            text="Average_rating",
            color_continuous_scale="Viridis",  # A sequential color palette
        )

        fig.update_layout(
            xaxis_tickangle=-60,
            yaxis_title="Average Rating (1=A, 7=G)",
            xaxis_title="Aircraft Variant",
            legend_title="Aircraft Manufacturer",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        st.plotly_chart(fig, use_container_width=True)

with loadfactor:

    st.header("🛩️ Load Factor Insights 🛩️")

    load1, load2 = st.columns(2)

    with load1:
        # Group by airline and calculate average load factor
        airline_load_factors = (
            df.groupby("Operator")["Loadfactor (%)"]
            .mean()
            .reset_index()
            .sort_values(by="Loadfactor (%)", ascending=False)
        )

        # Select top 10 and bottom 10 airlines
        top_10_airlines = airline_load_factors.head(10)
        bottom_10_airlines = airline_load_factors.tail(10)

        # Create and display bar chart for top 10 airlines
        st.subheader("Top 10 Airlines by Average Load Factor")
        fig_top_10 = px.bar(
            top_10_airlines,
            x="Operator",
            y="Loadfactor (%)",
            title="Top 10 Airlines by Average Load Factor",
            labels={"Operator": "Airline", "Loadfactor (%)": "Average Load Factor"},
            color="Loadfactor (%)",
            color_continuous_scale="RdBu",
            text="Loadfactor (%)",
        )
        st.plotly_chart(fig_top_10, use_container_width=True)

    with load2:
        # Create and display bar chart for bottom 10 airlines
        st.subheader("Bottom 10 Airlines by Average Load Factor")
        fig_bottom_10 = px.bar(
            bottom_10_airlines,
            x="Operator",
            y="Loadfactor (%)",
            title="Bottom 10 Airlines by Average Load Factor",
            labels={"Operator": "Airline", "Loadfactor (%)": "Average Load Factor"},
            color="Loadfactor (%)",
            color_continuous_scale="RdBu",
            text="Loadfactor (%)",
        )
        st.plotly_chart(fig_bottom_10, use_container_width=True)

    # st.subheader("Average Load Factor Per Airline")
    fig = px.bar(
        airline_load_factors,
        x="Operator",
        y="Loadfactor (%)",
        title="Average Load Factor Per Airline",
        labels={"Operator": "Airline", "Loadfactor (%)": "Average Load Factor"},
        color="Loadfactor (%)",
        color_continuous_scale="RdBu",
        text="Loadfactor (%)",  # Show values on top of bars
    )
    fig.update_layout(
        xaxis_tickangle=-90,  # Rotate x-axis labels for better readability
        yaxis_title="Average Load Factor",
        xaxis_title="Airline",
    )

    st.plotly_chart(fig, use_container_width=True)


# Story in Sidebar
# Introduction Section
st.sidebar.title("Introduction 📖")
st.sidebar.write(
    """
In this project, we address the pressing issue of environmental impact in aviation by creating a labeling system to evaluate the sustainability of airline routes. 
Inspired by established practices in emissions labeling, we assigned sustainability grades (from **A to G**) to various routes and airlines based on their environmental performance.
"""
)

# Key Metrics Section
st.sidebar.header("Key Metrics 📌")
st.sidebar.write(
    """
Our labeling system is built upon three core ratings:
- **NOx Rating:** Evaluates the impact on air quality and environmental health caused by nitrogen oxide emissions.
- **CO₂ Rating:** Reflects the contribution of carbon dioxide emissions to global climate change.
- **Fuel Flow Rating:** Assesses fuel efficiency, measured as fuel flow per kilometer, to determine overall energy usage.

By combining these metrics, we calculated an average rating that determines each route’s overall sustainability grade, making it easier to compare different options.
"""
)

# Goal Section
st.sidebar.header("Our Goal 🎯")
st.sidebar.write(
    """
Offer actionable insights to **municipalities, governments, and airports** to support data-driven policymaking and promote sustainability initiatives.
"""
)

# Features Section
st.sidebar.header("Features of Our Tool 💡")
st.sidebar.write(
    """
- **Dynamic Labels:** Each route is assigned a sustainability grade (A = Most sustainable, G = Least sustainable) based on its combined NOx, CO₂, and fuel flow ratings.
- **Comparative Insights:** Analyze how airlines perform on the same route to identify the most sustainable operators.
- **Geographical Analysis:** Discover how sustainability varies across different regions and routes.
- **Interactive Visualizations:** Explore trends, such as the influence of aircraft type, engine model, and flight distance on sustainability.
"""
)

# Why It Matters Section
st.sidebar.header("Why It Matters 🏔️")
st.sidebar.write(
    """
Aviation is a significant contributor to greenhouse gas emissions, and as demand for air travel grows, addressing its environmental impact becomes increasingly important. 
By making sustainability metrics transparent, this tool aligns with global efforts to create a greener and more sustainable aviation industry.

We invite you to explore the labels, compare airlines and routes, and gain insights into the environmental impact of air travel. Together, we can work toward a more sustainable future for aviation! 🚀
"""
)
