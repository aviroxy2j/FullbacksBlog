import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
data_path = "player_data.csv"
player_data = pd.read_csv(data_path)

# Sidebar filters
st.sidebar.header("Filters")

# Competition filter
competition = st.sidebar.selectbox(
    "Competition",
    options=["Any"] + list(player_data['Comp'].dropna().unique())
)

# Squad filter (dependent on competition)
if competition != "Any":
    filtered_squads = player_data[player_data['Comp'] == competition]['Squad'].dropna().unique()
else:
    filtered_squads = player_data['Squad'].dropna().unique()
squad = st.sidebar.selectbox("Squad", options=["Any"] + list(filtered_squads))

# Nation and Position filters
nation = st.sidebar.selectbox("Nation", options=["Any"] + list(player_data['Nation'].dropna().unique()))
position = st.sidebar.selectbox("Position", options=["Any"] + list(player_data['Pos'].dropna().unique()))

# Age range filter
age_range = st.sidebar.slider("Age Range", min_value=16, max_value=40, value=(18, 30))

# Apply filters
filtered_data = player_data[
    ((player_data['Comp'] == competition) | (competition == "Any")) &
    ((player_data['Squad'] == squad) | (squad == "Any")) &
    ((player_data['Nation'] == nation) | (nation == "Any")) &
    ((player_data['Pos'] == position) | (position == "Any")) &
    (player_data['Age'].between(age_range[0], age_range[1]))
]

# Autocomplete for highlighting multiple players
st.sidebar.header("Highlight Players")
num_players_to_highlight = st.sidebar.number_input(
    "Number of Players to Highlight (max: 6)",
    min_value=1, max_value=6, value=1, step=1
)

highlighted_players = []
for i in range(num_players_to_highlight):
    player_input = st.sidebar.text_input(
        f"Player {i + 1} Name", placeholder=f"Enter name for Player {i + 1}"
    )
    matching_players = player_data[
        player_data['Player'].str.contains(player_input, case=False, na=False)
    ]['Player'].unique()
    selected_player = st.sidebar.selectbox(
        f"Select Player {i + 1}",
        options=["None"] + list(matching_players),
        key=f"player_select_{i}"
    )
    if selected_player != "None":
        highlighted_players.append(selected_player)

# Scatterplot metric selection
st.sidebar.header("Scatterplot Metrics")
x_metric = st.sidebar.selectbox("X-axis Metric", options=player_data.columns[7:])  # Metrics from column 7 onward
y_metric = st.sidebar.selectbox("Y-axis Metric", options=player_data.columns[7:])

# General Analysis Header
st.header("General Analysis")
st.write(f"Filtered Data: {filtered_data.shape[0]} players")
st.dataframe(filtered_data)

# Dynamic Scatterplot
# Dynamic Scatterplot
st.subheader(f"Scatterplot: {x_metric} vs {y_metric}")
fig = px.scatter(
    filtered_data,
    x=x_metric,
    y=y_metric,
    hover_data=["Player", "Squad", "Age", "Pos"],
    title=f"{x_metric} vs {y_metric}",
    color_discrete_sequence=["blue"],  # All points default to blue
)

# Add highlighted players, all in red
for player in highlighted_players:
    player_data_row = filtered_data[filtered_data["Player"] == player]
    if not player_data_row.empty:
        fig.add_scatter(
            x=player_data_row[x_metric],
            y=player_data_row[y_metric],
            mode="markers",
            marker=dict(color="red", size=12),
            name=player
        )

# Update axis ranges dynamically
fig.update_layout(
    xaxis=dict(title=x_metric, range=[filtered_data[x_metric].min(), filtered_data[x_metric].max()]),
    yaxis=dict(title=y_metric, range=[filtered_data[y_metric].min(), filtered_data[y_metric].max()]),
    showlegend=True
)
st.plotly_chart(fig)
