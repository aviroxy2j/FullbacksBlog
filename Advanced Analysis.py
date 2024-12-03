import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load the dataset
data_path = "player_data.csv"
player_data = pd.read_csv(data_path)

# Sidebar: Choose analysis type
analysis_type = st.sidebar.selectbox(
    "Choose Advanced Analysis",
    [
        "Linear Regression",
        "Clustering",
        "Player Similarity",
        "Correlation Heatmap",
        "Player Archetype Mapping"
    ]
)

# Analysis Header
st.header(f"Advanced Analysis: {analysis_type}")

if analysis_type == "Linear Regression":
    st.subheader("Linear Regression Analysis")
    
    # Sidebar: Select variables
    dependent_var = st.sidebar.selectbox("Dependent Variable (Y)", options=player_data.columns[7:])
    independent_vars = st.sidebar.multiselect("Independent Variables (X)", options=player_data.columns[7:], default=player_data.columns[8:10])
    
    # Highlight players
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
        matching_players = player_data[player_data['Player'].str.contains(player_input, case=False, na=False)]['Player'].unique()
        selected_player = st.sidebar.selectbox(
            f"Select Player {i + 1}",
            options=["None"] + list(matching_players),
            key=f"regression_player_select_{i}"
        )
        if selected_player != "None":
            highlighted_players.append(selected_player)

    if dependent_var and independent_vars:
        # Prepare data
        X = player_data[independent_vars].fillna(0)
        y = player_data[dependent_var].fillna(0)
        
        # Fit Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Display results
        st.write("**Regression Coefficients:**")
        coef_df = pd.DataFrame({"Variable": independent_vars, "Coefficient": model.coef_})
        st.table(coef_df)
        
        st.write(f"**R² Score:** {model.score(X, y):.4f}")
        
        # Scatterplot (only if one independent variable is selected)
        if len(independent_vars) == 1:
            predicted = model.predict(X)
            fig = px.scatter(
                x=X[independent_vars[0]], y=y,
                title=f"Linear Regression: {dependent_var} vs {independent_vars[0]}",
                labels={"x": independent_vars[0], "y": dependent_var}
            )
            fig.add_scatter(x=X[independent_vars[0]], y=predicted, mode="lines", name="Regression Line")

            # Add highlighted players
            for player in highlighted_players:
                player_data_row = player_data[player_data["Player"] == player]
                if not player_data_row.empty:
                    fig.add_scatter(
                        x=player_data_row[independent_vars[0]],
                        y=player_data_row[dependent_var],
                        mode="markers",
                        marker=dict(color="red", size=12),
                        name=player
                    )

            st.plotly_chart(fig)


elif analysis_type == "Clustering":
    st.subheader("Clustering Analysis")
    
    # Sidebar: Select variables for clustering
    cluster_vars = st.sidebar.multiselect("Select Metrics for Clustering", options=player_data.columns[7:], default=player_data.columns[8:10])
    num_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
    
    # Highlight players
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
        matching_players = player_data[player_data['Player'].str.contains(player_input, case=False, na=False)]['Player'].unique()
        selected_player = st.sidebar.selectbox(
            f"Select Player {i + 1}",
            options=["None"] + list(matching_players),
            key=f"clustering_player_select_{i}"
        )
        if selected_player != "None":
            highlighted_players.append(selected_player)

    if cluster_vars:
        # Prepare data
        X = player_data[cluster_vars].fillna(0)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        player_data["Cluster"] = kmeans.fit_predict(X)
        
        # Display cluster centroids
        st.write("**Cluster Centroids:**")
        st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_vars))
        
        # Scatterplot (only if two metrics are selected)
        if len(cluster_vars) == 2:
            fig = px.scatter(
                player_data,
                x=cluster_vars[0], y=cluster_vars[1],
                color="Cluster",
                hover_data=["Player", "Squad"],
                title="Clustering Analysis",
                labels={"x": cluster_vars[0], "y": cluster_vars[1]}
            )

            # Add highlighted players
            for player in highlighted_players:
                player_data_row = player_data[player_data["Player"] == player]
                if not player_data_row.empty:
                    fig.add_scatter(
                        x=player_data_row[cluster_vars[0]],
                        y=player_data_row[cluster_vars[1]],
                        mode="markers",
                        marker=dict(color="red", size=12),
                        name=player
                    )

            st.plotly_chart(fig)


elif analysis_type == "Player Similarity":
    st.subheader("Player Similarity")

    # Sidebar: Select player and metrics
    selected_player = st.sidebar.selectbox("Select Player", options=player_data['Player'].unique())
    similarity_metrics = st.sidebar.multiselect(
        "Select Metrics for Similarity",
        options=["KP", "xAG", "PPA", "Cmp", "finalthird", "PrgP", "SCA90", "PassLive",
                 "TO", "Def", "Tkl+Int", "Touches", "PrgC", "PrgDist", "Rec", "PrgR"],
        default=["xAG", "KP"]
    )

    # Updated sensitivity values based on the refined table
    metric_sensitivity = {
        "KP": 0.650735,
        "xAG": 0.126838,
        "PPA": 0.617647,
        "Cmp": 14.407954,
        "finalthird": 2.073978,
        "PrgP": 1.878594,
        "SCA90": 1.170608,
        "PassLive": 0.805147,
        "TO": 0.183140,
        "Def": 0.044776,
        "Tkl+Int": 0.990000,
        "Touches": 15.493076,
        "PrgC": 1.847458,
        "PrgDist": 41.120512,
        "Rec": 13.960035,
        "PrgR": 2.540323
    }

    if selected_player and similarity_metrics:
        # Get the selected player's data
        player_row = player_data[player_data['Player'] == selected_player]
        if player_row.empty:
            st.error("Selected player not found in the data.")
        else:
            position = player_row['Pos'].iloc[0]
            st.write(f"**Position of Selected Player:** {position}")
            
            # Filter players by the same position
            filtered_data = player_data[player_data['Pos'] == position]

            # Apply similarity filter
            for metric in similarity_metrics:
                sensitivity = metric_sensitivity.get(metric, 0.1)  # Default sensitivity
                player_value = player_row[metric].iloc[0]
                filtered_data = filtered_data[
                    (filtered_data[metric] >= player_value - sensitivity) &
                    (filtered_data[metric] <= player_value + sensitivity)
                ]

            # Remove the selected player from the list of similar players
            similar_players = filtered_data[filtered_data['Player'] != selected_player]

            # Display similar players
            st.write(f"**Players Similar to {selected_player}:**")
            st.table(similar_players[['Player', 'Squad', 'Nation', 'Pos'] + similarity_metrics])

            # Scatterplot for similarity (if two metrics are selected)
            if len(similarity_metrics) == 2:
                fig = px.scatter(
                    player_data,
                    x=similarity_metrics[0],
                    y=similarity_metrics[1],
                    hover_data=["Player", "Squad", "Nation"],
                    title=f"Player Similarity: {selected_player}",
                    labels={"x": similarity_metrics[0], "y": similarity_metrics[1]}
                )
                # Highlight the selected player
                fig.add_scatter(
                    x=player_row[similarity_metrics[0]],
                    y=player_row[similarity_metrics[1]],
                    mode="markers",
                    marker=dict(color="red", size=12),
                    name="Selected Player"
                )
                # Highlight similar players
                fig.add_scatter(
                    x=similar_players[similarity_metrics[0]],
                    y=similar_players[similarity_metrics[1]],
                    mode="markers",
                    marker=dict(color="blue", size=10),
                    name="Similar Players"
                )
                st.plotly_chart(fig)



elif analysis_type == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")

    # Sidebar: Select metrics for the heatmap
    heatmap_metrics = st.sidebar.multiselect(
        "Select Metrics for Correlation Heatmap",
        options=player_data.columns[7:], default=player_data.columns[8:12]
    )

    if heatmap_metrics:
        # Compute correlation matrix
        correlation_matrix = player_data[heatmap_metrics].corr()

        # Plot heatmap using seaborn
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            xticklabels=heatmap_metrics, yticklabels=heatmap_metrics, ax=ax
        )
        st.pyplot(fig)


elif analysis_type == "Player Archetype Mapping":
    st.subheader("Player Archetype Mapping")

    # Sidebar: Position and Metrics Selection
    positions = player_data["Pos"].unique()
    selected_position = st.sidebar.selectbox("Select Position", options=positions, index=0)
    position_filtered_data = player_data[player_data["Pos"].str.contains(selected_position, na=False)]

    archetype_metrics = st.sidebar.multiselect(
        "Select Metrics for Archetype Mapping",
        options=position_filtered_data.columns[7:],  # Columns starting from 'KP' onwards
        default=position_filtered_data.columns[8:10]  # Default selection
    )
    num_archetypes = st.sidebar.slider("Number of Archetypes", min_value=2, max_value=8, value=4)

    if archetype_metrics and not position_filtered_data.empty:
        # Prepare data for clustering
        archetype_data = position_filtered_data[archetype_metrics].fillna(0)

        # Fit KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_archetypes, random_state=42)
        position_filtered_data["Archetype"] = kmeans.fit_predict(archetype_data)

        # Display archetype summary
        st.write("**Archetype Centers:**")
        st.table(pd.DataFrame(kmeans.cluster_centers_, columns=archetype_metrics))

        # Scatterplot for archetypes (if two metrics are selected)
        if len(archetype_metrics) == 2:
            fig = px.scatter(
                position_filtered_data,
                x=archetype_metrics[0],
                y=archetype_metrics[1],
                color="Archetype",
                hover_data=["Player", "Squad"],
                title=f"Player Archetype Mapping for Position: {selected_position}",
                labels={"x": archetype_metrics[0], "y": archetype_metrics[1]}
            )
            st.plotly_chart(fig)

        # Allow highlighting specific players
        selected_players = st.sidebar.multiselect(
            "Select Players to Highlight",
            options=position_filtered_data["Player"].unique()
        )

        if selected_players:
            # Highlight selected players in the scatterplot
            for player in selected_players:
                player_row = position_filtered_data[position_filtered_data["Player"] == player]
                if not player_row.empty:
                    fig.add_scatter(
                        x=[player_row[archetype_metrics[0]].values[0]],
                        y=[player_row[archetype_metrics[1]].values[0]],
                        mode="markers+text",
                        marker=dict(color="red", size=12, symbol="x"),
                        text=player,
                        textposition="top center",
                        name=f"Highlighted: {player}"
                    )
            st.plotly_chart(fig)

        # Display full player data for the selected archetype
        st.write("**Player Archetypes Data:**")
        st.dataframe(position_filtered_data[["Player", "Squad", "Archetype"] + archetype_metrics])

    else:
        st.error("Please select at least one metric for archetype mapping.")
