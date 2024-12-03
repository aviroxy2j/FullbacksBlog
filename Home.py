import streamlit as st

# Home page layout
st.title("⚽ Football Analysis App")
st.markdown(
    """
    Welcome to the **Football Analysis App**! 🎉

    This app allows you to explore and analyze football player performance data with ease. 
    Choose from the following features:
    
    - **General Analysis**: Perform descriptive and exploratory analysis on key player metrics.
    - **Advanced Analysis**: Dive deeper with regression models, clustering, similarity analysis, and more.

    Use the sidebar to navigate between pages and start your journey!
    """
)

# Add a cool image or video (optional)
st.image("Destiny.jpg")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "General Analysis", "Advanced Analysis"])


