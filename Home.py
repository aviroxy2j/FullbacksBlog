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

# Load the selected page
if page == "Home":
    home_page_path = Path("pages/Home.py")  # Make sure 'H' is capitalized to match the file name
    exec(home_page_path.read_text())
elif page == "General Analysis":
    general_page_path = Path("pages/General Analysis.py")
    exec(general_page_path.read_text())
elif page == "Advanced Analysis":
    advanced_page_path = Path("pages/Advanced Analysis.py")
    exec(advanced_page_path.read_text())

