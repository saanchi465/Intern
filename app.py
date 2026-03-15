import streamlit as st
import project1
import project2
import project3

# 1. PAGE SETUP: Sets the title in the browser tab and wide layout
st.set_page_config(
    page_title="Data Science Portfolio | Saanchi",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. STYLING: Adding a bit of custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR NAVIGATION
st.sidebar.title("📌 Project Hub")
st.sidebar.markdown("Navigate through the different phases of my Data Science work.")

# Use a radio or selectbox for the main menu
menu = ["🏠 Dashboard Home", "🚢 Titanic: Survival Prediction", "🎬 IMDb: Cinema Analytics", "🌸 Iris: Model Comparison"]
selection = st.sidebar.radio("Go to:", menu)

st.sidebar.divider()
st.sidebar.markdown(f"""
**Student:** Saanchi  
**Year:** B.Tech 2nd Year (DS)  
**Submission:** Data Science Project  
""")

# 4. PAGE LOGIC
if selection == "🏠 Dashboard Home":
    st.title("📊 Multi-Domain Data Science Dashboard")
    st.subheader("Welcome to my technical portfolio.")
    
    st.markdown("""
    This application serves as a comprehensive showcase of data science methodologies applied to three distinct datasets. 
    Each module explores different aspects of the data pipeline:
    ---
    """)

    # Project Summary Cards using Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🚢 Predictive Modeling")
        st.info("**Dataset:** Titanic (Kaggle)")
        st.write("Focused on binary classification and feature engineering using Random Forest.")
        if st.button("Explore Titanic Model"):
            st.toast("Navigate to 'Titanic' in the sidebar!")

    with col2:
        st.markdown("### 🎬 Exploratory Data Analysis")
        st.success("**Dataset:** IMDb India")
        st.write("In-depth statistical analysis of 100 years of cinema trends and correlations.")
        if st.button("Explore IMDb Analytics"):
            st.toast("Navigate to 'IMDb' in the sidebar!")

    with col3:
        st.markdown("### 🌸 Algorithm Benchmarking")
        st.warning("**Dataset:** Iris Species")
        st.write("A comparative study of Logistic Regression, SVM, and Random Forest.")
        if st.button("Explore Iris Comparison"):
            st.toast("Navigate to 'Iris' in the sidebar!")

    st.divider()
    st.markdown("#### 🛠️ Tech Stack & Methodology")
    st.table({
        "Category": ["Language", "Libraries", "Deployment", "Version Control"],
        "Tools Used": ["Python 3.x", "Pandas, Scikit-Learn, Seaborn, Matplotlib", "Streamlit", "Git / VS Code"]
    })

elif selection == "🚢 Titanic: Survival Prediction":
    project1.run_project()

elif selection == "🎬 IMDb: Cinema Analytics":
    project2.run_project()

elif selection == "🌸 Iris: Model Comparison":
    project3.run_project()

# 5. FOOTER
st.markdown("---")
st.caption("Developed by Saanchi | B.Tech Data Science 2026")