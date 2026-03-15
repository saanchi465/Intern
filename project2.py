import streamlit as st
import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ROBUST DATA LOADING: Cleans strings before converting to numbers
@st.cache_data
def load_data():
    try:
        with zipfile.ZipFile("archive.zip") as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, encoding="latin1")
        
        df.columns = df.columns.str.strip()

        # Step A: Clean 'Year' - remove brackets like (2023)
        # We use .str.extract to just pull out the digits
        df['Year'] = df['Year'].str.extract('(\d+)').astype(float)

        # Step B: Clean 'Duration' - remove ' min' text
        df['Duration'] = df['Duration'].str.replace(' min', '', regex=False)
        df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

        # Step C: Clean 'Votes' - remove commas like 1,234
        df['Votes'] = df['Votes'].str.replace(',', '', regex=False)
        df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

        # Step D: Clean 'Rating'
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

        # Step E: Fill missing values so calculations don't fail
        df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
        df['Votes'] = df['Votes'].fillna(df['Votes'].median())
        df['Duration'] = df['Duration'].fillna(df['Duration'].median())
        
        # Drop rows where we definitely need a Name or Year to make sense
        df = df.dropna(subset=['Name', 'Year'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame() # Return empty if file is missing

def run_project():
    st.title("🎬 IMDb India Cinema Analysis")
    st.write("An analysis of trends, ratings, and top performers in Indian Cinema.")

    df = load_data()

    # Safety Check: If cleaning failed and df is empty, stop here
    if df.empty:
        st.warning("The dataset is empty. Please check if the CSV file path is correct.")
        return

    # --- TOP ROW: KEY METRICS ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # We use .idxmax() on the grouped means
        yearly_rating = df.groupby('Year')['Rating'].mean()
        if not yearly_rating.empty:
            best_year_val = yearly_rating.idxmax()
            st.metric("Best Year (Avg Rating)", int(best_year_val))
    
    with col2:
        if 'Director' in df.columns and not df['Director'].isnull().all():
            top_director = df['Director'].value_counts().index[0]
            st.metric("Most Active Director", top_director)
        
    with col3:
        actors = pd.concat([df['Actor 1'], df['Actor 2'], df['Actor 3']]).dropna()
        if not actors.empty:
            top_actor = actors.value_counts().index[0]
            st.metric("Most Active Actor", top_actor)

    st.divider()

    # --- MIDDLE SECTION: TOP MOVIES ---
    tab1, tab2 = st.tabs(["🏆 Top 10 Overall", "📅 Top Movies Per Year"])
    
    with tab1:
        st.subheader("Highest Rated Movies of All Time")
        top10 = df.sort_values(by='Rating', ascending=False)[['Name', 'Year', 'Rating', 'Votes']].head(10)
        st.dataframe(top10, use_container_width=True)

    with tab2:
        st.subheader("Annual Cinema Leaders")
        # Ensure we sort by Year first for a clean view
        top_per_year = df.sort_values(['Year', 'Rating'], ascending=[True, False]).groupby('Year').head(1)
        st.dataframe(top_per_year[['Year', 'Name', 'Rating', 'Votes']], use_container_width=True)

    # --- BOTTOM SECTION: TRENDS ---
    st.subheader("Industry Trends")
    
    # Chart: Popular Movies (Rating >= 8) Per Year
    popular_count = df[df['Rating'] >= 8].groupby('Year').size().reset_index(name='Count')
    
    if not popular_count.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=popular_count, x='Year', y='Count', color="#E50914", ax=ax)
        ax.set_title("Rise of Highly Rated Movies (Rating 8+)")
        st.pyplot(fig)

    # Correlation Info
    correlation = df['Duration'].corr(df['Rating'])
    st.info(f"**Insight:** The correlation between Movie Duration and Rating is **{correlation:.2f}**. " 
            "Values close to 0 suggest that a longer movie doesn't necessarily mean a better rating!")

    # Search Feature
    st.subheader("🔍 Search for a Movie")
    search_query = st.text_input("Enter movie name to see details:")
    if search_query:
        results = df[df['Name'].str.contains(search_query, case=False, na=False)]
        if not results.empty:
            st.write(results)
        else:
            st.write("No movies found with that name.")