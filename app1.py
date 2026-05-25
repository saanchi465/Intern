import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("🏥 Healthcare Analytics Dashboard")

# Load dataset
df = pd.read_csv(r"C:\Users\navee\Desktop\saanchi\Intern\hospital data analysis.csv")

# ----------------------------
# DATASET
# ----------------------------

st.subheader("Complete Patient Dataset")

st.dataframe(
    df,
    use_container_width=True
)

st.write("Total Patients:", len(df))

# ----------------------------
# KPI CARDS
# ----------------------------

st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Patients", len(df))

col2.metric("Average Age", round(df['Age'].mean(), 1))

col3.metric(
    "Total Revenue",
    f"₹{int(df['Cost'].sum())}"
)

# ----------------------------
# GENDER DISTRIBUTION
# ----------------------------

st.subheader("Gender Distribution")

fig, ax = plt.subplots()

sns.countplot(x='Gender', data=df, ax=ax)

st.pyplot(fig)

# ----------------------------
# CONDITION ANALYSIS
# ----------------------------

st.subheader("Condition Analysis")

fig2, ax2 = plt.subplots(figsize=(8, 5))

sns.countplot(
    y='Condition',
    data=df,
    order=df['Condition'].value_counts().index,
    ax=ax2
)

st.pyplot(fig2)

# ----------------------------
# SATISFACTION ANALYSIS
# ----------------------------

st.subheader("Patient Satisfaction")

fig3, ax3 = plt.subplots(figsize=(8, 5))

sns.histplot(df['Satisfaction'], bins=10, ax=ax3)

st.pyplot(fig3)