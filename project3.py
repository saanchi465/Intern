import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

@st.cache_data
def load_and_prep():
    df = pd.read_csv("IRIS.csv")
    X = df.drop("species", axis=1)
    y = df["species"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def run_project():
    st.title("🌸 Iris Flower Classifier Comparison")
    st.write("Compare three different algorithms to see which one identifies flower species most accurately.")

    df, X_train, X_test, y_train, y_test, scaler, le = load_and_prep()

    # --- MODEL TRAINING ---
    lr = LogisticRegression().fit(X_train, y_train)
    svm = SVC(probability=True).fit(X_train, y_train)
    rf = RandomForestClassifier().fit(X_train, y_train)

    # --- ACCURACY COMPARISON ---
    st.subheader("Model Performance")
    c1, c2, c3 = st.columns(3)
    
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    acc_rf = accuracy_score(y_test, rf.predict(X_test))

    c1.metric("Logistic Regression", f"{acc_lr:.2%}")
    c2.metric("SVM", f"{acc_svm:.2%}")
    c3.metric("Random Forest", f"{acc_rf:.2%}")

    st.divider()

    # --- INTERACTIVE PREDICTION ---
    st.subheader("Try it yourself!")
    st.write("Adjust the sliders below to see how the models classify a specific flower.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s_len = st.slider("Sepal Length", float(df['sepal_length'].min()), float(df['sepal_length'].max()))
        s_wid = st.slider("Sepal Width", float(df['sepal_width'].min()), float(df['sepal_width'].max()))
    with col_s2:
        p_len = st.slider("Petal Length", float(df['petal_length'].min()), float(df['petal_length'].max()))
        p_wid = st.slider("Petal Width", float(df['petal_width'].min()), float(df['petal_width'].max()))

    # Process input for prediction
    user_input = np.array([[s_len, s_wid, p_len, p_wid]])
    user_input_scaled = scaler.transform(user_input)
    
    prediction = rf.predict(user_input_scaled)
    predicted_species = le.inverse_transform(prediction)[0]

    st.success(f"**Random Forest Prediction:** This flower is a **{predicted_species}**")

    # --- VISUALIZATION ---
    st.divider()
    st.subheader("Data Distribution")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', ax=ax)
    ax.scatter(p_len, p_wid, color='red', marker='X', s=200, label='Your Input')
    plt.legend()
    st.pyplot(fig)