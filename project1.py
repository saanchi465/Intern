import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# We use cache_data so the model doesn't re-train on every click
@st.cache_data
def train_model():
    # 1. Load Data
    df = pd.read_csv("Titanic-Dataset.csv")

    # 2. Preprocessing
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df = df.drop(columns=['SibSp', 'Parch'])

    # 3. Splitting
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Training
    param_grid = {'n_estimators': [200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    return y_test, y_pred, df, best_model

def run_project():
    st.header("🚢 Titanic Survival Prediction")
    st.write("This project uses a Random Forest Classifier to predict which passengers survived the Titanic disaster.")

    # Execute the training function
    with st.spinner('Training model... please wait.'):
        y_test, y_pred, df, model = train_model()

    # --- SHOW RESULTS ---
    
    # 1. Metrics
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{acc:.2%}")

    # 2. Layout with Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    with col2:
        st.subheader("Sample Predictions")
        prediction_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        st.dataframe(prediction_df.head(10), use_container_width=True)

    # 3. Data Preview
    if st.checkbox("Show Raw Data (Processed)"):
        st.write(df.head(20))