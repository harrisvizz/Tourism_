import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

#Load Datasets
@st.cache_data
def load_data():
    users = pd.read_excel("users.xlsx")  
    transactions = pd.read_excel("transactions.xlsx")
    attractions = pd.read_excel("attractions.xlsx")
    return users, transactions, attractions

users, transactions, attractions = load_data()

#Data Preprocessing
def preprocess_data(users, transactions, attractions):
    # Merge datasets
    df = transactions.merge(users, on="UserID", how="left")
    df = df.merge(attractions, on="AttractionID", how="left")

    # Handle missing values
    df.fillna({'Rating': df['Rating'].median()}, inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in ['VisitMode', 'ContentId', 'CountryId', 'AttractionID']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize numerical columns
    scaler = StandardScaler()
    df[['Rating']] = scaler.fit_transform(df[['Rating']])

    return df, label_encoders, scaler

df, label_encoders, scaler = preprocess_data(users, transactions, attractions)

#Exploratory Data Analysis (EDA)
def plot_eda(df):
    st.subheader("User Distribution by Continent")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='ContentId')
    st.pyplot(plt)

    st.subheader("Visit Mode Distribution")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='VisitMode')
    st.pyplot(plt)

if st.sidebar.button("Run EDA"):
    plot_eda(df)

#Regression Model - Predict Attraction Ratings
X_reg = df[['UserID', 'AttractionID', 'VisitMode', 'ContentId']]
y_reg = df['Rating']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

st.sidebar.subheader("Regression Model - Predict Ratings")
st.sidebar.write(f"Mean Squared Error: {reg_mse:.2f}")

#Classification Model - Predict Visit Mode
X_cls = df[['UserID', 'AttractionID', 'ContentId']]
y_cls = df['VisitMode']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
cls_model = RandomForestClassifier(n_estimators=100)
cls_model.fit(X_train_cls, y_train_cls)
y_pred_cls = cls_model.predict(X_test_cls)
cls_accuracy = accuracy_score(y_test_cls, y_pred_cls)

st.sidebar.subheader("Classification Model - Predict Visit Mode")
st.sidebar.write(f"Accuracy: {cls_accuracy:.2f}")

#Recommendation System
st.sidebar.subheader("Recommendation System")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserID', 'AttractionID', 'Rating']], reader)
svd = SVD()
cross_validate(svd, data, cv=5)

# Prediction Function
def recommend_attractions(user_id, top_n=5):
    user_data = df[df['UserID'] == user_id]
    attractions_not_visited = df[~df['AttractionID'].isin(user_data['AttractionID'])]['AttractionID'].unique()
    
    predictions = [svd.predict(user_id, aid) for aid in attractions_not_visited]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_recommendations = [pred.iid for pred in predictions[:top_n]]
    return attractions[attractions['AttractionID'].isin(top_recommendations)][['AttractionType', 'AttractionID']]

user_id_input = st.number_input("Enter User ID for Recommendations", min_value=1, step=1)
if st.button("Get Recommendations"):
    recommended = recommend_attractions(user_id_input)
    st.write(recommended)
