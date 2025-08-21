import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("Customer Segmentation using K-Means Clustering")

# Upload CSV
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Select features
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select k
    k = st.slider("Select number of clusters (k)", 2, 10, 5)

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Show results
    st.write("### Cluster Counts", df['Cluster'].value_counts())

    # Scatter plot
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df['Annual Income (k$)'], 
        y=df['Spending Score (1-100)'],
        hue=df['Cluster'],
        palette="Set2",
        ax=ax
    )
    plt.title("Customer Segmentation (Income vs Spending)")
    st.pyplot(fig)
