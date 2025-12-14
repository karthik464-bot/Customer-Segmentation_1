import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide"
)

st.title("🛍️ Customer Segmentation using K-Means")
st.markdown("An end-to-end Data Science project using Unsupervised Learning")

# --------------------------------------
# DATA UPLOAD
# --------------------------------------
st.header("1️⃣ Upload Dataset")
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------
    # FEATURE SELECTION
    # --------------------------------------
    st.header("2️⃣ Feature Selection")
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features]

    st.write("Selected Features:", features)

    # --------------------------------------
    # FEATURE SCALING
    # --------------------------------------
    st.header("3️⃣ Feature Scaling")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.success("Data has been standardized using StandardScaler")

    # --------------------------------------
    # ELBOW METHOD
    # --------------------------------------
    st.header("4️⃣ Elbow Method to Find Optimal K")

    wcss = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K_range, wcss, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")

    st.pyplot(fig)
    st.info("Elbow is usually observed at K = 5")

    # --------------------------------------
    # SILHOUETTE SCORE
    # --------------------------------------
    st.header("5️⃣ Silhouette Score Analysis")

    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        st.write(f"Clusters = {k} → Silhouette Score = {score:.3f}")

    # --------------------------------------
    # FINAL MODEL
    # --------------------------------------
    st.header("6️⃣ Final K-Means Model")

    k = st.slider("Select Number of Clusters (K)", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.success(f"K-Means Model trained with K = {k}")

    # --------------------------------------
    # 2D VISUALIZATION
    # --------------------------------------
    st.header("7️⃣ 2D Cluster Visualization")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="viridis",
        ax=ax2
    )
    ax2.set_title("Customer Segments (Income vs Spending)")
    st.pyplot(fig2)

    # --------------------------------------
    # CLUSTER SUMMARY
    # --------------------------------------
    st.header("8️⃣ Cluster Summary")
    summary = df.groupby("Cluster")[features].mean()
    st.dataframe(summary)

    # --------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------
    st.header("9️⃣ Business Insights")

    st.markdown("""
    - 🎯 **High Income & High Spending** → Premium customers  
    - 💰 **High Income & Low Spending** → Upselling opportunity  
    - 🛒 **Low Income & High Spending** → Discount-driven customers  
    - 📉 **Low Value Segments** → Cost-effective marketing  
    """)

else:
    st.warning("Please upload the dataset to proceed.")
