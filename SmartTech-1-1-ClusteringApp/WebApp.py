'''1) Implement a WEB APP and a GUI application which loads different data sets and applies our clustering algorithms we have discussed in the lecture (kmeans, meanshift, BIRCH...). 
Show the differences between using a PCA and no PCA. 
Use avalonia for the GUI APP, and github pages for the web-app
'''

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, Birch





# Load different datasets
def load_dataset(dataset_name):
    if dataset_name == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    elif dataset_name == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    elif dataset_name == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    else:
        st.error("Dataset not found!")
        return None
    return df

# Apply kmeans clustering
def apply_kmeans(df, n_clusters, use_pca):
    if use_pca:
        pca = PCA(n_components=2)
        df_reduced = pca.fit_transform(df)
    else:
        df_reduced = df.values

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df_reduced)
    return kmeans.labels_

#Apply meanshift clustering
def apply_meanshift(df, use_pca):
    if use_pca:
        pca = PCA(n_components=2)
        df_reduced = pca.fit_transform(df)
    else:
        df_reduced = df.values

    meanshift = MeanShift()
    meanshift.fit(df_reduced)
    return meanshift.labels_

# Apply BIRCH clustering
def apply_birch(df, n_clusters, use_pca):
    if use_pca:
        pca = PCA(n_components=2)
        df_reduced = pca.fit_transform(df)
    else:
        df_reduced = df.values

    birch = Birch(n_clusters=n_clusters)
    birch.fit(df_reduced)
    return birch.labels_

# Visualize clustering results in two dimensions
def visualize_clustering_results(df, labels, use_pca):
    if use_pca:
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(df)
        chart_data = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
        chart_title = "Clusters after PCA"
    else:
        chart_data = df.iloc[:, :2].copy()
        chart_data.columns = ["Feature 1", "Feature 2"]
        chart_title = "Clusters using the first two original features"

    chart_data["Cluster"] = labels.astype(str)
    st.subheader(chart_title)
    st.scatter_chart(chart_data, x=chart_data.columns[0], y=chart_data.columns[1], color="Cluster")

    if not use_pca:
        st.caption("The clustering uses all original features; this chart displays only the first two.")


# Streamlit web app
def main():
    st.title("Clustering Algorithms Web App")
    
    # Sidebar for dataset selection
    dataset_name = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine", "Breast Cancer"])
    df = load_dataset(dataset_name)
    
    if df is not None:
        st.write(f"Dataset: {dataset_name}")
        st.write(df.head())
        
        # Sidebar for clustering algorithm selection
        algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "MeanShift", "BIRCH"])
        
        # Sidebar for PCA option
        use_pca = st.sidebar.checkbox("Use PCA", value=True)
        
        # Sidebar for number of clusters (only for KMeans and BIRCH)
        n_clusters = 3
        if algorithm in ["KMeans", "BIRCH"]:
            n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        
        # Apply selected clustering algorithm
        if algorithm == "KMeans":
            labels = apply_kmeans(df, n_clusters, use_pca)
        elif algorithm == "MeanShift":
            labels = apply_meanshift(df, use_pca)
        elif algorithm == "BIRCH":
            labels = apply_birch(df, n_clusters, use_pca)
        
        # Display clustering results
        st.write(f"Clustering Results using {algorithm}:")
        df['Cluster'] = labels
        st.write(df)

        visualize_clustering_results(df.drop(columns=['Cluster']), labels, use_pca)
            
        
            
if __name__ == "__main__":
    main()