# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from io import StringIO

import warnings
warnings.filterwarnings('ignore')

# EDA
# Univariate_Categorical
def plot_countplots(data):
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=data, x=col, ax=ax)
        plt.title(f'Count Plot of {col}')
        
        total = len(data[col])
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height()/total)
            x = p.get_x() + p.get_width() / 2 - 0.05
            y = p.get_y() + p.get_height() + 1
            ax.annotate(percentage, (x, y), ha='center')

        st.pyplot(fig)
    return data

# Univariate_Numerical_Histogram
def plot_histograms(data):
    plt.figure()
    num_cols = data.select_dtypes(include=['int64']).columns
    nrows = 1 
    ncols = 3
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        data[col].hist(ax=axes[i], bins=20)
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    return data

# Univariate_Numerical_Boxplot
def plot_boxplots(data):
    plt.figure()
    num_cols = data.select_dtypes(include=['int64']).columns
    nrows = 1  
    ncols = 3  
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() 

    for i, col in enumerate(num_cols):
        data[[col]].boxplot(ax=axes[i]) 
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel('Value')

    plt.tight_layout()
    st.pyplot(fig)
    return data

# Bivariate_Numerical_Scatterplot
def plot_scatterplots(data):
    plt.figure()
    combinations = [
        ('Age', 'Annual Income (k$)'),
        ('Age', 'Spending Score (1-100)'),
        ('Annual Income (k$)', 'Spending Score (1-100)')
    ]
    nrows = 1
    ncols = 3
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5)) 
    axes = axes.flatten() 
    
    for i, (x_col, y_col) in enumerate(combinations):
        axes[i].scatter(data[x_col], data[y_col])
        axes[i].set_title(f'{y_col} vs. {x_col}')
        axes[i].set_xlabel(x_col)
        axes[i].set_ylabel(y_col)
    
    plt.tight_layout()
    st.pyplot(fig)
    return data

# Bivariate_Numerical_CorrelationHeatMap
def plot_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and a set of subplots
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns  # Include float64 for more general applicability
    sns.heatmap(data[num_cols].corr().round(2), annot=True, cmap='coolwarm', mask=np.triu(np.ones_like(data[num_cols].corr(), dtype=bool)), ax=ax)
    plt.title('Heatmap of Correlation Matrix')
    st.pyplot(fig)

# Data Preprocessing
# 1) Remove Index
def remove_index(data):
    return data.reset_index(drop=True)

# 2) Remove Outliers
def remove_outliers(data):
    Q1 = data['Annual Income (k$)'].quantile(0.25)
    Q3 = data['Annual Income (k$)'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data_cleaned = data[(data['Annual Income (k$)'] >= lower_bound) & (data['Annual Income (k$)'] <= upper_bound)]

    return data_cleaned

# 3) One-Hot Encoding
def ohe(data):
    cat_cols = data.select_dtypes(include=['object']).columns
    num_cols = data.select_dtypes(include=['int64']).columns
    
    encoder = OneHotEncoder(sparse=False) 
    encoded_categories = encoder.fit_transform(data[cat_cols])
    encoded_data = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(input_features=cat_cols))
    data_cleaned = pd.concat([encoded_data, data[num_cols]], axis=1)
    
    return data_cleaned

# Clustering Models
# k-Means
def kmeans_clustering(data, k, scaler):
    cols_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if scaler == 'MinMaxScaling':
        minmax_scaler = MinMaxScaler()
        data_scaled = minmax_scaler.fit_transform(data[cols_to_scale])
        data_scaled = pd.DataFrame(data_scaled, columns=cols_to_scale)
        data_scaled = pd.concat([data_scaled, data.drop(cols_to_scale, axis=1)], axis=1)
    elif scaler == 'StandardScaling':
        standard_scaler = StandardScaler()
        data_scaled = standard_scaler.fit_transform(data[cols_to_scale])
        data_scaled = pd.DataFrame(data_scaled, columns=cols_to_scale)
        data_scaled = pd.concat([data_scaled, data.drop(cols_to_scale, axis=1)], axis=1)
    else:
        data_scaled = data.copy()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_scaled)
    data_scaled['Cluster'] = kmeans.labels_
    return data_scaled

# hierarchical agglomerative clustering (HAC)
def hac_clustering(data, k, scaler, linkage):
    cols_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if scaler == 'MinMaxScaling':
        minmax_scaler = MinMaxScaler()
        data_scaled = minmax_scaler.fit_transform(data[cols_to_scale])
        data_scaled = pd.DataFrame(data_scaled, columns=cols_to_scale)
        data_scaled = pd.concat([data_scaled, data.drop(cols_to_scale, axis=1)], axis=1)
    elif scaler == 'StandardScaling':
        standard_scaler = StandardScaler()
        data_scaled = standard_scaler.fit_transform(data[cols_to_scale])
        data_scaled = pd.DataFrame(data_scaled, columns=cols_to_scale)
        data_scaled = pd.concat([data_scaled, data.drop(cols_to_scale, axis=1)], axis=1)
    else:
        data_scaled = data.copy()

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    clusters_predict = model.fit_predict(data_scaled)
    data_scaled['Cluster'] = clusters_predict
    return data_scaled

# Model Validation using CVI
def compute_cvi(data):
    X = data.drop('Cluster', axis=1)
    y = data['Cluster']
    st.write(f"Davies bouldin score: {davies_bouldin_score(X, y)}")
    st.write(f"Calinski Score: {calinski_harabasz_score(X, y)}")
    st.write(f"Silhouette Score: {silhouette_score(X, y)}")

# Clustering Result Visualization using PCA
import streamlit as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Clustering Result Visualization using PCA for a single dataset
def plot_pca(data, title):
    fig, ax = plt.subplots(figsize=(18, 6))
    
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data.drop('Cluster', axis=1))

    scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.grid(True)
    st.pyplot(fig)
 
# Pre-computed Elbow Plot to determine k   
def elbow_plots(data):
    cols_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    inertia_dict = {}

    # Case 1: No preprocessing
    inertia_values_base = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        inertia_values_base.append(kmeans.inertia_)
    inertia_dict['Base_Case'] = inertia_values_base

    # Case 2: MinMaxScaling
    minmax_scaler = MinMaxScaler()
    data_minmax = minmax_scaler.fit_transform(data[cols_to_scale])
    data_minmax = pd.DataFrame(data_minmax, columns=cols_to_scale)
    data_minmax = pd.concat([data_minmax, data.drop(cols_to_scale, axis=1)], axis=1)
    inertia_values_minmax = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_minmax)
        inertia_values_minmax.append(kmeans.inertia_)
    inertia_dict['MinMaxScaling'] = inertia_values_minmax

    # Case 3: StandardScaling
    standard_scaler = StandardScaler()
    data_standard = standard_scaler.fit_transform(data[cols_to_scale])
    data_standard = pd.DataFrame(data_standard, columns=cols_to_scale)
    data_standard = pd.concat([data_standard, data.drop(cols_to_scale, axis=1)], axis=1)
    inertia_values_standard = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_standard)
        inertia_values_standard.append(kmeans.inertia_)
    inertia_dict['StandardScaling'] = inertia_values_standard

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['ElbowPlot_OHE_None', 'ElbowPlot_OHE_MinMaxScaling', 'ElbowPlot_OHE_StandardScaling']
    for i, (key, values) in enumerate(inertia_dict.items()):
        axes[i].plot(range(2, 11), values, marker='o')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('k')
        axes[i].set_ylabel('SSE')
    
    plt.tight_layout()
    st.pyplot(fig)

# Pre-computed Dendrograms to determine k
def hierarchical_dendrograms(data):
    cols_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scalers = {
        'OHE_None': None,
        'OHE_MinMaxScaling': MinMaxScaler(),
        'OHE_StandardScaling': StandardScaler()
    }

    linkage_methods = ['complete', 'single', 'average', 'ward']

    fig, axes = plt.subplots(len(scalers), len(linkage_methods), figsize=(15, 15))
    for i, (scaler_name, scaler) in enumerate(scalers.items()):
        for j, linkage_method in enumerate(linkage_methods):
            if scaler_name == 'OHE_None':
                data_scaled = data.copy()
            else:
                data_scaled = scaler.fit_transform(data[cols_to_scale])
                data_scaled = pd.DataFrame(data_scaled, columns=cols_to_scale)
                data_scaled = pd.concat([data_scaled, data.drop(cols_to_scale, axis=1)], axis=1)
            
            linkage_matrix = linkage(data_scaled, method=linkage_method)

            ax = axes[i, j]
            dendrogram(linkage_matrix, labels=None, ax=ax, orientation='top', distance_sort='ascending')
            ax.set_title(f'Dendrogram_{scaler_name}_{linkage_method.capitalize()}')
            ax.set_xlabel('Customer Instance')
            ax.set_ylabel('Distance')
            ax.set_xticks([])

    plt.tight_layout()
    st.pyplot(fig)
    
# Results Summary
def results_summary(datasets, titles):
    db = []
    ch = []
    sil = []
    for data in datasets:
        X = data.drop('Cluster', axis=1)
        y = data['Cluster']
        db.append(davies_bouldin_score(X, y))
        ch.append(calinski_harabasz_score(X, y))
        sil.append(silhouette_score(X, y))
    results = pd.DataFrame({'Davies Bouldin': db, 'Calinski Harabasz': ch, 'Silhouette': sil}, index=titles)
    st.dataframe(results)
    
def plot_pca_2d_kmeans_comparison(datasets, titles):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, data in enumerate(datasets):
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(data.drop('Cluster', axis=1))
        
        ax = axes[i]
        scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7, edgecolor='k')
        ax.set_title(titles[i])
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
def plot_pca_2d_hac_comparison(datasets, titles):
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    
    for i, data in enumerate(datasets):
        row = i // 4
        col = i % 4

        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(data.drop('Cluster', axis=1))
        
        ax = axes[row, col]
        scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7, edgecolor='k')
        ax.set_title(titles[i])
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
# Summary Statistics
def generate_summary_statistics(raw_data, best_data):
    result = raw_data
    result = result.pipe(remove_index).pipe(remove_outliers)
    result['Cluster'] = best_data['Cluster']
    
    df_group = result.groupby('Cluster').agg(
        {
            'Gender': lambda x: x.value_counts().index[0],
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
        }
    )
    df_group = df_group.reset_index()
    st.dataframe(df_group)
    
# Snake Plot
def plot_snake_plot(best_data):
    result_snake = best_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    result_snake['Cluster'] = best_data['Cluster']
    result_snake = pd.melt(result_snake.reset_index(), id_vars=['Cluster'], value_vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], var_name='Feature', value_name='Value')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=result_snake, x='Feature', y='Value', hue='Cluster', palette='viridis', ci=None, ax=ax)
    ax.set_title('Snake plot')
    st.pyplot(fig)


# Main Function
def main():
    st.title('Shopping Mall Customer Segmentation')
    st.sidebar.title("Navigation")
    sections = ["EDA", "Clustering", "Elbow Plots", "Dendrograms", "Results Summary", "Best Model"]
    selected_section = st.sidebar.radio("Go to", sections)

    df = pd.read_csv('Mall_Customers.csv', index_col='CustomerID')
    df_cleaned = df.pipe(remove_index).pipe(remove_outliers).pipe(ohe)
    
    kmeans_base = kmeans_clustering(df_cleaned, 5, None)
    kmeans_minmax = kmeans_clustering(df_cleaned, 4, 'MinMaxScaling')
    kmeans_standard = kmeans_clustering(df_cleaned, 4, 'StandardScaling')
    
    hac_base_complete = hac_clustering(df_cleaned, 2, None, 'complete')
    hac_base_single = hac_clustering(df_cleaned, 2, None, 'single')
    hac_base_average = hac_clustering(df_cleaned, 5, None, 'average')
    hac_base_ward = hac_clustering(df_cleaned, 3, None, 'ward')

    hac_minmax_complete = hac_clustering(df_cleaned, 2, 'MinMaxScaling', 'complete')
    hac_minmax_single = hac_clustering(df_cleaned, 2, 'MinMaxScaling', 'single')
    hac_minmax_average = hac_clustering(df_cleaned, 2, 'MinMaxScaling', 'average')
    hac_minmax_ward = hac_clustering(df_cleaned, 2, 'MinMaxScaling', 'ward')

    hac_standard_complete = hac_clustering(df_cleaned, 4, 'StandardScaling', 'complete')
    hac_standard_single = hac_clustering(df_cleaned, 2, 'StandardScaling', 'single')
    hac_standard_average = hac_clustering(df_cleaned, 2, 'StandardScaling', 'average')
    hac_standard_ward = hac_clustering(df_cleaned, 2, 'StandardScaling', 'ward')

    if selected_section == "EDA":
        st.header("Exploratory Data Analysis (EDA)")
        st.subheader("Data Overview")
        st.markdown('<h4 style="margin-bottom:0px;">First 5 rows</h4>', unsafe_allow_html=True)
        st.write(df.head())
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.markdown('<h4 style="margin-bottom:0px;">Dataframe Information</h4>', unsafe_allow_html=True)
        st.text(s)
        st.markdown('<h4 style="margin-bottom:0px;">Missing values</h4>', unsafe_allow_html=True)
        missing_values = df.isna().sum().reset_index()
        missing_values.columns = ['Feature', 'Number of Missing Values']
        st.dataframe(missing_values)
        st.markdown('<h4 style="margin-bottom:0px;">Number of duplicated rows</h4>', unsafe_allow_html=True)
        st.markdown(df.duplicated().sum())
        
        st.subheader("Univariate Data Exploration")
        st.markdown('<h4 style="margin-bottom:0px;">Categorical</h4>', unsafe_allow_html=True)
        st.markdown('<h5 style="margin-bottom:0px;">Count Plot</h5>', unsafe_allow_html=True)
        plot_countplots(df.copy())

        st.markdown('<h4 style="margin-bottom:0px;">Numerical</h4>', unsafe_allow_html=True)
        st.markdown('<h5 style="margin-bottom:0px;">Histogram</h5>', unsafe_allow_html=True)
        plot_histograms(df.copy())

        st.markdown('<h5 style="margin-bottom:0px;">Boxplot</h5>', unsafe_allow_html=True)
        plot_boxplots(df) # Adjust this function to use st.pyplot()

        st.subheader("Bivariate Data Exploration")
        st.markdown('<h5 style="margin-bottom:0px;">Scatter Plot</h5>', unsafe_allow_html=True)
        plot_scatterplots(df) # Adjust this function to use st.pyplot()

        st.markdown('<h5 style="margin-bottom:0px;">Correlation Matrix Heatmap</h5>', unsafe_allow_html=True)
        plot_heatmap(df) # Adjust this function to use st.pyplot()

    elif selected_section == "Clustering":
        st.header("Clustering")

        clustering_methods = ["K-Means", "Hierarchical Agglomerative Clustering (HAC)"]
        selected_method = st.selectbox("Select Clustering Method", clustering_methods)

        if selected_method == "K-Means":
            st.subheader("K-Means Clustering")
            scaling_method = st.selectbox("Select a data normalization approach", ["None", "MinMaxScaling", "StandardScaling"])
            
            if scaling_method == "None":
                clustered_data = kmeans_base
            elif scaling_method == "MinMaxScaling":
                clustered_data = kmeans_minmax
            else:
                clustered_data = kmeans_standard

            plot_pca(clustered_data, f'PCA_OHE_{scaling_method}_kMeans')
            compute_cvi(clustered_data)

        elif selected_method == "Hierarchical Agglomerative Clustering (HAC)":
            st.subheader("Hierarchical Agglomerative Clustering (HAC)")
            scaling_method = st.selectbox("Select a data normalization approach", ["None", "MinMaxScaling", "StandardScaling"])
            linkage = st.selectbox("Select a linkage", ["complete", "single", "average", "ward"])
            
            if scaling_method == "None":
                if linkage == "complete":
                    clustered_data = hac_base_complete
                elif linkage == "single":
                    clustered_data = hac_base_single
                elif linkage == "average":
                    clustered_data = hac_base_average
                elif linkage == "ward":
                    clustered_data = hac_base_ward
            elif scaling_method == "MinMaxScaling":
                if linkage == "complete":
                    clustered_data = hac_minmax_complete
                elif linkage == "single":
                    clustered_data = hac_minmax_single
                elif linkage == "average":
                    clustered_data = hac_minmax_average
                elif linkage == "ward":
                    clustered_data = hac_minmax_ward
            else:
                if linkage == "complete":
                    clustered_data = hac_standard_complete
                elif linkage == "single":
                    clustered_data = hac_standard_single
                elif linkage == "average":
                    clustered_data = hac_standard_average
                elif linkage == "ward":
                    clustered_data = hac_standard_ward

            # Plot PCA visualization and compute CVI scores
            plot_pca(clustered_data, f'PCA_OHE_{scaling_method}_{linkage}_HAC')
            compute_cvi(clustered_data)
    elif selected_section == "Elbow Plots":
        st.header("Elbow Plots")
        elbow_plots(df_cleaned)
    elif selected_section == "Dendrograms":
        st.header("Dendrograms")
        hierarchical_dendrograms(df_cleaned)
    elif selected_section == "Results Summary":
        st.header("Results Summary")
        results_summary([kmeans_base, kmeans_minmax, kmeans_standard, 
                         hac_base_complete, hac_base_single, hac_base_average, hac_base_ward,
                         hac_minmax_complete, hac_minmax_single, hac_minmax_average, hac_minmax_ward,
                         hac_standard_complete, hac_standard_single, hac_standard_average, hac_standard_ward], 
                        ['OHE_None_kMeans', 'OHE_MinMaxScaling_kMeans', 'OHE_StandardScaling_kMeans', 
                         'OHE_None_complete_HAC', 'OHE_None_single_HAC', 'OHE_None_average_HAC', 'OHE_None_ward_HAC',
                         'OHE_MinMaxScaling_complete_HAC', 'PCA_OHE_MinMaxScaling_single_HAC', 'PCA_OHE_MinMaxScaling_average_HAC', 'PCA_OHE_MinMaxScaling_ward_HAC',
                         'OHE_StandardScaling_complete_HAC', 'PCA_OHE_StandardScaling_single_HAC', 'PCA_OHE_StandardScaling_average_HAC', 'PCA_OHE_StandardScaling_ward_HAC'])
        st.subheader("k-Means (PCA)")
        plot_pca_2d_kmeans_comparison([kmeans_base, kmeans_minmax, kmeans_standard],
                                      ['PCA_OHE_None_kMeans', 'PCA_OHE_MinMaxScaling_kMeans', 'PCA_OHE_StandardScaling_kMeans'])
        st.subheader("Hierarchical Agglomerative Clustering (PCA)")
        datasets = [hac_base_complete, hac_base_single, hac_base_average, hac_base_ward,
                    hac_minmax_complete, hac_minmax_single, hac_minmax_average, hac_minmax_ward,
                    hac_standard_complete, hac_standard_single, hac_standard_average, hac_standard_ward]
        titles = ['PCA_OHE_None_complete_HAC', 'PCA_OHE_None_single_HAC', 'PCA_OHE_None_average_HAC', 'PCA_OHE_None_ward_HAC',
                  'PCA_OHE_MinMaxScaling_complete_HAC', 'PCA_OHE_MinMaxScaling_single_HAC', 'PCA_OHE_MinMaxScaling_average_HAC', 'PCA_OHE_MinMaxScaling_ward_HAC',
                'PCA_OHE_StandardScaling_complete_HAC', 'PCA_OHE_StandardScaling_single_HAC', 'PCA_OHE_StandardScaling_average_HAC', 'PCA_OHE_StandardScaling_ward_HAC']
        plot_pca_2d_hac_comparison(datasets, titles)
    elif selected_section == "Best Model":
        st.header("Summary Statistics")
        generate_summary_statistics(df.copy(), kmeans_base)
        st.header("Snake Plot")
        plot_snake_plot(kmeans_base)
 
# Call to run the Streamlit application
if __name__ == "__main__":
    main()
