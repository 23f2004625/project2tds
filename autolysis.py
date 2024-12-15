# /// script
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "requests",
#   "chardet",
#   "numpy",
#   "joblib",
#   "folium",
#   "plotly",
#   "Pillow",
#   "geopy",
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import requests
import argparse
from typing import List, Dict

# === Validate and retrieve the AI Proxy Token ===
try:
    AI_PROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
except KeyError:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# === Utility Functions ===
def detect_encoding(filepath: str) -> str:
    """
    Detect the encoding of a CSV file to ensure it is read correctly.
    """
    try:
        with open(filepath, 'rb') as f:
            import chardet
            result = chardet.detect(f.read())
            return result['encoding']
    except ImportError:
        os.system(f'{sys.executable} -m pip install chardet')
        import chardet
        with open(filepath, 'rb') as f:
            result = chardet.detect(f.read())
            return result['encoding']

def save_plot(filename: str, fig=None):
    """
    Save a plot with compression.
    """
    if fig is None:
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the dataset for basic requirements.
    """
    if df.empty:
        raise ValueError("Dataset is empty!")
    if df.isnull().all().all():
        raise ValueError("Dataset contains only null values!")

# === Data Analysis Functions ===
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file, ensuring correct encoding.
    """
    try:
        encoding = detect_encoding(filepath)
        df = pd.read_csv(filepath, encoding=encoding)
        validate_dataset(df)
        return df
    except Exception as e:
        print(f"Error loading dataset {filepath}: {e}")
        sys.exit(1)

def clean_and_select_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only numeric columns for analysis, excluding any non-numeric columns (e.g., dates).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df

def perform_correlation_analysis(df: pd.DataFrame, save_path: str):
    """
    Perform correlation analysis and save the heatmap.
    """
    numeric_df = clean_and_select_numeric_columns(df)
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=90)  # Rotate the x-labels to be vertical
    save_plot(save_path)

def perform_pca_and_clustering(df: pd.DataFrame, save_path: str):
    """
    Perform PCA and KMeans clustering, then save the scatter plot.
    """
    numeric_df = clean_and_select_numeric_columns(df)
    if numeric_df.empty:
        print("No numeric columns available for PCA and Clustering.")
        return

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(numeric_df.dropna())  # Ensure no NaN values are present
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(reduced_data)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="viridis")
    plt.title("PCA with KMeans Clustering")
    save_plot(save_path)

def detect_and_plot_outliers(data, output_file):
    """
    Detect outliers using KMeans and plot the results.
    Outliers are identified based on distance from the centroids of the clusters.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Handle missing data by imputing with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(numeric_data)
    
    # Standardize the data to have zero mean and unit variance
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    # Fit KMeans with 3 clusters (you can modify n_clusters based on your data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data_scaled)
    
    # Compute the distances of points to their respective centroids
    distances = kmeans.transform(data_scaled).min(axis=1)
    
    # Define the threshold for outliers (3 standard deviations above the mean distance)
    threshold = distances.mean() + 3 * distances.std()
    outliers = distances > threshold

    # Plotting the outliers vs non-outliers
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot non-outliers in blue
    ax.scatter(np.where(~outliers)[0], distances[~outliers], c='blue', label="Non-Outliers", alpha=0.6)
    
    # Plot outliers in red
    ax.scatter(np.where(outliers)[0], distances[outliers], c='red', label="Outliers", alpha=0.6)
    
    ax.set_title("Outlier Detection (KMeans)", fontsize=16)
    ax.set_xlabel("Data Points", fontsize=12)
    ax.set_ylabel("Distance from Centroid", fontsize=12)
    ax.legend()
    
    # Save the plot
    save_plot(output_file, fig)

# === Narrative Generation with AI Proxy ===
def generate_story(summary: Dict, visualizations: List[str]) -> str:
    """
    Generate a comprehensive narrative using the AI Proxy API.
    """
    summary_str = f"Summary: {summary['summary_stats']}\nMissing: {summary['missing_values']}"
    prompt = f"""
    Given the following data analysis results:
    {summary_str}
    And the visualizations generated:
    - Outlier Detection
    - Correlation Heatmap
    - PCA Clustering
    - Time Series Analysis (if present)
    - Geographic Analysis (if present)

    Provide a comprehensive narrative with key findings and insights.
    """
    headers = {"Authorization": f"Bearer {AI_PROXY_TOKEN}", "Content-Type": "application/json"}
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500},
        headers=headers,
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Failed to generate story. Error {response.status_code}: {response.text}"

def summarize_data(df: pd.DataFrame) -> Dict:
    """
    Summarize the dataset with key statistics and missing values.
    """
    summary_stats = df.describe().to_dict()
    missing_values = df.isnull().sum().to_dict()
    return {
        'summary_stats': summary_stats,
        'missing_values': missing_values
    }

def create_markdown_report(story: str, output_path: str, visualizations: List[str]):
    """
    Save the narrative as a Markdown file.
    The visualizations are linked in the Markdown file.
    """
    with open(output_path, 'w') as f:
        f.write("# Analysis Report\n\n")
        f.write(story + "\n\n")
        f.write("## Visualizations\n")
        for viz in visualizations:
            f.write(f"![{viz}]({viz})\n")

# === Main Workflow ===
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze a CSV dataset and generate a report.")
    parser.add_argument("csv_file", help="Path to the CSV file to analyze")
    args = parser.parse_args()
    
    dataset_path = args.csv_file
    output_dir = "."

    print(f"Processing dataset: {dataset_path}")
    try:
        df = load_data(dataset_path)
        summary = summarize_data(df)

        # Correlation Heatmap
        heatmap_path = os.path.join(output_dir, f"{dataset_path}_correlation.png")
        perform_correlation_analysis(df, heatmap_path)

        # PCA and KMeans
        pca_path = os.path.join(output_dir, f"{dataset_path}_pca.png")
        perform_pca_and_clustering(df, pca_path)

        # Outlier Detection
        outliers_path = os.path.join(output_dir, f"{dataset_path}_outliers.png")
        detect_and_plot_outliers(df, outliers_path)

        # Generate Story
        visualizations = [heatmap_path, pca_path, outliers_path]
        story = generate_story(summary, visualizations)

        # Save Markdown Report as README.md
        report_path = os.path.join(output_dir, "README.md")
        create_markdown_report(story, report_path, visualizations)

        print(f"Analysis for {dataset_path} completed successfully!")

    except Exception as e:
        print(f"Error processing {dataset_path}: {e}")

if __name__ == "__main__":
    main()
