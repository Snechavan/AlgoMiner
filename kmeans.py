import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_data():
    # Open file dialog for user to select dataset
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV Files", "*.csv")])

    # Load selected dataset
    try:
        df = pd.read_csv(file_path)
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def train_model(df, n_clusters):
    # Select features (all columns except last)
    X = df.iloc[:, :-1]

    # Create KMeans clustering model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit KMeans model
    kmeans.fit(X)

    # Evaluate model
    silhouette = silhouette_score(X, kmeans.labels_)
    print(f"KMeans Clustering Silhouette Score: {silhouette:.2f}")

    return X, kmeans


def visualize_clusters(X, kmeans):
    # Convert DataFrame to NumPy array
    X_array = X.values

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot clusters
    ax.scatter(X_array[:, 0], X_array[:, 1], c=kmeans.labels_, cmap='viridis')

    # Plot centroids
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*',
               label='Centroids')

    # Show legend
    ax.legend()

    return fig

def run_kmeans():
    df = load_data()
    if df is not None:
        # Create main window
        window = tk.Tk()
        window.title("KMeans Clustering")

        # Get number of clusters from user
        n_clusters = int(input("Enter number of clusters: "))

        # Train model
        X, kmeans = train_model(df, n_clusters)

        # Visualize clusters
        fig = visualize_clusters(X, kmeans)

        # Add plot to window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Run application
        window.mainloop()

if __name__ == "__main__":
    run_kmeans()