import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import tree
import numpy as np

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

def train_model(df):
    # Select target variable (last column)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=42)

    # Train Decision Tree classifier
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Model Accuracy: {accuracy:.2f}")

    return X_test, y_test, y_pred, dt, df.iloc[:, -1].unique()

def visualize_results(X_test, y_test, y_pred, dt, class_names):
    # Create figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot confusion matrix
    ax[0].imshow(confusion_matrix(y_test, y_pred), interpolation='nearest')
    ax[0].set_title('Confusion Matrix')
    ax[0].set_xlabel('Predicted labels')
    ax[0].set_ylabel('True labels')

    # Plot decision tree
    tree.plot_tree(dt, feature_names=X_test.columns, class_names=class_names, filled=True)
    ax[1].set_title('Decision Tree')

    return fig

def run_decision_tree():
    df = load_data()
    if df is not None:
        X_test, y_test, y_pred, dt, class_names = train_model(df)
        fig = visualize_results(X_test, y_test, y_pred, dt, class_names)

        # Create main window
        window = tk.Tk()
        window.title("Decision Tree Classifier")

        # Add plot to window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Run application
        window.mainloop()

if __name__ == "__main__":
    run_decision_tree()