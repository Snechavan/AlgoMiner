import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
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

def train_model(df):
    # Select features (all columns except last)
    X = df.iloc[:, :-1]

    # Select target variable (last column)
    y = df.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ask user for hyperparameters
    while True:
        try:
            n_estimators = int(input("Enter number of estimators: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")

    while True:
        try:
            max_depth = int(input("Enter maximum depth: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")

    while True:
        criterion = input("Enter criterion (gini, entropy): ")
        if criterion in ['gini', 'entropy']:
            break
        print("Invalid criterion. Please enter 'gini' or 'entropy'.")

    # Create Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

    # Train Random Forest Classifier
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy:.2f}")

    return X_test, y_test, y_pred

def visualize_results(X_test, y_test, y_pred):
    # Create figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot confusion matrix
    ax[0].imshow(confusion_matrix(y_test, y_pred), interpolation='nearest')
    ax[0].set_title('Confusion Matrix')
    ax[0].set_xlabel('Predicted labels')
    ax[0].set_ylabel('True labels')

    # Plot feature importance
    ax[1].bar(X_test.columns, RandomForestClassifier().fit(X_test, y_test).feature_importances_)
    ax[1].set_title('Feature Importance')
    ax[1].set_xlabel('Features')
    ax[1].set_ylabel('Importance')

    return fig

def run_rf():
    root = tk.Tk()
    root.withdraw()
    df = load_data()
    if df is not None:
        X_test, y_test, y_pred = train_model(df)
        fig = visualize_results(X_test, y_test, y_pred)

        # Create main window
        window = tk.Tk()
        window.title("Random Forest Classifier")

        # Add plot to window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Run application
        window.mainloop()

if __name__ == "__main__":
    run_rf()