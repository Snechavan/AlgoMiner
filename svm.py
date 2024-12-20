import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog

def load_data():
    # Open file dialog for user to select dataset
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

    # Ask user for kernel type and regularization parameter (C)
    allowed_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    while True:
        kernel_type = input(f"Enter kernel type ({', '.join(allowed_kernels)}): ")
        if kernel_type in allowed_kernels:
            break
        print("Invalid kernel type. Please try again.")

    while True:
        try:
            C = float(input("Enter regularization parameter (C): "))
            break
        except ValueError:
            print("Invalid input. Please enter a float value.")

    # Create SVM classifier
    svm_model = svm.SVC(kernel=kernel_type, C=C)

    # Train SVM classifier
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Model Accuracy: {accuracy:.2f}")

def run_svm():
    root = tk.Tk()
    root.withdraw()
    df = load_data()
    if df is not None:
        train_model(df)

if __name__ == "__main__":
    run_svm()