import tkinter as tk
from tkinter import ttk, messagebox
import knn
import kmeans
import decision_tree
import svm  # Import SVM module
import RandomForest
from PIL import Image, ImageTk

# Login Page
class LoginPage:
    def __init__(self):
        self.window =tk.Tk()
        self.window.title("Data Mining Tool - Login")
        self.window.geometry("800x800")

        # Background image
        image_path = r"C:\Users\ADMIN\Desktop\dmnew.jpg"
        image = Image.open(image_path)
        resized_image = image.resize((1200, 800))  # Set desired size
        self.background = ImageTk.PhotoImage(resized_image)
        self.background_label = tk.Label(self.window, image=self.background)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Login frame
        self.login_frame = tk.Frame(self.window, bg="white", highlightthickness=2, highlightbackground="gray")
        self.login_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=300, height=200)

        # Username label and entry
        self.username_label = tk.Label(self.login_frame, text="Username:", font=("Arial", 12))
        self.username_label.grid(row=0, column=0, padx=10, pady=10)
        self.username_entry = tk.Entry(self.login_frame, font=("Arial", 12), width=20)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)

        # Password label and entry
        self.password_label = tk.Label(self.login_frame, text="Password:", font=("Arial", 12))
        self.password_label.grid(row=1, column=0, padx=10, pady=10)
        self.password_entry = tk.Entry(self.login_frame, font=("Arial", 12), show="*", width=20)
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)

        # Login button
        self.login_button = tk.Button(self.login_frame, text="Login", command=self.check_credentials,
                                      font=("Arial", 12), bg="blue", fg="white")
        self.login_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def check_credentials(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if username == "admin" and password == "admin":
            self.window.destroy()
            home_page = HomePage()
            home_page.run()
        else:
            messagebox.showerror("Invalid Credentials", "Username or password is incorrect.")

    def run(self):
        self.window.mainloop()

# Home Page
class HomePage:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Data Mining Tool - Home")
        self.window.geometry("800x600")

        # Background image
        image_path = r"C:\Users\ADMIN\Desktop\dmm.jpg"
        image = Image.open(image_path)
        resized_image = image.resize((800, 600))
        self.background = ImageTk.PhotoImage(resized_image)
        self.background_label = tk.Label(self.window, image=self.background)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Create start button to open new window with tabs
        self.start_button = tk.Button(self.window, text="Start", command=self.open_tabs)

        # Pack the button to the center
        self.start_button.pack(expand=True)

    def open_tabs(self):
        # Create new window with tabs
        self.tabs_window = tk.Toplevel(self.window)
        self.tabs_window.title("Data Mining Tool - Tabs")

        # Create tabs
        self.tab_control = ttk.Notebook(self.tabs_window)
        self.knn_tab = ttk.Frame(self.tab_control)
        self.kmeans_tab = ttk.Frame(self.tab_control)
        self.decision_tree_tab = ttk.Frame(self.tab_control)
        self.svm_tab = ttk.Frame(self.tab_control)
        self.rf_tab = ttk.Frame(self.tab_control)  # Add Random Forest tab

        # Add tabs
        self.tab_control.add(self.knn_tab, text="KNN")
        self.tab_control.add(self.kmeans_tab, text="K-Means")
        self.tab_control.add(self.decision_tree_tab, text="Decision Tree")
        self.tab_control.add(self.svm_tab, text="SVM")
        self.tab_control.add(self.rf_tab, text="Random Forest")  # Add Random Forest tab
        self.tab_control.pack(expand=1, fill="both")

        # Create algorithm implementation buttons
        self.knn_button = tk.Button(self.knn_tab, text="Run KNN", command=knn.run_knn)
        self.kmeans_button = tk.Button(self.kmeans_tab, text="Run K-Means", command=kmeans.run_kmeans)
        self.decision_tree_button = tk.Button(self.decision_tree_tab, text="Run Decision Tree",
                                              command=decision_tree.run_decision_tree)
        self.svm_button = tk.Button(self.svm_tab, text="Run SVM", command=svm.run_svm)
        self.rf_button = tk.Button(self.rf_tab, text="Run Random Forest", command=RandomForest.run_rf)  # Add Random Forest button

        # Layout
        self.knn_button.pack()
        self.kmeans_button.pack()
        self.decision_tree_button.pack()
        self.svm_button.pack()
        self.rf_button.pack()  # Layout Random Forest button

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    login_page = LoginPage()
    login_page.run()