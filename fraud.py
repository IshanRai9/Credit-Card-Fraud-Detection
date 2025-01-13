import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load dataset
def load_data():
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)
            messagebox.showinfo("File Loaded", "Dataset successfully loaded.")
            return data
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    return None


# Function to preprocess the data and split into train/test
def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Function to train the model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy


# Function to visualize fraud vs non-fraud distribution with pie chart
def plot_fraud_distribution(data):
    fraud_counts = data['Class'].value_counts()
    labels = ['Non-Fraud (0)', 'Fraud (1)']
    plt.figure(figsize=(6, 6))
    plt.pie(fraud_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'],
            explode=[0, 0.1])
    plt.title("Fraud vs Non-Fraud Transactions Distribution")
    plt.show()
    plt.close()


# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    plt.close()


# Function to display metrics
def plot_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Detection in Financial Transactions")
        self.root.configure(bg="#f0f0f0")

        # Create gradient background
        self.canvas = tk.Canvas(self.root, bg="white")
        self.canvas.pack(fill="both", expand=True)
        self.create_gradient_background()

        # Create main frame
        self.main_frame = tk.Frame(self.root, bg="#ffffff", bd=5, relief=tk.RAISED)
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Add title
        self.title_label = tk.Label(self.main_frame, text="Fraud Detection App", font=("Arial", 30, "bold"),
                                    bg="#ffffff")
        self.title_label.pack(pady=20)

        # Load Data button
        self.load_button = tk.Button(self.main_frame, text="Load Dataset", command=self.load_dataset, bg="#4CAF50",
                                     fg="white", font=("Arial", 16), width=20)
        self.load_button.pack(pady=10)

        # Train Model button
        self.train_button = tk.Button(self.main_frame, text="Train Model", command=self.train_model, state=tk.DISABLED,
                                      bg="#2196F3", fg="white", font=("Arial", 16), width=20)
        self.train_button.pack(pady=10)

        # Show Accuracy button
        self.accuracy_button = tk.Button(self.main_frame, text="Show Accuracy", command=self.show_accuracy,
                                         state=tk.DISABLED, bg="#FF9800", fg="white", font=("Arial", 16), width=20)
        self.accuracy_button.pack(pady=10)

        # Show Fraud Distribution button
        self.distribution_button = tk.Button(self.main_frame, text="Fraud Distribution",
                                             command=self.plot_fraud_distribution, state=tk.DISABLED, bg="#FF5722",
                                             fg="white", font=("Arial", 16), width=20)
        self.distribution_button.pack(pady=10)

        # Show Confusion Matrix button
        self.confusion_button = tk.Button(self.main_frame, text="Confusion Matrix", command=self.plot_confusion_matrix,
                                          state=tk.DISABLED, bg="#9C27B0", fg="white", font=("Arial", 16), width=20)
        self.confusion_button.pack(pady=10)

        # Show Metrics button
        self.metrics_button = tk.Button(self.main_frame, text="Show Metrics", command=self.plot_metrics,
                                        state=tk.DISABLED, bg="#3F51B5", fg="white", font=("Arial", 16), width=20)
        self.metrics_button.pack(pady=10)

        # Show Fraud Transactions button
        self.fraud_button = tk.Button(self.main_frame, text="Show Fraud Transactions",
                                      command=self.show_fraud_transactions, state=tk.DISABLED, bg="#E91E63", fg="white",
                                      font=("Arial", 16), width=20)
        self.fraud_button.pack(pady=10)

        self.data = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def create_gradient_background(self):
        self.gradient_colors = ["#ffcccb", "#ffb3ba", "#ff677d", "#d4a5a5", "#392f5a"]
        height = self.root.winfo_height()
        width = self.root.winfo_width()
        for i, color in enumerate(self.gradient_colors):
            self.canvas.create_rectangle(0, i * (height / len(self.gradient_colors)), width,
                                         (i + 1) * (height / len(self.gradient_colors)), fill=color, outline=color)

    def load_dataset(self):
        self.data = load_data()
        if self.data is not None:
            self.train_button.config(state=tk.NORMAL)
            self.distribution_button.config(state=tk.NORMAL)
            self.fraud_button.config(state=tk.NORMAL)

    def train_model(self):
        if self.data is not None:
            X_train, self.X_test, y_train, self.y_test = preprocess_data(self.data)
            self.model = train_model(X_train, y_train)
            self.y_pred, self.accuracy = evaluate_model(self.model, self.X_test, self.y_test)
            messagebox.showinfo("Model Trained", "The model has been trained.")
            self.accuracy_button.config(state=tk.NORMAL)
            self.confusion_button.config(state=tk.NORMAL)
            self.metrics_button.config(state=tk.NORMAL)

    def show_accuracy(self):
        if self.accuracy is not None:
            messagebox.showinfo("Accuracy", f"Model Accuracy: {self.accuracy * 100:.2f}%")

    def plot_fraud_distribution(self):
        if self.data is not None:
            plot_fraud_distribution(self.data)

    def plot_confusion_matrix(self):
        if self.y_test is not None and self.y_pred is not None:
            plot_confusion_matrix(self.y_test, self.y_pred)

    def plot_metrics(self):
        self.accuracy, self.precision, self.recall, self.f1 = plot_metrics(self.y_test, self.y_pred)
        if self.accuracy is not None and self.precision is not None and self.recall is not None and self.f1 is not None:
            messagebox.showinfo("Metrics",
                                f"Model Accuracy: {self.accuracy * 100:.2f}% \nModel Precision: {self.precision * 100:.2f}% \nModel Recall: {self.recall * 100:.2f}% \nModel F-1 score: {self.f1 * 100:.2f}%")

    def show_fraud_transactions(self):
        if self.y_pred is not None and self.y_test is not None:
            # True positives (fraud detected correctly)
            true_positives = np.sum((self.y_pred == 1) & (self.y_test == 1))

            # False positives (incorrectly flagged as fraud)
            false_positives = np.sum((self.y_pred == 1) & (self.y_test == 0))

            # True fraud transactions in the dataset
            actual_fraud_count = np.sum(self.y_test == 1)

            messagebox.showinfo(
                "Fraud Transactions",
                f"Number of Actual Fraudulent Transactions: {actual_fraud_count}\n"
                f"Correctly Predicted Fraudulent Transactions (True Positives): {true_positives}\n"
                f"Incorrectly Predicted Fraudulent Transactions (False Positives): {false_positives}"
            )

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    root.attributes('-fullscreen', True)  # Set the window to fullscreen
    app = FraudDetectionApp(root)
    root.mainloop()
