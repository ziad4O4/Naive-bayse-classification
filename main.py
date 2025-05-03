import pandas as pd
import numpy as np
import tkinter as tk
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
from tkinter import filedialog, messagebox, simpledialog
from collections import Counter


# === GUI Setup ===
root = tk.Tk()
root.title("Naive Bayes Classifier")
root.geometry("600x700")
root.configure(bg="#f0f4f7")

df = None
input_entries = {}
label_encoders = {}
model = None
X_columns = []
target_column = ""
y_true = []
y_pred = []
accuracy_computed = False

def load_dataset(file_path=None):
    global label_encoders, model, X_columns, target_column, y_true, y_pred, accuracy_computed, df

    if not file_path:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")
        return

    target_column_input = simpledialog.askstring("Target Column", "Enter the name of the target column:")
    if target_column_input not in df.columns:
        messagebox.showerror("Error", "Target column not found in dataset.")
        return

    target_column = target_column_input
    encoded_df = df.copy()
    label_encoders = {}
    
    for column in df.columns:
        if column == target_column:
            # Always label encode the target column
            le = LabelEncoder()
            encoded_df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        else:
            if df[column].dtype == 'object':
                # Only encode categorical features
                le = LabelEncoder()
                encoded_df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            else:
                # If numeric, don't encode, just remember it
                label_encoders[column] = None

    X = encoded_df.drop(target_column, axis=1)
    y = encoded_df[target_column]
    X_columns.clear()
    X_columns.extend(X.columns.tolist())

    model = CategoricalNB()
    model.fit(X, y)

    y_pred.clear()
    y_true.clear()
    y_pred.extend(model.predict(X))
    y_true.extend(y)

    accuracy_computed = False

    # Create input fields
    for widget in input_frame.winfo_children():
        widget.destroy()
    input_entries.clear()
    for col in X_columns:
        lbl = tk.Label(input_frame, text=col, bg="#ffffff", fg="#333333", font=("Arial", 10))
        lbl.pack(pady=3)
        entry = tk.Entry(input_frame, font=("Arial", 10))
        entry.pack(pady=3)
        input_entries[col] = entry

    messagebox.showinfo("Success", "Dataset loaded and model trained successfully!")
    result_label.config(text="")
    accuracy_label.config(text="")

def classify():
    global accuracy_computed
    if model is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    input_data = {}
    try:
        for col in X_columns:
            val = input_entries[col].get()
            if not val:
                messagebox.showerror("Error", f"Please enter a value for: {col}")
                return

            le = label_encoders[col]
            if le is not None:
                input_data[col] = le.transform([val])[0]
            else:
                # Numeric column, convert manually
                input_data[col] = float(val)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input for column '{col}': {e}")
        return

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    predicted_class = label_encoders[target_column].inverse_transform(prediction)
    result_label.config(text=f"Predicted Class: {predicted_class[0]}", fg="#1e8449")

    #accuracy  
    if y_true and y_pred and not accuracy_computed:
        acc = accuracy_score(y_true, y_pred)
        accuracy_label.config(text=f"Model Accuracy: {acc:.2f}", fg="#2c3e50")
        accuracy_computed = True

def show_steps():
    if model is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    steps_window = tk.Toplevel(root)
    steps_window.title("Naive Bayes Classification Steps")
    steps_window.geometry("400x600")
    steps_window.configure(bg="#f0f4f7")

    canvas = tk.Canvas(steps_window, bg="#f0f4f7")
    scrollbar = tk.Scrollbar(steps_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="#ffffff")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    title = tk.Label(scrollable_frame, text="Naive Bayes Detailed Steps", font=("Arial", 18, "bold"), bg="#ffffff", fg="#2c3e50")
    title.pack(pady=10)

    # Prior Probabilities
    prior_label = tk.Label(scrollable_frame, text="Prior Probabilities:", font=("Arial", 14, "bold"), bg="#ffffff", fg="#2980b9")
    prior_label.pack(pady=10)

    priors = Counter(y_true)
    total_samples = len(y_true)

    for cls, count in priors.items():
        cls_name = label_encoders[target_column].inverse_transform([cls])[0]
        prob = count / total_samples
        lbl = tk.Label(scrollable_frame, text=f"P({cls_name}) = {count}/{total_samples} = {prob:.4f}", bg="#ffffff", font=("Arial", 12))
        lbl.pack()

    # Likelihoods (per feature)
    likelihood_label = tk.Label(scrollable_frame, text="\nLikelihoods (P(Feature=Value | Class)):", font=("Arial", 14, "bold"), bg="#ffffff", fg="#27ae60")
    likelihood_label.pack(pady=10)

    for feature in X_columns:
        feature_frame = tk.LabelFrame(scrollable_frame, text=f"Feature: {feature}", bg="#ecf0f1", fg="#2c3e50", font=("Arial", 12, "bold"), padx=10, pady=10)
        feature_frame.pack(pady=5, padx=10, fill="x", expand=True)

        unique_feature_values = np.unique([row for row in df[feature]])
        for value in unique_feature_values:
            for cls in np.unique(y_true):
                count_value_and_class = ((df[feature] == value) & (df[target_column] == label_encoders[target_column].inverse_transform([cls])[0])).sum()
                count_class = (df[target_column] == label_encoders[target_column].inverse_transform([cls])[0]).sum()

                if count_class != 0:
                    likelihood = count_value_and_class / count_class
                else:
                    likelihood = 0

                lbl = tk.Label(feature_frame, text=f"P({feature}={value} | {label_encoders[target_column].inverse_transform([cls])[0]}) = {count_value_and_class}/{count_class} = {likelihood:.4f}", bg="#ecf0f1", font=("Arial", 11))
                lbl.pack(anchor="w")

    # Prediction Formula Info
    formula_label = tk.Label(scrollable_frame, text="\nPrediction Formula:", font=("Arial", 14, "bold"), bg="#ffffff", fg="#c0392b")
    formula_label.pack(pady=10)

    formula_info = tk.Label(scrollable_frame, text="Posterior Probability = Prior × (Product of Likelihoods)", bg="#ffffff", font=("Arial", 12))
    formula_info.pack()

# === GUI Components ===
title_label = tk.Label(root, text="Naive Bayes Classifier", font=("Arial", 16, "bold"), bg="#f0f4f7", fg="#2c3e50")
title_label.pack(pady=10)

frame = tk.Frame(root, bg="#e8eff5", bd=2, relief="groove")
frame.pack(padx=20, pady=10, fill="x")

load_btn = tk.Button(frame, text="Import Dataset", command=lambda: load_dataset(), bg="#3498db", fg="white", font=("Arial", 10))
load_btn.pack(pady=10)

input_section_label = tk.Label(root, text="Enter Feature Values for Prediction:", bg="#f0f4f7", fg="#2c3e50", font=("Arial", 12, "bold"))
input_section_label.pack(pady=10)

input_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="sunken")
input_frame.pack(pady=5, padx=20, fill="both", expand=True)

predict_btn = tk.Button(root, text="Classify", command=classify, bg="#27ae60", fg="white", font=("Arial", 12, "bold"))
predict_btn.pack(pady=15)

result_label = tk.Label(root, text="", bg="#f0f4f7", font=("Arial", 14))
result_label.pack(pady=10)

accuracy_label = tk.Label(root, text="", bg="#f0f4f7", font=("Arial", 12))
accuracy_label.pack(pady=10)

show_steps_btn = tk.Button(root, text="Show Steps", command=show_steps, bg="#27ae60", fg="white", font=("Arial", 12, "bold"))
show_steps_btn.pack(pady=15)

root.mainloop()