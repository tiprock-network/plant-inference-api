from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, name):

    acc = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{name} Results")
    print("Accuracy:", acc)
    print("Macro Precision:", precision_macro)
    print("Macro Recall:", recall_macro)
    print("Macro F1:", f1_macro)
    print("Weighted Precision:", precision_weighted)
    print("Weighted Recall:", recall_weighted)
    print("Weighted F1:", f1_weighted)

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))


import time
import os
import pickle
import pandas as pd
import numpy as np

# Create a directory to save models
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Global list to store comparison data
model_comparison_data = []

def track_model_performance(model, model_name, X_test, y_test, y_pred, train_time):
    """
    Saves model, calculates size, latency, and stores all metrics for the final table.
    """
    # 1. Save and Get Size (MB)
    file_path = f'saved_models/{model_name.replace(" ", "_")}.pkl'
    
    # Special handling for TensorFlow/Keras models if needed
    if "TensorFlow" in model_name:
        file_path = f'saved_models/{model_name.replace(" ", "_")}.h5'
        model.save(file_path)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # 2. Measure Latency (Time per sample)
    start_lat = time.time()
    _ = model.predict(X_test)
    end_lat = time.time()
    avg_latency = (end_lat - start_lat) / len(X_test)
    
    # 3. Collect Metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 4. Append to comparison list
    return{
        "Model": model_name,
        "Size (MB)": round(size_mb, 4),
        "Training Time (s)": round(train_time, 4),
        "Latency (s/sample)": f"{avg_latency:.6f}",
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    }
    
    



def utils_confusion_matrix(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()