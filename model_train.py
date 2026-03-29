import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
from utils.model_eval import evaluate_model, track_model_performance


    
#TODO: This needs to have an online batch query instead of static data path to train the examples
def load_and_preprocess_data(data_path: str):
    # 1. Load and create an explicit copy to avoid SettingWithCopyWarning
    df = pd.read_csv(data_path)
    cols = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Crop']
    
    # .copy() ensures sensor_df is its own object in memory
    sensor_df = df[cols].copy()
    
    # 2. Clean and Encode Target
    sensor_df["Crop"] = sensor_df["Crop"].str.lower()
    
    # Using a dictionary comprehension is cleaner than a manual loop
    # In load_and_preprocess_data
    CROP_MAPPING = {'orange': 0, 'pomegranate': 1, 'apple': 2, 'banana': 3, 'watermelon': 4, 'coconut': 5, 'chickpea': 6, 'sugarcane': 7, 'mango': 8, 'mothbeans': 9, 'grapes': 10, 'potato': 11, 'rice': 12, 'cotton': 13, 'blackgram': 14, 'kidneybeans': 15, 'jute': 16, 'lentil': 17, 'papaya': 18, 'coffee': 19, 'wheat': 20, 'tomato': 21, 'muskmelon': 22, 'mungbean': 23, 'maize': 24, 'pigeonpeas': 25}
    
    sensor_df["Crop"] = sensor_df["Crop"].str.lower().map(CROP_MAPPING)

    
    
    # 3. Encode Variety
    #le_variety = LabelEncoder()
    #sensor_df['Variety'] = le_variety.fit_transform(sensor_df['Variety'])

    # 4. Split Features and Target
    X = sensor_df.drop(columns=["Crop"])
    y = sensor_df["Crop"]
    feature_names = X.columns.tolist()

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 6. Apply Log Transformation
    # We create copies here to keep the original X_train/X_test intact
    skewed_features = ["Nitrogen","Phosphorus", "Potassium", "Humidity"]
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()

    for col in skewed_features:
        X_train_log[col] = np.log1p(X_train_log[col])
        X_test_log[col] = np.log1p(X_test_log[col])
    
    return X, y, X_train, X_test, y_train, y_test, feature_names


def train_model_mlflow(X_train, X_test, y_train, y_test, y_all):
   

    xgb_model = XGBClassifier(
        objective="multi:softmax",
        num_class=y_all.nunique(),
        eval_metric="mlogloss",
        random_state=42
    )


    start_train = time.time()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
        )
    end_train = time.time() - start_train

    y_pred_xgb = xgb_model.predict(X_test)

    #MINI TEST PHASE 1
    # Predict on BOTH sets
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    # Calculate Accuracy
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")
    print(f"Gap: {train_acc - test_acc:.4f}")

    xgboost_model_result = track_model_performance(xgb_model, 
                                                   "XGBoost Classifier", 
                                                   X_test, 
                                                   y_test, 
                                                   y_pred_xgb, 
                                                   end_train)
    
    results = xgb_model.evals_result()
    

    print(f"MODEL EVALUATION:\n")
    print("--"*30)
    evaluate_model(y_test, y_pred_xgb, "XGBoost")
    print("--"*30)
    

    

    print(f"\nMODEL SYSTEM PERFORMANCE:\n")
    
    print(xgboost_model_result)



if __name__ == "__main__":

    X, y, X_train_log,X_test_log,y_train,y_test,feature_names = load_and_preprocess_data(data_path="data/sensor_Crop_Dataset.csv")
    train_model_mlflow(
        X_train=X_train_log,
        X_test=X_test_log,
        y_train=y_train,
        y_test=y_test,
        y_all=y
    )
