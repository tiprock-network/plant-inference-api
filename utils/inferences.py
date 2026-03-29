import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# -----------------------------
# LOAD ENCODER (you hardcoded it)
# -----------------------------
le_variety = LabelEncoder()
le_variety.classes_ = np.array([
    'Soft Red','Beefsteak','Co 86032','Co 0238','Sweet',
    'Yukon Gold','Hard Red','Flint','Basmati','Co 99004',
    'Dent','Roma','Russet','Jasmine','Cherry',
    'Arborio','Red','Durum'
])


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess_input(data: list):
    columns = [
        'Nitrogen','Phosphorus','Potassium',
        'Temperature','Humidity','pH_Value'
    ]

    
    

    # create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # encode Variety
    #df["Variety"] = le_variety.transform(df["Variety"])

    # log transform
    #for col in ["Phosphorus", "Potassium", "Humidity"]:
    #    df[col] = np.log1p(df[col])

    return df.astype(float)


# -----------------------------
# INFERENCE
# -----------------------------
# -----------------------------
# UPDATED INFERENCE
# -----------------------------
def inference_fn(model_path: str, data: list):
    crops = ["wheat", "tomato", "sugarcane", "maize", "potato", "rice"]

    # load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # preprocess
    X = preprocess_input(data)

    # 1. Use predict_proba to get the confidence scores for all classes
    # probabilities will be an array like [[0.1, 0.05, 0.8, ...]]
    probabilities = model.predict_proba(X)

    # 2. Get the index of the highest probability
    best_index = np.argmax(probabilities, axis=1)[0]

    # 3. Get the actual probability value (score)
    confidence_score = np.max(probabilities, axis=1)[0]

    # 4. Map the index to the crop name
    crop_recommendation = crops[best_index]

    return {
        "recommendation": crop_recommendation,
        "confidence": float(confidence_score),
    }