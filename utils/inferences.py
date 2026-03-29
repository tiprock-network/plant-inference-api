import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder



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

    

    # log transform
    for col in ["Phosphorus", "Potassium", "Humidity"]:
        df[col] = np.log1p(df[col])

    return df.astype(float)


# -----------------------------
# INFERENCE
# -----------------------------
# -----------------------------
# UPDATED INFERENCE
# -----------------------------
def inference_fn(model_path: str, data: list):

    crops = [
        'orange',
        'pomegranate',
        'apple',
        'banana',
        'watermelon',
        'coconut',
        'chickpea',
        'sugarcane',
        'mango',
        'mothbeans',
        'grapes',
        'potato',
        'rice',
        'cotton',
        'blackgram',
        'kidneybeans',
        'jute',
        'lentil',
        'papaya',
        'coffee',
        'wheat',
        'tomato',
        'muskmelon',
        'mungbean',
        'maize',
        'pigeonpeas'
    ]

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