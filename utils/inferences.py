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
        'Temperature','Humidity','pH_Value','Variety'
    ]

    
    

    # create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # encode Variety
    df["Variety"] = le_variety.transform(df["Variety"])

    # log transform
    for col in ["Phosphorus", "Potassium", "Humidity"]:
        df[col] = np.log1p(df[col])

    return df.astype(float)


# -----------------------------
# INFERENCE
# -----------------------------
def inference_fn(model_path: str, data: list):
    crops = ["wheat","tomato","sugarcane","maize","potato","rice"]

    # load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # preprocess
    X = preprocess_input(data)

    # predict
    preds = model.predict(X)
    

    crop_recommendation = crops[int(preds.tolist()[0])]
    return crop_recommendation