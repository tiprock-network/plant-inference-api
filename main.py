from fastapi import FastAPI
from utils.inferences import inference_fn
from model.inferenceBody import InferencePost, InferenceResponse

app = FastAPI()

#get the prediction
@app.post('/api/v1/on-device/inference', response_model=InferenceResponse)
async def inference(inferenceBody: InferencePost):
    model_input = inferenceBody.input
    

    preds = inference_fn("saved_models/XGBoost_Classifier.pkl", [model_input])
    preds_dict = InferenceResponse(recommendation=preds["recommendation"], confidence=preds["confidence"])

    return preds_dict