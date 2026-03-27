from fastapi import FastAPI
from utils.inferences import inference_fn
from model.inferenceBody import InferencePost

app = FastAPI()

#get the prediction
@app.post('/api/v1/on-device/inference')
async def inference(inferenceBody: InferencePost):
    model_input = inferenceBody.input
    print(model_input)

    preds = inference_fn("saved_models/XGBoost_Classifier.pkl", [model_input])

    return preds