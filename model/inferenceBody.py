from pydantic import BaseModel

class InferencePost(BaseModel):
    input: list

class InferenceResponse(BaseModel):
    recommendation: str
    confidence: float