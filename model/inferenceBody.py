from pydantic import BaseModel

class InferencePost(BaseModel):
    input: list