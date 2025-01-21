import torch
import json
from fastapi import FastAPI 
from model import EngagementPredictor
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    data: str

def load_model():
    try:
        # Map the saved class to our current module
        model_data = torch.load(
            "./saved_model.pth",
            map_location=torch.device('cpu'),
            weights_only=True  # For security, only load weights
        )
        model = EngagementPredictor()  # Create a new instance
        model.load_state_dict(model_data)  # Load just the state dict
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Load model at startup
model = load_model()
model.eval()

@app.post("/predictor/engagement")
def get_engage_pred(data:Item):
    data = json.loads(data.data)
    return {
        "score": model(data)[:,0].tolist()
    }
