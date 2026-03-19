import pickle
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("spam-detector")

# load
log.info("Loading model...")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
log.info("Model loaded successfully")

# setup
app = FastAPI(title="Spam Detector API")

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# schema
class Message(BaseModel):
    text: str


# serve html
@app.get("/", response_class=HTMLResponse)
def home():
    log.info("Browser opened the UI")
    with open("spam.html", "r") as f:
        return f.read()


# predict
@app.post("/predict")
async def predict(msg: Message, request: Request):
    client = request.client.host
    log.info(f"Request from {client} → '{msg.text}'")

    result = model.predict([msg.text])[0]          # classify
    confidence = model.predict_proba([msg.text])   # probability
    score = round(max(confidence[0]) * 100, 2)     # percent

    log.info(f"Prediction → {result.upper()} ({score}%)")

    return {
        "message": msg.text,
        "prediction": result,
        "confidence": f"{score}%"
    }


# start
if __name__ == "__main__":
    import uvicorn
    log.info("Starting server → http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)