# Spam Message Detector

A simple spam classifier trained with Scikit-learn and served via FastAPI.

---

## Project Structure

```
spam_detector/
├── train_model.py   ← trains and saves the model
├── main.py          ← FastAPI app that loads model and serves predictions
├── model.pkl        ← saved model (generated after training)
└── requirements.txt ← dependencies
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This creates a `model.pkl` file.

### 3. Start the API
```bash
uvicorn main:app --reload
```

---

## Usage

### Check if API is running
```
GET http://127.0.0.1:8000/
```

### Predict a message
```
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "text": "Win a free iPhone now click here"
}
```

### Example Response
```json
{
  "message": "Win a free iPhone now click here",
  "prediction": "spam",
  "confidence": "97.3%"
}
```

---

## How it Works

```
User Message
    ↓
TF-IDF Vectorizer  →  converts text to numbers
    ↓
Naive Bayes Model  →  classifies as spam or ham
    ↓
FastAPI Response   →  returns prediction + confidence
```

- **TF-IDF** — scores each word based on how important it is in the message
- **Naive Bayes** — calculates the probability of spam vs ham based on word scores
- **Pipeline** — chains both steps so it works as one unit

---

## API Docs

FastAPI auto-generates interactive docs. Open in browser after starting the server:
```
http://127.0.0.1:8000/docs
```
