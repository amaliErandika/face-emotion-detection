from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from keras.models import load_model
import os
import gdown

app = FastAPI()

MODEL_PATH = "final_model_30epochs.h5"
model = None  


# Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1Np_qs8KOBr1nAIl7tg6G1hWKT2j6YxVI"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)


# Lazy load model (important for Render)
def get_model():
    global model
    if model is None:
        download_model()
        model = load_model(MODEL_PATH)
    return model


labels_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}


faceDetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()  # ✅ load here

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"emotion": "No face detected"}

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    resized = cv2.resize(face, (48, 48))
    normalized = resized.astype('float32') / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    result = model.predict(reshaped)
    label = int(np.argmax(result))
    confidence = float(np.max(result))

    return {
        "emotion": labels_dict[label],
        "confidence": confidence
    }