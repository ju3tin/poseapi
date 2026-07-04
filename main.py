from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np

app = FastAPI(title="MediaPipe Holistic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://motionplay.vercel.app", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Holistic Solution (Face + Hands + Pose in one model)
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,           # 0 = lite, 1 = full, 2 = heavy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Extract data
        data = {
            "pose": [],
            "hands": [],
            "face": []
        }

        # Pose
        if results.pose_landmarks:
            data["pose"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]

        # Hands (left + right)
        if results.left_hand_landmarks:
            data["hands"].append({
                "side": "left",
                "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in results.left_hand_landmarks.landmark]
            })
        if results.right_hand_landmarks:
            data["hands"].append({
                "side": "right",
                "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in results.right_hand_landmarks.landmark]
            })

        # Face
        if results.face_landmarks:
            data["face"] = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in results.face_landmarks.landmark]

        return data

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
