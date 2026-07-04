from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import cv2
import numpy as np

app = FastAPI(title="Pose + Hands API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_detector = mp_pose.Pose(
    static_image_mode=False,
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

        hand_results = hands_detector.process(image_rgb)
        pose_results = pose_detector.process(image_rgb)

        hands_data = []
        if hand_results.multi_hand_landmarks:
            for handLms in hand_results.multi_hand_landmarks:
                hands_data.append({
                    "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in handLms.landmark]
                })

        pose_data = []
        if pose_results.pose_landmarks:
            pose_data = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in pose_results.pose_landmarks.landmark
            ]

        return {
            "hands": hands_data,
            "pose": pose_data
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
