import mediapipe as mp
import cv2
from collections import deque

class SmileSpeed:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh()
        self.history = deque(maxlen=5)
        self.min_val = 0.08
        self.max_val = 0.11

    def get(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        if not res.multi_face_landmarks:
            return 1.0

        lm = res.multi_face_landmarks[0].landmark

        mouth_width = abs(lm[291].x - lm[61].x)
        mouth_open  = abs(lm[14].y - lm[13].y)

        raw = mouth_width - mouth_open
        norm = (raw - self.min_val) / (self.max_val - self.min_val)
        norm = max(0, min(1, norm))

        speed = 1 + norm
        self.history.append(speed)

        return sum(self.history)/len(self.history)
