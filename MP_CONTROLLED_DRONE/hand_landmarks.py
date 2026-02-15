import cv2
import mediapipe as mp
import numpy as np

class HandExtractor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None

        lm = res.multi_hand_landmarks[0].landmark
        points = np.array([[p.x, p.y, p.z] for p in lm])

        wrist = points[0]
        points -= wrist

        scale = np.linalg.norm(points[12])
        if scale > 0:
            points /= scale

        return points.flatten().astype(np.float32)
