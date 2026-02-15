import cv2
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_PATH = "gesture_cnn_model_version2.h5"
IMAGE_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

class_names = list(model.class_names) if hasattr(model, "class_names") else [
    "FIST","OPEN_PALM","INDEX_UP","INDEX_DOWN","TWO_FINGERS","THREE_FINGERS","I_AM_COOL"
]

cam = cv2.VideoCapture(0)
cam.set(3,960)
cam.set(4,540)

history = deque(maxlen=7)

def smooth(values):
    if len(values) == 0:
        return "NONE"
    return max(set(values), key=values.count)

while True:
    ok, frame = cam.read()
    if not ok:
        break

    img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = preds[idx]
    raw_gesture = class_names[idx]

    history.append(raw_gesture)
    gesture = smooth(history)

    cv2.putText(frame, f"Gesture : {gesture}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.putText(frame, f"Confidence : {confidence:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("CNN Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
