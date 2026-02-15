import csv, time, json
import cv2
from camera import Camera
from hand_landmarks import HandExtractor

SAMPLES = 60
CSV_FILE = "gesture_data.csv"

with open("gesture_map.json") as f:
    GESTURES = json.load(f)

cam = Camera()
hand = HandExtractor()

with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    for name, label in GESTURES.items():
        print(f"\nRecording {name} in 3 seconds...")
        time.sleep(3)
        count = 0

        while count < SAMPLES:
            frame = cam.read()
            feats = hand.extract(frame)

            if feats is not None:
                writer.writerow(list(feats) + [label])
                count += 1

            cv2.putText(frame,f"{name}: {count}/{SAMPLES}",
                        (20,40),0,1,(0,255,0),2)
            cv2.imshow("Recording",frame)

            if cv2.waitKey(1)==27:
                break

cam.release()
cv2.destroyAllWindows()
print("CSV saved.")
