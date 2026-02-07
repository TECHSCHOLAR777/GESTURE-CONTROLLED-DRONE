import cv2
import mediapipe as mp
import time
import speech_recognition as sr
import threading
from collections import deque

cam = cv2.VideoCapture(0)
cam.set(3, 960)
cam.set(4, 540)
mpHands = mp.solutions.hands
handAI = mpHands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)

face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

voiceAI = sr.Recognizer()
spoken = ""
action = "NONE"
listening = False

airborne = False
current_speed = 1.0

store = deque(maxlen=7)
seq = deque(maxlen=6)

def hear_loop():
    global spoken
    with sr.Microphone() as mic:
        voiceAI.adjust_for_ambient_noise(mic, duration=0.2)
        while True:
            try:
                audio = voiceAI.listen(mic, timeout=1, phrase_time_limit=1)
                spoken = voiceAI.recognize_google(audio).lower()
            except:
                spoken = ""
            time.sleep(0.05)

threading.Thread(target=hear_loop, daemon=True).start()

def smooth(arr):
    if len(arr) == 0:
        return "NONE"
    return max(set(arr), key=arr.count)

def fingers(p):
    return [p[8].y < p[6].y,
            p[12].y < p[10].y,
            p[16].y < p[14].y,
            p[20].y < p[18].y]

def thumb(p):
    return p[4].x > p[3].x

def gesture(p):
    f = fingers(p)
    t = thumb(p)
    if f == [1,1,1,1]: return "OPEN PALM"
    if f == [0,0,0,0] and not t: return "FIST"
    if f == [1,0,0,0]: return "INDEX UP"
    if p[8].y > p[6].y and f == [0,0,0,0]: return "INDEX DOWN"
    if f == [1,1,0,0]: return "TWO UP"
    if f == [1,1,1,0]: return "THREE UP"
    if t and f == [1,0,0,1]: return "I AM COOL"
    return "NONE"

def power(t, j):
    v = (j.y - t.y) * 30
    if v < 0: v = 0
    if v > 1: v = 1
    return v

def confidence(p, g):
    if g == "OPEN PALM":
        return (power(p[8],p[6])+power(p[12],p[10])+power(p[16],p[14])+power(p[20],p[18]))/4
    if g == "FIST":
        return 1-confidence(p,"OPEN PALM")
    if g == "INDEX UP":
        return power(p[8],p[6])
    if g == "INDEX DOWN":
        return power(p[6],p[8])
    if g == "TWO UP":
        return (power(p[8],p[6])+power(p[12],p[10]))/2
    if g == "THREE UP":
        return (power(p[8],p[6])+power(p[12],p[10])+power(p[16],p[14]))/3
    if g == "I AM COOL":
        return (power(p[8],p[6])+power(p[20],p[18]))/2
    return 0

def check_sequence():
    global action, airborne
    p = list(seq)
    if p[-3:] == ["OPEN PALM","INDEX UP","OPEN PALM"]:
        action = "KILL"
    if p[-3:] == ["INDEX UP","FIST","INDEX UP"]:
        action = "BOOST"
    if p[-2:] == ["OPEN PALM","FIST"]:
        action = "EMERGENCY LAND"
        airborne = False

last_t = 0
last_voice = 0
gap = 1.5

dots = ""
dot_t = 0

while True:
    ok, frame = cam.read()
    if not ok: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    out_face = faceMesh.process(rgb)
    smile_strength = 0
    if out_face.multi_face_landmarks:
        face_landmarks = out_face.multi_face_landmarks[0].landmark
        
        left_mouth = face_landmarks[61]
        right_mouth = face_landmarks[291]
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]

        mouth_width = abs(right_mouth.x - left_mouth.x)
        mouth_open = abs(lower_lip.y - upper_lip.y)

        smile_strength = mouth_width - mouth_open

    if smile_strength < 0: smile_strength = 0
    if smile_strength > 1: smile_strength = 1

    min_val = 0.08
    max_val = 0.11

    norm = (smile_strength - min_val) / (max_val - min_val)
    if norm < 0: norm = 0
    if norm > 1: norm = 1

    current_speed = 1.0 + norm * (2.0 - 1.0)

    out = handAI.process(rgb)
    g = "NONE"
    conf = 0

    if out.multi_hand_landmarks:
        h = out.multi_hand_landmarks[0]
        pts = h.landmark
        raw = gesture(pts)
        store.append(raw)
        g = smooth(store)
        conf = confidence(pts,g)
        if g != "NONE":
            if len(seq) == 0 or seq[-1] != g:
                seq.append(g)
                if airborne:
                    check_sequence()

    now = time.time()

    if action not in ["KILL","BOOST","EMERGENCY LAND"]:
        if conf > 0.75 and g != "NONE":
            if not listening and now-last_voice > gap:
                listening = True
                last_voice = now

            if g=="OPEN PALM" and "take off" in spoken:
                airborne=True; action="TAKEOFF"
            if g=="FIST" and "land" in spoken:
                airborne=False; action="LAND"
            if airborne:
                if g=="INDEX UP" and "forward" in spoken:
                    action=f"FORWARD {round(current_speed,2)}"
                if g=="INDEX DOWN" and "backward" in spoken:
                    action=f"BACKWARD {round(current_speed,2)}"
                if g=="TWO UP" and "left" in spoken:
                    action=f"LEFT {round(current_speed,2)}"
                if g=="THREE UP" and "right" in spoken:
                    action=f"RIGHT {round(current_speed,2)}"
            if g=="I AM COOL" and ("frontflip" in spoken or "front flip" in spoken):
                action="FRONTFLIP"

    if listening:
        if time.time()-dot_t > 0.4:
            dot_t = time.time()
            dots = "." if dots=="" else ".." if dots=="." else "..." if dots==".." else ""
    else:
        dots = ""

    t = time.time()
    fps = int(1/(t-last_t)) if last_t else 0
    last_t = t

    cv2.putText(frame,"Gesture : "+g,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(frame,"Sequence : "+" ".join(seq),(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(180,180,180),2)
    cv2.putText(frame,"Speed : "+str(round(current_speed,2)),(20,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    cv2.putText(frame,"FPS : "+str(fps),(20,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.putText(frame,"Confidence : "+str(round(conf,2)),(20,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame,"Voice : "+spoken,(20,240),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
    cv2.putText(frame,"Listening "+dots,(20,280),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200),2)
    cv2.putText(frame,"Command : "+action,(20,320),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

    cv2.imshow("Intelligent Drone Control",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cam.release()
cv2.destroyAllWindows()
