import cv2, time
from collections import deque

from camera import Camera
from hand_landmarks import HandExtractor
from predictor import Predictor
from smoother import Smoother
from voice import Voice
from speed_control import SmileSpeed

cam = Camera()
hand = HandExtractor()
ann = Predictor()
smooth = Smoother(7)
voice = Voice()
speed_ai = SmileSpeed()

sequence = deque(maxlen=6)

airborne = False
action = "NONE"
last_t = 0

gesture_names = {
    0:"OPEN PALM",
    1:"FIST",
    2:"INDEX UP",
    3:"INDEX DOWN",
    4:"TWO UP",
    5:"THREE UP",
    6:"I AM COOL"
}

def check_sequence(seq, airborne):
    s = list(seq)

    if s[-3:] == ["OPEN PALM","INDEX UP","OPEN PALM"]:
        return "KILL", airborne

    if s[-3:] == ["INDEX UP","FIST","INDEX UP"]:
        return "BOOST", airborne

    if s[-2:] == ["OPEN PALM","FIST"]:
        return "EMERGENCY LAND", False

    return None, airborne

while True:
    frame = cam.read()
    if frame is None:
        break

    feats = hand.extract(frame)
    if feats is not None:
        smooth.push(ann.predict(feats))

    g_id = smooth.get()
    g = gesture_names.get(g_id,"NONE")

    if g!="NONE" and (len(sequence)==0 or sequence[-1]!=g):
        sequence.append(g)
        if airborne:
            act, airborne = check_sequence(sequence, airborne)
            if act:
                action = act

    speed = speed_ai.get(frame)
    spoken = voice.text

    if g=="OPEN PALM" and "take off" in spoken:
        airborne=True; action="TAKEOFF"

    if g=="FIST" and "land" in spoken:
        airborne=False; action="LAND"

    if airborne:
        if g=="INDEX UP" and "forward" in spoken:
            action=f"FORWARD {round(speed,2)}"
        if g=="INDEX DOWN" and "backward" in spoken:
            action=f"BACKWARD {round(speed,2)}"
        if g=="TWO UP" and "left" in spoken:
            action=f"LEFT {round(speed,2)}"
        if g=="THREE UP" and "right" in spoken:
            action=f"RIGHT {round(speed,2)}"

    if g=="I AM COOL" and "flip" in spoken:
        action="FRONTFLIP"

    now=time.time()
    fps=int(1/(now-last_t)) if last_t else 0
    last_t=now

    cv2.putText(frame,"Gesture : "+g,(20,40),0,1,(0,0,255),2)
    cv2.putText(frame,"Sequence : "+" ".join(sequence),(20,80),0,1,(180,180,180),2)
    cv2.putText(frame,"Speed : "+str(round(speed,2)),(20,120),0,1,(0,255,255),2)
    cv2.putText(frame,"FPS : "+str(fps),(20,160),0,1,(255,0,0),2)
    cv2.putText(frame,"Voice : "+spoken,(20,200),0,1,(255,0,255),2)
    cv2.putText(frame,"Command : "+action,(20,240),0,1,(255,255,0),2)

    cv2.imshow("Intelligent Drone Control",frame)

    if cv2.waitKey(1)==ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
