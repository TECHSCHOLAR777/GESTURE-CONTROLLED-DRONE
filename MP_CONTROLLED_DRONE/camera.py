import cv2

class Camera:
    def __init__(self, index=0, w=960, h=540):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(3, w)
        self.cap.set(4, h)

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
