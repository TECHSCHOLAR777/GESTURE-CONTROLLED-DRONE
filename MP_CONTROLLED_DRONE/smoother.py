from collections import deque

class Smoother:
    def __init__(self,n=7):
        self.buf = deque(maxlen=n)

    def push(self,v):
        self.buf.append(v)

    def get(self):
        return max(set(self.buf), key=self.buf.count) if self.buf else None
