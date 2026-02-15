import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import threading

MODEL_PATH = "vosk-model-small-en-us-0.15"

COMMANDS = [
    "take off",
    "land",
    "forward",
    "backward",
    "left",
    "right",
    "flip",
    "frontflip"
]

class Voice:
    def __init__(self):
        self.text = ""
        self.model = Model(MODEL_PATH)
        grammar = json.dumps(COMMANDS)
        self.rec = KaldiRecognizer(self.model, 16000, grammar)
        self.q = queue.Queue()
        threading.Thread(target=self.listen, daemon=True).start()

    def callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def listen(self):
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.callback
        ):
            while True:
                data = self.q.get()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    self.text = result.get("text", "")
