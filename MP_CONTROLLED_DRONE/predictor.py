import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,7)
        )

    def forward(self,x): return self.net(x)

class Predictor:
    def __init__(self):
        self.model = ANN()
        self.model.load_state_dict(torch.load("gesture_model.pt"))
        self.model.eval()

    def predict(self, features):
        x = torch.tensor(features).float().unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        return int(out.argmax(1))
