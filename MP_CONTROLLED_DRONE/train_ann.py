import torch, csv
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn

X, y = [], []

with open("gesture_data.csv") as f:
    for row in csv.reader(f):
        X.append(row[:-1])
        y.append(row[-1])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2)

train = DataLoader(TensorDataset(
    torch.tensor(Xtr), torch.tensor(ytr)), batch_size=64, shuffle=True)

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

model = ANN()
opt = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for e in range(40):
    for xb,yb in train:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    print("Epoch",e+1,"loss",loss.item())

torch.save(model.state_dict(),"gesture_model.pt")
print("Model saved.")
