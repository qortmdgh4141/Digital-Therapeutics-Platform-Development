import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init

# 데이터셋 클래스를 정의
class MyDataset(Dataset):
    def __init__(self, df):
        self.df_list = df.values.tolist()

    def __len__(self):
        return len(self.df_list)

    def __getitem__(self, idx):
        # 입력과 출력 데이터를 정의
        inputs = self.df_list[idx][1:]
        targets = [self.df_list[idx][0]]

        # 입력과 타겟 데이터를 파이토치 텐서로 변환
        inputs = torch.tensor(inputs)
        targets = torch.LongTensor(targets)

        return inputs, targets

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Xavier initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

def evaluate(model, dataloader, criterion):
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    acc = 100 * correct / total
    loss = running_loss / len(dataloader)

    return loss, acc

# TSV 파일을 불러온 후 데이터셋 객체 생성
file_path  = "merged_sample1_tsv\\all_TRAIN.tsv"
df = pd.read_csv(file_path, delimiter="\t", header=None)
dataset = MyDataset(df)

# 데이터 로더를 생성합니다.
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

best_acc = 0
best_loss = float('inf')

# 학습
for epoch in range(10):
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # 데이터 표준화
        inputs = F.normalize(inputs, dim=1)

        optimizer.zero_grad()
        outputs = model(inputs.float())

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

    train_acc = 100 * correct / total
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    if val_loss < best_loss:
        best_loss = val_loss
        best_acc = val_acc
        best_model_state = model.state_dict()

        print("-------------------- BEST -------------------- ")
        print('[%d] loss: %.3f, val_loss: %.3f, acc: %.3f %%, val_acc: %.3f %%' %
              (epoch + 1, running_loss / len(train_loader), val_loss,
               train_acc, val_acc))

    else:
        print('[%d] loss: %.3f, val_loss: %.3f, acc: %.3f %%, val_acc: %.3f %%' %
              (epoch + 1, running_loss / len(train_loader), val_loss,
               train_acc, val_acc))





