import os
import math
import visdom
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        self.transform = transform
        self.file_names = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".npz")]
        self.num_files = len(self.file_names)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.folder_path, file_name)
        loaded = np.load(file_path)

        input_data = loaded["input"]  # 입력 데이터
        target_data = loaded["target"]  # 대상 데이터

        if self.transform is not None:
            input_data = self.transform(input_data)

        # s parameter 순서 변경, "s11, s21, s12, s22" -> "s11, s22, s21, s12"
        input_data = input_data[:, [0, 3, 1, 2], :, :]

        input_data = torch.from_numpy(input_data)  # 입력 데이터
        target_data = torch.from_numpy(target_data)  # 대상 데이터

        # 차원 순서 변경 (36, 4 ,2 ,1001) -> (2, 36, 4 ,1001)
        #input_data = input_data.permute((2, 0, 1, 3))

        return input_data, target_data, file_name

class Signal_Transform(object):
    def __init__(self, folder_path, total_angle_num=36, angle_interval=10):
        data = []
        file_names = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".npz")]
        # 데이터를 로드하여 data에 추가
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            loaded = np.load(file_path)
            input_data = loaded["input"]
            data.append(input_data)
        data = np.array(data)
        print(data.shape)

        self.mean = np.mean(data, axis=0) # 각 채널별 평균 계산
        self.std = np.std(data, axis=0)  # 각 채널별 표준편차 계산

        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

        self.angle_num = (total_angle_num * 10) // angle_interval
        self.angle_indices = np.linspace(0, total_angle_num - 1, self.angle_num, dtype=int)  # 선택할 각도 인덱스 계산

    def reshape_by_angle_interval(self, data):
        new_data = np.zeros((self.angle_num, *data.shape[1:]), dtype=data.dtype)  # 새로운 데이터셋 생성
        # 각도 인덱스를 이용하여 데이터 복사
        for i, angle_idx in enumerate(self.angle_indices):
            new_data[i, :] = data[angle_idx, :]

        return new_data

    def __call__(self, data):
        #transformed_data = (data - self.mean) / self.std  # 데이터를 표준화합니다.
        transformed_data = (data - self.min) / (self.max - self.min)  # 데이터를 정규화합니다.
        if self.angle_num != 36:
            transformed_data = self.reshape_by_angle_interval(transformed_data)

        return transformed_data

class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')

        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')

        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False

class cnn_clf:
    def __init__(self, train_path, val_path, test_path, MyDataset, transform):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.MyDataset = MyDataset
        self.transform = transform

    def train(self, net, criterion, optimizer, lr_scheduler, batch_size):

        train_dataset = self.MyDataset(folder_path=self.train_path, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        net.train()

        total = 0
        running_loss = 0.0
        running_corrects = 0

        iter_num = len(train_loader)
        log_step = math.ceil(iter_num * 0.25)

        for i, (input_data, target_data, file_name) in enumerate(train_loader):
            input, target = input_data.to("cuda", dtype=torch.float32), target_data.to("cuda", dtype=torch.float32)
            optimizer.zero_grad()

            output = net(input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            batch_size = target.shape[0]
            train_loss = loss.item()

            preds = (output >= 0.5).float()
            train_corrects = torch.sum(preds == target)

            total += batch_size
            running_loss += (train_loss * batch_size)
            running_corrects += train_corrects

            train_iter_acc = (train_corrects / batch_size).item()

            if (i in [0, iter_num-1]) or ((i + 1) % log_step == 0):
                print(f" [Batch: {i + 1} / {len(train_loader)}]  Loss: {train_loss:.4f}  Acc: {train_iter_acc:.4f}")

        print(" -------------------------------------------")
        lr_scheduler.step()
        train_epoch_loss = running_loss / total
        train_epoch_acc = (running_corrects / total).item()

        return train_epoch_loss, train_epoch_acc

    def val(self, net, criterion, batch_size):
        val_dataset = self.MyDataset(folder_path=self.val_path, transform=self.transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        net.eval()

        total = 0
        running_loss = 0.0
        running_corrects = 0

        for i, (input_data, target_data, file_name) in enumerate(val_loader):
            input, target = input_data.to("cuda", dtype=torch.float32), target_data.to("cuda", dtype=torch.float32)

            with torch.no_grad():
                output = net(input)
                loss = criterion(output, target)

            batch_size = target.shape[0]
            total += batch_size
            running_loss += (loss.item() * batch_size)
            preds = (output >= 0.5).float()
            running_corrects += torch.sum(preds == target)

        val_epoch_loss = running_loss / total
        val_epoch_acc = (running_corrects / total).item()
        print(f" ※\tVal Loss: {val_epoch_loss:.4f}\t/\tVal Acc: {val_epoch_acc:.4f}", end="\n\n\n")

        return val_epoch_loss, val_epoch_acc

    def test(self, net, criterion, batch_size):
        test_dataset = self.MyDataset(folder_path=self.test_path, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        net.eval()

        total = 0
        running_loss = 0.0
        running_corrects = 0

        for i, (input_data, target_data, file_name) in enumerate(test_loader):
            input, target = input_data.to("cuda", dtype=torch.float32), target_data.to("cuda", dtype=torch.float32)

            with torch.no_grad():
                output = net(input)
                loss = criterion(output, target)

            batch_size = target.shape[0]
            total += batch_size
            running_loss += (loss.item() * batch_size)
            preds = (output >= 0.5).float()

            print(preds == target)
            print(file_name)
            running_corrects += torch.sum(preds == target)

        test_epoch_loss = running_loss / total
        test_epoch_acc = (running_corrects / total).item()
        print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        print(f"┃\t\t\t  Test Loss: {test_epoch_loss:.4f}  \t\t\t┃")
        print(f"┃\t\t\t  Test ACC : {test_epoch_acc:.4f}  \t\t\t┃")
        print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        return test_epoch_loss, test_epoch_acc


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # input_shape = (36, 4, 2, 1001)  # (channel, depth, height, width)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=36, out_channels=36, kernel_size=(2, 2, 4), stride=(2, 1, 2), padding=(0, 1, 0)),
            nn.BatchNorm3d(36),
            nn.LeakyReLU(0.01),
            nn.Dropout3d(0.7),
            nn.MaxPool3d((1, 1, 4))
        )

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Sequential(
            nn.Linear(36 * 2 * 3 * 124, 36),
            nn.BatchNorm1d(36),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.7),
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(36, 1),
            nn.Sigmoid()
        )

        # 각 레이어의 가중치를 He 초기화로 설정
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)


    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.flatten(x)

        x = self.linear_1(x)

        x = self.linear_2(x)

        return x

def plot_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc, model_name):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 6))

    # Loss 그래프
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title(f'{model_name} Model - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 그래프
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title(f'{model_name} Model - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.gca().legend(loc='upper right')
    plt.tight_layout()

    plt.show()

# sample4
train_path = "C:/Users/82104/PycharmProjects/Digital-Therapeutics-Platform-Development/training_test_data_generator/merged_sample4/all/angel360_step10/train_aug"
val_path = "C:/Users/82104/PycharmProjects/Digital-Therapeutics-Platform-Development//training_test_data_generator/merged_sample4/all/angel360_step10/val"
test_path = "C:/Users/82104/PycharmProjects/Digital-Therapeutics-Platform-Development//training_test_data_generator/merged_sample4/all/angel360_step10/test"

# 모델 초기화 및 GPU 설정
net = CNN_Model().to("cuda")

learning_rate=0.001

# 손실 함수, 옵티마이저, 학습률 스케줄러, 배치사이즈, 에폭 설정
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.1)
lr_scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
batch_size=128
epochs = 5

transform = Signal_Transform(folder_path=train_path)
#transform = None
cnn = cnn_clf(train_path=train_path, val_path=val_path, test_path=test_path, MyDataset=MyDataset, transform=transform)
train_loss_list, train_acc_list, val_loss_list, val_acc_list, test_loss_list, test_acc_list = [], [], [], [] ,[], []

# Visdom 설정
# python -m visdom.server
#viz_loss= visdom.Visdom()
#viz_acc= visdom.Visdom()

for epoch in range(epochs):
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print(f"┃\t\t\t\t   Epoch {epoch + 1}/{epochs}   \t\t\t\t┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

    train_loss, train_acc = cnn.train(net=net, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, batch_size=batch_size)
    val_loss, val_acc = cnn.val(net=net, criterion=criterion, batch_size=36)
    cnn.test(net=net, criterion=criterion, batch_size=36)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    # 현재 epoch의 손실 및 정확도를 Visdom에 업데이트
    '''viz_loss.line(Y=np.column_stack([train_loss_list,val_loss_list]),
             X=np.column_stack((np.arange(0, epoch + 1), np.arange(0, epoch + 1))),
             opts=dict(title='Training and Validation Loss', legend=['Training Loss', 'Validation Loss'],
                       xlabel='Epoch', ylabel='Loss'), win=0)

    viz_acc.line(Y=np.column_stack([train_acc_list, val_acc_list]),
                  X=np.column_stack((np.arange(0, epoch + 1), np.arange(0, epoch + 1))),
                  opts=dict(title='Training and Validation Accuracy', legend=['Training Accuracy', 'Validation Accuracy'],
                            xlabel='Epoch', ylabel='Accuracy'), win=1)'''

#viz_loss.close()
#viz_acc.close()


# CNN 모델 결과 그래프 출력
plot_loss_and_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list, 'CNN')

# 데이터를 파일로 저장
data = {'train_loss': train_loss_list, 'train_acc': train_acc_list, 'val_loss': val_loss_list, 'val_acc': val_acc_list}
data_file_path = 'result/loss_acc.pkl'
model_file_path = 'result/cnn.pth'

while True:
    user_input = input("\nDo you want to save the data and model to files? (y/n): ")

    if user_input.lower() == 'y':
        # Save the data to a file
        with open(data_file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"The data has been successfully saved to {data_file_path}.")

        # Save the model to a file
        torch.save(net.state_dict(), model_file_path)
        print(f"The model has been successfully saved to {model_file_path}.")
        break

    elif user_input.lower() == 'n':
        print("Data and model saving skipped.")
        break

    else:
        print("Invalid input. Please enter 'y' or 'n'.")
