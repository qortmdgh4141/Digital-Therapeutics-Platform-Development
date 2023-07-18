import pickle
import matplotlib.pyplot as plt

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
    # plt.legend()

    plt.gca().legend(loc='upper right')

    plt.tight_layout()

    plt.show()

# 저장된 데이터를 파일에서 로드합니다.
with open('loss_acc.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# 로드된 데이터에서 각 리스트를 추출합니다.
train_loss_list = loaded_data['train_loss']
train_acc_list = loaded_data['train_acc']
val_loss_list = loaded_data['val_loss']
val_acc_list = loaded_data['val_acc']

# CNN 모델 결과 그래프 출력
plot_loss_and_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list, 'CNN')