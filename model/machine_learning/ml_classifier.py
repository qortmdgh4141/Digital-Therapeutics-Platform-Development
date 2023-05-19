import os
import random
import numpy as np

from sklearn import tree, svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class ml_clf:
    def __init__(self, train_dir_path, test_dir_path):
        self.train_dir_path = train_dir_path
        self.test_dir_path = test_dir_path

    def main(self, reshaped_dataset):
        x_train, y_train = self.load_dataset(dir_path=self.train_dir_path, mode="TRAIN")
        x_test, y_test = self.load_dataset(dir_path=self.test_dir_path, mode="TEST")

        if reshaped_dataset == True:
            x_train, y_train = self.reshape_data(x_train, y_train)
            x_test, y_test = self.reshape_data(x_test, y_test)

        (x_train_std, x_test_std) = self.standardize_data(x_train, x_test, with_mean=False)

        knn_result = self.train_test_knn(x_train_std, y_train, x_test_std, y_test)
        svm_result = self.train_test_svm(x_train_std, y_train, x_test_std, y_test)
        c50_result = self.train_test_c50(x_train, y_train, x_test, y_test)
        average_result = round(np.mean([knn_result, svm_result, c50_result]), 2)

        return knn_result, svm_result, c50_result, average_result

    def load_dataset(self, dir_path, mode):
        x_train_list = []
        y_train = np.array([])

        random.seed(42)
        sample_size = len(os.listdir(dir_path))
        population = list(range(sample_size))
        samples_num = random.sample(population, sample_size)

        for num in samples_num:
            loaded = np.load(f"{dir_path}\\{mode}_({num}).npz")
            data = loaded["data"]
            label = loaded["label"]

            x_train_list.append(data)
            y_train = np.concatenate([y_train, label])

        x_train = np.stack(x_train_list, axis=0)

        return x_train, y_train

    def reshape_data(self, x_data, y_data):
        x_data_reshaped = x_data.reshape((-1, x_data.shape[2], x_data.shape[3], x_data.shape[4])) # x_train 변환
        y_data_repeated = np.repeat(y_data, 36)  # y_train 변환

        # 라벨 매핑 검증
        for i in range(x_data.shape[0]):
            start_index = i * 36
            end_index = (i + 1) * 36
            labels = np.unique(y_data_repeated[start_index:end_index])
            if labels[0] != y_data[i]:
                print("Label mapping error for x_train[{}], y_train[{}].".format(i, i))

        return x_data_reshaped, y_data_repeated

    def standardize_data(self, x_train, x_test, with_mean=True):
        if with_mean == True:
            train_mean = x_train.mean()
            train_std = x_train.std()
            x_train_std = (x_train - train_mean) / train_std
            x_test_std = (x_test - train_mean) / train_std
        else:
            train_std = x_train.std()
            x_train_std = x_train / train_std
            x_test_std = x_test / train_std

        return x_train_std, x_test_std

    def normalize_data(self, x_train, x_test):
        train_min = x_train.min()
        train_max = x_train.max()
        x_train_normalized = (x_train - train_min) / (train_max - train_min)
        x_test_normalized = (x_test - train_min) / (train_max - train_min)
        return x_train_normalized, x_test_normalized

    def train_test_knn(self, x_train, y_train, x_test, y_test):
        knn_train_accuracy = []
        knn_test_accuracy = []
        # num_neighbors = range(1, x_train.shape[0])
        num_neighbors = range(1, 36)

        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        print(f"Finding the optimal K value: ", end="")
        for k in num_neighbors:
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn.fit(x_train, y_train)
            knn_train_accuracy.append(knn.score(x_train, y_train))
            knn_test_accuracy.append(knn.score(x_test, y_test))
            print(f"{k}", end=" ")

        max_accuracy = 0
        for num_k, accuracy in enumerate(knn_test_accuracy):
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                best_k = num_k + 1

        knn_result = round(max_accuracy * 100, 2)
        print(f"최적의 k 값 : {best_k}")
        print(f'KNN 알고리즘을 이용한 분류 정확도 {knn_result}%')

        return knn_result

    def train_test_svm(self, x_train, y_train, x_test, y_test):
        x_train = x_train.reshape(x_train.shape[0], -1)
        clf = svm.SVC(kernel='linear')

        clf.fit(x_train, y_train)

        x_test = x_test.reshape(x_test.shape[0], -1)
        svm_test_acuaracy = clf.score(x_test, y_test)

        svm_result = round(svm_test_acuaracy * 100, 2)
        print(f'SVM 알고리즘을 이용한 분류 정확도 {svm_result}%')

        return svm_result

    def train_test_c50(self, x_train, y_train, x_test, y_test):
        x_train = x_train.reshape(x_train.shape[0], -1)
        clf = tree.DecisionTreeClassifier(criterion="entropy")

        clf.fit(x_train, y_train)

        x_test = x_test.reshape(x_test.shape[0], -1)
        y_pred = clf.predict(x_test)

        c50_result = round(accuracy_score(y_test, y_pred) * 100, 2)
        print(f'C5.0 알고리즘을 이용한 분류 정확도 {c50_result}%')

        return c50_result