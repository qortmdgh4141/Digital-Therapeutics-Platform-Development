import os
import random
import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from minirocket_multivariate import fit, transform

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class minirocket_clf:
    def __init__(self, train_dir_path, test_dir_path, num_runs=2, num_kernels=10000):
        self.train_dir_path = train_dir_path
        self.test_dir_path = test_dir_path
        self.num_runs = num_runs
        self.num_kernels = num_kernels

    def main(self, element = None):
        x_train, y_train = self.load_dataset(dir_path=self.train_dir_path, mode="TRAIN", shuffle = True)
        x_test, y_test = self.load_dataset(dir_path=self.test_dir_path, mode="TEST", shuffle = False)

        x_train = self.reshape_data(x_train)
        x_test = self.reshape_data(x_test)

        minirocket_result = self.train_test_minirocket(x_train, x_test, y_train, y_test, element)

        return minirocket_result


    def load_dataset(self, dir_path, mode, shuffle):
        x_train_list = []
        y_train = np.array([])

        if shuffle == True:
            random.seed(42)
            sample_size = len(os.listdir(dir_path))
            population = list(range(sample_size))
            samples_num = random.sample(population, sample_size)
        else :
            sample_size = len(os.listdir(dir_path))
            samples_num = list(range(sample_size))

        for num in samples_num:
            loaded = np.load(f"{dir_path}\\{mode}_({num}).npz")
            data = loaded["data"]
            label = loaded["label"]

            x_train_list.append(data)
            y_train = np.concatenate([y_train, label])

        x_train = np.stack(x_train_list, axis=0).astype(np.float32)
        y_train = y_train.astype(np.float32)

        return x_train, y_train

    def reshape_data(self, data, norm=False, std=False, with_mean=True):
        if std == True:
            if with_mean==True:
                mean = np.mean(data, axis=(0, 1, 2, 3, 4), keepdims=True)
                std = np.std(data, axis=(0, 1, 2, 3, 4), keepdims=True)
                data = (data - mean) / std
            else:
                std = np.std(data, axis=(0, 1, 2, 3, 4), keepdims=True)
                data = data / std
        elif norm == True:
            min = data.min()
            max = data.max()
            data = (data - min) / (max - min)

        reshaped_x_data = np.reshape(data, (data.shape[0], data.shape[1], -1))

        return reshaped_x_data

    def train_test_minirocket(self,x_train, x_test, y_train, y_test, element):
        if element == None:
            _results = np.zeros(self.num_runs)
        else:
            _results = np.zeros((6, self.num_runs))

        # Perform runs
        print("Minirocket - RUNNING".center(80, "="))
        for i in range(self.num_runs):
            parameters = fit(x_train)

            # -- transform training -----------------------------------------------
            x_train_transform = transform(x_train, parameters)
            # -- transform test ---------------------------------------------------
            x_test_transform = transform(x_test, parameters)

            # -- training ---------------------------------------------------------
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier = make_pipeline(StandardScaler(with_mean=True), classifier)
            classifier.fit(x_train_transform, y_train)
            # -- test -------------------------------------------------------------

            if element == None:
                _results[i] = classifier.score(x_test_transform, y_test)
            else:
                x_test_all, y_test_all= self.split_y_test(x_train_transform , y_test, element)

                print("y_test_all", len(y_test_all))
                print("y_test_all[0]", len(y_test_all[0]))

                count = 0
                for num_1 in range(len(y_test_all)):
                    for num_2 in range(len(y_test_all[0])):
                        _results[count][i] = classifier.score(x_test_all[num_1][num_2], y_test_all[num_1][num_2])
                        print(_results[count][i])
                        print(x_test_all[num_1][num_2])
                        print(y_test_all[num_1][num_2])
                        count+=1

        if element == None:
            minirocket_result = round(_results.mean()*100, 2)
            print(f'Minirocket을 이용한 분류 정확도 {minirocket_result}%')
            return minirocket_result
        else:
            minirocket_split_result = _results.mean(axis=-1)
            mean_value = np.mean(minirocket_split_result)
            minirocket_result = np.insert(minirocket_split_result , 0, mean_value)

            return minirocket_result

    def split_y_test(self, x_test, y_test, element):
        if element != None:
            indices_1 = [0, 1, 2, 9, 10, 11]; indices_2 = [3, 4, 5, 12, 13, 14]; indices_3 = [6, 7, 8, 15, 16, 17]
            x_test_1 = [x_test[indices_1], x_test[indices_2], x_test[indices_3]]
            y_test_1 = [y_test[indices_1], y_test[indices_2], y_test[indices_3]]

            ind_1 = [0, 3, 6, 9, 12, 15]; ind_2 = [1, 4, 7, 10, 13, 16]; ind_3 = [2, 5, 8, 11, 14, 17]
            x_test_2 = [x_test[ind_1], x_test[ind_2], x_test[ind_3]]
            y_test_2 = [y_test[ind_1], y_test[ind_2], y_test[ind_3]]

            x_test_all = [x_test_1, x_test_2]
            y_test_all = [y_test_1, y_test_2]

            return x_test_all, y_test_all