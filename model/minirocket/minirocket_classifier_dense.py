import os
import sys
import random
import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from minirocket_multivariate import fit, transform

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from keras import initializers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv3D, MaxPooling3D, MaxPooling2D, Conv2D, BatchNormalization, Conv3DTranspose, Permute, Activation, LeakyReLU
from keras import regularizers

class minirocket_clf:
    def __init__(self, train_dir_path, test_dir_path, num_runs=1, num_kernels=10000):
        self.train_dir_path = train_dir_path
        self.test_dir_path = test_dir_path
        self.num_runs = num_runs
        self.num_kernels = num_kernels

    def main(self, augmentation, element, total_angle_num, angle_interval):
        x_train, y_train = self.load_dataset(dir_path=self.train_dir_path, mode="TRAIN", shuffle=True)
        x_test, y_test = self.load_dataset(dir_path=self.test_dir_path, mode="TEST", shuffle=False)
        #print(1)
        #print(x_train.shape)

        x_train = self.reshape_data(x_train, norm=False, std=False, with_mean=False)
        x_test = self.reshape_data(x_test, norm=False, std=False, with_mean=False)
        #print(2)
        #print(x_train.shape)

        if augmentation == True:
            x_train, y_train, augment_num_angles = self.augment_data(x_train, y_train, shuffle=True)
            x_test, y_test, augment_num_angles = self.augment_data(x_test, y_test, shuffle=False)
        else:
            augment_num_angles = 1

        #print(3)
        #print(x_train.shape)

        if angle_interval != 10:
            x_train = self.reshape_by_angle_interval(x_train, total_angle_num=total_angle_num, angle_interval=angle_interval)
            x_test = self.reshape_by_angle_interval(x_test, total_angle_num=total_angle_num, angle_interval=angle_interval)

        #print(4)
        #print(x_train.shape)

        print()

        minirocket_result = self.train_test_minirocket(x_train, y_train, x_test, y_test, element, augment_num_angles)

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

    def reshape_data(self, data, norm, std, with_mean):
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
        # 호호 : 정신기 선배님 조언으로 I/Q 차원 변경
        # reshaped_x_data = np.reshape(data, (data.shape[0], data.shape[1], -1))
        reshaped_x_data = np.reshape(data, (data.shape[0], data.shape[3], -1))

        return reshaped_x_data

    def augment_data(self, x_data, y_data, shuffle):
        num_samples = x_data.shape[0]  # 배치 사이즈
        num_angles = x_data.shape[1]  # 각도의 개수

        # 어규멘테이션된 데이터를 저장할 변수 초기화
        augmented_x_data = np.zeros((num_samples * num_angles, *x_data.shape[1:]), dtype=x_data.dtype)
        augmented_y_data = np.zeros((num_samples * num_angles), dtype=x_data.dtype)

        for i in range(num_samples):
            for j in range(num_angles):
                # 어규멘테이션 데이터
                augmented_x_data[i * num_angles + j] = np.roll(x_data[i], -j, axis=0)
                augmented_y_data[i * num_angles + j] = y_data[i]

        # 라벨 매핑 검증
        for i in range(num_samples):
            start_index = i * num_angles
            end_index = (i + 1) * num_angles
            labels = np.unique(augmented_y_data[start_index:end_index])

            if labels[0] != y_data[i]:
                print("Label mapping error for x_train[{}], y_train[{}].".format(i, i))

        # augmented_x_train의 배치사이즈를 기준으로 데이터 순서를 셔플
        if shuffle == True:
            shuffle_indices = np.random.permutation(augmented_x_data.shape[0])
            augmented_x_data = augmented_x_data[shuffle_indices]
            augmented_y_data = augmented_y_data[shuffle_indices]

        print("\t - Original to Augmented 'x_data / y_data' shape:", x_data.shape, "/", y_data.shape, "->", augmented_x_data.shape, "/", augmented_y_data.shape)

        return augmented_x_data, augmented_y_data, num_angles

    def reshape_by_angle_interval(self, data, total_angle_num, angle_interval):
        angle_num = (total_angle_num * 10) // angle_interval

        # 선택할 각도 인덱스 계산
        angle_indices = np.linspace(0, total_angle_num - 1, angle_num, dtype=int)

        # 새로운 데이터셋 생성
        new_data = np.zeros((data.shape[0], angle_num, *data.shape[2:]), dtype=data.dtype)

        # 각도 인덱스를 이용하여 데이터 복사
        for i, angle_idx in enumerate(angle_indices):
            new_data[:, i, :] = data[:, angle_idx, :]

        print(f"\t - X_Data reshaped by angle interval (angle interval - {angle_interval} degree) : ", new_data.shape)

        return new_data

    def train_test_minirocket(self, x_train, y_train, x_test, y_test, element, augment_num_angles):
        if element == None:
            _results = np.zeros(self.num_runs)
        else:
            _results = np.zeros((6, self.num_runs))

        # Perform runs
        print("Minirocket - RUNNING".center(80, "="))
        for i in range(self.num_runs):
            parameters = fit(x_train)

            #호호
            #print(x_train.shape)
            #print(x_test.shape)

            # Ouput shape - x_train_shape : (36, 36, 8008) -> x_train_transform_shape : (36, 9996)
            # -- transform training -----------------------------------------------
            x_train_transform = transform(x_train, parameters)

            # -- transform test ---------------------------------------------------
            x_test_transform = transform(x_test, parameters)

            # 호호
            #print("---")
            #print(x_train_transform.shape)
            #print(x_test_transform.shape)

            # -- training ---------------------------------------------------------
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier = make_pipeline(StandardScaler(with_mean=True), classifier)
            classifier.fit(x_train_transform, y_train)

            # -- test -------------------------------------------------------------

            if element == None:
                _results[i] = classifier.score(x_test_transform, y_test)
            else:
                x_test_all, y_test_all= self.split_y_test(x_test_transform , y_test, element, augment_num_angles)

                count = 0
                for num_1 in range(len(y_test_all)):
                    for num_2 in range(len(y_test_all[0])):
                        _results[count][i] = classifier.score(x_test_all[num_1][num_2], y_test_all[num_1][num_2])
                        count+=1

            percent_complete = (i + 1) / self.num_runs * 100
            sys.stdout.write("\r")
            sys.stdout.write(f"1) Training in progress : Epoch {i + 1}/{self.num_runs} ({int(percent_complete)})% completed")
            sys.stdout.flush()

        if element == None:
            minirocket_result = round(_results.mean()*100)
            print(f"\n2) Minirocket classification accuracy: {minirocket_result}%")

            return minirocket_result

        else:
            minirocket_split_result = _results.mean(axis=-1)
            mean_value = np.mean(minirocket_split_result[0:3])
            minirocket_result = np.round(np.insert(minirocket_split_result, 0, mean_value) * 100)
            print(f"\n2) Minirocket classification accuracy : {minirocket_result[0]}%   ==>   ({', '.join([f'{acc}%' for acc in minirocket_result[1:4]])}  /  {', '.join([f'{acc}%' for acc in minirocket_result[4:7]])})")

            return minirocket_result

    def split_y_test(self, x_test, y_test, element, augment_num_angles):
        if element != None:
            indices_1 = [0, 1, 2, 9, 10, 11]; indices_2 = [3, 4, 5, 12, 13, 14]; indices_3 = [6, 7, 8, 15, 16, 17]
            ind_1 = [0, 3, 6, 9, 12, 15]; ind_2 = [1, 4, 7, 10, 13, 16]; ind_3 = [2, 5, 8, 11, 14, 17]
            aug_indices_list = [np.array([], dtype=int) for _ in range(3)]
            aug_ind_list = [np.array([], dtype=int) for _ in range(3)]

            for num, indices in enumerate([indices_1, indices_2, indices_3]):
                for i in indices:
                    aug_indices_list[num] = np.concatenate((aug_indices_list[num], np.arange(i * augment_num_angles, i * augment_num_angles + augment_num_angles)))
            for num, ind in enumerate([ind_1, ind_2, ind_3]):
                for i in ind:
                    aug_ind_list[num] = np.concatenate((aug_ind_list[num], np.arange(i * augment_num_angles, i * augment_num_angles + augment_num_angles)))

            x_test_1 = [x_test[aug_indices_list[0]], x_test[aug_indices_list[1]], x_test[aug_indices_list[2]]]
            y_test_1 = [y_test[aug_indices_list[0]], y_test[aug_indices_list[1]], y_test[aug_indices_list[2]]]

            x_test_2 = [x_test[aug_ind_list[0]], x_test[aug_ind_list[1]], x_test[aug_ind_list[2]]]
            y_test_2 = [y_test[aug_ind_list[0]], y_test[aug_ind_list[1]], y_test[aug_ind_list[2]]]

            x_test_all = [x_test_1, x_test_2]
            y_test_all = [y_test_1, y_test_2]

            return x_test_all, y_test_all