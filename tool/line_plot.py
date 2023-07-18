import numpy as np
import matplotlib.pyplot as plt
import math



def plot_graph(path0, path1):
    epochs = range(1, 1001 + 1)
    plt.figure(figsize=(12, 3))
    # "s11, s21, s12, s22"
    loaded = np.load(path0)
    input_data0 = loaded["input"]  # 입력 데이터

    loaded = np.load(path1)
    input_data1 = loaded["input"]  # 입력 데이터

    name = ["s11", "s21", "s12", "s22"]

    for i in range(0, 4):
        plt.subplot(4, 1, i+1)
        data0 = input_data0[0, i, 0, :] + 1j*input_data0[0, i, 1, :]
        data1 = input_data1[0, i, 0, :] + 1j*input_data1[0, i, 1, :]

        plt.plot(epochs, data0, 'b', label='label 0 : I/Q Value')
        plt.plot(epochs, data1, 'r', label='label 1 : I/Q Value')

        plt.xlabel(f'Frequency of {name[i]}\n\n\n\n')
        plt.ylabel('I/Q Value')

        plt.legend()
        plt.gca().legend(loc='upper right')
        plt.subplots_adjust(wspace=0, hspace=0.5)

    plt.show()


def scatter_graph(path0):
    epochs = range(1, 1001 + 1)
    plt.figure(figsize=(12, 3))
    # "s11, s21, s12, s22"
    loaded = np.load(path0)
    input_data0 = loaded["input"]  # 입력 데이터

    name = ["s11", "s21", "s12", "s22"]



    for i in range(0, 4):
        #for v in range(1001):
        plt.subplot(4, 1, i+1)
        I=input_data0[0, i, 0, :250]
        Q=input_data0[0, i, 1, :250]
        plt.scatter(x=I, y=Q, color='red', label='label 0 : 0~250')

        I = input_data0[0, i, 0, 250:500]
        Q = input_data0[0, i, 1, 250:500]
        plt.scatter(x=I, y=Q, color='green', label='label 0 : 250~500 Value')

        I = input_data0[0, i, 0, 500:750]
        Q = input_data0[0, i, 1, 500:750]
        plt.scatter(x=I, y=Q, color='yellow', label='label 0 : 500~750 Value')


        I = input_data0[0, i, 0, 750:1001]
        Q = input_data0[0, i, 1, 750:1001]
        plt.scatter(x=I, y=Q, color='blue', label='label 0 : 750~1001 Value')


        plt.xlabel(f'Frequency of {name[i]}\n\n\n\n')
        plt.ylabel('I/Q Value')

        plt.legend()
        plt.gca().legend(loc='upper right')
        plt.subplots_adjust(wspace=0, hspace=0.5)

    plt.show()


sample4 = True
if sample4:
    file_path0 = r"C:\Users\82104\PycharmProjects\Digital-Therapeutics-Platform-Development\training_test_data_generator\merged_sample4\all\angel360_step10" \
                 r"\test" \
                r"\label-0-container_200ml-captube_0.5ml-distance_1cm-height_0cm.npz"

    file_path1 = r"C:\Users\82104\PycharmProjects\Digital-Therapeutics-Platform-Development\training_test_data_generator\merged_sample4\all\angel360_step10" \
                 r"\test" \
                r"\label-1-container_200ml-captube_0.5ml-distance_1cm-height_0cm.npz"
else:
    file_path0 = r"C:\Users\82104\PycharmProjects\Digital-Therapeutics-Platform-Development\training_test_data_generator\merged_sample3\all\angel360_step10" \
                 r"\train" \
                 r"\label-0-container_200ml-captube_1.5ml-distance_1cm-height_0cm.npz"
    file_path1 = r"C:\Users\82104\PycharmProjects\Digital-Therapeutics-Platform-Development\training_test_data_generator\merged_sample3\all\angel360_step10" \
                 r"\train" \
                 r"\label-1-container_200ml-captube_1.5ml-distance_1cm-height_0cm.npz"


#plot_graph(path0=file_path0, path1=file_path1)
scatter_graph(file_path0)