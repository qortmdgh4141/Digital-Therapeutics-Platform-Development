import os
import sys
import time
import tqdm
import serial
import pyautogui

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QApplication, QProgressBar

# Prograss Bar 조작
class Pyqt_Bar(QMainWindow):
    def __init__(self):
        super().__init__()

        # 출력 위치 설정
        self.win_width, self.win_height = pyautogui.size()[0], pyautogui.size()[1]
        self.setGeometry(0, self.win_height-160, self.win_width, 100)  # setting geometry

        # 상단 바 설정
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # ProgressBar 위젯 설정
        self.bar = QProgressBar(self)  # creating progress bar
        self.bar.setGeometry(0, 0, self.win_width, 100)  # setting geometry to progress bar
        self.bar.setValue(0)  # setting the value to maximum (100)
        self.bar.setAlignment(Qt.AlignCenter)  # setting alignment to center
        font = QFont('Segoe UI', 30)
        font.setWeight(75)  # 두께 설정
        self.bar.setFont(font)
        self.bar.setStyleSheet("QProgressBar" "{" "border : 10px solid green" "}")

        self.show()  # showing all the widgets

    def update_value(self, value):
        percentage = round((value + 1) / 36 * 100)
        self.bar.setValue(percentage)
        if percentage == 100:
            self.bar.setValue(percentage)
            self.bar.setStyleSheet("QProgressBar" "{" "border : 10px solid red" "}"
                                   "QProgressBar::chunk" "{" "background-color: red;" "}")

        QApplication.processEvents()

# Picovna Software 조작
def click_position(x, y):
    pyautogui.moveTo(x=x, y=y)
    pyautogui.click(button='left')

def automate_data_save_picovna(write_list, positions):
    # 1) File 클릭  -->  2) Save Measurements 클릭  -->  3) Real + Imag 클릭  -->  4) Save 클릭
    for position in positions[:4]:
        click_position(*position)

    # 5) 경로 클릭 후, 파일명 입력  -->  6) 파일 이름(N) 클릭 후, 파일명 입력 및 저장
    for i, write_position in enumerate(positions[4:6]):
        click_position(*write_position)
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('backspace')
        pyautogui.typewrite(write_list[i])
        pyautogui.press('enter')

    pyautogui.press('esc')

    # 올바르게 파일이 저장 안될 시, 프로그램 종료
    data_path = os.path.join(write_list[0], f"{write_list[1]}.dat")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} does not exist.")

# 올바른 딕셔너리 경로를 설정하였는지 검사
def check_dict(dict):
    true_count = 0
    key_name = None
    for key, value in dict.items():
        if value is True:
            true_count += 1
            key_name = key

    if true_count == 0:
        raise ValueError("The dictionary must have at least one True value.")
    elif true_count > 1:
        raise ValueError("The dictionary can only have one True value.")

    return key_name

App = QApplication(sys.argv)
pyqt_bar = Pyqt_Bar()  # create the instance of our Window

positions = [
    (32, 59),  # 1) File 클릭
    (117, 295),  # 2) Save Measurements 클릭
    (477, 321),  # 3) Real + Imag 클릭
    (325, 586),  # 4) Save 클릭
    (382, 240),  # 5) 경로 클릭 후, 파일명 입력
    (404, 808),  # 6) 파일 이름(N) 클릭 후, 파일명 입력 및 저장
]

T=True; F=False
org_path = "C:\\Users\\82104\\Desktop\\DTX\Digital-Therapeutics-Platform-Development\\data_collection\\sample_data\\sample5"

class_tumor = {
                "benign_tumor"    : T,
                "malignant_tumor" : F}
container = {
                "container_100ml" : T,
                "container_130ml" : F,

                "container_150ml" : F,
                "container_200ml": F,

                "container_250ml": F,
                "container_300ml": F}

captube = {
                "captube_0.2ml" : T,
                "captube_0.5ml" : F}

distance = {
                "distance_1cm" : T,
                "distance_3cm" : F,
                "distance_5cm" : F}

height = {
                "height_0cm" : T,
                "height_2cm" : F,
                "height_4cm" : F}

# 딕셔너리당 True가 한개가 아닐시 오류출력 (+Caps Lock 키가 켜져있으면 오류가 나니, 반드시 비활성화)
key_list = []
dic_list = [class_tumor, container, captube, distance, height]
for dict in dic_list:
    key_name = check_dict(dict)
    key_list.append(key_name)

# 디렉토리 내 파일 개수 확인 후, 파일 개수가 0보다 큰 경우 오류 출력 (==빈디렉토리가 아니면 오류출력)
save_directory = os.path.join(org_path, *key_list)
if len(os.listdir(save_directory)) > 0:
    raise ValueError(f"The {save_directory} directory is not empty.")

# Arduino 조작
if __name__ == "__main__":
    try:
        comPort = "COM4"
        angleSet = range(0, 360, 10)

        arduino = serial.Serial(port=comPort, baudrate=9600, timeout=1)
        print("\nArduino connection here!")
        print("-" * 115)

        for num, angle in enumerate(angleSet):

            if angle % 30 == 0:
                command = "1"
            else:
                command = "2"

            print(f"\n[angle={angle}°]", end="\t:\t")
            print(f"Picovna is automatically saving the file {num}.dat...")
            write_list = [save_directory, str(num)]
            automate_data_save_picovna(write_list=write_list, positions=positions)
            pyqt_bar.update_value(value=num)

            progress_bar = tqdm.tqdm(range(4))
            progress_bar.set_description("Please wait a moment while the PicoVNA is measuring the frequency  ")
            for _ in progress_bar:
                time.sleep(4)

            while True:
                arduino.write(command.encode())
                if arduino.readable():
                    value = arduino.readline()
                    if value.decode()[:6] == "rotate":
                        break

        # 최종 저장된 파일 개수가 36개가 아닐 시, 오류 출력
        if len(os.listdir(save_directory)) != 36:
            raise ValueError(f"The number of files in {save_directory} is not equal to 36.")

        sys.exit(App.exec())  # start the app

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[main:{tb.tb_lineno}] {ex}\n\n")
