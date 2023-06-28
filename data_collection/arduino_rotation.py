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
def automate_data_save_picovna(file_path, num, positions):
    # 1) File 클릭  -->  2) Save Measurements 클릭  -->  3) Real + Imag 클릭  -->  4) Save 클릭
    for position in positions[:4]:
        click_position(*position)

    # 5) 파일 이름(N) 클릭 후, 파일명 입력
    click_position(*positions[4])
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.typewrite(str(num))
    click_position(*positions[5])

    # 6) 저장(S) 클릭
    click_position(*positions[6])
    pyautogui.press('esc')
    pyautogui.press('esc')

    # 올바르게 파일이 저장 안될 시, 프로그램 종료
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def click_position(x, y):
    pyautogui.moveTo(x=x, y=y)
    pyautogui.click(button='left')

App = QApplication(sys.argv)
pyqt_bar = Pyqt_Bar()  # create the instance of our Window

positions = [
    (32, 59),   # 1) File 클릭
    (117, 295),  # 2) Save Measurements 클릭
    (477, 321),  # 3) Real + Imag 클릭
    (325, 586),  # 4) Save 클릭
    (404, 808),  # 5) 파일 이름(N) 클릭 후, 파일명 입력
    (372, 890),  # 5) 파일 이름(N) 클릭 후, 파일명 입력 후, 빈공간 클릭
    (1238, 914),  # 6) 저장(S) 클릭
]

comPort = "COM4"
angleSet = range(0, 360, 10)
cache_directory = "C:\\Users\\82104\\Desktop\\RF 기반 생체 정보 및 헬스케어\\데이터 수집\\sample_data\\cache"

# Arduino 조작
if __name__ == "__main__":
    try:
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
            file_path = os.path.join(cache_directory, f"{num}.dat")

            automate_data_save_picovna(file_path=file_path, num=num, positions=positions)
            pyqt_bar.update_value(value=num)

            progress_bar = tqdm.tqdm(range(4))
            progress_bar.set_description("Please wait a moment while the PicoVNA is measuring the frequency  ")
            for _ in progress_bar:
                time.sleep(1)

            while True:
                arduino.write(command.encode())
                if arduino.readable():
                    value = arduino.readline()
                    if value.decode()[:6] == "rotate":
                        break

        sys.exit(App.exec())  # start the app

    except Exception as ex:
        _, _, tb = sys.exc_info()
        print(f"\n\n[main:{tb.tb_lineno}] {ex}\n\n")