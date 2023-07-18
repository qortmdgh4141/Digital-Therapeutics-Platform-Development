from conver_npz import convert_dat_to_npz

# "Select Train or Val or Test"
data_type = "train"

if data_type == "train":
    augmentation = True
    Container_Set = ["container_100ml", "container_130ml", "container_250ml", "container_300ml"]

elif data_type == "val":
    augmentation = False
    Container_Set = ["container_150ml"]
elif data_type == "test":
    augmentation = False
    Container_Set = ["container_200ml"]
else:
    raise AssertionError("Invalid data_type value. Please choose data_type as 'train', 'val', or 'test'.")

# sample 폴더의 하위디렉토리 구성
Label_Set = ["benign_tumor", "malignant_tumor"]
#Container_Set = ["container_100ml", "container_130ml", "container_150ml", "container_200ml", "container_250ml", "container_300ml"]
Captube_Set = ["captube_0.2ml", "captube_0.5ml"]
Distance_Set = ["distance_1cm", "distance_3cm", "distance_5cm"]
Height_Set = ["height_0cm", "height_2cm", "height_4cm"]

# 학습 데이터셋으로 사용할 RF S 매개변수 선택
modeS11 = {"use": False, "fileName": "s11", "reset": False, "s11": True, "s21": False, "s12": False, "s22": False}
modeS21 = {"use": False, "fileName": "s21", "reset": False, "s11": False, "s21": True, "s12": False, "s22": False}
modeS12 = {"use": False, "fileName": "s12", "reset": False, "s11": False, "s21": False, "s12": True, "s22": False}
modeS22 = {"use": False, "fileName": "s22", "reset": False, "s11": False, "s21": False, "s12": False, "s22": True}

modeS11_21 = {"use": False, "fileName": "s11_21", "reset": False, "s11": True, "s21": True, "s12": False, "s22": False}
modeS12_22 = {"use": False, "fileName": "s12_22", "reset": False, "s11": False, "s21": False, "s12": True, "s22": True}
modeS21_12 = {"use": False, "fileName": "s21_12", "reset": False, "s11": False, "s21": True, "s12": True, "s22": False}
modeS11_22 = {"use": False, "fileName": "s11_22", "reset": False, "s11": True, "s21": False, "s12": False, "s22": True}

modeAll = {"use": True, "fileName": "all", "reset": False, "s11": True, "s21": True, "s12": True, "s22": True}

mode_dir_list = (modeS11, modeS21, modeS12, modeS22, modeS11_21, modeS12_22, modeS21_12, modeS11_22, modeAll)

# mode 하나만 True이도록 예외처리
true_count = sum(mode["use"] for mode in mode_dir_list)
if true_count != 1:
    raise ValueError("Only one mode should be set to True.")

# original_sample_list 폴더의 sample 폴더 선택
sample_num_name = "sample4"

# 각도의 최댓값을 나타내는 변수
angleCount = 360
# 각 반복에서 각도가 증가하는 크기를 나타내는 변수
angleStep = 10

dat_to_npz = convert_dat_to_npz(Label_Set=Label_Set, Container_Set=Container_Set, Captube_Set=Captube_Set, Distance_Set=Distance_Set, Height_Set= Height_Set,
                                mode_dir_list=mode_dir_list, sample_num_name=sample_num_name, angleCount=angleCount, angleStep=angleStep, data_type=data_type, augmentation=augmentation)

dat_to_npz.main()