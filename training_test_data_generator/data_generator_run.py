# from conver_csv import convert_dat_to_tsv
from conver_npz import convert_dat_to_npz

def make_comb(Set):
    x = [[Set[0], Set[1]], [Set[2]]]
    y = [[Set[0], Set[2]], [Set[1]]]
    z = [[Set[1], Set[2]],[Set[0]]]
    return [x, y, z]

def reset():
    Type_Set[:] = ["type_1.5ml", "type_2ml", "type_4ml"]
    Distance_Set[:] = ["distance_1cm", "distance_3cm", "distance_5cm"]
    Height_Set[:] = ["height_0cm", "height_2cm", "height_4cm"]

# sample 폴더의 하위디렉토리 구성
#Label_Set = ["benign_tumor", "malignant_tumor"]
#Type_Set = ["type_1.5ml", "type_2ml", "type_4ml"]
#Distance_Set = ["distance_1cm", "distance_3cm", "distance_5cm"]
#Height_Set = ["height_0cm", "height_2cm", "height_4cm"]

Label_Set = ["benign_tumor", "malignant_tumor"]
Type_Set = ["type_2ml"]
Distance_Set = ["distance_7cm"]
Height_Set = ["height_0cm"]

# 학습 데이터셋으로 사용할 RF S 매개변수 선택
modeS11 = {"use": False, "fileName": "s11", "reset": False, "s11": True, "s21": False, "s12": False, "s22": False}
modeS21 = {"use": False, "fileName": "s21", "reset": False, "s11": False, "s21": True, "s12": False, "s22": False}
modeS12 = {"use": False, "fileName": "s12", "reset": False, "s11": False, "s21": False, "s12": True, "s22": False}
modeS22 = {"use": False, "fileName": "s22", "reset": False, "s11": False, "s21": False, "s12": False, "s22": True}

modeS11_21 = {"use": False, "fileName": "s11_21", "reset": False, "s11": True, "s21": True, "s12": False, "s22": False}
modeS12_22 = {"use": False, "fileName": "s12_22", "reset": False, "s11": False, "s21": False, "s12": True, "s22": True}

modeS21_12 = {"use": False, "fileName": "s21_12", "reset": False, "s11": False, "s21": True, "s12": True, "s22": False}
modeS11_22 = {"use": False, "fileName": "s11_22", "reset": False, "s11": True, "s21": False, "s12": False, "s22": True}

#modeRx2 = {"use": False, "fileName": "rx2", "reset": False, "s11": False, "s21": True, "s12": False, "s22": True}
#modeRx1 = {"use": False, "fileName": "rx1", "reset": False, "s11": True, "s21": False, "s12": True, "s22": False}
#modeRx2 = {"use": False, "fileName": "rx2", "reset": False, "s11": False, "s21": True, "s12": False, "s22": True}
modeAll = {"use": True, "fileName": "all", "reset": False, "s11": True, "s21": True, "s12": True, "s22": True}

# mode_dir_list = (modeS11, modeS21, modeS12, modeS22, modeRx1, modeRx2, modeAll)
mode_dir_list = (modeS11, modeS21, modeS12, modeS22, modeS11_21, modeS12_22, modeS21_12, modeS11_22, modeAll)

# mode 하나만 True이도록 예외처리
true_count = sum(mode["use"] for mode in mode_dir_list)
if true_count != 1:
    raise ValueError("Only one mode should be set to True.")


# original_sample_list 폴더의 sample 폴더 선택
#sample_num_name = "sample3"
sample_num_name = "test_distanc_7cm(sample3)"

# 각도의 최댓값을 나타내는 변수
angleCount = 360
# 각 반복에서 각도가 증가하는 크기를 나타내는 변수
angleStep = 10

# 학습/테스트 데이터셋을 나눌지 결정
train_test_split = False

dat_to_npz = convert_dat_to_npz(Label_Set=Label_Set, Type_Set=Type_Set, Distance_Set=Distance_Set, Height_Set= Height_Set,
                                mode_dir_list=mode_dir_list, sample_num_name=sample_num_name, angleCount=angleCount, angleStep=angleStep, train_test_split=train_test_split)

if train_test_split:
    for set in [Height_Set, Type_Set, Distance_Set]:
        Feature_Set = set
        comb_sets = make_comb(set)
        for comb_set in comb_sets :
            train_feature = comb_set[0]
            test_feature = comb_set[1]
            print(test_feature)
            dat_to_npz.main(Feature_Set, train_feature, test_feature)
            reset()
else:
    dat_to_npz.main()


