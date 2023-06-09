import os
import time
import pandas as pd
from minirocket_classifier import minirocket_clf

def train_test_dir_path(dir_path):
    split_path = dir_path.split('\\')
    feature = split_path[-2]  # 맨 오른쪽에서 두번째에 있는 "distance"
    feature_train_test = split_path[-1]  # 맨 오른쪽 첫번째에 있는 문자열 "distance(1&5-3)"

    extracted = feature_train_test[feature_train_test.find("(")+1 : feature_train_test.find(")")]  # "()" 안의 문자열 "1&5-3"
    split_extracted = extracted.split("-")
    train_part = split_extracted[0]  # "1&5"
    test_part = split_extracted[1]  # "3"

    feature_train = 'train_' + feature + "(" + train_part + ")"
    feature_test = 'test_' + feature + "(" + test_part + ")"

    train_dir_path = os.path.join(dir_path, feature_train)
    test_dir_path = os.path.join(dir_path, feature_test)

    return train_dir_path, test_dir_path, feature_train_test

all_dir_path = []

# 5/22 미팅자료 실험을 위해 수정해야 하는 부분
dir_paths = [os.path.join("../../training_test_data_generator/merged_sample3/all/angle360_step10", feature) for feature in ["distance", "height", "type"]]
# angle_interval = 360
# 기본 36개의 각도를 다 사용하는 angle_interval = 10
angle_interval = 10

split_accuracy = True
augmentation = False
total_angle_num = 36

for dir_path in dir_paths :
    subdir_name = next(os.walk(dir_path))[1]
    for name in subdir_name:
        subdir_path = os.path.join(dir_path, name)
        all_dir_path.append(subdir_path)

# 컬럼명만 있는 데이터프레임 생성
if split_accuracy==True:
    result_df = pd.DataFrame(columns=['Train/Test Set Composition', 'Minirocket Accuracy', "Accuracy_1-1", "Accuracy_1-2", "Accuracy_1-3", "Accuracy_2-1", "Accuracy_2-2", "Accuracy_2-3"])
else:
    result_df = pd.DataFrame(columns=['Train/Test Set Composition', 'Minirocket Accuracy'])

# 빈 행 추가#
empty_row = pd.DataFrame([{}], columns=result_df.columns)
result_df = pd.concat([result_df, empty_row], ignore_index=True)

for num, dir_path in enumerate(all_dir_path):
    (train_path, test_path, train_test_set_composition) = train_test_dir_path(dir_path)

    print(f"Start ({num+1} / {len(all_dir_path)})     :       {train_test_set_composition}    ({dir_path})")
    start_time = time.time()
    minirocket = minirocket_clf(train_path, test_path)

    if split_accuracy == True:
        element = train_test_set_composition.split('(')[0]
        minirocket_result = minirocket.main(augmentation=augmentation, element=element, total_angle_num=total_angle_num, angle_interval=angle_interval)
    else :
        element = None
        minirocket_result = minirocket.main(augmentation=augmentation, element=element, total_angle_num=total_angle_num, angle_interval=angle_interval)

    # 새로운 값 추가
    if split_accuracy == True:
        new_values = [train_test_set_composition, minirocket_result[0], minirocket_result[1], minirocket_result[2], minirocket_result[3], minirocket_result[4], minirocket_result[5], minirocket_result[6]]
        result_df.loc[result_df.index[-1] + 1] = new_values
    else:
        new_values = [train_test_set_composition, minirocket_result]
        result_df.loc[result_df.index[-1] + 1] = new_values

    # 빈 행 추가
    if (num - 2) % 3 == 0:
        empty_row = pd.DataFrame([{}], columns=result_df.columns)
        result_df = pd.concat([result_df, empty_row], ignore_index=True)

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Time elapsed: {minutes} minutes {seconds} seconds\n")
    print("\n")

# 데이터프레임을 엑셀 파일로 저장
result_df.to_excel('sample_result_s11_s22.xlsx', index=False)