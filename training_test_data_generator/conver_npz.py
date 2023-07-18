import os
import shutil
import numpy as np

class convert_dat_to_npz:
    def __init__(self, Label_Set, Container_Set, Captube_Set, Distance_Set, Height_Set,
                mode_dir_list, sample_num_name, angleCount, angleStep, data_type, augmentation):
        self.Label_Set = Label_Set
        self.Container_Set = Container_Set
        self.Captube_Set = Captube_Set
        self.Distance_Set = Distance_Set
        self.Height_Set = Height_Set
        self.mode_dir_list = mode_dir_list
        self.sample_num_name = sample_num_name
        self.angleCount = angleCount
        self.angleStep = angleStep
        self.data_type = data_type
        self.augmentation = augmentation

        self.maxValue = 4
        self.maxLine = 1001
        self.original_dataPath = f"original_sample_list\\{self.sample_num_name}"
        self.angle_range_str = 'angel'+ str(self.angleCount) + '_step' + str(self.angleStep)

    def main(self):
        modenames = [mode["fileName"] for mode in self.mode_dir_list if mode["use"]][0]

        newPath = "cache_folder"
        merge_newpath = f"merged_{self.sample_num_name}\\{modenames}\\{self.angle_range_str}\\{self.data_type}\\"
        aug_merge_newpath = f"merged_{self.sample_num_name}\\{modenames}\\{self.angle_range_str}\\{self.data_type}_aug\\"

        for path in [newPath, merge_newpath]:
            self.recreate_directory(path)
        if self.augmentation==True:
            self.recreate_directory(aug_merge_newpath)
        newPath = os.path.join(newPath, f"{modenames}_{self.angle_range_str}")
        self.recreate_directory(newPath)



        print("1) start : run function")
        self.run(newPath=newPath)
        print(f"2) start : run_move function")
        self.run_move(newPath=newPath, merge_newpath=merge_newpath, aug_merge_newpath=aug_merge_newpath)

    def recreate_directory(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"※ Deleted path : {path}")
        os.makedirs(path)
        print(f"※ Created empty directory : {path}\n")

    def run(self, newPath):
        for label in range(len(self.Label_Set)):
            for container in self.Container_Set:
                for captube in self.Captube_Set:
                    for distance in self.Distance_Set:
                        for height in self.Height_Set:
                                total_data = []
                                for angle in range(0, self.angleCount, self.angleStep):
                                    file = open(f"{self.original_dataPath}\\{self.Label_Set[label]}\\{container}\\{captube}\\{distance}\\{height}\\{angle//10}.dat", "r")
                                    data = []
                                    for _ in range(self.maxLine):
                                        line = file.readline().rstrip("\n").split()
                                        line = [float(i) for i in line]
                                        curData = []
                                        for y in range(self.maxValue):
                                            value1 = line[2 * y + 1] # I Value
                                            value2 = line[2 * y + 2] # Q Value
                                            curData.append(np.array([value1, value2]))
                                        data.append(np.array(curData)) # len(data) = maxLine= 1001 / curData shape = (maxValue, 2) = (4, 2)
                                    total_data.append(np.stack(data, axis=-1)) # len(total_data) = angleCount = 36  / np.stack(data, axis=-1) shape = (maxValue, 2, maxline) = (4, 2, 1001)
                                    file.close()

                                data_arr = np.stack(total_data, axis=0) # data_arr shape = (angleCount, amaxValue, 2, maxline) = (36, 4, 2, 1001)

                                identifier_name = f"label-{label}-{container}-{captube}-{distance}-{height}"
                                for mode in self.mode_dir_list:
                                    self.makeFile(mode=mode, newPath=newPath, data_arr=data_arr, label=label, identifier_name=identifier_name)

    def makeFile(self, mode, newPath, data_arr, label, identifier_name):
        if mode["use"] is False : return

        mode_index = [mode[mode_name] for mode_name in ["s11", "s21", "s12", "s22"]]
        curData = data_arr[:, mode_index]

        np.savez(f"{newPath}\\{identifier_name}.npz", input=curData, target=[label])

    def run_move(self, newPath, merge_newpath, aug_merge_newpath):
        for mode in self.mode_dir_list:
            if mode["use"] is False: continue

            src_dir = newPath # 원본 디렉토리 경로
            dst_dir = merge_newpath # 대상 디렉토리 경로

            # 원본 디렉토리에서 npz 파일을 찾아서 복제하여 대상 디렉토리로 이동
            for file_name in os.listdir(src_dir):
                if file_name.endswith('.npz'):  # 파일 이름이 .npz로 끝나는 경우
                    src_path = os.path.join(src_dir, file_name)  # 원본 파일 경로
                    dst_path = os.path.join(dst_dir, file_name)  # 대상 파일 경로
                    shutil.copy(src_path, dst_path)  # 파일 복제

                    if self.augmentation == True:
                        file_name_without_extension = os.path.splitext(file_name)[0]
                        self.augment_data(dst_path=dst_path, aug_path=aug_merge_newpath, name=file_name_without_extension)

            print(f"\n- Created files in aug_merge_new_path: {merge_newpath}")
            print(f"- Created files in aug_merge_new_path: {aug_merge_newpath}")

    def augment_data(self, dst_path, aug_path, name):
        loaded = np.load(dst_path)
        input_data = loaded["input"]  # 입력 데이터
        target_data = loaded["target"]  # 대상 데이터

        num_angles = input_data.shape[0]  # augmentaion할 각도 갯수(ex.36)

        for i, angle_num in enumerate(range(num_angles)):
            # 어규멘테이션 데이터
            augmented_input_data = np.roll(input_data, -angle_num, axis=0)
            save_path = os.path.join(aug_path, f"{name}_{i}.npz")
            np.savez(save_path, input=augmented_input_data, target=target_data)