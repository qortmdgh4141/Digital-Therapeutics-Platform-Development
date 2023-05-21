import os
import shutil
import numpy as np

class convert_dat_to_npz:
    def __init__(self, Label_Set, Type_Set, Distance_Set, Height_Set,
                mode_dir_list, sample_num_name, angleCount, angleStep, train_test_split):
        self.Label_Set = Label_Set
        self.Type_Set = Type_Set
        self.Distance_Set = Distance_Set
        self.Height_Set = Height_Set
        self.mode_dir_list = mode_dir_list
        self.sample_num_name = sample_num_name
        self.angleCount = angleCount
        self.angleStep = angleStep
        self.train_test_split = train_test_split

        self.maxValue = 4
        self.maxLine = 1001
        self.original_dataPath = f"original_sample_list\\{self.sample_num_name}"
        self.names_npzs = ["TRAIN", "TEST"]

        self.angle_range_str = str(self.angleCount) + '_(' + str(self.angleStep) + ')'

    def main(self, Feature_Set=None, train_feature=None, test_feature=None):
        modenames = [mode["fileName"] for mode in self.mode_dir_list if mode["use"]][0]

        cache_subdir_paths = [os.path.join("cache_folder\\", cache_subdir) for cache_subdir in ["cache_train", "cache_test", "cache_undivided"]]
        for cache_subdir_path in cache_subdir_paths:
            if os.path.exists(cache_subdir_path):
                shutil.rmtree(cache_subdir_path)
                print(f"Deleted directory: {cache_subdir_path}")

        if self.train_test_split:
            feature_name = train_feature[0].split('_')[0]
            train_num = self.extract_numbers_from_strings(train_feature)
            test_num = self.extract_numbers_from_strings(test_feature)
            train_test_num = train_num + "-" + test_num

            train_path = f"cache_folder\\cache_train\\data({train_num})"
            test_path = f"cache_folder\\cache_test\\data({test_num})"
            newPath = [train_path, test_path]
            merge_newpath = [f"merged_{self.sample_num_name}\\{modenames}\\angle{self.angleCount}_step{self.angleStep}\\{feature_name}\\{feature_name}({train_test_num})\\train_{feature_name}({train_num})",
                             f"merged_{self.sample_num_name}\\{modenames}\\angle{self.angleCount}_step{self.angleStep}\\{feature_name}\\{feature_name}({train_test_num})\\test_{feature_name}({test_num})"]

            print("\n 1) start : run function")

            for i, feature in enumerate([train_feature, test_feature]):
                if not os.path.exists(merge_newpath[i]):
                    os.makedirs(newPath[i])
                Feature_Set[:] = feature
                self.run(newPath=newPath[i], names_npz=self.names_npzs[i])

            for i, names_npz in enumerate(self.names_npzs):
                print(f"\n 2-{i + 1}) start-{names_npz} : merge function")
                if not os.path.exists(merge_newpath[i]):
                    os.makedirs(merge_newpath[i])
                self.run_move(new_path=newPath[i], merge_newpath=merge_newpath[i], names_npz=names_npz)

        else:
            newPath = ["cache_folder\\cache_undivided\\data(undivided)"]
            merge_newpath = [f"merged_{self.sample_num_name}\\{modenames}\\angle{self.angleCount}_step{self.angleStep}\\undivided"]

            if not os.path.exists(merge_newpath[0]):
                os.makedirs(newPath[0])

            print("\n 1) start : run function")

            self.run(newPath=newPath[0], names_npz=self.names_npzs[0]+'_'+self.names_npzs[1])

            for i, names_npz in enumerate(["TRAIN_TEST"]):
                print(f"\n 2-{i+1}) start-{names_npz} : merge function")
                if not os.path.exists(merge_newpath[i]):
                    os.makedirs(merge_newpath[i])
                self.run_move(new_path=newPath[i], merge_newpath=merge_newpath[i], names_npz=names_npz)

    def extract_numbers_from_strings(self, feature_list):
        num_list = []
        for string in feature_list:
            num_str = ''
            for c in string:
                if c.isdigit() or c == '.':
                    num_str += c
            num_list.append(num_str)
        if len(num_list) == 2:
            num_string = num_list[0]+"&"+num_list[1]
        else :
            num_string = num_list[0]
        return num_string

    def run(self, newPath, names_npz):
        file_num = 0
        for label in range(len(self.Label_Set)):
            for type in self.Type_Set:
                for distance in self.Distance_Set:
                    for height in self.Height_Set:
                            total_data = []
                            for angle in range(0, self.angleCount, self.angleStep):
                                file = open(f"{self.original_dataPath}\\{self.Label_Set[label]}\\{type}\\{distance}\\{height}\\{angle//10}.dat", "r")
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

                            for mode in self.mode_dir_list:
                                self.makeFile(mode=mode, newPath=newPath, names_npz=names_npz, data_arr=data_arr, label=label, file_num=file_num)
                            file_num += 1

    def makeFile(self, mode, newPath, names_npz, data_arr, label, file_num):
        if mode["use"] is False : return

        mode_newpath = f"{newPath}\\{mode['fileName']}_{self.angle_range_str}"
        if not os.path.exists(mode_newpath):
            os.makedirs(mode_newpath)

        mode_index = [mode[mode_name] for mode_name in ["s11", "s21", "s12", "s22"]]
        curData = data_arr[:, mode_index]

        np.savez(f"{mode_newpath}\\{names_npz}_({file_num}).npz", data=curData, label=[label])

    def run_move(self, new_path, merge_newpath, names_npz):

        for mode in self.mode_dir_list:
            if mode["use"] is False: continue

            src_dir = os.path.join(new_path, f"{mode['fileName']}_{self.angle_range_str}")  # 원본 디렉토리 경로
            dst_dir = merge_newpath # 대상 디렉토리 경로

            # 원본 디렉토리에서 npz 파일을 찾아서 복제하여 대상 디렉토리로 이동
            for file_name in os.listdir(src_dir):
                if file_name.endswith('.npz'):  # 파일 이름이 .npz로 끝나는 경우
                    src_path = os.path.join(src_dir, file_name)  # 원본 파일 경로
                    dst_path = os.path.join(dst_dir, file_name)  # 대상 파일 경로
                    shutil.copy(src_path, dst_path)  # 파일 복제

            print(f"create ({names_npz}) : {merge_newpath}")