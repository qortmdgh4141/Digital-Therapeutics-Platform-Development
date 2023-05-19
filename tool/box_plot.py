import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
# file_path = 'merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/test_distance(5)/TEST.tsv'
# df = pd.read_csv(file_path, delimiter='\t', header=None)
file_path1 = '../merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/test_distance(5)/TEST.tsv'
file_path2 = '../merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/train_distance(1&3)/TRAIN.tsv'

# 두 데이터프레임을 행 방향으로 합치기
df1 = pd.read_csv(file_path1, delimiter='\t', header=None)
df2 = pd.read_csv(file_path2, delimiter='\t', header=None)
df = pd.concat([df1, df2], axis=0)

# 표준화
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.iloc[:, 1:])

# 이상치 탐지 함수 정의
def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1)

    return outliers

# 이상치 탐지
outliers = detect_outliers(pd.DataFrame(scaled_df))

# 이상치를 제외한 데이터와 이상치를 분리
clean_data = df.iloc[~outliers.values, :]

print(f'원본 데이터 개수 : {len(df)}')
print(f'이상치를 제외한 데이터 개수 : {len(clean_data)}')
print(f'--> {round(len(clean_data)/len(df)*100, 2)}%')

# 이상치를 포함한 원본 데이터로 박스플롯 그리기
ax1 = plt.subplot(1, 2, 1)
sns.boxplot(data=df.iloc[:, 1:])
plt.xlabel('Data')
plt.ylabel('Value')
plt.title('Original Data')
ax1.set_xticklabels(['S11', 'S21', 'S12', 'S22'])

# 이상치를 제외한 데이터로 박스플롯 그리기
ax2 = plt.subplot(1, 2, 2)
sns.boxplot(data=clean_data.iloc[:, 1:])
plt.xlabel('Data')
plt.ylabel('Value')
plt.title('Data Excluding Outliers')
ax2.set_xticklabels(['S11', 'S21', 'S12', 'S22'])

plt.suptitle('Compare Distributions of Data', fontsize=16)
plt.tight_layout()
plt.show()






