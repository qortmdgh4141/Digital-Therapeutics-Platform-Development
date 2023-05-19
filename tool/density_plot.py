import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
#file_path = '../merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/test_distance(5)/TEST.tsv'
#df = pd.read_csv(file_path, delimiter='\t', header=None)
file_path1 = 'merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/test_distance(5)/TEST.tsv'
file_path2 = 'merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/train_distance(1&3)/TRAIN.tsv'

# 두 데이터프레임을 행 방향으로 합치기
df1 = pd.read_csv(file_path1, delimiter='\t', header=None)
df2 = pd.read_csv(file_path2, delimiter='\t', header=None)
df = pd.concat([df1, df2], axis=0)

new_col_name = ['label', 'S11', 'S21', 'S12', 'S22']
df.columns = new_col_name

df = df.iloc[:, 1:]

# StandardScaler 객체 생성
scaler = StandardScaler()

# 각 열의 값들을 표준화
df[:] = scaler.fit_transform(df[:])

# 그래프 그리기
colors = ['r', 'b', 'g', 'm'] # 각 열별로 다른 색상 리스트
fig, axs = plt.subplots(ncols=4, figsize=(18, 4), sharey=True) # sharey 옵션을 사용해 y축 범위 동일하게 설정
fig.subplots_adjust(wspace=0.3) # 그래프 간격 조정

for i, col in enumerate(df.columns):
    sns.kdeplot(df[col], ax=axs[i], color=colors[i], shade=True, alpha=0.4, linewidth=2, legend=False)
    axs[i].set_xlabel(col, fontsize=12)
    axs[i].set_ylabel('Density')
    axs[i].spines['right'].set_visible(False)  # 오른쪽 라인 제거
    axs[i].spines['top'].set_visible(False)  # 위쪽 라인 제거
    axs[i].spines['bottom'].set_linewidth(0.5)  # 아래쪽 라인 두께 조절
    axs[i].spines['left'].set_linewidth(0.5)  # 왼쪽 라인 두께 조절
    axs[i].tick_params(axis='x', rotation=45)
    axs[i].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    axs[i].set_xlim(left=-4.0, right=4.0)

plt.suptitle('Density Plots with Standardization', fontsize=16)  # suptitle 추가
plt.tight_layout()

plt.show()

