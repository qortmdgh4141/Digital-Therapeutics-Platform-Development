import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_path1 = '../merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/test_distance(5)/TEST.tsv'
file_path2 = '../merged_sample3_tsv_version/angle360_step10/multivariate/distance/distance(1&3-5)/train_distance(1&3)/TRAIN.tsv'

# 두 데이터프레임을 행 방향으로 합치기
df1 = pd.read_csv(file_path1, delimiter='\t', header=None)
df2 = pd.read_csv(file_path2, delimiter='\t', header=None)
df = pd.concat([df1, df2], axis=0)

new_col_name = ['label', 's11', 's21', 's12', 's22']
df.columns = new_col_name

# 상관계수 행렬 생성
corr_matrix = df.iloc[:, 1:].corr()
print(corr_matrix)


# heatmap으로 상관계수 행렬 시각화
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.4f', square=True, cbar=True,
            vmin=0, vmax=1, cmap='RdBu_r', annot_kws= {'size':18}, ax=ax, linewidths=4, linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=17)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=17)

plt.title("Correlation Heatmap", fontsize=19, pad=40)

plt.tight_layout()

plt.show()


