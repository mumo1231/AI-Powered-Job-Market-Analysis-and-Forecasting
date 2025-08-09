import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 读取数据
file = "C:\\Users\\21783\\Desktop\\kczy\\jobb.csv"
data = pd.read_csv(file)

# 识别数值型和分类型特征列
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

print("数值型特征:", numeric_cols)
print("分类型特征:", categorical_cols)

# 准备数据：数值特征在前，分类特征在后
X = data[numeric_cols + categorical_cols].values
categorical_indices = [i for i, col in enumerate(numeric_cols + categorical_cols)
                       if col in categorical_cols]

# 使用 K-Prototypes 算法进行聚类
kproto = KPrototypes(n_clusters=3, init='Huang', random_state=100)
clusters = kproto.fit_predict(X, categorical=categorical_indices)

# 获取聚类中心
cluster_centers = kproto.cluster_centroids_
feature_names = numeric_cols + categorical_cols

print("\n聚类中心坐标：")
for i, center in enumerate(cluster_centers):
    print(f"\n类别 {i} 中心点：")
    for j, (feature, value) in enumerate(zip(feature_names, center)):
        # 检查当前特征是否是数值型
        if j < len(numeric_cols):  # 或者 if feature in numeric_cols
            print(f"{feature}: {float(value):.4f} (数值型)")
        else:
            print(f"{feature}: {value} (分类型)")

# 将数据转换回数值格式以计算评估指标
# 对于评估指标，我们需要将分类变量转换为数值表示
if categorical_cols:
    X_for_metrics = data.copy()
    for col in categorical_cols:
        X_for_metrics[col] = pd.factorize(X_for_metrics[col])[0]
    X_for_metrics = X_for_metrics.values
else:
    X_for_metrics = X

# 计算聚类评估指标
try:
    silhouette = silhouette_score(X_for_metrics, clusters)
    dbi = davies_bouldin_score(X_for_metrics, clusters)
    ch = calinski_harabasz_score(X_for_metrics, clusters)

    print("\n聚类效果评估指标：")
    print(f"轮廓系数(Silhouette Score): {silhouette:.4f} (越接近1越好)")
    print(f"DBI指数(Davies-Bouldin Index): {dbi:.4f} (越小越好)")
    print(f"CH指数(Calinski-Harabasz Index): {ch:.4f} (越大越好)")
except Exception as e:
    print("\n计算评估指标时出错:", e)

# 将聚类结果添加回原始数据
data['Cluster'] = clusters
print("\n添加聚类标签后的数据样例：")
print(data.head())