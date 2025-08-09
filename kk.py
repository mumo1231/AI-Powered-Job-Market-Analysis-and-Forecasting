import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

file = "C:\\Users\\21783\\Desktop\\kczy\\jobb.csv"
data = pd.read_csv(file)

# Change objective type to numerical type
cat_cols = []
for col in data.columns:
    if data[col].dtypes == "object":
        cat_cols.append(col)

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

xgbr = XGBRegressor(n_estimators=171, max_depth=9, colsample_bytree=0.579)
xgbr.fit(X_train, y_train)
y_xgbr = xgbr.predict(X_test)
y_xgbr_rounded = np.round(y_xgbr)

# 计算MAE
print("MAE Score:", mean_absolute_error(y_xgbr_rounded, y_test))


# 计算加权准确率
def weighted_accuracy(y_true, y_pred):
    # 四舍五入到最近的整数
    y_true_rounded = np.round(y_true).astype(int)
    y_pred_rounded = np.round(y_pred).astype(int)

    # 计算每个类别的权重（基于真实值的分布）
    unique_classes, counts = np.unique(y_true_rounded, return_counts=True)
    class_weights = counts / len(y_true_rounded)

    # 创建权重字典
    weight_dict = dict(zip(unique_classes, class_weights))

    # 计算每个样本的权重
    sample_weights = np.array([weight_dict[c] for c in y_true_rounded])

    # 计算加权准确率
    correct = (y_true_rounded == y_pred_rounded).astype(int)
    weighted_acc = np.sum(correct * sample_weights) / np.sum(sample_weights)

    return weighted_acc


# 计算并打印加权准确率
w_acc = weighted_accuracy(y_test, y_xgbr)
print(f"加权准确率: {w_acc:.4f}")

