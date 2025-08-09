import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from scipy import stats

# 加载和预处理数据
file = "C:\\Users\\21783\\Desktop\\kczy\\jobb.csv"
data = pd.read_csv(file)

# 将分类变量转换为数值类型
cat_cols = [col for col in data.columns if data[col].dtypes == "object"]
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 修正后的参数分布 - 使用 colsample_bytree 替代 max_features
param_dist = {
    'n_estimators': stats.randint(50, 500),
    'max_depth': stats.randint(3, 15),
    'colsample_bytree': stats.uniform(0.1, 0.8)  # 范围0.1-0.9
}

# 创建XGBoost模型
model = XGBRegressor(random_state=42)

# 设置随机搜索参数
n_iter = 50
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=n_iter,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1  # 使用所有CPU核心加速
)

# 执行随机搜索
print("Starting randomized search...")
random_search.fit(X_train, y_train)

# 输出最佳参数
print("\nOptimal Parameters Found:")
best_params = random_search.best_params_
print(f"n_estimators: {best_params['n_estimators']}")
print(f"max_depth: {best_params['max_depth']}")
print(f"colsample_bytree: {best_params['colsample_bytree']:.3f}")

# 评估最终模型
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nTest MAE with best parameters: {mae:.4f}")


