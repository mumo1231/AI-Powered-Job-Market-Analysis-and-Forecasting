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

