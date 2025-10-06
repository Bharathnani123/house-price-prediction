# =====================================
# 1️⃣ Import Libraries
# =====================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================
# 2️⃣ Load Online Dataset (from seaborn)
# =====================================
# 'penguins' or 'diamonds' are small toy datasets — we'll use seaborn's "diamonds" as proxy for house price style
# For actual housing data, sklearn's California Housing dataset is better
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Show first few rows
print(df.head())
print("\nColumns:", df.columns)

# =====================================
# 3️⃣ Explore Data
# =====================================
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic Stats:\n", df.describe())

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# =====================================
# 4️⃣ Define Features and Target
# =====================================
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# =====================================
# 5️⃣ Split Data
# =====================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================
# 6️⃣ Preprocess (Scaling)
# =====================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# 7️⃣ Train Models
# =====================================
# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# =====================================
# 8️⃣ Evaluate Models
# =====================================
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name} Model Evaluation:")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# =====================================
# 9️⃣ Visualize Actual vs Predicted
# =====================================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest - Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# =====================================
# 🔟 Feature Importance (Random Forest)
# =====================================
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.sort_values(ascending=True).plot(kind='barh', figsize=(8,5))
plt.title("Feature Importance (Random Forest)")
plt.show()