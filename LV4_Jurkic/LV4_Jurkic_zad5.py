import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error


df = pd.read_csv('cars_processed.csv')


if 'name' in df.columns:
    df = df.drop(['name'], axis=1)

print(df.info())


X = df[['km_driven', 'year', 'engine', 'max_power']]  # numeričke ulazne veličine
y = df['selling_price']  # ciljna veličina


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train_s, y_train)


y_pred_train = linear_model.predict(X_train_s)
y_pred_test = linear_model.predict(X_test_s)

print("=== Evaluacija na test skupu ===")
print("R2 test:", r2_score(y_test, y_pred_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Max error test:", max_error(y_test, y_pred_test))
print("MAE test:", mean_absolute_error(y_test, y_pred_test))


plt.figure(figsize=[13,10])
sns.regplot(x=y_pred_test, y=y_test, line_kws={'color':'green'})
plt.xlabel('Predikcija')
plt.ylabel('Stvarna vrijednost')
plt.title('Rezultati na testnim podacima')
plt.show()

# 6. Utjecaj broja ulaznih značajki (primjer)
print("\n=== Utjecaj broja ulaznih značajki ===")
feature_sets = [
    ['km_driven', 'year'], 
    ['km_driven', 'year', 'engine', 'max_power']
]

for features in feature_sets:
    X_small = df[features]
    X_train_s, X_test_s, y_train, y_test = train_test_split(X_small, y, test_size=0.2, random_state=300)
    X_train_s = scaler.fit_transform(X_train_s)
    X_test_s = scaler.transform(X_test_s)
    
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    print(f"Značajke: {features} -> MAE test: {mean_absolute_error(y_test, y_pred):.2f}, RMSE test: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")