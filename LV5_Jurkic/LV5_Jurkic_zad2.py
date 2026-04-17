import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

data = pd.read_csv("occupancy_processed.csv")

data.columns = data.columns.str.strip()

print("Stupci:")
print(data.columns)

X = data.iloc[:, 0:2]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print("Točnost:", accuracy_score(y_test, y_pred))
print("\nIzvjesce:")
print(classification_report(y_test, y_pred))

print("\nUtjecaj broja susjeda (K):")
for k in [1, 3, 5, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    print(f"K={k}, accuracy={accuracy_score(y_test, y_pred_k)}")

print("\nBez skaliranja:")

knn_no_scale = KNeighborsClassifier(n_neighbors=5)
knn_no_scale.fit(X_train, y_train)

y_pred_no_scale = knn_no_scale.predict(X_test)

print("Točnost bez skaliranja:", accuracy_score(y_test, y_pred_no_scale))
print("\nIzvjesce (bez skaliranja):")
print(classification_report(y_test, y_pred_no_scale))