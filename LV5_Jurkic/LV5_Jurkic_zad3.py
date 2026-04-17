import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

data = pd.read_csv("occupancy_processed.csv")
data.columns = data.columns.str.strip()

print("Stupci:", data.columns)

X = data.iloc[:, 0:2]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

y_pred = dt.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Stablo odlucivanja")
plt.show()

print("Točnost:", accuracy_score(y_test, y_pred))
print("\nIzvjesce:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True)
plt.title("Stablo odlucivanja")
plt.show()

print("\nUtjecaj max_depth:")

for depth in [1, 2, 3, 5, 10]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_d = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred_d)
    print(f"max_depth={depth}, accuracy={acc}")

print("\nBez skaliranja:")

dt_no_scale = DecisionTreeClassifier(random_state=42)
dt_no_scale.fit(X_train, y_train)

y_pred_no = dt_no_scale.predict(X_test)

print("Točnost bez skaliranja:", accuracy_score(y_test, y_pred_no))
print("\nIzvjesce:")
print(classification_report(y_test, y_pred_no))