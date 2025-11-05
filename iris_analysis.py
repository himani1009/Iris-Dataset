# iris_analysis.py — Himani Agarwal (PCE23AD024)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = sns.load_dataset("iris")
print(df.head())

# Quick EDA plots (optional when running locally)
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
plt.title("Petal Length vs Width by Species")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Feature Correlations")
plt.show()

# Split
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM-Linear": SVC(kernel="linear", random_state=42),
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results.append((name, acc))
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, pred))

best_name, _ = max(results, key=lambda t: t[1])
best_model = models[best_name]
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nModel Accuracies:")
for n, a in results:
    print(f"{n}: {a:.4f}")
