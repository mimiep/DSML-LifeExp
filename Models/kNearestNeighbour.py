# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

# For Print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Read File
file_path = r"../Data_Cleaned.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)

# Drop unwanted columns immediately
columns_to_drop = [
    "Country", "Region", "Life_expectancy", "Year",
    "Economy_status_Developed", "Economy_status_Developing"
]
y = df["Region"].astype("category")
df_knn = df.drop(columns=columns_to_drop)

# Zielvariable-Verteilung
print("\nVerteilung der Zielvariable (Regionen) im Gesamt-Datensatz:")
print(y.value_counts())
y.value_counts().plot(kind='bar', title="Region Distribution in Full Dataset")
plt.xlabel("Region")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.show()

# Skalieren
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_knn)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=123, stratify=y
)

# KNN mit bestem k
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')

print("\nCross-Validation Ergebnisse:")
print("Accuracy pro Fold:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Standardabweichung:", np.std(scores))

# Plot zur k-Auswahl
k_values = range(1, 21)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
    cv_scores.append(np.mean(scores))

plt.plot(k_values, cv_scores, marker='o')
plt.title("KNN Cross-Validation Accuracy für verschiedene k-Werte")
plt.xlabel("k")
plt.ylabel("Mean Accuracy (5-fold CV)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Modell trainieren und testen
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=y.cat.categories, yticklabels=y.cat.categories)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Welche Regionen wurden gar nicht korrekt erkannt?
unique_regions = set(y.cat.categories)
predicted_regions = set(np.unique(y_pred))
missed_regions = unique_regions - predicted_regions

print("\nRegionen, die im Testset vorhanden waren, aber nie korrekt vorhergesagt wurden:")
for region in missed_regions:
    print("-", region)


# Vergleich tatsächlicher vs. vorhergesagter Verteilungen
test_distribution = pd.Series(y_test.values, name="Actual")
pred_distribution = pd.Series(y_pred, name="Predicted")

compare_df = pd.concat([test_distribution.value_counts().sort_index(),
                        pred_distribution.value_counts().sort_index()], axis=1, keys=["Actual", "Predicted"])
print("\nVerteilung der Klassen im Testset vs. Vorhersage:")
print(compare_df)

compare_df.plot(kind='bar', title="Actual vs. predicted Klassenverteilung")
plt.xlabel("Region")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Klassenspezifischer Recall (Genauigkeit pro Region)
recalls = {label: round(metrics["recall"] * 100, 2) for label, metrics in report.items() if label in y.cat.categories}
print("\nRecall per Region (hitquote):")
for region, recall in recalls.items():
    print(f"{region}: {recall}%")

# Feature Importance Plot
perm_importance = permutation_importance(knn, X_test, y_test, scoring='accuracy', random_state=123)
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 7))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [df_knn.columns[i] for i in sorted_idx])
plt.title("Feature Importance (Permutation Importance)")
plt.xlabel("Mean Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

print("\nTop 5 most important Features from Permutation Importance:")
top_indices = sorted_idx[::-1][:5]
for i in top_indices:
    print(f"{df_knn.columns[i]}: {perm_importance.importances_mean[i]:.4f}")


# t-SNE-Plot zur Visualisierung
tsne = TSNE(n_components=2, perplexity=30, random_state=123)
X_tsne = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame()
tsne_df["x"] = X_tsne[:, 0]
tsne_df["y"] = X_tsne[:, 1]
tsne_df["Region"] = y.values

plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x="x", y="y", hue="Region", palette="tab10", s=60)
plt.title("t-SNE Visualisierung der Regionen")
plt.xlabel("t-SNE Komponente 1")
plt.ylabel("t-SNE Komponente 2")
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


