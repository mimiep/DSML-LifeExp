# ============================================
# K-Nearest Neighbors (KNN) Classification
# ============================================

"""
In this section, we apply the K-Nearest Neighbors (KNN) algorithm to classify countries
based on health-related indicators. KNN is a simple, non-parametric classifier that
assigns labels based on the majority vote of the nearest data points.
Its interpretability and effectiveness make it a suitable choice for our multiclass classification task.
"""

# --------------------------------------------------
# Import required libraries for data handling, modeling, and visualization
# --------------------------------------------------
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

# --------------------------------------------------
# Improve DataFrame print readability (for development/debugging)
# --------------------------------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# --------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------
# The dataset was preprocessed and cleaned in a separate script (data_cleaning.py).
# We now load the cleaned version for further modeling steps.
file_path = r"Data_Cleaned.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)

# --------------------------------------------------
# Drop irrelevant or redundant columns
# --------------------------------------------------
# To allow a direct comparison with our Decision Tree model,
# we drop the same set of features here.
#
# Some features like 'Country', 'Region', 'Year', 'Life_expectancy',
# 'Economy_status_Developed', and 'Economy_status_Developing'
# were immediately excluded after a group discussion, as they either
# leak target information or are not informative for prediction.
#
# The remaining features (e.g. mortality, GDP, schooling, etc.)
# were identified as less helpful or redundant based on results
# from data exploration (data_exploration.py) and various experiments.

'''
columns_to_drop = [
    "Country", "Region", "Life_expectancy", "Year",
    "Economy_status_Developed", "Economy_status_Developing"
]
'''
columns_to_drop = [
    "Country", "Region", "Life_expectancy", "Year",
    "Economy_status_Developed", "Economy_status_Developing",
    "Adult_mortality", "Infant_deaths", "Under_five_deaths",
    "Population_mln", "GDP_per_capita", "Schooling"
]

# Set target and features
y = df["Region"].astype("category")
df_knn = df.drop(columns=columns_to_drop)

# --------------------------------------------------
# Distribution of the target variable (Region)
# --------------------------------------------------
# Here we inspect the distribution of our target variable 'Region'.
#
# As seen below, the class distribution is imbalanced. The 'Africa'
# region clearly dominates the dataset, which is important for model
# interpretation later.
#
# This also explains why we observe higher classification performance
# for 'Africa' in the final results — the model has more data to learn
# from for that region.
print("\nDistribution of the target variable (Region) in the dataset:")
print(y.value_counts())
y.value_counts().plot(kind='bar', title="Region Distribution in Full Dataset")
plt.xlabel("Region")
plt.ylabel("Amount")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Feature Scaling and Train/Test Split
# --------------------------------------------------
# We apply MinMax scaling to normalize all feature values to the [0, 1] range.
#
# This is crucial for KNN, which is a distance-based algorithm and sensitive
# to the scale of the input features. Without normalization, features with
# larger value ranges would dominate the distance calculation.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_knn)

# We then split the data into training and test sets.
#
# Stratified sampling ensures the class distribution of 'Region'
# is preserved in both sets, which is important due to the imbalance
# noted earlier.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=123, stratify=y
)

# KNN with chosen best k

# In order to evaluate the performance of our K-Nearest Neighbors (KNN) classifier, we used 5-fold stratified cross-validation.
# Cross-validation is a robust method to assess the generalization ability of a model by partitioning the data into several subsets (folds).
# The model is trained on k-1 folds and validated on the remaining fold. This process is repeated k times, and the results are averaged.
# We chose stratified cross-validation to ensure that each fold contains approximately the same class distribution as the original dataset.
# This is especially important for imbalanced datasets to avoid biased performance estimates.

# The plot generated below suggests that k = 1 yields the highest cross-validation accuracy.
# However, k = 1 can be highly sensitive to noise in the data and prone to overfitting.
# Since the mean accuracy only decreases slightly for k = 5 and remains relatively stable,
# we selected k = 5 as a more balanced and robust choice for the final model.

best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')

print("\nCross-Validation Results:")
print("Accuracy per Fold:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Standard Deviation:", np.std(scores))

# Plotting the mean accuracy for different k values to visualize the selection process
k_values = range(1, 21)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
    cv_scores.append(np.mean(scores))

plt.plot(k_values, cv_scores, marker='o')
plt.title("KNN Cross-Validation Accuracy for Different k Values")
plt.xlabel("k")
plt.ylabel("Mean Accuracy (5-fold CV)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Training and evaluating the final KNN model on the test set
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# Confusion Matrix Analysis

# A confusion matrix is a useful tool to evaluate the performance of classification models.
# It shows the counts of true positive, true negative, false positive, and false negative predictions for each class.
# This detailed breakdown helps us understand not only the overall accuracy but also how the model performs for each individual class.
# In real-world applications, confusion matrices are widely used to identify which classes are often confused by the model,
# helping to diagnose issues like class imbalance or systematic misclassification.

# Here, we visualize the confusion matrix as a heatmap to easily identify which regions (classes) are predicted correctly
# and which are frequently mistaken for others. This provides deeper insight into the strengths and weaknesses of our KNN model.

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=y.cat.categories, yticklabels=y.cat.categories)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# Comparison of Actual vs. Predicted Class Distributions

# This section compares the distribution of the target classes in the test dataset with the distribution
# of the classes predicted by the model. While the confusion matrix shows detailed prediction results for each instance,
# this aggregated comparison gives a simpler overview of how well the model preserves the overall class proportions.

# By plotting the actual and predicted class counts side by side, we can visually inspect if the model
# tends to over- or under-predict certain regions. This helps to detect potential bias or imbalance in predictions.

test_distribution = pd.Series(y_test.values, name="Actual")
pred_distribution = pd.Series(y_pred, name="Predicted")

compare_df = pd.concat([test_distribution.value_counts().sort_index(),
                        pred_distribution.value_counts().sort_index()], axis=1, keys=["Actual", "Predicted"])
print("\nVerteilung der Klassen im Testset vs. Vorhersage:")
print(compare_df)

compare_df.plot(kind='bar', title="Actual vs. predicted Class distribution")
plt.xlabel("Region")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.show()

# Classification Report

# The classification report provides a detailed summary of the model's performance for each class,
# including precision, recall, f1-score, and support.
#
# - Precision measures the accuracy of positive predictions for each class.
# - Recall (also called sensitivity) measures how many actual positives were correctly identified.
# - The F1-score is the harmonic mean of precision and recall, balancing the two.
#
# This report helps to evaluate the strengths and weaknesses of the model in distinguishing between regions,
# especially useful when class distributions are imbalanced.
print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Recall per Region (Hit Rate)

# Recall represents the proportion of actual instances of each class (region) that were correctly identified by the model.
# It is an important metric especially for imbalanced datasets because it shows how well the model detects each specific class.
#
# Reporting recall per region helps to identify which regions the model performs well on and which it struggles with,
# providing insights for potential improvements or further investigation.
recalls = {label: round(metrics["recall"] * 100, 2) for label, metrics in report.items() if label in y.cat.categories}
print("\nRecall per Region (hitquote):")
for region, recall in recalls.items():
    print(f"{region}: {recall}%")

# Feature Importance using Permutation Importance

# Permutation importance measures the impact of each feature on the model's accuracy by randomly shuffling feature values
# and observing the decrease in performance. Features causing a larger drop are considered more important.
#
# This method provides an intuitive way to understand which features the KNN model relies on most for classification,
# helping to interpret the model and guide potential feature selection or engineering.
#
# The plot visualizes the importance scores for all features, while the printed list highlights the top 5 most influential features.
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


# t-SNE Plot for Visualization
# t-SNE is a dimensionality reduction technique that projects high-dimensional data into two dimensions,
# allowing us to visually inspect how well the different classes (regions) separate from each other.
# In this plot, we can see clusters corresponding to different regions. However, some regions,
# like 'European Union' and 'Rest of Europe', often overlap significantly.
# This overlap occurs because countries such as Austria and Switzerland share very similar feature profiles,
# making them hard to distinguish in the feature space.
# Such overlaps reflect the real-world similarity between these regions and explain why the model might find
# it challenging to clearly separate them.
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


