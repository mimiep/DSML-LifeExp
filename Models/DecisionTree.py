#Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ShuffleSplit, cross_validate, \
    GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#For Prints
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#Read File
file_path = r"../Data_Cleaned.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)
print('Complete DataSet:')
print(df.head())

#Categories
categorical_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing"]

for col in categorical_columns:
    df[col] = df[col].astype("category") #define type category

#print('Categorical:')
#print(df.dtypes)

#Drop: Country, Region, Life_Expectancy,(Year), (Developed/Developing) ?
#Target: Region
X = df.drop(columns=['Region', 'Country', 'Year', 'Economy_status_Developing', 'Adult_mortality', 'Under_five_deaths', 'Infant_deaths', 'Population_mln', 'GDP_per_capita'])
#very little data for a country per year, very easy to predict region based on country, #economy status redundant as there are 2 columns for it, population_million indirect indicator for region?, gdp per capita also indirect indicator for region
y = df['Region']

print("X shape:", X.shape)

print("\n------------------------ check which validation procedure fits data best ------------------------------------------")
print('\n------------------------ 1. Holdout validation (stratified) ------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems accour since data is in a specific order!
)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nHoldout-Testset Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Balanced Accuracy:")
print(round(balanced_accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Classification report:")
print(classification_report(y_test, y_pred))

print("\n------------------------ 2. k-Fold Cross-Validation ------------------------------------------")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
bacc_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')
print(f"5-Fold CV Balanced Accuracy: {bacc_scores.mean():.3f} ± {bacc_scores.std():.3f}")

fold = 1
for train_idx, val_idx in cv.split(X, y):
    # Trainings-/Validierungs-Sets für diesen Fold
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    # Trainieren & Vorhersagen
    clf.fit(X_train_cv, y_train_cv)
    y_pred_cv = clf.predict(X_val_cv)

    # Report ausgeben
    #print(f"\n=== Fold {fold} ===")
    #print(classification_report(y_val_cv, y_pred_cv))
    fold += 1

print("\n--------------------------------------- 3. Monte Carlo Cross-Validation (ShuffleSplit) ------------------------------")
mc_cv = ShuffleSplit(n_splits=10,test_size=0.2) #20% Test-Splitrandom_state=42
results_mc = cross_validate(
    clf, X, y,
    cv=mc_cv,
    scoring=['accuracy','balanced_accuracy'],
    return_train_score=False
)

print("\n=== Monte Carlo CV (10x Random Splits) ===")
print(f"Accuracy (mean ± std):          "
      f"{results_mc['test_accuracy'].mean():.3f} ± {results_mc['test_accuracy'].std():.3f}")
print(f"Balanced Accuracy (mean ± std): "
      f"{results_mc['test_balanced_accuracy'].mean():.3f} ± {results_mc['test_balanced_accuracy'].std():.3f}")


print("\n------------------------------------------------ 4. Principal Component Analysis ------------------------------------------------")
# 1) data scaling important for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) PCA-Instanz erzeugen und fitten
pca_model = PCA()
pca_model = pca_model.fit(X_scaled)

# 4) Scree-Plot
cum_var = np.cumsum(pca_model.explained_variance_ratio_)
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(cum_var)+1), cum_var, marker='o')
plt.xticks(np.arange(1, len(cum_var)+1))
plt.xlabel('Anzahl der Komponenten')
plt.ylabel('kumulative erklärte Varianz')
plt.axhline(0.90, color='gray', linestyle='--', label='90 % Varianz')
plt.legend()
plt.grid(True)
plt.show()


n_components_list = [9, 10, 11, 12]
# - Ganzzahl = fixe Zahl der Komponenten

# CV-Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'balanced_accuracy']

results = []
for n_comp in n_components_list:
    # Pipeline: PCA + Decision Tree
    pca = PCA(n_components=n_comp)
    pipe = Pipeline([
        ('pca', pca),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    # Cross-Validate
    cv_res = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    results.append({
        'n_components': n_comp,
        'Accuracy_mean': cv_res['test_accuracy'].mean(),
        'Accuracy_std': cv_res['test_accuracy'].std(),
        'BalancedAcc_mean': cv_res['test_balanced_accuracy'].mean(),
        'BalancedAcc_std': cv_res['test_balanced_accuracy'].std()
    })

# In DataFrame zusammenfassen und schön formatieren
res_df = pd.DataFrame(results)
res_df['Accuracy_mean'] = res_df['Accuracy_mean'].round(3)
res_df['Accuracy_std'] = res_df['Accuracy_std'].round(3)
res_df['BalancedAcc_mean'] = res_df['BalancedAcc_mean'].round(3)
res_df['BalancedAcc_std'] = res_df['BalancedAcc_std'].round(3)

print("\nSummary after Principal Component Analysis:")
print(res_df)

print("\n------------------------------------------------ Top 10 features ---------------------------------------------")
importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))

# Paarweise Korrelation
#print(X.corr().round(2))

print("\n------------------------------------ Drop most important feature: Life expectancy ------------------------------------------------")

X = X.drop(columns=['Life_expectancy'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems accour since data is in a specific order!
)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nHoldout-Testset Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Balanced Accuracy:")
print(round(balanced_accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Classification report:")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
bacc_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')
print(f"5-Fold CV Balanced Accuracy: {bacc_scores.mean():.3f} ± {bacc_scores.std():.3f}")

importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))

print("\n------------------------------------ Drop second most important feature: Economy_status_Developed ------------------------------------------------")

X = X.drop(columns=['Economy_status_Developed'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems accour since data is in a specific order!
)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nHoldout-Testset Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Balanced Accuracy:")
print(round(balanced_accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Classification report:")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
bacc_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')
print(f"5-Fold CV Balanced Accuracy: {bacc_scores.mean():.3f} ± {bacc_scores.std():.3f}")

importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))

print("\n------------------------------------ Drop schooling to only model health related measures ------------------------------------------------")

X = X.drop(columns=['Schooling'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems accour since data is in a specific order!
)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nHoldout-Testset Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Balanced Accuracy:")
print(round(balanced_accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Classification report:")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
bacc_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')
print(f"5-Fold CV Balanced Accuracy: {bacc_scores.mean():.3f} ± {bacc_scores.std():.3f}")

importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))

print("\n------------------------------------ Drops for comparison with kNN ------------------------------------------------")

X = df.drop(columns=[
    "Country", "Region", "Life_expectancy", "Year",
    "Economy_status_Developed", "Economy_status_Developing"
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems accour since data is in a specific order!
)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nHoldout-Testset Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Balanced Accuracy:")
print(round(balanced_accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Classification report:")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
bacc_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')
print(f"5-Fold CV Balanced Accuracy: {bacc_scores.mean():.3f} ± {bacc_scores.std():.3f}")

importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))

print("----------------------------------------------------- Try to improve model -------------------------------------------------------")
# 1) Basis-Estimator
base_clf = DecisionTreeClassifier(random_state=42)

# 2) Parameter-Grid definieren
param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
}

# 3) Cross-Validation-Strategie
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4) Grid-Search aufsetzen
grid_search = GridSearchCV(
    estimator=base_clf,
    param_grid=param_grid,
    scoring='balanced_accuracy',   # oder 'accuracy', 'f1_macro', ...
    n_jobs=-1,
    cv=cv,
    verbose=1
)

# 5) Suche auf dem Trainingsset ausführen
grid_search.fit(X_train, y_train)

# 6) Beste Parameter & Performance
print("Bestes Parameter-Set:", grid_search.best_params_)
print("Best Score (Balanced Accuracy):", grid_search.best_score_)

# 7) Finalmodell evaluieren
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("---------------------------------------------------- show confusions between regions and countries ----------------------------------")
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

#plot confision matrix -> check which regions are confused
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax)   # verwendet default-Farbskala
plt.xticks(rotation=45, ha='right')
plt.title("Confusion Matrix der Regionen")
plt.tight_layout()
plt.show()


#show which countries get confused
countries = df["Country"]
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
    X, y, countries,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4) Modell trainieren und vorhersagen
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 5) DataFrame aller Test-Beispiele mit Country, true, pred
df_test = pd.DataFrame({
    "Country": c_test,
    "TrueRegion": y_test,
    "PredRegion": y_pred
})

# 6a) Alle Fehlklassifikationen anzeigen
mis = df_test[df_test["TrueRegion"] != df_test["PredRegion"]]
print("Fehlklassifizierte Länder:")
print(mis)

# 6b) Häufigkeiten pro (TrueRegion, PredRegion)-Paar
conf_pairs = (
    mis
    .groupby(["TrueRegion","PredRegion"])["Country"]
    .apply(list)
    .reset_index(name="Countries")
)
print("\nÜbersicht, welche Länder zwischen welchen Regionen verwechselt werden:")
print(conf_pairs)