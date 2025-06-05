#Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ShuffleSplit, cross_validate, \
    GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''
In the following script, a decision tree (classification) is trained to predict a geographical region based on health and economical data of countries
data and performance evaluation is done based on holdout and cross validation (Monte Carlo and k-Fold)
It will be tested which features are most influencial, if Principal Component Analysis is useful and whether Hyperparameter optimization
can improve the model performance
The performance will also be compared with kNN as both are supervised models and can therefore be compared with the same performance measures
It will be tested if the model can still perform well when important features are dropped
Classification errors will be examined to clarify possible causes
'''

#For Prints: the following 4 columns make sure that the console outputs are readable more easily
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#Read File: Kaggle data was downloaded as *.csv file, therefore the read_csv function from pandas was used:
#Please note that data exploration and cleaning was already done in the DataHandling section. Now, the cleaned csv is imported:
file_path = r"Data_Cleaned.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)
print('Complete DataSet:')
print(df.head())

#Categororical columns:
categorical_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing"]
for col in categorical_columns:
    df[col] = df[col].astype("category") #define type category

#print('Categorical:')
#print(df.dtypes)

#Drop features:
'''
Country: avoid data leakage... model could simply "remember" that Austria belongs to European Union
Year: there are only a few entries per country and year. Therefore, it is assumed that there is not enough data.
Economy_status_Developing: there is also another feature Economy_status_Developed which is simply the opposisite of Economy_status_Developing (redundant information if both features are used for training)
Adult_mortality, Under_five_deaths, Infant_deaths: those are assumed to give a clear hint for the region, therefore they are dropped as well (although Economy_status_Developing is also a clear hint but not dropped from the beginning but it will be dropped later on to see how the model performance is influenced by it)
Population_mln and GDP_per_capita: also assumed that they give away too much info about the region
'''
X = df.drop(columns=['Region', 'Country', 'Year', 'Economy_status_Developing', 'Adult_mortality', 'Under_five_deaths', 'Infant_deaths', 'Population_mln', 'GDP_per_capita'])
y = df['Region'] #Target: Region

print("X shape:", X.shape)

'''
It will be tested how different validation techniques influence the model performance
Since decision trees easily overfit, it is assumed that cross validation will be the preferred option over holdout validation (and resubstitution will not be considered because of the high overfitting risk)
'''

print("\n------------------------ 1. check which validation procedure fits data best ------------------------------------------")
print('\n------------------------ 1.1. Holdout validation (stratified) ------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, #20% of the data will be used for performance evaluation (which are approx. 600 entries)
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems occur since data is in a specific order! E.g. it would be problematic if the model is trained only based on south american countries and tested on asian countries
)
'''
For the decision tree below, the default parameters are used:
max_depth=None (tree can become very large, therefore it has to be checked if the tree overfits the data)
min_samples_split=2
min_samples_leaf=1
max_leaf_nodes=None
min_impurity_decrease=0.0 (another split is made under the condition that the gini impurity decreases)
Later on in this script, hyperparemeters will be tested to see whether the performance can be improved
'''
clf = DecisionTreeClassifier(random_state=42) #default: gini index is used for identifying tree nodes
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) #predict labeled test data
print("\nHoldout-Testset Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Balanced Accuracy:")
print(round(balanced_accuracy_score(y_test, y_pred), 3))
print("\nHoldout-Testset Classification report:")
print(classification_report(y_test, y_pred))

print("\n------------------------ 1.2. k-Fold Cross-Validation ------------------------------------------")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
bacc_scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')
print(f"5-Fold CV Balanced Accuracy: {bacc_scores.mean():.3f} ± {bacc_scores.std():.3f}")

fold = 1
for train_idx, val_idx in cv.split(X, y):
    #training and validation set for the current fold:
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    #Train (training data) and predict (tast data):
    clf.fit(X_train_cv, y_train_cv)
    y_pred_cv = clf.predict(X_val_cv)

    #Print report (currently not done because lots of information is printed in console already)
    #print(f"\n=== Fold {fold} ===")
    #print(classification_report(y_val_cv, y_pred_cv))
    fold += 1

print("\n--------------------------------------- 1.3. Monte Carlo Cross-Validation (ShuffleSplit) ------------------------------")
mc_cv = ShuffleSplit(n_splits=5, test_size=0.2) #20% Test
results_mc = cross_validate(
    clf, X, y,
    cv=mc_cv,
    scoring=['accuracy','balanced_accuracy'],
    return_train_score=False
)

print(f"Accuracy (mean ± std):          "
      f"{results_mc['test_accuracy'].mean():.3f} ± {results_mc['test_accuracy'].std():.3f}")
print(f"Balanced Accuracy (mean ± std): "
      f"{results_mc['test_balanced_accuracy'].mean():.3f} ± {results_mc['test_balanced_accuracy'].std():.3f}")

'''
Based on the performance evaluation results above, one can see:
Accuracy for Holdout validation in slightly higher than Cross validation (but very little difference)
Accuracy for k-Fold and Monte Carlo Cross Validation are almost the same
First conclusion:
    the decision tree seems to be a good classification model for the data
    as cross validation has an accuracy above 95% (because of CV, overfitting should also not play a huge role for this dataset)

'''

print("\n------------------------------------------------ 2. Principal Component Analysis ------------------------------------------------")
'''
Here, PCA is tried for dimensionality reduction:
It will be checked how much of the performance is lost by conducting cross validation after applying PCA
'''
#data scaling important for PCA:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#create instance and fit X_scaled:
pca_model = PCA()
pca_model = pca_model.fit(X_scaled)

#Scree-Plot for visualisations
cum_var = np.cumsum(pca_model.explained_variance_ratio_)
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(cum_var)+1), cum_var, marker='o')
plt.xticks(np.arange(1, len(cum_var)+1))
plt.xlabel('Number of components')
plt.ylabel('cumulative explained variance')
plt.axhline(0.90, color='gray', linestyle='--', label='90% variance')
plt.legend()
plt.grid(True)
plt.show()

n_components_list = [9, 10, 11, 12] #different number of principal components will be tried

#Cross Validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'balanced_accuracy']

results = []
for n_comp in n_components_list: #loop over number of components
    #Pipeline: PCA + Decision Tree
    pca = PCA(n_components=n_comp)
    pipe = Pipeline([ #used to combine PCA and classificator
        ('pca', pca),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    #Cross-Validate
    cv_res = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1 #parallel computations on all CPU cores
    )

    results.append({ #collect results
        'n_components': n_comp,
        'Accuracy_mean': cv_res['test_accuracy'].mean(),
        'Accuracy_std': cv_res['test_accuracy'].std(),
        'BalancedAcc_mean': cv_res['test_balanced_accuracy'].mean(),
        'BalancedAcc_std': cv_res['test_balanced_accuracy'].std()
    })

#create dataFrame with results
res_df = pd.DataFrame(results)
res_df['Accuracy_mean'] = res_df['Accuracy_mean'].round(3)
res_df['Accuracy_std'] = res_df['Accuracy_std'].round(3)
res_df['BalancedAcc_mean'] = res_df['BalancedAcc_mean'].round(3)
res_df['BalancedAcc_std'] = res_df['BalancedAcc_std'].round(3)

print("\nSummary after Principal Component Analysis:")
print(res_df)

'''
Conclusion:
    Even with 12 components (which is the same dimensionality as before because of 12 features), the Accuracy drops from 95% to 88%
    With fewer number of components, it drops even further: e.g. 82% accuracy for 9 components
    Therefore, it is concluded that PCA is not effective in combination with this dataset and decision trees
    The following steps will not be conducted with the principal components
'''

'''
The next goal is to improve the performance of the model with the help of Hyperparameters.
But since the model performance is already high, some of the features will be dropped (e.g. it is assumed that based on Economy_status_Developed, the model can already predict a lot of regions)
Moreover, it should be checked whether the assumption is true that only some of the features can be used for the classification and which of them have a stronger influence on the result
'''

print("\n------------------------------------------------ 3. Top 10 features ---------------------------------------------")
importances = pd.Series(clf.feature_importances_, index=X.columns) #calculate which features are of importantce
print("\nTop 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10))

#pairwise correclation:
#print(X.corr().round(2))

print("\n------------------------------------ 3.1. Drop most important feature: Life expectancy ------------------------------------------------")

X = X.drop(columns=['Life_expectancy'])

'''
As the classification report shows performance measures within the targets (regions) and it is more handy to use only one classification report (instead of one for every fold),
holdout validation will be used for the classification reports (the tests from above have shown that holdout and cross validation show very similar results but it has to be
kept in mind that holdout validation is generally not the ideal choice)
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems occur since data is in a specific order!
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

'''
The accuracy only drops by approx. 0.5% after dropping the most important feature (Life expectancy)
As Life expectancy was assumed to give major hints for the region, it was shown that the classification results are nearly as good without the feature
'''

print("\n------------------------------------ 3.2. Drop second most important feature: Economy_status_Developed ------------------------------------------------")
'''
This feature was chosen to drop as it also should be relatively easy to rule out some regions simply based on this feature
'''

X = X.drop(columns=['Economy_status_Developed'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems occur since data is in a specific order!
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

'''
Conclusion:
    The accuracy drops by about 1%
    What's interesting is that some of the regions improved while others lost precision
        e.g. after dropping Economy_status_Developed: Classifications for Africa and South america improved while the performance got worse for European Union and Rest of Europe
        and the reason might be the following: regions where the Economy_status_Developed is the same for every country, benefit from dropping the feature
        (e.g. Africa has only Economy_status_Developed = 0 whereas Rest Of Europe has both developed countries and undeveloped ones (they benefit from this additional information))
'''

print("\n------------------------------------ 3.3. Drop schooling to only model health related measures ------------------------------------------------")

'''
Schooling will also be dropped to see wheter the model works solely on health based indicators
'''
X = X.drop(columns=['Schooling'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems occur since data is in a specific order!
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

'''
The accuracy drops by arround 1%
Especially Asia, Oceania, Middle East and South America loose precision which indicates that the countries in those regions have different access to education
'''

print("\n------------------------------------ 3.4. Drops for comparison with kNN ------------------------------------------------")

'''
Here, other features are used to make a direct comparison with the kNN model (those exact features were used for kNN)
'''
X = df.drop(columns=[
    "Country", "Region", "Life_expectancy", "Year",
    "Economy_status_Developed", "Economy_status_Developing"
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True #without shuffle, problems occur since data is in a specific order!
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

'''
Conclusion:
    For those features, the performance of decision tree and kNN is almost the same
    But when using only health related data, the decision tree performed slightly better
        This shows that there is not the one model which is best for all kinds of data and underlines the No free lunch theorem
'''

print("----------------------------------------------------- 4. Try to improve model -------------------------------------------------------")
'''
It should be tried whether the model performance can be improved by trying different Hyperparameters (grid search)
For a compromise between speed and performance evaluation holdout analog is used: cross validation to find the best Hyperparameters and holdout validation to test the performance
'''
#model with default parameters
base_clf = DecisionTreeClassifier(random_state=42)

#parameter grid to try different hyperparameters:
param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2'], #how many features should be considered per split
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1] #pruning parameter -> higher = more pruning
}

#cross validation for performance comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Define grid search:
grid_search = GridSearchCV(
    estimator=base_clf,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=1
)

grid_search.fit(X_train, y_train)

#Show best hyperparameters from grid:
print("Bestes Parameter-Set:", grid_search.best_params_)
print("Best Score (Balanced Accuracy):", grid_search.best_score_)

# 7)evaluate performance based on hyperparameters:
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''
The best hyperparameters from the grid are:
    ccp_alpha': 0.0, 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2
Those are the default values which have been used in the above models. Therefore, the hyperparameter selection did not lead to improved performance
It was confusing at first as the balanced accuracy is now better but this might be due to the fact that .best_estimator_ uses the best model from the loop
Therefore, the default model will be used in the following
'''

print("---------------------------------------------------- 5. show confusions between regions and countries ----------------------------------")
'''
Finally, it should be determined, which regions get confused with each other and which countries cause those confusions
'''
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_) #create confusion matrix

#plot confusion matrix:
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax)   # verwendet default-Farbskala
plt.xticks(rotation=45, ha='right')
plt.title("Confusion Matrix der Regionen")
plt.tight_layout()
plt.show()


#show which countries get confused: create new split including country labels
countries = df["Country"]
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
    X, y, countries,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#train model when country labels are included:
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Create dataFrame:
df_test = pd.DataFrame({
    "Country": c_test,
    "TrueRegion": y_test,
    "PredRegion": y_pred
})

#show all false classified countries:
mis = df_test[df_test["TrueRegion"] != df_test["PredRegion"]]
print("Misidentified countries:")
print(mis)

#show relative frequencies:
conf_pairs = (
    mis
    .groupby(["TrueRegion","PredRegion"])["Country"]
    .apply(list)
    .reset_index(name="Countries")
)
print("\nOverview of which countries are confused between which regions")
print(conf_pairs)

'''
Most of the wrong classifications seem logical after seeing them:
- Most of the false classifications are where the true region is Asia and the predicted regions are Central America and Caribbean
    - happend for the following countries: Pakistan, Brunei Darussalam, Azerbaijan, Brunei Darussalam
- Finland, Bulgaria, Latvia were identified as European Union but they belong to the class Rest Of Europe
    - this assumption turned out to be true as the structure of health access is very similar in central europe even for countries not belonging to the EU
- Some confusions are also because of Asia and Middle East: Pakistan is identified as Middle East as an example

-> the health structure is not solely dependent on the region of a country but it can be shown that countries within regions show very similar structures
'''