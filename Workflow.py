import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

'''
========================================
            INTRODUCTION
========================================

In this project, we aim to predict the Region a country belongs to, based solely on various health, economic, and demographic indicators. By analyzing patterns in national statistics, our goal is to classify countries into their corresponding regions, even when the country name is unknown or hidden.

The dataset used for this task is publicly available on Kaggle:
https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated

It includes data from 179 countries, collected between 2000 and 2015, and consists of 21 variables sourced from international organizations such as the World Health Organization (WHO), the World Bank, and the University of Oxford.

### Problem Definition 

Classification:
    -Target: Region
    -Goal: Predict the region a country belongs using provided features

Models used: 
    -kNN
    -Decision Tree
    
    For deeper insight:
    -Clustering 
'''

# %%
'''
========================================
            DATA CLEANING
========================================
In this section, we perform a thorough data cleaning process to ensure the quality and consistency of the dataset before further analysis.

### Step-by-Step Explanation:

    1. Imports & Reading the Dataset 
    2. Handling Missing Values
    3. Duplicate Detection
    4. Categorical Conversion
    5. Saving Cleaned Data

    -> more details are in the document itself 
'''

exec(open("DataHandling/DataCleaning.py").read())


'''
Conclusion:

The dataset is now clean, consistent, and ready for analysis. All missing values and potential anomalies (e.g., zero entries in critical columns) were handled appropriately, and no duplicates were found. Categorical variables were properly formatted, and the cleaned data has been saved for the next steps in the workflow.

'''

# %%
'''
========================================
            DATA EXPLORATION
========================================
### Explanation: 
    This document focuses on understanding patterns, relationships, and anomalies within the cleaned dataset. 
    We explore temporal trends in life expectancy, investigate feature correlations, detect outliers, and analyze 
    regional and global patterns across key health and socioeconomic indicators.
    
    The main goal is/was to gain insights that inform feature selection, modeling strategies, and hypothesis formulation.
    
    Visualizations include heatmaps, scatter plots, boxplots, time series, and cluster analysis.
    
        -> more details are in the document itself 
'''

exec(open("DataHandling/DataExploration.py").read())

'''
Conclusion: 
    - Life expectancy generally increased from 2000 to 2015, but a few countries showed a decline
    - Regional trends reveal strong disparities
    - Strong positive correlations were observed between Life Expectancy and variables such as Schooling, GDP per Capita, and low HIV incidence
    - Highly correlated variables (e.g., Infant_deaths and Under_five_deaths) may be redundant and suitable for dimensionality reduction
    - Temporal feature analysis showed that Schooling and Alcohol consumption changed notably in several countries and could influence regional classification
    
    These findings directly support our modeling goal: predicting a country's Region based on health, demographic, and economic data using classification techniques.
'''


'''
========================================
                Models
========================================
'''

# %%
'''
---------------Clustering---------------

Explain: 
Clustering was used to explore hidden structures and similarities between countries, independent of their official regional labels.
The main goal shifted from "predicting regions" to "understanding latent groupings" based on health, demographic, and economic indicators.

We tested:
k-Means with the number of clusters equal to the number of known regions
Agglomerative hierarchical clustering
Explorative clustering with k = 5 (based on silhouette score and elbow method)

Evaluation Metrics:
Adjusted Rand Index (ARI): measured alignment with true region labels
Silhouette Score: assessed internal cluster cohesion
Confusion matrices and cluster profiles: used for interpretation

Findings:
Region-based clustering produced mixed results (e.g., Africa and EU were well grouped, Oceania poorly)
Hierarchical clustering performed similarly, with no clear improvement
Clustering with k = 5 revealed clearer and more interpretable clusters with distinct feature profiles

'''

exec(open("Models/Clustering.py").read())

'''
Conlusion:
Clustering was not well-suited for directly predicting region labels, but it provided valuable insights into shared health and development patterns across countries.
Using k = 5, we identified interpretable and diverse clusters, such as:
A cluster with high life expectancy, high schooling, high vaccination coverage, and low infant mortality
A cluster with very high adult mortality, high HIV incidence, and low life expectancy
A cluster with high BMI and education, but low alcohol use and malnutrition
A cluster marked by low schooling, high infant deaths, and very large population sizes
And one with high malnutrition indicators and low HIV and BMI levels

These clusters illustrate that countries across different geographic regions can share similar health and development profiles.

'''

# %%
'''
---------------Decision Tree---------------

Explain: 
A classification tree was used to predict the region of a country based on several health and nutrition indicators.
Several techniques were used to find an optimal decision tree:
    Comparison of different validation techniques
    PCA
    Feature selection
    Hyperparameter tuning
Finally, false classification results were shown with the help of a confusion matrix

'''

exec(open("Models/DecisionTree.py").read())

'''
Conlusion:

It was possible to classify the region with an accuracy of well above 90%
    The model containing most of the original features shows a balanced accuracy of 95%
    Even after dropping important features, the balanced accuracy was 93% (only using health related features)
After PCA, the performance got worse (even with a large number of componentes), therefore it was not used further
Hyperparameter tuning showed that the default parameters were already the best choice 

Comparison with kNN: for health related features only, the accuracy was higher with 95% (compared to 90%)
    However, when using more features, the difference in accuracy was very little which shows that the performance of a model is highly dependent on the data

The classification report shows strong performance across most regions:
- recall for Asia is 90%
- recalls above 95% for all other regions
- Precision and F1-scores are also high, indicating that the model performs well in both identifying and correctly labeling the regions.
'''
# %%
'''
---------------kNearestNeighbour---------------

Explain: 
We used a K-Nearest Neighbors (KNN) classifier to predict the region of a country based on several health and nutrition indicators.
KNN is a simple algorithm that classifies a sample based on the majority class of its 'k' closest neighbors in the feature space.
We used 5-fold stratified cross-validation to select the best value of 'k'. This method splits the data into 5 parts while keeping the class distribution consistent in each fold.
Based on the cross-validation plot, we chose k = 5. This value provided a good balance between model complexity and accuracy.

Model Evaluation:
The mean cross-validation accuracy was about 89.8%, with a low standard deviation, showing consistent performance across folds.
On the test set, we achieved a final accuracy of 90%.

The classification report shows strong performance across most regions:
- Very high recall for regions like Africa (98.77%), European Union (96.51%) and Middle East (100%).
- Slightly lower recall for South America (71.05%) and Rest of Europe (79.17%), possibly due to overlapping features between countries like Austria and Switzerland.
- Precision and F1-scores are also high, indicating that the model performs well in both identifying and correctly labeling the regions.

'''

exec(open("Models/kNearestNeighbour.py").read())

'''
Conlusion:
KNN is a strong baseline model for this classification task, reaching 90% accuracy with relatively simple logic.
It performs particularly well for distinct regions with strong feature signals, but struggles more when regions are similar in their indicators.
Overall, KNN offers good interpretability and decent performance, making it a useful method to start exploring the regional classification problem.

'''

'''
========================================
                Deployment
========================================

- Try to interpret your model from the domain expert view. Is the model good
enough?  What needs to be monitored after deployment?

    ->Presentation

- Discuss model deployment from a societal point of view. What does it mean for
society if such a model would be deployed at large scale? Are there any dangers?

    ->It could help policymakers identify development needs or allocate resources more fairly by focusing on indicators rather than national identity.
    However, there are also dangers: the model might reinforce existing biases, oversimplify complex social realities, or lead to decisions based solely on data without considering local context.

'''

'''
========================================
            Presentation
========================================

Also added in the upload.
'''

