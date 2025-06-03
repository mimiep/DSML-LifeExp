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


'''

exec(open("Models/Clustering.py").read())

'''
Conlusion:



'''

# %%
'''
---------------Decision Tree---------------

Explain: 


'''

exec(open("Models/DecisionTree.py").read())

'''
Conlusion:



'''
# %%
'''
---------------kNearestNeighbour---------------

Explain: 


'''

exec(open("Models/kNearestNeighbour.py").read())

'''
Conlusion:



'''

'''
========================================
                Models
========================================

- Discuss model deployment from a technical point of view

- Try to interpret your model from the domain expert view. Is the model good
enough? What are possible consequences of deploying the model? What needs to
be monitored after deployment?

- Discuss model deployment from a societal point of view. What does it mean for
society if such a model would be deployed at large scale? Are there any dangers?




'''


