#---------------Data Cleaning-------------------

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For Print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#-----------------------------------------------------

#Read File
file_path = r"../Data.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)
print('Complete DataSet:')
print(df.head())
print()

#-> no skipfooter or skiprow necessarry

#-----------------------------------------------------

#Print Datatypes
print('Datatypes:')
print(df.dtypes)
print()

#-----------------------------------------------------

#Print Shape
print('Shape:')
print(df.shape)
print()

#-----------------------------------------------------

#Missing Values
missing_values = ["-999", "NaN", "", "nan", "NULL", "null", "na", "missing"]

critical_columns = [
    "Country", "Region", "Year", "Infant_deaths", "Under_five_deaths",
    "Adult_mortality", "Hepatitis_B", "Measles", "BMI", "Polio", "Diphtheria",
    "Incidents_HIV", "GDP_per_capita", "Population_mln",
    "Thinness_ten_nineteen_years", "Thinness_five_nine_years",
    "Schooling", "Life_expectancy"
]

print("Before:")
print(df.isna().sum())
print()

df.replace(missing_values, np.nan, inplace=True)

for col in critical_columns:
    if df[col].dtype != object:
        df[col] = df[col].replace(0, np.nan)

print("After:")
print(df.isna().sum())
print()

#-> there were and are no missing values in the dataset

#-----------------------------------------------------

#Find duplicates
print("Duplicates:")
duplicates = df[df.duplicated(keep=False)]  # zeigt alle Duplikate, nicht nur die zweiten Vorkommen
print(duplicates)

df_cleaned = df.drop_duplicates()

#-> no duplicates found

#-----------------------------------------------------

#Categories
categorical_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing"]

for col in categorical_columns:
    df[col] = df[col].astype("category")

print('Categorical:')
print(df.dtypes)
print()

#-> no duplicates found

#SPEICHERN
output_path = r"../Data_Cleaned.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"Daten wurden erfolgreich gespeichert unter: {output_path}")

#-----------------------------------------------------