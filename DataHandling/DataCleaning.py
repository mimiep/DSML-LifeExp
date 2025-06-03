# ============================================================
#                     DATA CLEANING
# ============================================================

# Goal: Prepare the dataset for analysis by handling missing values, duplicates, and data types.

# ------------------ Imports ------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Adjust Print Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ------------------ Load Dataset ------------------

file_path = r"Data.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)
print('Complete DataSet:')
print(df.head())
print()

#Conclusion:
#   -> no skipfooter or skiprow necessarry

# ------------------ Basic Dataset Info ------------------

#Print Datatypes
print('Datatypes:')
print(df.dtypes)
print()

#Print Shape
print('Shape:')
print(df.shape)
print()

# ------------------ Handle Missing Values ------------------

#Define indicators for missing data
missing_values = ["-999", "NaN", "", "nan", "NULL", "null", "na", "missing"]

#Columns where missing values are likely
critical_columns = [
    "Country", "Region", "Year", "Infant_deaths", "Under_five_deaths",
    "Adult_mortality", "Hepatitis_B", "Measles", "BMI", "Polio", "Diphtheria",
    "Incidents_HIV", "GDP_per_capita", "Population_mln",
    "Thinness_ten_nineteen_years", "Thinness_five_nine_years",
    "Schooling", "Life_expectancy"
]

print("Missing Values BEFORE cleaning:")
print(df.isna().sum())
print()

# Replace missing indicators with np.nan
df.replace(missing_values, np.nan, inplace=True)

#Replace zeros in critical numerical columns with np.nan
for col in critical_columns:
    if df[col].dtype != object:
        df[col] = df[col].replace(0, np.nan)

print("Missing Values AFTER cleaning:")
print(df.isna().sum())
print()

#Conclusion:
#   -All known missing value placeholders have been replaced.
#   -> there were and are no missing values in the dataset

# ------------------ Check for Duplicates ------------------

#Find duplicates
print("Duplicates:")
duplicates = df[df.duplicated(keep=False)]   #shows duplicates
print(duplicates)

# Drop any duplicates just in case
df_cleaned = df.drop_duplicates()

#Conclusion:
#   -> No duplicate rows found

# ------------------ Convert to Categorical ------------------

#Categories
categorical_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing"]

for col in categorical_columns:
    df[col] = df[col].astype("category")

print('Categorical:')
print(df.dtypes)
print()

#Conclusion:
#   Converting relevant columns to 'category'
#   -> Improves Memory Efficiency and simplifies later analysis

# ------------------ Save Cleaned Dataset ------------------
# Save as CSV
output_path = r"Data_Cleaned.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"Daten wurden erfolgreich gespeichert unter: {output_path}")

# Save as Excel
output_path = r"Data_Cleaned.xlsx"
(df_cleaned.to_excel(output_path, index=False))
print(f"Daten wurden erfolgreich gespeichert unter: {output_path}")

# ------------------ Class Distribution: Region ------------------

print("Class Distribution of the Target Variable 'Region'")

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Region", order=df["Region"].value_counts().index, palette="viridis")
plt.title("Distribution of Countries by Region", fontsize=14)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Number of Records", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

region_counts = df["Region"].value_counts()
print("\nRegion Distribution (Number of Rows):")
print(region_counts)

# ------------------ Final Conclusion ------------------
# The dataset has been successfully cleaned:
#   - Missing values replaced
#   - No duplicates
#   - Categorical data converted
#   - Cleaned dataset saved in both CSV and Excel formats
#   - A lot of records of Africa, and few of North America