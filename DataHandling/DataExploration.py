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

#Read File
file_path = r"../Data_Cleaned.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)
print('Complete DataSet:')
print(df.head())


#Outliers & Plots
def plot_life_expectancy_outliers(df):
    plt.figure(figsize=(10, 4))

    # Histogramm + KDE
    sns.histplot(df['Life_expectancy'], kde=True, color='skyblue')
    plt.title('Verteilung der Lebenserwartung')
    plt.xlabel('Lebenserwartung')
    plt.ylabel('Anzahl / Dichte')
    plt.grid(True)
    plt.show()

    # Boxplot
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=df['Life_expectancy'], color='lightgreen')
    plt.title('Boxplot der Lebenserwartung')
    plt.grid(True)
    plt.show()


plot_life_expectancy_outliers(df)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Year', y='Life_expectancy', hue='Region', alpha=0.7)
plt.title('Lebenserwartung nach Jahr und Region')
plt.xlabel('Jahr')
plt.ylabel('Lebenserwartung')
plt.grid(True)
plt.show()

#-----------------------------------------------------
#Outliers (declining Life Expectancy)
df_sorted = df.sort_values(by=["Country", "Year"])

df_sorted["Life_expectancy_diff"] = df_sorted.groupby("Country")["Life_expectancy"].diff()

decline_df = df_sorted[df_sorted["Life_expectancy_diff"] < 0]

print("Länder mit sinkender Lebenserwartung:")
print(decline_df[["Country", "Year", "Life_expectancy", "Life_expectancy_diff"]])

#-----------------------------------------------------
#Outliers (declining Life Expectancy)

#2000 vs 2015
first = df_sorted.groupby("Country").first()
last = df_sorted.groupby("Country").last()

life_change = last["Life_expectancy"] - first["Life_expectancy"]
decline_countries = life_change[life_change < 0]

print("\nLänder, bei denen die Lebenserwartung insgesamt gesunken ist:")
print(decline_countries)


