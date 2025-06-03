#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

#Categories (because they are not saved)
categorical_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing"]

for col in categorical_columns:
    df[col] = df[col].astype("category")

print('Categorical:')
print(df.dtypes)
print()

#========================================
#           DATA-Exploration
#=======================================

# Display a heatmap of the correlation matrix for all numerical features
plt.figure(figsize=(14,10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlationmatrix")
plt.show()

# --> Shows which variables are positively or negatively correlated with each other

#-----------------------------------

#Change in Life Expectancy Over Time (2000-2015)
first = df.groupby("Country").first()
last = df.groupby("Country").last()

life_change = last["Life_expectancy"] - first["Life_expectancy"]
decline_countries = life_change[life_change < 0]

print("\nCountries where life expectancy decreased overall:")
print(decline_countries)

top_decline = decline_countries.sort_values().head(5).index.tolist()

for country in top_decline:
    country_df = df[df["Country"] == country]
    plt.plot(country_df["Year"], country_df["Life_expectancy"], label=country)

plt.title("Countries with Strong Declines in Life Expectancy (2000-2015)")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.legend()
plt.grid(True)
plt.show()

#-----------------------------------

#Regional Life Expectancy Trends
region_year = df.groupby(["Region", "Year"])["Life_expectancy"].mean().reset_index()

plt.figure(figsize=(14,8))
sns.lineplot(data=region_year, x="Year", y="Life_expectancy", hue="Region")
plt.title("Average Life Expectancy by Region Over Time")
plt.show()

#-----------------------------------

#Boxplots for Numerical Features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    plt.grid(True)
    plt.show()

#-----------------------------------

#Outliers (declining Life Expectancy)

#2000 vs 2015
first = df.groupby("Country").first()
last = df.groupby("Country").last()

life_change = last["Life_expectancy"] - first["Life_expectancy"]
decline_countries = life_change[life_change < 0]

print("\nLänder, bei denen die Lebenserwartung insgesamt gesunken ist:")
print(decline_countries)

#-----------------------------------

#Scatterplot: Life Expectancy vs GDP
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="GDP_per_capita", y="Life_expectancy", hue="Region", alpha=0.7)
plt.xscale('log')
plt.title("Life Expectancy vs. GDP per Capita (Log Scale)")
plt.grid(True)
plt.show()

#-----------------------------------
#Scatterplot: Life Expectancy vs HIV Incidence
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Incidents_HIV", y="Life_expectancy", hue="Region", alpha=0.6)
plt.yscale('linear')
plt.xscale('log')
plt.title("HIV Incidence vs Life Expectancy")
plt.grid(True)
plt.show()

#-----------------------------------
#Schooling by Region
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="Region", y="Schooling")
plt.title("Years of Schooling by Region")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#-----------------------------------
#Correlation Over Time (GDP vs Life Expectancy)
years = sorted(df["Year"].unique())
corrs = []

for year in years:
    subset = df[df["Year"] == year]
    corrs.append(subset["GDP_per_capita"].corr(subset["Life_expectancy"]))

plt.plot(years, corrs, marker='o')
plt.title("Correlation Between GDP and Life Expectancy Over Time")
plt.xlabel("Year")
plt.ylabel("Correlation Coefficient")
plt.grid(True)
plt.show()

#-----------------------------------
#Top & Bottom Countries in 2015
year_df = df[df["Year"] == 2015]

top = year_df.sort_values(by="Life_expectancy", ascending=False).head(10)
bottom = year_df.sort_values(by="Life_expectancy").head(10)

plt.figure(figsize=(12,6))
sns.barplot(x="Life_expectancy", y="Country", data=pd.concat([top, bottom]))
plt.title("Top & Bottom 10 Countries by Life Expectancy in 2015")
plt.grid(True)
plt.show()

#-----------------------------------
#Clustering Countries Based on Health & Socioeconomic Features

features = ["GDP_per_capita", "Schooling", "Alcohol_consumption", "Incidents_HIV", "Life_expectancy"]
X = df[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
X["Cluster"] = kmeans.fit_predict(X_scaled)

sns.pairplot(X, hue="Cluster", palette="tab10")
plt.suptitle("Clusters of Countries by Health & Social Indicators", y=1.02)
plt.show()

#-----------------------------------

#Feature Stability Over Time
feature_changes = []

for col in ["Alcohol_consumption", "BMI", "Schooling"]:
    first = df.groupby("Country").first()[col]
    last = df.groupby("Country").last()[col]
    diff = (last - first).abs().sort_values(ascending=False)
    feature_changes.append((col, diff.mean()))

for name, mean_change in feature_changes:
    print(f"{name}: Average Change = {mean_change:.2f}")

#-----------------------------------

#Redundant Features (Highly Correlated)
sns.scatterplot(data=df, x="Infant_deaths", y="Under_five_deaths")
plt.title("Infant Deaths vs Under-5 Deaths")
plt.show()

print("Correlation:", df["Infant_deaths"].corr(df["Under_five_deaths"]))

#-----------------------------------
#Features with Low Variance (e.g., Hepatitis_B)
for col in df.columns:
    if df[col].dtype != "object" and df[col].nunique() < 5:
        print(f"{col}: {df[col].value_counts(normalize=True).round(2)}\n")

#-----------------------------------

# KDE Plot: GDP vs Life Expectancy
sns.jointplot(data=df, x="GDP_per_capita", y="Life_expectancy", kind="kde", fill=True)
plt.suptitle("KDE Plot: GDP vs Life Expectancy", y=1.05)
plt.show()

#-----------------------------------

# Pairplot of Selected Features
sns.pairplot(df[["Life_expectancy", "GDP_per_capita", "Schooling", "Adult_mortality", "Alcohol_consumption"]])
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

#-----------------------------------

# Joint Histogram: Schooling vs Life Expectancy
sns.jointplot(data=df, x="Schooling", y="Life_expectancy", kind="hist")
plt.suptitle("Histogram: Schooling vs Life Expectancy", y=1.05)
plt.show()


#========================================
#           Conclusion
#=======================================

# - Strong positive correlations were found between GDP, Schooling, and Life Expectancy.
# - Countries with decreasing life expectancy were mainly affected by HIV and low GDP.
# - Regions show differing trends in health indicators over time.
# - Some features like Infant_deaths and Under_five_deaths are highly correlated and could be redundant.
# - Hepatitis_B and other low-variance features may be excluded from certain analyses.
# - Clustering reveals meaningful groupings of countries by socioeconomic and health data.
