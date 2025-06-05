# ============================================================
#                     DATA EXPLORATION
# ============================================================

#Goal: Explore the cleaned dataset to uncover patterns, correlations, outliers, and other maybe meaningful informations related to life expectancy.

# ------------------ Imports ------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Adjust Print Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ------------------ Load Cleaned Dataset ------------------
file_path = r"Data_Cleaned.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)
print('Complete DataSet:')
print(df.head())

#Restore categorical types (because they are not saved)
categorical_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing"]

for col in categorical_columns:
    df[col] = df[col].astype("category")

print('Categorical:')
print(df.dtypes)
print()

#Sort the values
df = df.sort_values(by=["Country", "Year"])

# ===========================================================
#                    EXPLORATION
# ===========================================================

# ------------------ Correlation Matrix ------------------
# Purpose: Identify relationships between features

plt.figure(figsize=(14,10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlationmatrix")
plt.show()

#Finding:
#   - Shows relationships between all numerical features.
#   - For Example
#       -> Under Five death & infant deaths
#       -> Diphteria & Polio
#       -> Adult Mortality & HIV Incidents

# ------------------ Life Expectancy Over Time ------------------
# Purpose: Identify countries with greatest improvement in Life Expectancy

# Compare first and last available life expectancy for each country
first = df.groupby("Country").first()
last = df.groupby("Country").last()

life_change = last["Life_expectancy"] - first["Life_expectancy"]

growth_countries = life_change[life_change > 0].sort_values(ascending=False)

# Print Top 10
print("Countries where life expectancy increased the most (2000–2015):")
for country, change in growth_countries.head(10).items():
    print(f"{country}: +{change:.2f} years")

# Plot top 5 with biggest improvement
top_growth = growth_countries.head(5)

plt.figure(figsize=(10,6))
for country in top_growth.index:
    country_df = df[df["Country"] == country]
    plt.plot(
        country_df["Year"],
        country_df["Life_expectancy"],
        label=f"{country} (+{top_growth[country]:.2f} yrs)"
    )

plt.title("Countries with Largest Increase in Life Expectancy (2000–2015)")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.legend()
plt.grid(True)
plt.show()

#Finding:
#   - This line plot shows the top 5 countries with the greatest improvement


# ------------------ Life Expectancy Decline ------------------
#Purpose: Identify countries with decreased Life Expectancy

#2000 vs 2015
first = df.groupby("Country").first()
last = df.groupby("Country").last()

life_change = last["Life_expectancy"] - first["Life_expectancy"]
decline_countries = life_change[life_change < 0]

print("\nCountries with decreased Life Expectancy:")
print(decline_countries)

#Finding:
#   - Only a few countries experienced a decline in life expectancy.
#   - This may reflect crises like war, pandemics, or systemic health issues.

# ------------------ Regional Trends ------------------
# Purpose: Examine average LE changes by region

region_year = df.groupby(["Region", "Year"])["Life_expectancy"].mean().reset_index()

plt.figure(figsize=(14,8))
sns.lineplot(data=region_year, x="Year", y="Life_expectancy", hue="Region")
plt.title("Average Life Expectancy by Region Over Time")
plt.show()

#Finding:
#   - Regional differences are visible — some regions consistently lag behind others.
#   - Africa and South-East Asia lag behind, though they show upward trends.
#   -> this can show the difference between 2000- 2015 when predicting the region

# ------------------ Boxplots for Numerical Features ------------------
# Purpose: Detect outliers and value distributions

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    plt.grid(True)
    plt.show()

# Finding:
#   - Outliers become visible
#   - HIV incidence and GDP have extreme outliers (but they are still very important)

# ------------------ Life Expectancy vs GDP ------------------
# Purpose: Look at correlation between Life Expectancy and GDP

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="GDP_per_capita", y="Life_expectancy", hue="Region", alpha=0.7)
plt.xscale('log')
plt.title("Life Expectancy vs. GDP per Capita (Log Scale)")
plt.grid(True)
plt.show()

# Finding:
#   - higher GDP per capita correlates with higher life expectancy
#   - Richer countries tend to have longer life expectancy.

# ------------------ Life Expectancy vs HIV ------------------
# Purpose: Look at correlation between Life Expectancy and HIV

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Incidents_HIV", y="Life_expectancy", hue="Region", alpha=0.6)
plt.yscale('linear')
plt.xscale('log')
plt.title("HIV Incidence vs Life Expectancy")
plt.grid(True)
plt.show()

# Finding:
#   - A negative relationship exists.
#   - High HIV incidence tends to correlates with lower life expectancy
#   -> relevant for Africa

# ------------------ Schooling by Region ------------------
# Purpose: Compare education levels across regions

plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="Region", y="Schooling")
plt.title("Years of Schooling by Region")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Finding:
#   - Large disparities between regions when it comes to school years
#   -> may explain life expectancy differences.

# ------------------ Correlation Over Time ------------------
# Purpose: See if LifeExpectancy and GDP is stable

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

# Finding:
#   - GDP is a stable predictor of life expectancy
#   - The correlation remains consistently strong

# ------------------ Top & Bottom Countries (2015) ------------------

year_df = df[df["Year"] == 2015]

top = year_df.sort_values(by="Life_expectancy", ascending=False).head(10)
bottom = year_df.sort_values(by="Life_expectancy").head(10)

plt.figure(figsize=(12,6))
sns.barplot(x="Life_expectancy", y="Country", data=pd.concat([top, bottom]))
plt.title("Top & Bottom 10 Countries by Life Expectancy in 2015")
plt.grid(True)
plt.show()

# Finding:
#   - Conformation: top countries are mostly developed, bottom are not

# ------------------ Clustering ------------------
# Purpose: Group countries by similar features

features = ["GDP_per_capita", "Schooling", "Alcohol_consumption", "Incidents_HIV", "Life_expectancy"]
X = df[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
X["Cluster"] = kmeans.fit_predict(X_scaled)

sns.pairplot(X, hue="Cluster", palette="tab10")
plt.suptitle("Clusters of Countries by Health & Social Indicators", y=1.02)
plt.show()

# Finding:
#   - Countries cluster into meaningful groups
#   -> Clustering becomes a model itself

# ------------------ Feature Change Over Time ------------------
#Purpose: Understand which features changed the most

feature_changes = []

for col in ["Alcohol_consumption", "BMI", "Schooling"]:
    first = df.groupby("Country").first()[col]
    last = df.groupby("Country").last()[col]
    diff = (last - first).abs().sort_values(ascending=False)
    feature_changes.append((col, diff.mean()))

print("Average Feature Change (2000–2015):")
for name, mean_change in feature_changes:
    print(f"{name}: {mean_change:.2f}")

# Finding:
#   - Some features shows improvements over time
#   - Helps assess whether development trends are improving.

# ------------------ Redundant or Low-Variance Features ------------------
# Same as the Correlationmatrix: Find out which features you can drop and which don't

sns.scatterplot(data=df, x="Infant_deaths", y="Under_five_deaths")
plt.title("Infant Deaths vs Under-5 Deaths")
plt.show()

print("Correlation:", df["Infant_deaths"].corr(df["Under_five_deaths"]))

# Finding:
#   - Very high correlation - Drop feature??

#Features with Low Variance (e.g., Hepatitis_B)
for col in df.columns:
    if df[col].dtype != "object" and df[col].nunique() < 5:
        print(f"{col}: {df[col].value_counts(normalize=True).round(2)}\n")

# Finding:
#   - Low-variance features (may not contribute much to modeling)

# ------------------ Additional Visual Insights ------------------


# KDE Plot: GDP vs Life Expectancy
sns.jointplot(data=df, x="GDP_per_capita", y="Life_expectancy", kind="kde", fill=True)
plt.suptitle("KDE Plot: GDP vs Life Expectancy", y=1.05)
plt.show()

# Pairplot of Selected Features
sns.pairplot(df[["Life_expectancy", "GDP_per_capita", "Schooling", "Adult_mortality", "Alcohol_consumption"]])
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Joint Histogram: Schooling vs Life Expectancy
sns.jointplot(data=df, x="Schooling", y="Life_expectancy", kind="hist")
plt.suptitle("Histogram: Schooling vs Life Expectancy", y=1.05)
plt.show()


# ============================================================
#                     CONCLUSION
# ============================================================

#Key Findings:
#   - Life Expectancy is strongly correlated with GDP per capita and schooling.
#   - HIV is a major driver of life expectancy decline in certain countries.
#   - Regional trends show disparities in health and education.
#   - Certain features are highly correlated (e.g., Infant vs. Under-5 deaths).
#   - Low-variance features may be excluded from modeling.
#   - Clustering reveals meaningful groupings of countries for classification.
#   - Overall, the dataset is rich in interpretable and predictive features.
